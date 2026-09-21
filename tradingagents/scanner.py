"""Live movers scanner — find stocks moving fast on abnormal volume, then
check whether an announcement explains it (2026-08-22, user request).

**Purpose: forward paper testing, not backtesting.** The user wants to watch
this live and place paper trades by hand to build intuition. That is worth
more than another backtest here for a specific reason — **live scanning is
survivorship-free by construction**. Every retrospective study in this project
is contaminated because the universe is *today's* top 500 and delisted names
are absent (see `event_momentum.py`); a scan run at 11am on a Tuesday sees
exactly what was there at 11am on that Tuesday, including the names that later
fail.

**The thresholds come from measured results, not taste** (`event_momentum.py`):
  - move >= +8%: below that, volume carries NO continuation information at all
  - volume >= 3x normal: the gradient is monotone in volume and only the
    3x+ buckets separate from the no-news bucket
  - price 50c-$2 is the only band where the edge exceeded the round-trip tick
    cost, so rows are flagged by band rather than filtered — the user asked to
    see the market, and hiding the untradeable ones would hide the evidence

Rows are **flagged, never suppressed**. `cost_verdict` says whether the
historical edge for that price band survived its spread, so a 3c rocket still
appears and is plainly marked as uneconomic.

**Volume comparison is time-of-day aware.** Completed hourly windows are
compared with matching hours over up to 20 previous sessions, with a minimum
of 10. Outside trading, full-day volume uses a 50-session median.

Data: yfinance daily bars with `period='3mo'`, whose current-day row updates
intraday (delayed ~20 min for ASX). Delay is fine for a multi-day hold — the
measured effect plays out over 3-10 days, not seconds — and it avoids needing
a live market-data subscription on the IBKR gateway.
"""

from __future__ import annotations

import sqlite3
import logging
from zoneinfo import ZoneInfo
from datetime import datetime, time, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from tradingagents.backtest import tick_size
from tradingagents.yf_lock import YF_LOCK
from tradingagents.screener import load_candidates

ASX_DB = "/opt/asxbrief/data/asx.db"
SIGNALS_DB = None  # resolved lazily from asx_signals

# Measured thresholds -- see module docstring.
MIN_MOVE_PCT = 8.0
MIN_VOLUME_RATIO = 3.0
WATCH_MOVE_PCT = 5.0          # shown as "watch", below the measured effect
VOLUME_LOOKBACK_DAYS = 50

# ASX continuous trading, Sydney time.
SESSION_OPEN = time(10, 0)
SESSION_CLOSE = time(16, 0)
SYDNEY = ZoneInfo("Australia/Sydney")

TRADEABLE_BANDS = [
    (0.00, 0.05, "under 5c", "cost > edge (round trip ~10%)"),
    (0.05, 0.20, "5-20c", "cost > edge (round trip ~5.4%)"),
    (0.20, 0.50, "20-50c", "cost > edge (round trip ~3.2%)"),
    (0.50, 2.00, "50c-$2", "edge survived cost in backtest"),
    (2.00, 1e9, "over $2", "cost > edge (edge fades above $2)"),
]


def price_band(price: float) -> tuple[str, str]:
    for lo, hi, label, verdict in TRADEABLE_BANDS:
        if lo <= price < hi:
            return label, verdict
    return "unknown", "unknown"


def session_state(now_utc: datetime | None = None) -> tuple[str, float]:
    """Return session state and elapsed fraction using Sydney daylight saving.

    The fraction is retained for display; it does not scale relative volume.
    """
    now = now_utc or datetime.now(timezone.utc)
    syd = now.astimezone(SYDNEY)
    if syd.weekday() >= 5:
        return "weekend", 1.0
    t_ = syd.time()
    if t_ < SESSION_OPEN:
        return "pre-open", 1.0
    if t_ >= SESSION_CLOSE:
        return "closed", 1.0
    elapsed = (syd.hour * 60 + syd.minute) - (SESSION_OPEN.hour * 60)
    total = (SESSION_CLOSE.hour - SESSION_OPEN.hour) * 60
    return "open", max(0.02, min(1.0, elapsed / total))


def todays_announcements(db_path: str = ASX_DB) -> dict[str, list[dict[str, Any]]]:
    """Announcements released today, keyed by ticker, with the AI score
    attached where the classifier has already run.

    This is the piece that makes the scanner more than a price screen: volume
    says *something* happened, the announcement says *what*. `event_momentum.py`
    could not use this (3 days of history at the time) but a LIVE scan can,
    which is the other reason forward testing is worth more here.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        # Include releases since the previous weekday close, across weekends.
        now = datetime.now(SYDNEY)
        prior = now.date() - timedelta(days=1)
        while prior.weekday() >= 5:
            prior -= timedelta(days=1)
        since = datetime.combine(prior, SESSION_CLOSE, SYDNEY).astimezone(timezone.utc).isoformat()
        rows = conn.execute(
            "SELECT fingerprint, ticker, headline, released_at, price_sensitive, is_halt, url "
            "FROM announcements WHERE COALESCE(released_at, seen_at) >= ? "
            "ORDER BY released_at DESC", (since,),
        ).fetchall()
        conn.close()
    except Exception:
        return out

    anns = [dict(r) for r in rows]
    try:
        from tradingagents.asx_signals import get_signals_for
        scores = get_signals_for([a["fingerprint"] for a in anns])
    except Exception:
        scores = {}
    for a in anns:
        sig = scores.get(a["fingerprint"]) or {}
        a["ai_signal"] = sig.get("signal")
        a["ai_score"] = sig.get("score")
        a["reason"] = sig.get("reason")
        a["ai_classified_at"] = sig.get("classified_at")
        a["ai_model"] = sig.get("model")
        a["ai_prompt_version"] = sig.get("prompt_version")
        out.setdefault(a["ticker"], []).append(a)
    return out


def scan(limit: int = 500, batch: int = 60) -> dict[str, Any]:
    """Scan the universe for fast movers. Returns every row above
    WATCH_MOVE_PCT, flagged rather than filtered."""
    cands = load_candidates(limit)
    by_ticker = {c.ticker: c for c in cands}
    tickers = list(by_ticker)
    state, frac = session_state()
    # Compare each bar against the current Sydney date.
    session_day = datetime.now(SYDNEY).date().isoformat()
    anns = todays_announcements()
    try:
        from tradingagents.asx_signals import get_ticker_signals
        ticker_signals = get_ticker_signals()
    except Exception:
        ticker_signals = {}

    rows: list[dict[str, Any]] = []
    n_read = 0
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        with YF_LOCK:
            data = yf.download([f"{t}.AX" for t in chunk], period="3mo", interval="1d",
                               group_by="ticker", auto_adjust=False, threads=True, progress=False)
        for t in chunk:
            try:
                df = data[f"{t}.AX"].dropna(subset=["Close"])
            except KeyError:
                continue
            if len(df) < 2:
                continue
            n_read += 1
            today, prev = df.iloc[-1], df.iloc[-2]
            # **Which day is this bar actually from?** Never assume iloc[-1] is
            # today. yfinance's ASX data is delayed, so early in the session
            # today's bar does not exist yet and iloc[-1] is still YESTERDAY --
            # at which point every number below silently describes yesterday's
            # session while the page presents it as live. Caught 2026-08-25:
            # SKC showed +12% (its 24 Aug move) at 10:2x Sydney, and 37 of 39
            # rows were logged into mover_log under the 25th carrying the
            # 24th's prices. The bar's own date now travels with the row.
            bar_date = df.index[-1].date().isoformat()
            prev_bar_date = df.index[-2].date().isoformat()
            prev_close = float(prev["Close"])
            if not prev_close:
                continue
            last = float(today["Close"])
            move_pct = (last - prev_close) / prev_close * 100
            if move_pct < WATCH_MOVE_PCT:
                continue

            gap_pct = (float(today["Open"]) - prev_close) / prev_close * 100
            # Move measured from the OPEN rather than the previous close: what
            # the stock has done since the opening auction, i.e. the part of
            # the move still available once the gap had already happened. This
            # is the same quantity the gap-outcomes table calls Open→Close %,
            # measured live instead of after the close.
            open_px = float(today["Open"])
            move_from_open_pct = ((last - open_px) / open_px * 100) if open_px else None
            median_vol = float(df["Volume"].iloc[:-1].tail(VOLUME_LOOKBACK_DAYS).median() or 0)
            expected = median_vol
            vol_ratio = ((float(today["Volume"]) / expected)
                         if expected and state != "open" else np.nan)
            band, verdict = price_band(last)
            ticker_anns = anns.get(t, [])
            # **Never max() over a ticker's announcements.** 34% of scored
            # tickers have more than one, and they are usually the same event
            # split across documents -- so max() surfaces the most flattering
            # facet and hides the rest. GNG (2026-08-23) published six
            # documents for "FY26 results plus a dilutive equity raising",
            # scoring 20 to 68; max() showed 68 Buy and hid the raise entirely.
            # Prefer the net per-ticker score from asx_signals.ticker_signals;
            # fall back to the MOST MATERIAL individual score (furthest from
            # 50), never the highest.
            net = ticker_signals.get(t)
            if net and not set((net.get("fingerprints") or "").split(",")).issubset(
                    {a["fingerprint"] for a in ticker_anns}):
                net = None
            from .asx_signals import _worth_a_call
            pending_documents = any(_worth_a_call(a) and a.get("ai_score") is None for a in ticker_anns)
            best = max((a for a in ticker_anns if a.get("ai_score") is not None),
                       key=lambda a: abs(a["ai_score"] - 50), default=None)
            if pending_documents:
                net, best = None, None

            rows.append({
                "ticker": t, "company": by_ticker[t].company,
                "bar_date": bar_date, "prev_bar_date": prev_bar_date,
                "is_stale": bar_date != session_day,
                "last": round(last, 4), "prev_close": round(prev_close, 4),
                "open": round(float(today["Open"]), 4),
                "high": round(float(today["High"]), 4), "low": round(float(today["Low"]), 4),
                "move_pct": round(move_pct, 2), "gap_pct": round(gap_pct, 2),
                "move_from_open_pct": (round(move_from_open_pct, 2)
                                       if move_from_open_pct is not None else None),
                "volume": int(today["Volume"]),
                "volume_ratio": round(vol_ratio, 2) if vol_ratio == vol_ratio else None,
                "price_band": band, "cost_verdict": verdict,
                "tick_pct": round(tick_size(last) / last * 100, 2) if last else None,
                "n_announcements": len(ticker_anns),
                "has_price_sensitive": any(a["price_sensitive"] for a in ticker_anns),
                "has_halt": any(a["is_halt"] for a in ticker_anns),
                "top_headline": best["headline"] if best else (
                    ticker_anns[0]["headline"] if ticker_anns else None),
                "ai_signal": (net["signal"] if net else (best["ai_signal"] if best else None)),
                "ai_score": (net["score"] if net else (best["ai_score"] if best else None)),
                "ai_is_net": bool(net),
                "ai_scored_at": net["computed_at"] if net else (best.get("ai_classified_at") if best else None),
                "ai_model": net["model"] if net else (best.get("ai_model") if best else None),
                "ai_prompt_version": net["prompt_version"] if net else (best.get("ai_prompt_version") if best else None),
                "evidence_status": "awaiting_documents" if pending_documents else "document_read",
                "ai_reason": (net["reason"] if net else (best.get("reason") if best else None)),
                "ai_score_range": (
                    [min(a["ai_score"] for a in ticker_anns if a.get("ai_score") is not None),
                     max(a["ai_score"] for a in ticker_anns if a.get("ai_score") is not None)]
                    if any(a.get("ai_score") is not None for a in ticker_anns) else None),
                "announcement_url": best["url"] if best else (
                    ticker_anns[0]["url"] if ticker_anns else None),
                "announcement_at": (best or (ticker_anns[0] if ticker_anns else {})).get("released_at"),
            })

    if state == "open":
        from .scan_volume import for_tickers
        try:
            volumes = for_tickers([r["ticker"] for r in rows if not r["is_stale"]])
        except Exception:
            logging.getLogger(__name__).warning("Matched-window volume unavailable", exc_info=True)
            volumes = {}
        for r in rows:
            v = volumes.get(r["ticker"])
            r["volume_ratio"] = round(v["ratio"], 2) if v else None
            r["volume_through"] = v["through"] if v else None
            r["volume_baseline_sessions"] = v["sessions"] if v else None

    for r in rows:
        vr = r["volume_ratio"]
        r["meets_move"] = r["move_pct"] >= MIN_MOVE_PCT
        r["meets_volume"] = bool(vr is not None and vr >= MIN_VOLUME_RATIO)
        r["has_news"] = r["n_announcements"] > 0
        r["full_setup"] = bool(not r["is_stale"] and r["meets_move"] and r["meets_volume"] and r["has_news"])
    rows.sort(key=lambda r: (r["full_setup"], r["meets_move"] and r["meets_volume"],
                             r["move_pct"]), reverse=True)

    try:
        from tradingagents.mover_log import log_movers
        # Log each row under the date of the BAR it came from, not "today".
        # A scan that ran before today's bar existed then re-touches
        # yesterday's row with yesterday's own numbers -- a harmless no-op --
        # instead of fabricating a today row that is a copy of yesterday.
        for day in sorted({r["bar_date"] for r in rows}):
            log_movers([r for r in rows if r["bar_date"] == day], as_of=day)
    except Exception:
        logging.getLogger(__name__).exception("Mover observation logging failed")

    n_stale = sum(1 for r in rows if r["is_stale"])
    return {
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_fraction": round(frac, 3),
        "session_state": state,
        "session_day": session_day,
        "bar_dates": sorted({r["bar_date"] for r in rows}),
        "n_stale": n_stale,
        # How many candidates actually returned usable bars. Zero rows with
        # zero bars read is a broken fetch; zero rows with 400+ bars read is a
        # genuinely quiet market. The caller uses this to decide whether the
        # result is worth caching.
        "n_candidates": len(tickers),
        "n_bars_read": n_read,
        "thresholds": {"min_move_pct": MIN_MOVE_PCT, "min_volume_ratio": MIN_VOLUME_RATIO,
                       "watch_move_pct": WATCH_MOVE_PCT},
        "volume_ratio_note": (
            "During trading: completed hourly volume vs matching hours over up to 20 prior sessions "
            "(minimum 10), allowing for delayed data. Blank until a comparable window is available. "
            "Outside trading: full-day volume vs the previous 50-session median."),
        "price_note": (
            "Quotes are yfinance daily bars updating intraday, ~20 min delayed. Fine for a "
            "3-10 day hold, not for precise entry timing."),
        "n_rows": len(rows),
        "rows": rows,
    }


def print_scan(res: dict[str, Any]) -> None:
    print(f"scanned {res['scanned_at']}  session={res['session_state']} "
          f"(volume divisor {res['session_fraction']:.0%} of a full day)  rows={res['n_rows']}")
    print(f"thresholds: move>={MIN_MOVE_PCT}%  vol>={MIN_VOLUME_RATIO}x  (watch from {WATCH_MOVE_PCT}%)")
    print()
    if not res["rows"]:
        print("  no movers above the watch threshold")
        return
    hdr = (f"  {'tkr':<6}{'last':>9}{'move%':>8}{'gap%':>8}{'vol x':>8}{'band':>9}"
           f"{'news':>6}{'AI':>5}  {'setup':<7} headline")
    print(hdr)
    for r in res["rows"][:40]:
        setup = "FULL" if r["full_setup"] else ("move+vol" if r["meets_move"] and r["meets_volume"] else "watch")
        print(f"  {r['ticker']:<6}{r['last']:>9.3f}{r['move_pct']:>8.2f}{r['gap_pct']:>8.2f}"
              f"{(r['volume_ratio'] or 0):>8.1f}{r['price_band']:>9}"
              f"{r['n_announcements']:>6}{(r['ai_score'] if r['ai_score'] is not None else '-'):>5}  "
              f"{setup:<7} {(r['top_headline'] or '')[:60]}")


if __name__ == "__main__":
    print_scan(scan())
