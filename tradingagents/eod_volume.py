"""End-of-day volume scan: the tradeable half of the overnight-hold finding
(2026-08-28, user request).

**The strategy this serves.** Measured over 2 years of hourly bars on the top
500: buying a stock's closing auction and selling the next opening auction
returns ~+0.19% per night unconditionally, and **~+0.34% when the stock traded
>6x its normal volume that day** (8 of 9 quarters positive, not outlier-driven).
Holding into the next session instead destroys it -- selling at 11:00 gives back
69% of the gain. So the trade is two auction fills and nothing in between.

**Why the ratio is computed against the SAME HOUR of prior sessions.** Only
~43% of an ASX session's volume has traded by 15:00 -- the closing auction
carries much of the rest -- so comparing a partial day against prior FULL days
would understate every ratio. Scaling by an assumed intraday profile is no good
either: the profile's 10th-90th spread is 28%-70%, far too wide to point
estimate per stock. Comparing like with like removes the assumption entirely.
Validated: the full-day signal scores +0.366%/night and this partial signal
+0.337%, so almost nothing is lost by using only what is knowable at 15:45.

**Every scan is stored, and resolved the next morning.** That builds a live,
survivorship-free forward record to set against the backtest -- which is run on
today's top 500 and therefore biased. `resolve()` fills the next open once it
exists. Same pattern as `mover_log` + `forward_returns`.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "swing.db"
SYDNEY = "Australia/Sydney"
SESSION_OPEN_HOUR = 10
SIGNAL_HOUR = 15          # count volume traded before 15:00
DEFAULT_THRESHOLD = 6.0
VOL_WINDOW = 20
MIN_MEDIAN_TURNOVER_AUD = 500_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS eod_volume_candidates (
    scan_date     TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    company       TEXT,
    vol_ratio     REAL NOT NULL,
    price         REAL,
    price_1545    REAL,
    price_1545_at TEXT,
    price_1545_source TEXT,
    previous_close REAL,
    move_pct      REAL,
    signal_hour   INTEGER NOT NULL,
    threshold     REAL NOT NULL,
    scanned_at    TEXT NOT NULL,
    next_open     REAL,
    overnight_pct REAL,
    resolved_at   TEXT,
    announcement_score INTEGER,
    announcement_signal TEXT,
    announcement_session TEXT,
    announcement_headline TEXT,
    catalyst_lane TEXT,
    selection_score REAL,
    median_turnover_aud REAL,
    PRIMARY KEY (scan_date, ticker)
);

CREATE TABLE IF NOT EXISTS eod_volume_runs (
    scan_date       TEXT PRIMARY KEY,
    scanned_at      TEXT NOT NULL,
    threshold       REAL NOT NULL,
    signal_hour     INTEGER NOT NULL,
    n_universe      INTEGER NOT NULL,
    n_with_data     INTEGER NOT NULL,
    n_candidates    INTEGER NOT NULL,
    n_volume_candidates INTEGER NOT NULL DEFAULT 0,
    n_catalyst_candidates INTEGER NOT NULL DEFAULT 0,
    sydney_time     TEXT NOT NULL,
    window_complete INTEGER NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    have = {r["name"] for r in conn.execute("PRAGMA table_info(eod_volume_candidates)")}
    for col, typ in {
        "announcement_score": "INTEGER",
        "announcement_signal": "TEXT",
        "announcement_session": "TEXT",
        "announcement_headline": "TEXT",
        "catalyst_lane": "TEXT",
        "selection_score": "REAL",
        "median_turnover_aud": "REAL",
        "price_1545": "REAL",
        "price_1545_at": "TEXT",
        "price_1545_source": "TEXT",
        "previous_close": "REAL",
    }.items():
        if col not in have:
            conn.execute(f"ALTER TABLE eod_volume_candidates ADD COLUMN {col} {typ}")
    run_have = {r["name"] for r in conn.execute("PRAGMA table_info(eod_volume_runs)")}
    for col, typ in {
        "n_volume_candidates": "INTEGER NOT NULL DEFAULT 0",
        "n_catalyst_candidates": "INTEGER NOT NULL DEFAULT 0",
    }.items():
        if col not in run_have:
            conn.execute(f"ALTER TABLE eod_volume_runs ADD COLUMN {col} {typ}")
    return conn


def _sydney_today() -> str:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(SYDNEY)).date().isoformat()


def latest_stored_scan(scan_date: str | None = None) -> dict[str, Any]:
    """Return the frozen scheduled scan without fetching market data.

    Older databases only have candidate rows, so the fallback reconstructs
    the available metadata from those rows. New scans also write a run row,
    which lets a valid zero-candidate session be represented explicitly.
    """
    from zoneinfo import ZoneInfo

    with _connect() as conn:
        if scan_date:
            run = conn.execute(
                "SELECT * FROM eod_volume_runs WHERE scan_date=?", (scan_date,)
            ).fetchone()
        else:
            run = conn.execute(
                "SELECT * FROM eod_volume_runs ORDER BY scan_date DESC LIMIT 1"
            ).fetchone()

        if run:
            meta = dict(run)
            day = meta["scan_date"]
        else:
            if scan_date:
                legacy = conn.execute(
                    "SELECT scan_date, MAX(scanned_at) scanned_at, MIN(threshold) threshold,"
                    " MIN(signal_hour) signal_hour, COUNT(*) n_candidates"
                    " FROM eod_volume_candidates WHERE scan_date=? GROUP BY scan_date",
                    (scan_date,),
                ).fetchone()
            else:
                legacy = conn.execute(
                    "SELECT scan_date, MAX(scanned_at) scanned_at, MIN(threshold) threshold,"
                    " MIN(signal_hour) signal_hour, COUNT(*) n_candidates"
                    " FROM eod_volume_candidates GROUP BY scan_date"
                    " ORDER BY scan_date DESC LIMIT 1"
                ).fetchone()
            if not legacy:
                return {
                    "available": False, "stored": True, "is_current": False,
                    "n_candidates": 0, "candidates": [],
                }
            meta = dict(legacy)
            day = meta["scan_date"]
            local = datetime.fromisoformat(meta["scanned_at"]).astimezone(ZoneInfo(SYDNEY))
            meta.update({
                "n_universe": None,
                "n_with_data": None,
                "n_volume_candidates": meta["n_candidates"],
                "n_catalyst_candidates": 0,
                "sydney_time": local.strftime("%H:%M"),
                "window_complete": int(local.hour >= int(meta["signal_hour"])),
            })

        rows = [dict(r) for r in conn.execute(
            "SELECT ticker, company, vol_ratio, price, move_pct,"
            " price_1545, price_1545_at, price_1545_source, previous_close,"
            " announcement_score, announcement_signal, announcement_session,"
            " announcement_headline, catalyst_lane, selection_score, median_turnover_aud"
            " FROM eod_volume_candidates WHERE scan_date=?"
            " ORDER BY COALESCE(selection_score, vol_ratio) DESC, vol_ratio DESC",
            (day,),
        )]

    meta.update({
        "available": True,
        "stored": True,
        "is_current": day == _sydney_today(),
        "window_complete": bool(meta["window_complete"]),
        "n_candidates": len(rows),
        "candidates": rows,
    })
    return meta


def _announcement_overlay(scan_date: str, as_of: datetime) -> dict[str, dict[str, Any]]:
    """Return scored, price-sensitive news from today and the prior session.

    The measured volume signal remains the base rule. This overlay supplies a
    separate catalyst lane and only uses documents already scored by the
    classifier. `as_of` prevents a late manual run from using future news.
    """
    try:
        from .asx_feed import recent_announcements, tradeable_session_for
        from .asx_signals import attach_signals

        rows = []
        for row in recent_announcements(limit=3000, universe_only=False):
            stamp = row.get("released_at") or row.get("seen_at") or ""
            try:
                if datetime.fromisoformat(stamp) > as_of:
                    continue
            except (TypeError, ValueError):
                pass
            session = tradeable_session_for(stamp)
            if session and session <= scan_date:
                item = dict(row)
                item["tradeable_session"] = session
                rows.append(item)
        prior_day = datetime.fromisoformat(scan_date).date() - timedelta(days=1)
        while prior_day.weekday() >= 5:
            prior_day -= timedelta(days=1)
        prior = prior_day.isoformat()
        rows = [r for r in rows if r["tradeable_session"] in {scan_date, prior}]
        attach_signals(rows)
    except Exception:
        logger.exception("EOD announcement overlay failed")
        return {}

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row.get("ticker") or not row.get("price_sensitive"):
            continue
        try:
            score = int(row.get("score"))
        except (TypeError, ValueError):
            continue
        item = {
            "score": score,
            "signal": row.get("signal"),
            "session": "same_day" if row["tradeable_session"] == scan_date else "prior_session",
            "headline": row.get("headline"),
            "reason": row.get("signal_reason"),
        }
        current = out.get(row["ticker"])
        if current is None or abs(score - 50) > abs(current["score"] - 50):
            out[row["ticker"]] = item
    return out


def _catalyst_bonus(news: dict[str, Any] | None) -> float:
    """Small secondary ranking boost for a verified catalyst.

    The announcement history is currently too short to let news override the
    volume signal. Keeping this deliberately small prevents a single bullish
    headline from outranking a much stronger, measured volume event.
    """
    if not news:
        return 0.0
    score = float(news.get("score") or 50)
    if news.get("session") == "same_day" and score >= 75:
        return round(2.0 + (score - 75.0) * 0.03, 2)
    if news.get("session") == "prior_session" and score >= 80:
        return round(1.0 + (score - 80.0) * 0.03, 2)
    if score <= 35:
        return -3.0
    return 0.0


def _annotate_candidate(candidate: dict[str, Any], news: dict[str, Any] | None) -> None:
    candidate["announcement_score"] = news.get("score") if news else None
    candidate["announcement_signal"] = news.get("signal") if news else None
    candidate["announcement_session"] = news.get("session") if news else None
    candidate["announcement_headline"] = news.get("headline") if news else None
    candidate["catalyst_lane"] = news.get("session") if news and _catalyst_bonus(news) > 0 else None
    candidate["selection_score"] = round(candidate["vol_ratio"] + _catalyst_bonus(news), 2)


def _snapshot_1545(tickers: list[str], scan_date: str, as_of: datetime) -> dict[str, dict[str, Any]]:
    """Capture the latest 15-minute Yahoo bar available at the 15:45 scan.

    The hourly scan is the right efficient source for the volume ratio, but its
    last row is not a reliable 15:45 quote: Yahoo's hourly bars are stamped by
    their bar time and may still be provisional.  This separate snapshot is
    only fetched for candidates, and is stored with both timestamp and source.
    """
    if not tickers:
        return {}
    import yfinance as yf
    from zoneinfo import ZoneInfo
    from .yf_lock import YF_LOCK

    out: dict[str, dict[str, Any]] = {}
    local_cutoff = as_of.astimezone(ZoneInfo(SYDNEY))
    for i in range(0, len(tickers), 40):
        chunk = tickers[i:i + 40]
        with YF_LOCK:
            data = yf.download([f"{t}.AX" for t in chunk], period="5d", interval="15m",
                               group_by="ticker", auto_adjust=False, threads=True,
                               progress=False)
        for t in chunk:
            try:
                df = data[f"{t}.AX"].dropna(subset=["Close"]) if len(chunk) > 1 else data.dropna(subset=["Close"])
            except (KeyError, TypeError, AttributeError):
                continue
            if df.empty:
                continue
            idx = df.index.tz_convert(SYDNEY) if df.index.tz is not None else df.index.tz_localize(SYDNEY)
            df = df.copy()
            df.index = idx
            eligible = df[(df.index.date == local_cutoff.date()) & (df.index <= local_cutoff)]
            if eligible.empty:
                continue
            stamp = eligible.index[-1]
            price = float(eligible.iloc[-1]["Close"])
            if price > 0:
                out[t] = {
                    "price": round(price, 4),
                    "at": stamp.isoformat(timespec="minutes"),
                    "source": "yahoo_15m",
                }
    return out


def _apply_1545_snapshot(candidates: list[dict[str, Any]], snapshots: dict[str, dict[str, Any]], scanned_at: datetime) -> None:
    """Make the stored candidate price the explicitly labelled 15:45 price."""
    for candidate in candidates:
        snap = snapshots.get(candidate["ticker"])
        if snap:
            candidate["price_1545"] = snap["price"]
            candidate["price_1545_at"] = snap["at"]
            candidate["price_1545_source"] = snap["source"]
            candidate["price"] = snap["price"]
            if candidate.get("previous_close"):
                candidate["move_pct"] = round(
                    100 * (snap["price"] - candidate["previous_close"]) / candidate["previous_close"], 2)
        else:
            candidate["price_1545"] = candidate.get("price")
            from zoneinfo import ZoneInfo
            candidate["price_1545_at"] = scanned_at.astimezone(ZoneInfo(SYDNEY)).isoformat(timespec="minutes")
            candidate["price_1545_source"] = "hourly_fallback"


def scan(universe_limit: int = 500, threshold: float = DEFAULT_THRESHOLD,
         vol_window: int = VOL_WINDOW, signal_hour: int = SIGNAL_HOUR,
         store: bool = True) -> dict[str, Any]:
    """Names running above `threshold` x their normal volume-to-this-hour."""
    today = _sydney_today()
    if store:
        frozen = latest_stored_scan(today)
        if frozen.get("available") and frozen.get("window_complete"):
            return {**frozen, "cached": True}

    import numpy as np
    import yfinance as yf

    from .gap_study import _universe
    from .symbols import company_name
    from .yf_lock import YF_LOCK

    tickers = _universe(universe_limit)
    if not tickers:
        return {"error": "no universe available", "candidates": []}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    cands: list[dict[str, Any]] = []
    observed: dict[str, dict[str, Any]] = {}
    n_seen = 0
    for i in range(0, len(tickers), 40):
        chunk = tickers[i:i + 40]
        with YF_LOCK:
            data = yf.download([f"{t}.AX" for t in chunk], period="60d", interval="1h",
                               group_by="ticker", auto_adjust=False, threads=True,
                               progress=False)
        for t in chunk:
            try:
                df = data[f"{t}.AX"].dropna(subset=["Close"])
            except (KeyError, TypeError):
                continue
            if df.empty or "Volume" not in df:
                continue
            idx = df.index.tz_convert(SYDNEY) if df.index.tz is not None \
                else df.index.tz_localize(SYDNEY)
            days = [x.date().isoformat() for x in idx]
            hours = np.fromiter((x.hour for x in idx), dtype=int, count=len(idx))
            vol = df["Volume"].to_numpy(float)
            cl = df["Close"].to_numpy(float)

            starts = [0]
            for j in range(1, len(days)):
                if days[j] != days[j - 1]:
                    starts.append(j)
            ends = starts[1:] + [len(days)]
            if days[starts[-1]] != today:
                continue                      # no bars for today yet

            # **Match the hour set, not just the cutoff.** Run at 11:51, today
            # has only the 10:00 and 11:00 bars while prior sessions have
            # 10:00-14:00 -- comparing those understates every ratio and makes
            # the scan silently wrong at any time before 15:00. The baseline
            # uses exactly the hours today has so far, so the ratio means the
            # same thing whenever it is run.
            a0, b0 = starts[-1], ends[-1]
            today_hours = {int(hours[a0 + j]) for j in range(b0 - a0)
                           if hours[a0 + j] < signal_hour}
            if not today_hours:
                continue

            def vol_for(k: int) -> float:
                a, b = starts[k], ends[k]
                sel = [a + j for j in range(b - a) if int(hours[a + j]) in today_hours]
                return float(np.nansum(vol[sel])) if sel else 0.0

            partial = [vol_for(k) for k in range(len(starts))]
            last_close = [float(cl[ends[k] - 1]) for k in range(len(starts))]
            n_seen += 1
            base = [v for v in partial[max(0, len(starts) - 1 - vol_window):-1] if v > 0]
            if not base or not partial[-1]:
                continue
            prior_turnover = [
                float(np.nansum(vol[starts[k]:ends[k]] * cl[starts[k]:ends[k]]))
                for k in range(max(0, len(starts) - 1 - vol_window), len(starts) - 1)
            ]
            prior_turnover = [v for v in prior_turnover if v > 0]
            median_turnover = float(np.median(prior_turnover)) if prior_turnover else None
            ratio = partial[-1] / float(np.median(base))
            price = last_close[-1]
            prev = last_close[-2] if len(last_close) >= 2 else None
            observed[t] = {
                "ticker": t, "company": company_name(t, "AU"),
                "vol_ratio": round(ratio, 2), "price": round(price, 4),
                "previous_close": round(prev, 4) if prev else None,
                "move_pct": round(100 * (price - prev) / prev, 2) if prev else None,
                "median_turnover_aud": round(median_turnover) if median_turnover else None,
            }
            if ratio >= threshold and (median_turnover or 0) >= MIN_MEDIAN_TURNOVER_AUD:
                cands.append(dict(observed[t]))

    news = _announcement_overlay(today, datetime.now(timezone.utc))
    raw_tickers = {c["ticker"] for c in cands}
    for candidate in cands:
        _annotate_candidate(candidate, news.get(candidate["ticker"]))

    # News is secondary evidence only. The measured edge is the >6x volume
    # condition, and the announcement history is too short to justify letting
    # a 2x/3x catalyst lane bypass that hard gate. Strong news still annotates
    # and slightly re-ranks a volume-qualified candidate.
    catalyst_added = sum(1 for c in cands if c.get("catalyst_lane"))

    # Capture this after all volume/news lanes are known, but before ranking.
    # The snapshot is a measurement field; it does not alter the volume rule.
    snapshots = _snapshot_1545(sorted({c["ticker"] for c in cands}), today, datetime.fromisoformat(now))
    _apply_1545_snapshot(cands, snapshots, datetime.fromisoformat(now))

    cands.sort(key=lambda c: (-c["selection_score"], -c["vol_ratio"], c["ticker"]))
    from zoneinfo import ZoneInfo
    syd = datetime.now(ZoneInfo(SYDNEY))
    result = {
        "scan_date": today, "scanned_at": now, "threshold": threshold,
        "signal_hour": signal_hour, "n_universe": len(tickers),
        "n_with_data": n_seen, "n_candidates": len(cands),
        "n_volume_candidates": len(raw_tickers),
        "n_catalyst_candidates": catalyst_added,
        "min_median_turnover_aud": MIN_MEDIAN_TURNOVER_AUD,
        "sydney_time": syd.strftime("%H:%M"),
        # The ratio is only the one the strategy was measured on once the
        # full pre-15:00 window has elapsed. Earlier readings are a live
        # preview computed on fewer hours -- directionally useful, but a
        # different statistic.
        "window_complete": syd.hour >= signal_hour,
        "available": True, "stored": bool(store), "is_current": True,
        "cached": False, "candidates": cands,
    }
    if store:
        with _connect() as conn:
            # A stored preview may be replaced by the complete scheduled run.
            # This happens before the 16:25 booking job, so no trade fields can
            # be lost. Once a complete run exists, the early return above keeps
            # it frozen for the rest of the session.
            conn.execute("DELETE FROM eod_volume_candidates WHERE scan_date=?", (today,))
            conn.executemany(
                "INSERT INTO eod_volume_candidates (scan_date, ticker, company,"
                " vol_ratio, price, price_1545, price_1545_at, price_1545_source, previous_close, move_pct, signal_hour, threshold, scanned_at,"
                " announcement_score, announcement_signal, announcement_session,"
                " announcement_headline, catalyst_lane, selection_score, median_turnover_aud)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(today, c["ticker"], c["company"], c["vol_ratio"], c["price"],
                  c.get("price_1545"), c.get("price_1545_at"), c.get("price_1545_source"),
                  c.get("previous_close"), c["move_pct"], signal_hour, threshold, now,
                  c.get("announcement_score"), c.get("announcement_signal"),
                  c.get("announcement_session"), c.get("announcement_headline"),
                  c.get("catalyst_lane"), c.get("selection_score"),
                  c.get("median_turnover_aud")) for c in cands])
            conn.execute(
                "INSERT OR REPLACE INTO eod_volume_runs"
                " (scan_date, scanned_at, threshold, signal_hour, n_universe, n_with_data,"
                " n_candidates, n_volume_candidates, n_catalyst_candidates,"
                " sydney_time, window_complete) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (today, now, threshold, signal_hour, len(tickers), n_seen, len(cands),
                 len(raw_tickers), catalyst_added, syd.strftime("%H:%M"),
                 int(result["window_complete"])),
            )
            conn.commit()
    return result


def resolve(days_back: int = 10) -> dict[str, Any]:
    """Fill the next session's open for past scans -- the live track record.

    A candidate is only resolved from a bar dated AFTER its scan date; using
    `iloc[-1]` blindly would stamp today's open onto an unresolved older row,
    the same bug class that corrupted `mover_log` earlier.
    """
    import numpy as np
    import yfinance as yf

    from .yf_lock import YF_LOCK

    since = (datetime.now(timezone.utc) - timedelta(days=days_back)).date().isoformat()
    with _connect() as conn:
        pend = [dict(r) for r in conn.execute(
            "SELECT scan_date, ticker, price FROM eod_volume_candidates"
            " WHERE overnight_pct IS NULL AND scan_date>=?", (since,))]
    if not pend:
        return {"resolved": 0, "pending": 0}

    tickers = sorted({p["ticker"] for p in pend})
    with YF_LOCK:
        data = yf.download([f"{t}.AX" for t in tickers], period="1mo", interval="1d",
                           group_by="ticker", auto_adjust=False, threads=True,
                           progress=False)
    resolved, still = 0, 0
    with _connect() as conn:
        for p in pend:
            try:
                df = data[f"{p['ticker']}.AX"].dropna(subset=["Open"])
            except (KeyError, TypeError):
                still += 1
                continue
            dates = [d.date().isoformat() for d in df.index]
            after = [d for d in dates if d > p["scan_date"]]
            if not after:
                still += 1                    # next session has not happened yet
                continue
            nxt = float(df.iloc[dates.index(after[0])]["Open"])
            if not nxt or not p["price"]:
                still += 1
                continue
            conn.execute(
                "UPDATE eod_volume_candidates SET next_open=?, overnight_pct=?, resolved_at=?"
                " WHERE scan_date=? AND ticker=?",
                (round(nxt, 4), round(100 * (nxt - p["price"]) / p["price"], 4),
                 datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 p["scan_date"], p["ticker"]))
            resolved += 1
        conn.commit()
    return {"resolved": resolved, "pending": still}


def record(days: int = 90) -> dict[str, Any]:
    """The live forward record: what the scan picked and how it did."""
    import statistics as st
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    with _connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM eod_volume_candidates WHERE scan_date>=?"
            " ORDER BY scan_date DESC, vol_ratio DESC", (since,))]
    done = [r["overnight_pct"] for r in rows if r["overnight_pct"] is not None]
    return {
        "n": len(rows), "n_resolved": len(done),
        "mean_pct": round(st.mean(done), 4) if done else None,
        "median_pct": round(st.median(done), 4) if done else None,
        "win_rate_pct": round(100 * sum(1 for x in done if x > 0) / len(done), 1) if done else None,
        # The backtested expectation, for comparison. A live record that drifts
        # far from this is the signal that the edge has decayed or that the
        # backtest's survivorship bias was doing more work than assumed.
        "backtest_mean_pct": 0.337,
        "rows": rows,
    }


# --- automated paper trading on the candidates ------------------------------
#
# Deliberately NOT written into `momentum_trades`. That log exists to measure a
# discretionary decision -- its docstring is explicit that "the exit is the
# decision you are trying to learn" -- and this strategy has a mechanical exit
# with no decision in it. Mixing them would make per-strategy P&L unrecoverable,
# the same reason momentum_trades was kept apart from swing_trades.
#
# Execution model: a MARKET-ON-CLOSE order placed at ~15:55 fills at the closing
# auction price, which is not known when the order is placed. So selection
# happens on the 15:45 scan and the fill is recorded afterwards at the actual
# close. Recording the 15:55 quote as the entry would book a price that was
# never available -- the same look-ahead family as the errors documented in
# gap_study.
_TRADE_COLUMNS = {
    "selected": "INTEGER",        # chosen for the paper book
    "shares": "INTEGER",
    "entry_price": "REAL",        # the actual closing auction price
    "entry_at": "TEXT",
    "exit_price": "REAL",         # the actual opening auction price
    "exit_at": "TEXT",
    "pnl_pct": "REAL",
    "pnl_dollars": "REAL",
    # The ASX 200 close->open move over the SAME window. Ten long positions
    # overnight is partly a bet on the index gapping up; without this the book
    # cannot tell "the selection worked" from "the market opened higher", which
    # on any single night is mostly what it measures.
    "mkt_overnight_pct": "REAL",
}

# The ASX 200 ETF, NOT the ^AXJO index, for anything that touches an OPEN
# price. Verified 2026-08-31: ^AXJO's Open equals its previous Close on 21 of
# 21 sessions (sd 0.0000%) because the index is carried forward rather than
# struck at a traded auction. Benchmarking an overnight or open-to-close leg
# against it silently returns 0.0 for every row -- which is exactly what this
# book recorded until today. STW tracks the same index and has a genuine
# traded open (sd 0.268% over the same window).
MARKET_PROXY = "STW.AX"

DEFAULT_MAX_POSITIONS = 10
DEFAULT_DOLLARS = 20_000.0
MIN_PRICE = 0.50              # retain the user's price floor; liquidity is the hard gate


def _ensure_trade_columns(conn: sqlite3.Connection) -> None:
    have = {r["name"] for r in conn.execute("PRAGMA table_info(eod_volume_candidates)")}
    for col, typ in _TRADE_COLUMNS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE eod_volume_candidates ADD COLUMN {col} {typ}")


def open_positions(max_positions: int = DEFAULT_MAX_POSITIONS,
                   dollars: float = DEFAULT_DOLLARS,
                   min_price: float = MIN_PRICE,
                   scan_date: str | None = None) -> dict[str, Any]:
    """Select the day's candidates and book them at the ACTUAL closing price.

    Ranked by volume ratio and capped, because the backtest never limited how
    many positions a day produced and a live scan can surface 60+ names -- which
    is not a portfolio, it is the whole market.
    """
    import yfinance as yf

    from .yf_lock import YF_LOCK

    day = scan_date or _sydney_today()
    with _connect() as conn:
        _ensure_trade_columns(conn)
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM eod_volume_candidates WHERE scan_date=? AND selected IS NULL"
            " ORDER BY COALESCE(selection_score, vol_ratio) DESC, vol_ratio DESC", (day,))]
    if not rows:
        return {"opened": 0, "reason": "no unselected candidates for " + day}

    picked = [r for r in rows
              if (r["price"] or 0) >= min_price
              and (r.get("median_turnover_aud") is None
                   or r["median_turnover_aud"] >= MIN_MEDIAN_TURNOVER_AUD)][:max_positions]
    if not picked:
        return {"opened": 0, "reason": f"no candidates at or above ${min_price}"}

    # The real closing price, which the 15:45 scan could not have known.
    with YF_LOCK:
        data = yf.download([f"{p['ticker']}.AX" for p in picked], period="5d",
                           interval="1d", group_by="ticker", auto_adjust=False,
                           threads=True, progress=False)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    opened, skipped = [], []
    with _connect() as conn:
        _ensure_trade_columns(conn)
        # Everything not picked is marked selected=0 so it is never reconsidered
        # and the record shows what was passed over as well as what was taken.
        conn.execute("UPDATE eod_volume_candidates SET selected=0"
                     " WHERE scan_date=? AND selected IS NULL", (day,))
        for p in picked:
            close_px = None
            try:
                df = data[f"{p['ticker']}.AX"].dropna(subset=["Close"])
                dates = [d.date().isoformat() for d in df.index]
                if day in dates:
                    close_px = float(df.iloc[dates.index(day)]["Close"])
            except (KeyError, TypeError, IndexError):
                pass
            if not close_px:
                skipped.append(p["ticker"])
                continue
            shares = int(dollars // close_px)
            if shares < 1:
                skipped.append(p["ticker"])
                continue
            conn.execute(
                "UPDATE eod_volume_candidates SET selected=1, shares=?, entry_price=?,"
                " entry_at=? WHERE scan_date=? AND ticker=?",
                (shares, round(close_px, 4), now, day, p["ticker"]))
            opened.append({"ticker": p["ticker"], "shares": shares,
                           "entry": round(close_px, 4), "vol_ratio": p["vol_ratio"]})
        conn.commit()
    return {"scan_date": day, "opened": len(opened), "positions": opened,
            "skipped_no_close": skipped, "considered": len(rows)}


def close_positions(days_back: int = 10) -> dict[str, Any]:
    """Exit every open paper position at the ACTUAL next opening auction.

    The exit is the open and nothing later: selling at 11:00 instead was
    measured to give back 69% of the overnight gain.
    """
    import yfinance as yf

    from .yf_lock import YF_LOCK

    since = (datetime.now(timezone.utc) - timedelta(days=days_back)).date().isoformat()
    with _connect() as conn:
        _ensure_trade_columns(conn)
        open_rows = [dict(r) for r in conn.execute(
            "SELECT * FROM eod_volume_candidates WHERE selected=1 AND entry_price IS NOT NULL"
            " AND exit_price IS NULL AND scan_date>=?", (since,))]
    if not open_rows:
        return {"closed": 0, "still_open": 0}

    with YF_LOCK:
        data = yf.download([f"{r['ticker']}.AX" for r in open_rows] + [MARKET_PROXY],
                           period="1mo", interval="1d", group_by="ticker",
                           auto_adjust=False, threads=True, progress=False)

    def mkt_move(scan_date):
        # Index close(scan_date) -> open(next session): the same window the
        # trade was held over, so the subtraction is like for like.
        try:
            mdf = data[MARKET_PROXY].dropna(subset=["Open"])
        except (KeyError, TypeError):
            return None
        md = [d.date().isoformat() for d in mdf.index]
        if scan_date not in md:
            return None
        nxt = [d for d in md if d > scan_date]
        if not nxt:
            return None
        c = float(mdf.iloc[md.index(scan_date)]["Close"])
        o = float(mdf.iloc[md.index(nxt[0])]["Open"])
        return round(100 * (o - c) / c, 4) if c else None
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    closed, still = 0, 0
    with _connect() as conn:
        for r in open_rows:
            try:
                df = data[f"{r['ticker']}.AX"].dropna(subset=["Open"])
            except (KeyError, TypeError):
                still += 1
                continue
            dates = [d.date().isoformat() for d in df.index]
            after = [d for d in dates if d > r["scan_date"]]
            if not after:
                still += 1            # next session has not opened yet
                continue
            px = float(df.iloc[dates.index(after[0])]["Open"])
            if not px:
                still += 1
                continue
            pnl_pct = 100 * (px - r["entry_price"]) / r["entry_price"]
            conn.execute(
                "UPDATE eod_volume_candidates SET exit_price=?, exit_at=?, pnl_pct=?,"
                " pnl_dollars=?, mkt_overnight_pct=? WHERE scan_date=? AND ticker=?",
                (round(px, 4), now, round(pnl_pct, 4),
                 round((px - r["entry_price"]) * r["shares"], 2),
                 mkt_move(r["scan_date"]), r["scan_date"], r["ticker"]))
            closed += 1
        conn.commit()
    return {"closed": closed, "still_open": still}


def book(days: int = 90) -> dict[str, Any]:
    """The automated paper book: positions, P&L, and the backtest to beat."""
    import statistics as st
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    with _connect() as conn:
        _ensure_trade_columns(conn)
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM eod_volume_candidates WHERE selected=1 AND scan_date>=?"
            " ORDER BY scan_date DESC, vol_ratio DESC", (since,))]
    done = [r for r in rows if r["pnl_pct"] is not None]
    pnl = [r["pnl_pct"] for r in done]
    # Market-relative is the number that says whether SELECTION added anything.
    rel = [r["pnl_pct"] - r["mkt_overnight_pct"] for r in done
           if r["mkt_overnight_pct"] is not None]
    return {
        "n_positions": len(rows), "n_closed": len(done),
        "n_open": len(rows) - len(done),
        "mean_pct": round(st.mean(pnl), 4) if pnl else None,
        "median_pct": round(st.median(pnl), 4) if pnl else None,
        "win_rate_pct": round(100 * sum(1 for x in pnl if x > 0) / len(pnl), 1) if pnl else None,
        "total_dollars": round(sum(r["pnl_dollars"] or 0 for r in done), 2) if done else None,
        "n_market_matched": len(rel),
        "mean_vs_market_pct": round(st.mean(rel), 4) if rel else None,
        "win_rate_vs_market_pct": (round(100 * sum(1 for x in rel if x > 0) / len(rel), 1)
                                   if rel else None),
        "backtest_mean_pct": 0.337,
        # 381 trades is where a +0.337% mean against a 3.29% sd reaches t=2.
        # Stated here so the page cannot imply significance it does not have.
        "trades_for_significance": 381,
        "rows": rows,
    }
