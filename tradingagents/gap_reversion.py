"""Second automated paper book: buy the most-reverting names after a gap down,
entered mid-morning rather than at the open (2026-08-29, user request).

**Why this exists as paper trading rather than a live strategy.** The 10-year
study found a real, monotonic, decade-stable gap reversion, and found that
reversion strength is a persistent STOCK-LEVEL characteristic (rank stocks by
it on 2016-2020 and the ranking still holds on 2021-2026: spearman +0.46 in
the ASX 50, +0.65 in rank 101-300). But almost all of the measured profit sits
at the opening auction price, and that price cannot be traded on this signal:
the auction clears at a single price, orders must be in BEFORE it clears, and
the gap is not observable until after it has cleared. Entering at the open is
look-ahead bias. Re-measured at a realistic 11:00 entry, one variant survived:

    small-cap Q1, long gap < -3%, 11:00 entry -> +0.503%/trade, t 2.7, n 123
    ASX 50   Q1, long gap < -3%, 11:00 entry -> +0.228%/trade, t 1.1, n  58
    every short variant                      -> <= 0, some significantly

**Those figures assumed a 0.05% round trip and are too generous.** Corrected
2026-08-31 once brokerage was known ($2 per $20k = 2bp round trip) and spreads
were measured per instrument: the small_q1 cohort averages 0.270% all-in
(INA and ASB 0.41%, SLC 0.38%) and asx50_q1 averages 0.149% (REA 0.27%). On
the same gross returns the honest expectations are roughly +0.28% for
small_q1 and +0.13% for asx50_q1 -- still positive, roughly half what was
quoted, and for small_q1 no longer clear of its own trade-to-trade noise.

One survivor out of roughly a dozen variants tested is about what multiple
testing produces by itself, and buying stocks that just fell 3% is exactly the
strategy that survivorship bias flatters -- the universe is TODAY's ASX 300, so
names that gapped down and never recovered are absent. Hence: paper, not money.

**Both cohorts are booked** even though only one cleared significance, because
the ASX 50 list is the one whose result cannot be a microstructure artifact
(liquid mega-caps, tight spreads) and it deserves its own out-of-sample record.
They are tagged separately and never pooled.

**The lists are frozen.** They were selected on 2016-2020 data only and are
hard-coded here on purpose. Recomputing them from current data would quietly
turn an out-of-sample test into an in-sample one, which is how a paper book
starts agreeing with its own backtest.

**On the entry price and the 20-minute data delay.** The job runs at 11:25
Sydney and books the last COMPLETE 5-minute bar available then, which is
roughly the 11:00-11:05 price, and stores that bar's timestamp so the lag is
visible in the record rather than assumed away. The measured intraday shape
says this should cost little -- the ASX 50 reversion path is -0.194% by 11:00,
-0.245% by 12:00, -0.254% by 13:00, i.e. nearly flat after 11:00 -- but
`entry_bar_ts` is stored so that claim can be checked against real fills
instead of trusted.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "swing.db"
SYDNEY = "Australia/Sydney"

# Frozen quintile lists -- fit on 2016-08-30..2021-01-01, COVID (Feb-Dec 2020)
# excluded, ranked by the OLS slope of daytime return on overnight gap. Do not
# regenerate; see the module docstring.
COHORTS: dict[str, list[str]] = {
    "asx50_q1": ["ALL", "AMC", "CPU", "GMG", "NWS", "REA", "SOL", "WOW"],
    "small_q1": ["ARB", "ARF", "ASB", "AUB", "BWP", "CIP", "CLW", "CNI", "CQR",
                 "DDR", "EVT", "GOZ", "IMD", "INA", "MXT", "NHC", "SLC", "SSM",
                 "WPR"],
}
FIT_WINDOW = "2016-08-30..2021-01-01"

GAP_THRESHOLD = -3.0          # gap must be at or below this, in percent
DEFAULT_DOLLARS = 20_000.0
DEFAULT_MAX_POSITIONS = 5     # ~60 signals/year expected, so this rarely binds

# The ASX 200 ETF, NOT the ^AXJO index, for anything that touches an OPEN
# price. Verified 2026-08-31: ^AXJO's Open equals its previous Close on 21 of
# 21 sessions (sd 0.0000%) because the index is carried forward rather than
# struck at a traded auction. Benchmarking an overnight or open-to-close leg
# against it silently returns 0.0 for every row -- which is exactly what this
# book recorded until today. STW tracks the same index and has a genuine
# traded open (sd 0.268% over the same window).
MARKET_PROXY = "STW.AX"
MIN_PRICE = 1.00

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gap_reversion_trades (
    trade_date    TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    cohort        TEXT NOT NULL,
    prev_close    REAL,
    open_price    REAL,
    gap_pct       REAL,
    scanned_at    TEXT NOT NULL,
    selected      INTEGER,
    shares        INTEGER,
    entry_price   REAL,
    entry_bar_ts  TEXT,
    entry_at      TEXT,
    exit_price    REAL,
    exit_at       TEXT,
    return_pct    REAL,
    pnl           REAL,
    mkt_day_pct   REAL,
    PRIMARY KEY (trade_date, ticker)
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _sydney_today() -> str:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(SYDNEY)).date().isoformat()


def _all_tickers() -> list[str]:
    return sorted({t for names in COHORTS.values() for t in names})


def _cohort_of(ticker: str) -> str:
    for name, members in COHORTS.items():
        if ticker in members:
            return name
    return "unknown"


def scan(threshold: float = GAP_THRESHOLD,
         trade_date: str | None = None) -> dict[str, Any]:
    """Find today's gap-down names in the frozen cohorts.

    The gap uses today's OPEN against yesterday's CLOSE, both of which are
    settled facts by mid-morning -- unlike the entry price, this part of the
    signal carries no look-ahead.
    """
    import yfinance as yf

    from .yf_lock import YF_LOCK

    day = trade_date or _sydney_today()
    tickers = _all_tickers()
    with YF_LOCK:
        data = yf.download([f"{t}.AX" for t in tickers], period="7d", interval="1d",
                           group_by="ticker", auto_adjust=False, threads=True,
                           progress=False)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    found, no_bar = [], []
    with _connect() as conn:
        for t in tickers:
            try:
                df = data[f"{t}.AX"].dropna(subset=["Close"])
            except (KeyError, TypeError):
                no_bar.append(t)
                continue
            dates = [d.date().isoformat() for d in df.index]
            if day not in dates or dates.index(day) == 0:
                no_bar.append(t)
                continue
            i = dates.index(day)
            open_px = float(df.iloc[i]["Open"])
            prev_close = float(df.iloc[i - 1]["Close"])
            if not open_px or not prev_close:
                no_bar.append(t)
                continue
            gap = (open_px / prev_close - 1) * 100
            if gap > threshold:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO gap_reversion_trades"
                " (trade_date, ticker, cohort, prev_close, open_price, gap_pct, scanned_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (day, t, _cohort_of(t), round(prev_close, 4), round(open_px, 4),
                 round(gap, 3), now))
            found.append({"ticker": t, "cohort": _cohort_of(t),
                          "gap_pct": round(gap, 3), "open": round(open_px, 4)})
        conn.commit()
    found.sort(key=lambda r: r["gap_pct"])
    return {"trade_date": day, "threshold": threshold, "scanned": len(tickers),
            "candidates": len(found), "rows": found, "no_bar": no_bar}


def open_positions(dollars: float = DEFAULT_DOLLARS,
                   max_positions: int = DEFAULT_MAX_POSITIONS,
                   min_price: float = MIN_PRICE,
                   trade_date: str | None = None) -> dict[str, Any]:
    """Book today's candidates at the last complete 5-minute bar.

    Ranked by gap size (most negative first) when the cap binds. The 5m bar's
    own timestamp is stored as `entry_bar_ts` -- with ~20 minute delayed data
    the bar booked at 11:25 is a ~11:00 price, and that gap should be auditable
    rather than buried.
    """
    import yfinance as yf

    from .yf_lock import YF_LOCK

    day = trade_date or _sydney_today()
    with _connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM gap_reversion_trades WHERE trade_date=? AND selected IS NULL"
            " ORDER BY gap_pct ASC", (day,))]
    if not rows:
        return {"opened": 0, "reason": f"no unselected candidates for {day}"}

    picked = [r for r in rows if (r["open_price"] or 0) >= min_price][:max_positions]
    if not picked:
        return {"opened": 0, "reason": f"no candidates at or above ${min_price}"}

    with YF_LOCK:
        intraday = yf.download([f"{p['ticker']}.AX" for p in picked], period="1d",
                               interval="5m", group_by="ticker", auto_adjust=False,
                               threads=True, progress=False)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    opened, skipped = [], []
    with _connect() as conn:
        conn.execute("UPDATE gap_reversion_trades SET selected=0"
                     " WHERE trade_date=? AND selected IS NULL", (day,))
        for p in picked:
            px, bar_ts = None, None
            try:
                df = intraday[f"{p['ticker']}.AX"].dropna(subset=["Close"])
                if not df.empty:
                    px = float(df["Close"].iloc[-1])
                    bar_ts = df.index[-1].tz_convert(SYDNEY).isoformat(timespec="minutes")
            except (KeyError, TypeError, IndexError, AttributeError):
                pass
            if not px:
                skipped.append(p["ticker"])
                continue
            shares = int(dollars // px)
            if shares < 1:
                skipped.append(p["ticker"])
                continue
            conn.execute(
                "UPDATE gap_reversion_trades SET selected=1, shares=?, entry_price=?,"
                " entry_bar_ts=?, entry_at=? WHERE trade_date=? AND ticker=?",
                (shares, round(px, 4), bar_ts, now, day, p["ticker"]))
            opened.append({"ticker": p["ticker"], "cohort": p["cohort"],
                           "gap_pct": p["gap_pct"], "shares": shares,
                           "entry": round(px, 4), "entry_bar_ts": bar_ts})
        conn.commit()
    return {"trade_date": day, "opened": len(opened), "positions": opened,
            "skipped_no_price": skipped, "considered": len(rows)}


def close_positions(days_back: int = 5) -> dict[str, Any]:
    """Exit every open position at the ACTUAL closing price of its own day.

    Same-day exit: the backtest measured open/11:00 -> close, so a position
    left overnight is a different strategy and would silently pick up the
    overnight drift the other paper book is already testing.
    """
    import yfinance as yf

    from .yf_lock import YF_LOCK

    with _connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM gap_reversion_trades WHERE selected=1 AND exit_price IS NULL"
            " AND trade_date >= date('now', ?)", (f"-{days_back} days",))]
    if not rows:
        return {"closed": 0, "reason": "no open positions"}

    symbols = sorted({f"{r['ticker']}.AX" for r in rows}) + [MARKET_PROXY]
    with YF_LOCK:
        data = yf.download(symbols, period="7d", interval="1d", group_by="ticker",
                           auto_adjust=False, threads=True, progress=False)

    def bar(sym: str, day: str) -> dict[str, float] | None:
        try:
            df = data[sym].dropna(subset=["Close"])
        except (KeyError, TypeError):
            return None
        dates = [d.date().isoformat() for d in df.index]
        if day not in dates:
            return None
        row = df.iloc[dates.index(day)]
        return {"open": float(row["Open"]), "close": float(row["Close"])}

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    closed, pending = [], []
    with _connect() as conn:
        for r in rows:
            b = bar(f"{r['ticker']}.AX", r["trade_date"])
            if not b:
                pending.append(r["ticker"])
                continue
            exit_px = b["close"]
            ret = (exit_px / r["entry_price"] - 1) * 100
            pnl = (exit_px - r["entry_price"]) * (r["shares"] or 0)
            mkt = bar(MARKET_PROXY, r["trade_date"])
            mkt_day = ((mkt["close"] / mkt["open"] - 1) * 100) if mkt else None
            conn.execute(
                "UPDATE gap_reversion_trades SET exit_price=?, exit_at=?, return_pct=?,"
                " pnl=?, mkt_day_pct=? WHERE trade_date=? AND ticker=?",
                (round(exit_px, 4), now, round(ret, 3), round(pnl, 2),
                 round(mkt_day, 3) if mkt_day is not None else None,
                 r["trade_date"], r["ticker"]))
            closed.append({"ticker": r["ticker"], "cohort": r["cohort"],
                           "trade_date": r["trade_date"], "gap_pct": r["gap_pct"],
                           "entry": r["entry_price"], "exit": round(exit_px, 4),
                           "return_pct": round(ret, 3), "pnl": round(pnl, 2)})
        # Second pass: the proxy can publish a day behind the individual names in
        # yfinance, so a trade closed at 16:25 on its own day has no index bar
        # yet and would keep mkt_day_pct NULL forever (the main loop only looks
        # at rows still open). Backfill it here on a later run instead.
        stale = [dict(r) for r in conn.execute(
            "SELECT trade_date FROM gap_reversion_trades WHERE exit_price IS NOT NULL"
            " AND mkt_day_pct IS NULL AND trade_date >= date('now', ?)",
            (f"-{days_back + 5} days",))]
        filled = 0
        if stale:
            with YF_LOCK:
                idx = yf.download(MARKET_PROXY, period="1mo", interval="1d",
                                  auto_adjust=False, progress=False)
            if not idx.empty:
                if hasattr(idx.columns, "levels"):
                    idx.columns = idx.columns.droplevel(1)
                by_date = {d.date().isoformat(): row for d, row in idx.iterrows()}
                for day in {r["trade_date"] for r in stale}:
                    row = by_date.get(day)
                    if row is None or not float(row["Open"]):
                        continue
                    pct = (float(row["Close"]) / float(row["Open"]) - 1) * 100
                    filled += conn.execute(
                        "UPDATE gap_reversion_trades SET mkt_day_pct=?"
                        " WHERE trade_date=? AND mkt_day_pct IS NULL"
                        " AND exit_price IS NOT NULL", (round(pct, 3), day)).rowcount
        conn.commit()
    return {"closed": len(closed), "trades": closed, "pending_no_bar": pending,
            "market_backfilled": filled}


def book(days: int = 180) -> dict[str, Any]:
    """Every trade plus per-cohort summary. Cohorts are never pooled -- they
    are two separate hypotheses with different expected effect sizes."""
    with _connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM gap_reversion_trades WHERE selected=1"
            " AND trade_date >= date('now', ?) ORDER BY trade_date DESC, ticker",
            (f"-{days} days",))]

    summary = {}
    for name in COHORTS:
        done = [r for r in rows if r["cohort"] == name and r["return_pct"] is not None]
        if not done:
            summary[name] = {"n_closed": 0, "n_open":
                             sum(1 for r in rows if r["cohort"] == name)}
            continue
        rets = [r["return_pct"] for r in done]
        rel = [r["return_pct"] - r["mkt_day_pct"] for r in done
               if r["mkt_day_pct"] is not None]
        summary[name] = {
            "n_closed": len(done),
            "n_open": sum(1 for r in rows
                          if r["cohort"] == name and r["return_pct"] is None),
            "mean_return_pct": round(sum(rets) / len(rets), 3),
            "mean_vs_market_pct": round(sum(rel) / len(rel), 3) if rel else None,
            "win_rate_pct": round(100 * sum(1 for x in rets if x > 0) / len(rets), 1),
            "total_pnl": round(sum(r["pnl"] or 0 for r in done), 2),
            # Gross expectation less the cohort's MEASURED mean round-trip
            # cost, not the 0.05% flat assumption the backtest used.
            "backtest_expectation_pct": 0.283 if name == "small_q1" else 0.129,
            "assumed_cost_pct": 0.270 if name == "small_q1" else 0.149,
        }
    return {"days": days, "fit_window": FIT_WINDOW, "cohorts": COHORTS,
            "summary": summary, "trades": rows}
