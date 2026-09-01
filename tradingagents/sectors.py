"""S&P/ASX sector indices: the daily sector board, and the peer benchmark that
mover-day forward returns are measured against (2026-08-31, user request).

**Why this exists.** BC8 released FY26 results scoring 90 from the classifier
and fell 6.2% the same morning, which reads as a rejected report until you
notice ^AXGD (the gold index) fell 5.14% that day and the ASX 200 was flat.
Measured against the ASX 200 the announcement looks like a large miss;
measured against its own sector it barely moved. Scoring every announcement
against a broad index charges stock-specific classifiers for sector beta, and
the calibration work then blames the model for something it never predicted.

**Sectors are assigned by correlation, not by a metadata lookup.** asxbrief's
`universe.industry` column is entirely NULL, and yfinance's per-ticker sector
field costs one request per name. Correlating a ticker's daily returns against
each sector index and taking the best fit needs no extra requests, produces a
number (the correlation) that says how much to trust the assignment, and
naturally puts a gold miner in ^AXGD rather than the ^AXMJ bucket a GICS
lookup would give it. A ticker that correlates with nothing falls back to the
ASX 200, and `sector_corr` records which case a row is.

**^AXGD and ^AXJR overlap ^AXMJ on purpose.** They are sub-indices, not
disjoint GICS sectors. For a benchmark the tighter peer group is the better
comparison, and best-correlation picks it automatically; for the dashboard
panel seeing gold and broad resources side by side is the point.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

MARKET_INDEX = "^AXJO"

# Verified live 2026-08-31: every symbol here returns daily bars from yfinance.
# ^AXJS (Small Ordinaries) and ^AXBJ (Banks) 404 and are deliberately absent.
SECTOR_INDICES: dict[str, str] = {
    "^AXEJ": "Energy",
    "^AXMJ": "Materials",
    "^AXGD": "Gold",
    "^AXJR": "Resources",
    "^AXNJ": "Industrials",
    "^AXDJ": "Consumer Discretionary",
    "^AXSJ": "Consumer Staples",
    "^AXHJ": "Health Care",
    "^AXFJ": "Financials",
    "^AXIJ": "Information Technology",
    "^AXAT": "All Technology",
    "^AXTJ": "Communication Services",
    "^AXUJ": "Utilities",
    "^AXPJ": "A-REITs",
}

# **Two different uses, two different bars.** As a BENCHMARK, a sector index
# is only valid if the ticker actually tracks it -- below MIN_CORR the ASX 200
# is the honest comparison instead. As a LABEL ("which of these are gold
# miners?") correlation is beside the point: a micro-cap explorer is still a
# gold explorer even though its returns are mostly idiosyncratic noise.
#
# The map therefore always stores the BEST-FIT index and its correlation, and
# the benchmark rule is applied at read time by `benchmark_for()`. Storing the
# fallback instead lost the label for 910 of 1208 tickers on the first build,
# which emptied the dashboard's sector column for most of the feed.
MIN_CORR = 0.35

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sector_map (
    ticker       TEXT PRIMARY KEY,
    sector_index TEXT NOT NULL,
    sector_name  TEXT,
    sector_corr  REAL,
    n_days       INTEGER,
    computed_at  TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    from .mover_log import DB_PATH
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _download(symbols: list[str], period: str, interval: str = "1d"):
    import yfinance as yf

    from .yf_lock import YF_LOCK
    with YF_LOCK:
        return yf.download(symbols, period=period, interval=interval,
                           group_by="ticker", auto_adjust=False, threads=True,
                           progress=False)


def _close(data, symbol: str):
    try:
        s = data[symbol]["Close"].dropna()
    except (KeyError, TypeError):
        return None
    return s if len(s) > 20 else None


# ---------------------------------------------------------------------------
# The dashboard panel
# ---------------------------------------------------------------------------

_board_cache: dict[str, Any] = {"ts": 0.0, "data": None}
_BOARD_TTL = 120.0


def board() -> dict[str, Any]:
    """Every sector index with its day, week and month move.

    The ASX 200 is included as the first row rather than left implicit -- a
    sector down 5% means one thing on a flat day and another on a day the
    index is down 4%, and the comparison should not require a second glance
    at the panel above.
    """
    import time
    now = time.monotonic()
    if _board_cache["data"] is not None and now - _board_cache["ts"] < _BOARD_TTL:
        return _board_cache["data"]

    symbols = [MARKET_INDEX] + list(SECTOR_INDICES)
    data = _download(symbols, period="3mo")
    rows, as_of = [], None
    for sym in symbols:
        label = "ASX 200" if sym == MARKET_INDEX else SECTOR_INDICES[sym]
        s = _close(data, sym)
        entry = {"symbol": sym, "label": label, "is_market": sym == MARKET_INDEX,
                 "last": None, "day_pct": None, "week_pct": None, "month_pct": None}
        if s is not None and len(s) >= 2:
            last = float(s.iloc[-1])
            entry["last"] = last
            entry["day_pct"] = (last / float(s.iloc[-2]) - 1) * 100
            if len(s) > 5:
                entry["week_pct"] = (last / float(s.iloc[-6]) - 1) * 100
            if len(s) > 21:
                entry["month_pct"] = (last / float(s.iloc[-22]) - 1) * 100
            as_of = as_of or str(s.index[-1].date())
        rows.append(entry)

    market_day = next((r["day_pct"] for r in rows if r["is_market"]), None)
    for r in rows:
        r["vs_market_pct"] = (r["day_pct"] - market_day
                              if r["day_pct"] is not None and market_day is not None
                              else None)
    sectors = [r for r in rows if not r["is_market"]]
    sectors.sort(key=lambda r: (r["day_pct"] is None, -(r["day_pct"] or 0)))
    out = {"as_of": as_of, "market": next(r for r in rows if r["is_market"]),
           "sectors": sectors, "fetched_at": datetime.now(timezone.utc).isoformat(
               timespec="seconds")}
    _board_cache.update(ts=now, data=out)
    return out


# ---------------------------------------------------------------------------
# The ticker -> sector assignment
# ---------------------------------------------------------------------------

def build_map(tickers: list[str], period: str = "1y",
              batch: int = 40) -> dict[str, Any]:
    """Assign each ticker the sector index its daily returns track best.

    Correlation of daily RETURNS, not of levels -- two rising series correlate
    on level regardless of whether they move together day to day, which is the
    thing being asked here.
    """
    import numpy as np
    import pandas as pd

    idx_data = _download(list(SECTOR_INDICES), period=period)
    idx_ret = {}
    for sym in SECTOR_INDICES:
        s = _close(idx_data, sym)
        if s is not None:
            idx_ret[sym] = s.pct_change().dropna()
    if not idx_ret:
        return {"error": "no sector index history"}
    idx_frame = pd.DataFrame(idx_ret)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    assigned, fallback, missing = 0, 0, []
    with _connect() as conn:
        for i in range(0, len(tickers), batch):
            chunk = tickers[i:i + batch]
            data = _download([f"{t}.AX" for t in chunk], period=period)
            for t in chunk:
                s = _close(data, f"{t}.AX")
                if s is None:
                    missing.append(t)
                    continue
                r = s.pct_change().dropna()
                joined = idx_frame.join(r.rename("_t"), how="inner").dropna()
                if len(joined) < 60:
                    missing.append(t)
                    continue
                corrs = joined.drop(columns="_t").corrwith(joined["_t"])
                corrs = corrs.dropna()
                if corrs.empty:
                    missing.append(t)
                    continue
                best = corrs.idxmax()
                val = float(corrs[best])
                name = SECTOR_INDICES[best]
                if val < MIN_CORR:
                    fallback += 1      # label kept, benchmark falls back at read time
                else:
                    assigned += 1
                conn.execute(
                    "INSERT OR REPLACE INTO sector_map"
                    " (ticker, sector_index, sector_name, sector_corr, n_days, computed_at)"
                    " VALUES (?,?,?,?,?,?)",
                    (t, best, name, round(val, 3), int(len(joined)), now))
        conn.commit()
    return {"assigned": assigned, "below_min_corr": fallback,
            "no_data": missing, "n_tickers": len(tickers)}


def announcing_tickers() -> list[str]:
    """Every ticker that has appeared in the announcement feed.

    The map originally covered the top-500 universe, which is the set that can
    produce a SCORED announcement. But the feed itself shows the whole
    exchange, so the dashboard's sector column was blank for 405 of the 551
    tickers announcing on 2026-08-31. Coverage follows the feed instead.
    """
    from .asx_feed import DB_PATH as ASX_DB
    if not ASX_DB.exists():
        return []
    conn = sqlite3.connect(f"file:{ASX_DB}?mode=ro", uri=True, timeout=10.0)
    try:
        return sorted({r[0] for r in conn.execute(
            "SELECT DISTINCT ticker FROM announcements WHERE ticker IS NOT NULL") if r[0]})
    finally:
        conn.close()


def get_map(tickers: list[str] | None = None) -> dict[str, dict[str, Any]]:
    with _connect() as conn:
        if tickers:
            q = ",".join("?" * len(tickers))
            rows = conn.execute(
                f"SELECT * FROM sector_map WHERE ticker IN ({q})", tickers).fetchall()
        else:
            rows = conn.execute("SELECT * FROM sector_map").fetchall()
    return {r["ticker"]: dict(r) for r in rows}


def benchmark_for(ticker: str) -> str:
    """The index to MEASURE this ticker against: its sector when the
    correlation supports it, the ASX 200 otherwise."""
    r = get_map([ticker]).get(ticker) or {}
    if r.get("sector_index") and (r.get("sector_corr") or 0) >= MIN_CORR:
        return r["sector_index"]
    return MARKET_INDEX


def benchmark_index(row: dict[str, Any] | None) -> str:
    """Same rule, applied to an already-fetched map row -- avoids a query per
    ticker when benchmarking a whole batch."""
    row = row or {}
    if row.get("sector_index") and (row.get("sector_corr") or 0) >= MIN_CORR:
        return row["sector_index"]
    return MARKET_INDEX


# Kept as the old name so existing callers keep working.
sector_for = benchmark_for
