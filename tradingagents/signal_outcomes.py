"""Price outcomes for EVERY scored ticker-session, not just the ones that moved.

**Why this table exists.** Classifier IC was being measured on `mover_log`,
which holds only names that passed the scanner's move/volume screen -- 6 to 19
rows a session against the 130-166 tickers actually scored. Two problems, and
the second is worse than the first:

  * **Width.** IC needs a cross-section to rank. Simulated at a true rank-IC
    of 0.04 (about what published methods achieve), 15 names/day takes ~154
    trading days to reach t=2 while 100 names/day takes ~27. Total
    observations needed is roughly constant, so a wider daily cross-section
    buys calendar time almost linearly.
  * **Selection.** `mover_log` is conditioned on the OUTCOME. A high-scoring
    announcement whose stock did nothing never enters the sample at all, so
    the measurement cannot see the classifier's most important failure mode --
    predicting a move that never came. Every IC computed on movers answers the
    narrower question "among today's movers, did the score rank them", and
    quietly assumes the rest away.

**Attribution uses the TRADEABLE session, not the publication date.** A ticker
whose news landed after 16:00 cannot be judged on that day's move; see
`asx_feed.tradeable_session_for`. The stored `as_of` is the earliest session
in which any of that ticker-day's announcements could actually be acted on.

**Both return framings are stored.** `open_close_pct` is the honest window for
a pre-open announcement, `full_day_pct` for one released during the session --
and the sector leg of each is stored beside it, never pre-subtracted.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

HORIZONS = (1, 3, 5, 10)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signal_outcomes (
    ticker        TEXT NOT NULL,
    as_of         TEXT NOT NULL,      -- tradeable session, not publication date
    session_date  TEXT,               -- what ticker_signals filed it under
    score         INTEGER,
    signal        TEXT,
    n_announcements INTEGER,
    model         TEXT,
    prompt_version TEXT,
    prev_close    REAL,
    open          REAL,
    close         REAL,
    open_close_pct REAL,
    full_day_pct  REAL,
    sector_index  TEXT,
    sector_corr   REAL,
    sect_open_close_pct REAL,
    sect_full_day_pct   REAL,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (ticker, as_of)
);
"""
for _h in HORIZONS:
    _SCHEMA += (f"\nALTER TABLE signal_outcomes ADD COLUMN fwd_{_h}d_pct REAL;"
                f"\nALTER TABLE signal_outcomes ADD COLUMN sect_{_h}d_pct REAL;")


def _connect() -> sqlite3.Connection:
    from .mover_log import DB_PATH
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA.split("\nALTER")[0])
    have = {r[1] for r in conn.execute("PRAGMA table_info(signal_outcomes)")}
    for h in HORIZONS:
        for col in (f"fwd_{h}d_pct", f"sect_{h}d_pct"):
            if col not in have:
                conn.execute(f"ALTER TABLE signal_outcomes ADD COLUMN {col} REAL")
    return conn


def collect(days: int = 30, period: str = "3mo") -> dict[str, Any]:
    """Snapshot prices for every scored ticker-session in the last `days`."""
    import sqlite3 as sq
    from pathlib import Path

    from .asx_feed import tradeable_session_for
    from .screener import bulk_daily
    from .sectors import MARKET_INDEX, SECTOR_INDICES, benchmark_index, get_map
    from .forward_returns import _session_opens
    from .yf_lock import YF_LOCK

    sig_db = Path.home() / ".tradingagents" / "asx_signals.db"
    if not sig_db.exists():
        return {"error": "no asx_signals db"}
    sig = sq.connect(f"file:{sig_db}?mode=ro", uri=True)
    sig.row_factory = sq.Row
    nets = [dict(r) for r in sig.execute(
        "SELECT * FROM ticker_signals WHERE session_date >= date('now', ?)",
        (f"-{days} days",))]
    if not nets:
        return {"updated": 0, "reason": "no scored sessions in range"}

    # Announcement release times, to find the earliest ACTIONABLE session.
    from .asx_feed import DB_PATH as ASX_DB
    rel: dict[str, str] = {}
    if ASX_DB.exists():
        conn = sq.connect(f"file:{ASX_DB}?mode=ro", uri=True)
        rel = {r[0]: r[1] for r in conn.execute(
            "SELECT fingerprint, COALESCE(released_at, seen_at) FROM announcements")}
        conn.close()
    for n in nets:
        fps = [f for f in (n.get("fingerprints") or "").split(",") if f]
        sessions = sorted({tradeable_session_for(rel[f]) for f in fps if f in rel})
        n["as_of"] = sessions[0] if sessions else n["session_date"]

    tickers = sorted({n["ticker"] for n in nets})
    smap = get_map(tickers)
    with YF_LOCK:
        px = bulk_daily(tickers, period=period, batch=40)
    bench = sorted({benchmark_index(smap.get(t)) for t in tickers}
                   | {MARKET_INDEX})
    bench = [b for b in bench if b in SECTOR_INDICES or b == MARKET_INDEX]
    with YF_LOCK:
        bpx = bulk_daily([b.lstrip("^") for b in []], period=period) or {}
    import yfinance as yf
    with YF_LOCK:
        braw = yf.download(bench, period=period, interval="1d", group_by="ticker",
                           auto_adjust=False, threads=True, progress=False)
    bopens = _session_opens(bench)

    def frame(sym):
        try:
            df = braw[sym].dropna(subset=["Close"])
        except (KeyError, TypeError):
            return None, None
        return [d.date().isoformat() for d in df.index], df

    bench_f = {b: frame(b) for b in bench}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    written, skipped = 0, 0

    with _connect() as db:
        for n in nets:
            t, day = n["ticker"], n["as_of"]
            df = px.get(t)
            if df is None:
                skipped += 1
                continue
            dates = [d.date().isoformat() for d in df.index]
            if day not in dates:
                skipped += 1
                continue
            i = dates.index(day)
            o, c = float(df.iloc[i]["Open"]), float(df.iloc[i]["Close"])
            pc = float(df.iloc[i - 1]["Close"]) if i else None
            if not o or not c:
                skipped += 1
                continue
            m = smap.get(t) or {}
            bsym = benchmark_index(m)
            bd, bdf = bench_f.get(bsym, (None, None))

            vals: dict[str, Any] = {
                "session_date": n["session_date"], "score": n["score"],
                "signal": n["signal"], "n_announcements": n.get("n_announcements"),
                "model": n.get("model"), "prompt_version": n.get("prompt_version"),
                "prev_close": pc, "open": o, "close": c,
                "open_close_pct": round((c / o - 1) * 100, 4),
                "full_day_pct": round((c / pc - 1) * 100, 4) if pc else None,
                "sector_index": bsym, "sector_corr": m.get("sector_corr"),
                "updated_at": now,
            }
            if bd and day in bd:
                j = bd.index(day)
                bo = bopens.get(bsym, {}).get(day)
                bc = float(bdf.iloc[j]["Close"])
                bpc = float(bdf.iloc[j - 1]["Close"]) if j else None
                vals["sect_open_close_pct"] = round((bc / bo - 1) * 100, 4) if bo else None
                vals["sect_full_day_pct"] = round((bc / bpc - 1) * 100, 4) if bpc else None
                for h in HORIZONS:
                    k = j + h
                    vals[f"sect_{h}d_pct"] = (round((float(bdf.iloc[k]["Close"]) / bc - 1) * 100, 4)
                                              if k < len(bd) and bc else None)
            for h in HORIZONS:
                k = i + h
                vals[f"fwd_{h}d_pct"] = (round((float(df.iloc[k]["Close"]) / c - 1) * 100, 4)
                                         if k < len(dates) else None)
            cols = ["ticker", "as_of"] + list(vals)
            db.execute(
                f"INSERT OR REPLACE INTO signal_outcomes ({', '.join(cols)})"
                f" VALUES ({', '.join('?' * len(cols))})",
                [t, day, *vals.values()])
            written += 1
        db.commit()
    return {"updated": written, "skipped_no_price": skipped,
            "scored_sessions": len(nets), "tickers": len(tickers)}
