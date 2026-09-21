"""Forward outcomes for document-backed scores.

The first recorded net score for each ticker/session is evaluated from the
first actual ASX opening bar after computed_at. Only completed sessions are
used. These rows live in document_signal_outcomes, separate from the legacy
signal_outcomes table retained for retrospective research. They must not be
joined to retrospective A/B scores as if those scores existed at entry time.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
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


def _connect(*, prospective: bool = False) -> sqlite3.Connection:
    from .mover_log import DB_PATH
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.row_factory = sqlite3.Row
    table = "document_signal_outcomes" if prospective else "signal_outcomes"
    conn.executescript(_SCHEMA.split("\nALTER")[0].replace("signal_outcomes", table))
    have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    for h in HORIZONS:
        for col in (f"fwd_{h}d_pct", f"sect_{h}d_pct"):
            if col not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} REAL")
    for col in ("evaluation_basis", "score_available_at"):
        if col not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
    conn.commit()
    return conn


def collect(days: int = 30, period: str = "3mo") -> dict[str, Any]:
    """Snapshot prices for every scored ticker-session in the last `days`."""
    import sqlite3 as sq
    from .screener import bulk_daily
    from .sectors import MARKET_INDEX, SECTOR_INDICES, benchmark_index, get_map
    from .forward_returns import _session_opens
    from .yf_lock import YF_LOCK

    from .asx_signals import DB_PATH as sig_db
    if not sig_db.exists():
        return {"error": "no asx_signals db"}
    sig = sq.connect(f"file:{sig_db}?mode=ro", uri=True)
    sig.row_factory = sq.Row
    nets = [dict(r) for r in sig.execute(
        "SELECT * FROM ticker_signals WHERE session_date >= date('now', ?)",
        (f"-{days} days",))]
    if sig.execute("SELECT 1 FROM sqlite_master WHERE name='ticker_signal_history'").fetchone():
        nets += [dict(r) for r in sig.execute(
            "SELECT * FROM ticker_signal_history WHERE session_date >= date('now', ?)", (f"-{days} days",))]
    # Old rows lack document provenance and cannot be promoted into forward evidence.
    first = {}
    for n in sorted(nets, key=lambda r: r["computed_at"]):
        if n.get("prompt_version") in {
            "v6-document-required", "v7-ticker-memory", "v8-document-fast",
        }:
            first.setdefault((n["ticker"], n["session_date"]), n)
    nets = list(first.values())
    sig.close()
    # Continue maturing already-recorded retrospective observations without
    # substituting a later score or mixing them with the prospective sample.
    with _connect() as legacy:
        nets += [{**dict(r), "_legacy_day": r["as_of"]} for r in legacy.execute(
            "SELECT * FROM signal_outcomes WHERE as_of>=date('now', ?) AND fwd_10d_pct IS NULL",
            (f"-{days} days",))]
    if not nets:
        return {"updated": 0, "reason": "no scored sessions in range"}

    tickers = sorted({n["ticker"] for n in nets})
    smap = get_map(tickers)
    with YF_LOCK:
        px = bulk_daily(tickers, period=period, batch=40)
    bench = sorted({benchmark_index(smap.get(t)) for t in tickers}
                   | {MARKET_INDEX})
    bench = [b for b in bench if b in SECTOR_INDICES or b == MARKET_INDEX]
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

    cutoff = datetime.now(ZoneInfo("Australia/Sydney"))
    def completed(day):
        return datetime.fromisoformat(day + "T16:20:00").replace(tzinfo=ZoneInfo("Australia/Sydney")) < cutoff

    with _connect(prospective=True) as db:
        for n in nets:
            t = n["ticker"]
            df = px.get(t)
            if df is None:
                skipped += 1
                continue
            dates = [d.date().isoformat() for d in df.index]
            historical = bool(n.get("_legacy_day"))
            table = "signal_outcomes" if historical else "document_signal_outcomes"
            if historical:
                entries = [d for d in dates if d == n["_legacy_day"]]
            else:
                available = datetime.fromisoformat(n["computed_at"].replace("Z", "+00:00"))
                # Daily bars support the first actual opening after scoring.
                entries = [d for d in dates if datetime.fromisoformat(d + "T10:00:00").replace(
                    tzinfo=ZoneInfo("Australia/Sydney")) > available]
            if not entries:
                skipped += 1
                continue
            day = entries[0]
            if not completed(day):
                skipped += 1
                continue
            old = db.execute(f"SELECT * FROM {table} WHERE ticker=? AND as_of=?", (t, day)).fetchone()
            if not historical and old and old["evaluation_basis"] and old["score_available_at"] != n["computed_at"]:
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
                "evaluation_basis": "retrospective" if historical else "first_open_after_score",
                "score_available_at": n.get("computed_at"),
            }
            if bd and day in bd:
                j = bd.index(day)
                bo = bopens.get(bsym, {}).get(day)
                bc = float(bdf.iloc[j]["Close"])
                bpc = float(bdf.iloc[j - 1]["Close"]) if j else None
                bentry = bc if historical else bo
                vals["sect_open_close_pct"] = round((bc / bo - 1) * 100, 4) if bo else None
                vals["sect_full_day_pct"] = round((bc / bpc - 1) * 100, 4) if bpc else None
                for h in HORIZONS:
                    k = j + h
                    vals[f"sect_{h}d_pct"] = (round((float(bdf.iloc[k]["Close"]) / bentry - 1) * 100, 4)
                                              if k < len(bd) and completed(bd[k]) and bentry else None)
            for h in HORIZONS:
                k = i + h
                vals[f"fwd_{h}d_pct"] = (round((float(df.iloc[k]["Close"]) / (c if historical else o) - 1) * 100, 4)
                                         if k < len(dates) and completed(dates[k]) else None)
            cols = ["ticker", "as_of"] + list(vals)
            db.execute(
                f"INSERT OR REPLACE INTO {table} ({', '.join(cols)})"
                f" VALUES ({', '.join('?' * len(cols))})",
                [t, day, *vals.values()])
            written += 1
        db.commit()
    return {"updated": written, "skipped_no_price": skipped,
            "scored_sessions": len(nets), "tickers": len(tickers)}
