"""`/setups` -- the end-of-day technical setup scanner over the ASX-300 proxy.

**What it does.** At 15:40 Sydney (`asx-ta-setups.timer`) it pulls two years
of daily bars for the top 300 by market cap, runs every detector in
`ta_detectors` on today's provisional bar, and stores each detection. At
17:10 (`asx-ta-setups-resolve.timer`) `resolve()` re-reads the completed bar
-- what did the close actually do, did the setup still hold -- and then fills
forward returns at 1/3/5/10/20 sessions as they mature, plus a date-matched
control. `scorecard()` turns that into a per-setup record; `backfill()` is
the one-off historical pass that gives every setup a baseline and puts it
through `hypothesis.run()`.

**The signal is knowable before the close.** Unlike the opening-auction
entries that voided earlier gap studies, a detection made at 15:40 can be
acted on in the 16:10 closing auction. The scorecard measures from that
close. `delayed_*` is the same trade entered at the NEXT close, for the
`delayed_entry` gate.

**Provisional bar.** At 15:40 yfinance's daily row is ~20 minutes stale and
excludes the closing auction, so today's volume is a fraction of a full day.
Volume rules divide by `provisional_vol_scale()` -- the median of
`vol_at_scan / vol_eod` over resolved live rows, seeded at 0.6 until ten
sessions have resolved. `state_at_close` records what the completed bar
showed, so the page can say how often the 15:40 view was wrong.

**Control.** `ta_control` holds the equal-weight forward return of every
liquid ticker on each date. A setup's "excess" is against that, not against
zero: a long setup that fires on a day the whole market rose 2% did not find
anything. The sector index (`sectors.benchmark_index`) is stored alongside as
the second benchmark, the one the gate harness uses.

**Every number on the page is a record, not a recommendation.** Nothing here
proposes trades; the badges only say how much evidence there is.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ta_detectors import (MIN_BARS, SETUPS, STATES, Detection, detect_all,
                           detect_history, liquid_mask)
from .ta_indicators import indicator_frame

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "swing.db"
SYDNEY = "Australia/Sydney"
HORIZONS = (1, 3, 5, 10, 20)
PRIMARY_HORIZON = 5
DEFAULT_LIMIT = 300
DEFAULT_PERIOD = "2y"
BACKFILL_PERIOD = "3y"
SEED_VOL_SCALE = 0.6
MIN_VOL_SCALE_SESSIONS = 10
DEDUP_SESSIONS = 5           # first_in_window: no same detection in the prior 5 sessions
MFE_BARS = 10
MIN_DATES_LIVE = 30
MIN_DATES_EDGE = 100

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ta_detections (
    scan_date       TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    setup_id        TEXT NOT NULL,
    source          TEXT NOT NULL,           -- 'live' | 'backfill'
    state           TEXT NOT NULL,
    tier            INTEGER NOT NULL,
    direction       INTEGER NOT NULL,
    company         TEXT,
    level           REAL,
    level_kind      TEXT,
    distance_pct    REAL,
    context_json    TEXT,
    price_at_scan   REAL,
    bar_provisional INTEGER NOT NULL,
    bar_source      TEXT,
    vol_ratio_at_scan REAL,
    vol_at_scan     REAL,
    scanned_at      TEXT NOT NULL,
    first_in_window INTEGER NOT NULL DEFAULT 1,
    close_eod       REAL,
    scan_vs_close_pct REAL,
    vol_eod         REAL,
    vol_ratio_eod   REAL,
    state_at_close  TEXT,
    next_close      REAL,
    fwd_1d_pct REAL, fwd_3d_pct REAL, fwd_5d_pct REAL, fwd_10d_pct REAL, fwd_20d_pct REAL,
    delayed_5d_pct REAL, delayed_10d_pct REAL,
    mfe_10d_pct REAL, mae_10d_pct REAL,
    bench_symbol    TEXT,
    bench_1d_pct REAL, bench_3d_pct REAL, bench_5d_pct REAL, bench_10d_pct REAL, bench_20d_pct REAL,
    fwd_bars_available INTEGER,
    resolved_at     TEXT,
    PRIMARY KEY (scan_date, ticker, setup_id, state, source)
);
CREATE INDEX IF NOT EXISTS idx_ta_det_setup ON ta_detections(source, setup_id, state, scan_date);
CREATE TABLE IF NOT EXISTS ta_scan_runs (
    scan_date       TEXT NOT NULL,
    source          TEXT NOT NULL,
    scanned_at      TEXT NOT NULL,
    sydney_time     TEXT,
    period          TEXT,
    n_universe      INTEGER,
    n_with_data     INTEGER,
    n_today_bar     INTEGER,
    n_liquid        INTEGER,
    n_detections    INTEGER,
    n_tier1         INTEGER,
    n_tier2         INTEGER,
    bar_provisional INTEGER,
    bar_source      TEXT,
    vol_scale       REAL,
    elapsed_s       REAL,
    liquid_json     TEXT,
    PRIMARY KEY (scan_date, source)
);
CREATE TABLE IF NOT EXISTS ta_control (
    date            TEXT PRIMARY KEY,
    n               INTEGER,
    ew_fwd_1d_pct REAL, ew_fwd_3d_pct REAL, ew_fwd_5d_pct REAL, ew_fwd_10d_pct REAL, ew_fwd_20d_pct REAL,
    mkt_fwd_5d_pct REAL, mkt_fwd_10d_pct REAL,
    updated_at      TEXT
);
CREATE TABLE IF NOT EXISTS ta_setup_baseline (
    setup_id        TEXT NOT NULL,
    state           TEXT NOT NULL,
    computed_at     TEXT NOT NULL,
    period_start    TEXT,
    period_end      TEXT,
    n               INTEGER,
    n_dates         INTEGER,
    mean_5d REAL, mean_10d REAL, hit_rate_5d REAL,
    ctrl_mean_5d REAL, ctrl_mean_10d REAL,
    excess_5d REAL, excess_10d REAL, t_5d REAL, t_10d REAL,
    verdict         TEXT,
    killed_by       TEXT,
    blocked_by      TEXT,
    hypothesis_run_at TEXT,
    median_5d REAL, tail_share_5d REAL, mean_ex_tail_5d REAL,
    PRIMARY KEY (setup_id, state)
);
"""
_BASELINE_MIGRATIONS = {"median_5d": "REAL", "tail_share_5d": "REAL", "mean_ex_tail_5d": "REAL"}


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    have = {r["name"] for r in conn.execute("PRAGMA table_info(ta_setup_baseline)")}
    for col, typ in _BASELINE_MIGRATIONS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE ta_setup_baseline ADD COLUMN {col} {typ}")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sydney_now() -> datetime:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(SYDNEY))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_bars(limit: int = DEFAULT_LIMIT, period: str = DEFAULT_PERIOD,
               tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Daily OHLCV for the top-`limit` universe (or an explicit list), via the
    same batched yfinance path `/recommendations` uses each morning. Frames
    are trimmed to the five columns the detectors read."""
    from .gap_study import _universe
    from .screener import bulk_daily
    from .yf_lock import YF_LOCK

    if tickers is None:
        tickers = _universe(limit)
    if not tickers:
        return {}
    with YF_LOCK:
        hist = bulk_daily(tickers, period=period, batch=40)
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = hist.get(t)
        if df is None or df.empty:
            continue
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).copy()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        out[t] = df
    return out


def _ensure_today_bar(frames: dict[str, pd.DataFrame], today: str) -> dict[str, str]:
    """Which frames carry a bar for `today`. Frames that lack one get a bar
    aggregated from the last session's hourly bars (one batched call for
    just those tickers); returns {ticker: bar_source}. Tickers with neither
    are absent from the result and skipped by the scan."""
    import yfinance as yf

    from .yf_lock import YF_LOCK

    src: dict[str, str] = {}
    missing: list[str] = []
    today_ts = pd.Timestamp(today)
    for t, df in frames.items():
        if len(df) and df.index[-1] == today_ts:
            src[t] = "daily_partial"
        else:
            missing.append(t)
    if not missing:
        return src
    for i in range(0, len(missing), 40):
        chunk = missing[i:i + 40]
        try:
            with YF_LOCK:
                data = yf.download([f"{t}.AX" for t in chunk], period="5d", interval="1h",
                                   group_by="ticker", auto_adjust=True, threads=True,
                                   progress=False)
        except Exception as exc:
            logger.warning("hourly fallback failed for %s..%s: %s", chunk[0], chunk[-1], exc)
            continue
        for t in chunk:
            try:
                h = data[f"{t}.AX"].dropna(subset=["Close"])
            except (KeyError, TypeError):
                continue
            if h.empty:
                continue
            idx = h.index.tz_convert(SYDNEY) if h.index.tz is not None else h.index.tz_localize(SYDNEY)
            sel = [d.date().isoformat() == today for d in idx]
            hb = h[sel]
            if hb.empty:
                continue
            bar = pd.DataFrame({"Open": [float(hb["Open"].iloc[0])],
                                "High": [float(hb["High"].max())],
                                "Low": [float(hb["Low"].min())],
                                "Close": [float(hb["Close"].iloc[-1])],
                                "Volume": [float(hb["Volume"].sum())]}, index=[today_ts])
            frames[t] = pd.concat([frames[t], bar])
            src[t] = "hourly_agg"
    return src


def _index_bars(symbols: list[str], period: str) -> dict[str, tuple[list[str], np.ndarray]]:
    """{index symbol: (dates, closes)} for benchmark returns."""
    from .forward_returns import _bars, _series
    out: dict[str, tuple[list[str], np.ndarray]] = {}
    if not symbols:
        return out
    try:
        data = _bars(symbols, period=period)
    except Exception as exc:
        logger.warning("index bars failed: %s", exc)
        return out
    for sym in symbols:
        dates, df = _series(data, sym)
        if dates:
            out[sym] = (dates, df["Close"].to_numpy(float))
    return out


def _bench_symbols(tickers: list[str]) -> dict[str, str]:
    from .sectors import MARKET_INDEX, benchmark_index, get_map
    try:
        smap = get_map(tickers)
    except Exception:
        smap = {}
    return {t: benchmark_index(smap.get(t)) for t in tickers} | {"__market__": MARKET_INDEX}


def _fwd(closes: np.ndarray, i: int, h: int) -> float | None:
    if i + h < len(closes) and closes[i] > 0:
        return float((closes[i + h] / closes[i] - 1.0) * 100.0)
    return None


def _bench_fwd(bars: dict[str, tuple[list[str], np.ndarray]], sym: str, date: str, h: int) -> float | None:
    b = bars.get(sym)
    if not b:
        return None
    dates, closes = b
    i = bisect_left(dates, date)
    if i >= len(dates) or dates[i] != date:
        i -= 1                      # index has no bar that day: use the prior close
    if i < 0:
        return None
    return _fwd(closes, i, h)


def _forward_metrics(df: pd.DataFrame, i: int) -> dict[str, Any]:
    """Completed-bar outcomes for a detection on bar `i` of `df`."""
    closes = df["Close"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    out: dict[str, Any] = {"close_eod": float(closes[i]),
                           "fwd_bars_available": int(len(closes) - 1 - i)}
    for h in HORIZONS:
        out[f"fwd_{h}d_pct"] = _fwd(closes, i, h)
    out["next_close"] = float(closes[i + 1]) if i + 1 < len(closes) else None
    for h in (5, 10):
        out[f"delayed_{h}d_pct"] = _fwd(closes, i + 1, h) if i + 1 < len(closes) else None
    j = min(len(closes), i + 1 + MFE_BARS)
    if j > i + 1 and closes[i] > 0:
        out["mfe_10d_pct"] = float((highs[i + 1:j].max() / closes[i] - 1.0) * 100.0)
        out["mae_10d_pct"] = float((lows[i + 1:j].min() / closes[i] - 1.0) * 100.0)
    else:
        out["mfe_10d_pct"] = out["mae_10d_pct"] = None
    vol = df["Volume"].to_numpy(float)
    out["vol_eod"] = float(vol[i])
    med = float(np.nanmedian(vol[max(0, i - 20):i])) if i > 0 else float("nan")
    out["vol_ratio_eod"] = float(vol[i] / med) if med and med > 0 else None
    return out


# ---------------------------------------------------------------------------
# Live scan
# ---------------------------------------------------------------------------

def provisional_vol_scale() -> tuple[float, int]:
    """Median of vol_at_scan / vol_eod over resolved live rows -- what fraction
    of a full day's volume the 15:40 bar carries. (scale, n_sessions)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT scan_date, AVG(vol_at_scan * 1.0 / vol_eod) AS frac FROM ta_detections"
            " WHERE source='live' AND vol_at_scan IS NOT NULL AND vol_eod > 0"
            " AND bar_source='daily_partial' GROUP BY scan_date").fetchall()
    fr = [float(r["frac"]) for r in rows if r["frac"] is not None and 0 < r["frac"] <= 1.5]
    if len(fr) < MIN_VOL_SCALE_SESSIONS:
        return SEED_VOL_SCALE, len(fr)
    return float(np.median(fr)), len(fr)


def _first_in_window_live(conn: sqlite3.Connection, ticker: str, setup_id: str,
                          state: str, scan_date: str) -> int:
    since = (datetime.fromisoformat(scan_date) - timedelta(days=DEDUP_SESSIONS + 3)).date().isoformat()
    r = conn.execute(
        "SELECT 1 FROM ta_detections WHERE source='live' AND ticker=? AND setup_id=? AND state=?"
        " AND scan_date >= ? AND scan_date < ? LIMIT 1",
        (ticker, setup_id, state, since, scan_date)).fetchone()
    return 0 if r else 1


def scan(limit: int = DEFAULT_LIMIT, store: bool = True, period: str = DEFAULT_PERIOD,
         force: bool = False) -> dict[str, Any]:
    """Detect setups on today's provisional bar across the universe."""
    from .asx_feed import sydney_today
    from .market_hours import is_open
    from .symbols import company_name

    today = sydney_today()
    if store and not force:
        frozen = latest_stored(today)
        if frozen.get("available"):
            return {**frozen, "cached": True}
    if not force and is_open("^AXJO") is False:
        return {"skipped": "market closed", "scan_date": today, "available": False,
                "detections": []}

    t0 = time.time()
    vol_scale, n_scale = provisional_vol_scale()
    frames = _load_bars(limit, period)
    n_universe = limit
    n_with_data = len(frames)
    sources = _ensure_today_bar(frames, today)
    dets: list[Detection] = []
    liquid: list[str] = []
    vol_at: dict[str, float] = {}
    for t, df in frames.items():
        if t not in sources or len(df) < MIN_BARS:
            continue
        try:
            ind = indicator_frame(df)
        except Exception as exc:
            logger.warning("indicator frame failed for %s: %s", t, exc)
            continue
        if bool(liquid_mask(ind).iloc[-1]):
            liquid.append(t)
        vol_at[t] = float(df["Volume"].iloc[-1])
        try:
            dets.extend(detect_all(t, df, vol_scale=vol_scale, ind=ind))
        except Exception as exc:
            logger.warning("detect failed for %s: %s", t, exc)
    elapsed = time.time() - t0
    bar_source = ("daily_partial" if all(v == "daily_partial" for v in sources.values())
                  else "mixed") if sources else "none"
    scanned_at = _now()
    rows = []
    for d in dets:
        rows.append({**d.as_dict(), "company": company_name(d.ticker),
                     "vol_at_scan": vol_at.get(d.ticker),
                     "bar_source": sources.get(d.ticker), "first_in_window": 1})
    run = {
        "scan_date": today, "source": "live", "scanned_at": scanned_at,
        "sydney_time": _sydney_now().strftime("%H:%M"), "period": period,
        "n_universe": n_universe, "n_with_data": n_with_data, "n_today_bar": len(sources),
        "n_liquid": len(liquid), "n_detections": len(dets),
        "n_tier1": sum(1 for d in dets if d.tier == 1),
        "n_tier2": sum(1 for d in dets if d.tier == 2),
        "bar_provisional": 1, "bar_source": bar_source, "vol_scale": vol_scale,
        "vol_scale_sessions": n_scale, "elapsed_s": round(elapsed, 1),
    }
    if store:
        with _connect() as conn:
            for r in rows:
                r["first_in_window"] = _first_in_window_live(
                    conn, r["ticker"], r["setup_id"], r["state"], today)
                conn.execute(
                    "INSERT OR REPLACE INTO ta_detections (scan_date, ticker, setup_id, source,"
                    " state, tier, direction, company, level, level_kind, distance_pct,"
                    " context_json, price_at_scan, bar_provisional, bar_source,"
                    " vol_ratio_at_scan, vol_at_scan, scanned_at, first_in_window)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (today, r["ticker"], r["setup_id"], "live", r["state"], r["tier"],
                     r["direction"], r["company"], r["level"], r["level_kind"],
                     r["distance_pct"], json.dumps(r["context"]), r["price"], 1,
                     r["bar_source"], (r["context"] or {}).get("vol_ratio"),
                     r["vol_at_scan"], scanned_at, r["first_in_window"]))
            conn.execute(
                "INSERT OR REPLACE INTO ta_scan_runs (scan_date, source, scanned_at, sydney_time,"
                " period, n_universe, n_with_data, n_today_bar, n_liquid, n_detections, n_tier1,"
                " n_tier2, bar_provisional, bar_source, vol_scale, elapsed_s, liquid_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (today, "live", scanned_at, run["sydney_time"], period, n_universe,
                 n_with_data, len(sources), len(liquid), len(dets), run["n_tier1"],
                 run["n_tier2"], 1, bar_source, vol_scale, run["elapsed_s"],
                 json.dumps(liquid)))
            conn.commit()
        return latest_stored(today)
    return {"available": True, "run": run, "detections": _enrich(rows), "stored": False}


def _enrich(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        sd = SETUPS.get(r["setup_id"])
        out.append({**r, "label": sd.label if sd else r["setup_id"],
                    "tier": sd.tier if sd else r.get("tier")})
    return out


def latest_stored(scan_date: str | None = None) -> dict[str, Any]:
    with _connect() as conn:
        if scan_date:
            run = conn.execute("SELECT * FROM ta_scan_runs WHERE scan_date=? AND source='live'",
                               (scan_date,)).fetchone()
        else:
            run = conn.execute("SELECT * FROM ta_scan_runs WHERE source='live'"
                               " ORDER BY scan_date DESC LIMIT 1").fetchone()
        if not run:
            return {"available": False, "scan_date": scan_date, "detections": []}
        rows = conn.execute(
            "SELECT * FROM ta_detections WHERE scan_date=? AND source='live'"
            " ORDER BY tier, setup_id, ticker", (run["scan_date"],)).fetchall()
    run_d = dict(run)
    run_d["liquid_json"] = None
    dets = []
    for r in rows:
        d = dict(r)
        d["context"] = json.loads(d.pop("context_json") or "{}")
        d["price"] = d.get("price_at_scan")
        dets.append(d)
    return {"available": True, "run": run_d, "detections": _enrich(dets), "stored": True}


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------

def resolve(days_back: int = 40) -> dict[str, Any]:
    """Fill completed-bar outcomes for live detections and the live control.

    Idempotent: rows are revisited until 20 forward sessions exist, and the
    same-day fields are set on the first pass after the close."""
    from .sectors import MARKET_INDEX

    since = (_sydney_now().date() - timedelta(days=days_back)).isoformat()
    with _connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM ta_detections WHERE source='live' AND scan_date >= ?"
            " AND (fwd_bars_available IS NULL OR fwd_bars_available < ?)",
            (since, max(HORIZONS))).fetchall()]
        runs = [dict(r) for r in conn.execute(
            "SELECT scan_date, liquid_json FROM ta_scan_runs WHERE source='live'"
            " AND scan_date >= ?", (since,)).fetchall()]
    if not rows and not runs:
        return {"resolved": 0, "since": since}

    liquid_by_date = {r["scan_date"]: json.loads(r["liquid_json"] or "[]") for r in runs}
    tickers = sorted({r["ticker"] for r in rows} | {t for v in liquid_by_date.values() for t in v})
    frames = _load_bars(tickers=tickers, period=DEFAULT_PERIOD)
    bench = _bench_symbols(tickers)
    ibars = _index_bars(sorted(set(bench.values())), period="6mo")

    n_res = 0
    updates = []
    for r in rows:
        df = frames.get(r["ticker"])
        if df is None or df.empty:
            continue
        dates = [d.date().isoformat() for d in df.index]
        i = bisect_left(dates, r["scan_date"])
        if i >= len(dates) or dates[i] != r["scan_date"]:
            continue
        # Is the scan-day bar complete? Only once a later bar exists, or it is
        # after 16:30 Sydney on that day.
        now_syd = _sydney_now()
        complete = i + 1 < len(dates) or (
            now_syd.date().isoformat() == r["scan_date"] and now_syd.hour * 60 + now_syd.minute >= 16 * 60 + 30)
        if not complete:
            continue
        m = _forward_metrics(df, i)
        state_at_close = r.get("state_at_close")
        if state_at_close is None:
            state_at_close = _state_at_close(r["ticker"], df.iloc[:i + 1], r["setup_id"])
        sym = bench.get(r["ticker"], MARKET_INDEX)
        b = {h: _bench_fwd(ibars, sym, r["scan_date"], h) for h in HORIZONS}
        scan_vs_close = ((m["close_eod"] / r["price_at_scan"] - 1.0) * 100.0
                         if r.get("price_at_scan") else None)
        updates.append((
            m["close_eod"], scan_vs_close, m["vol_eod"], m["vol_ratio_eod"], state_at_close,
            m["next_close"], m["fwd_1d_pct"], m["fwd_3d_pct"], m["fwd_5d_pct"],
            m["fwd_10d_pct"], m["fwd_20d_pct"], m["delayed_5d_pct"], m["delayed_10d_pct"],
            m["mfe_10d_pct"], m["mae_10d_pct"], sym, b[1], b[3], b[5], b[10], b[20],
            m["fwd_bars_available"], _now(),
            r["scan_date"], r["ticker"], r["setup_id"], r["state"]))
        n_res += 1

    # Control rows for each live scan date.
    ctrl_rows = []
    for date, liq in liquid_by_date.items():
        acc: dict[int, list[float]] = {h: [] for h in HORIZONS}
        for t in liq:
            df = frames.get(t)
            if df is None:
                continue
            dates = [d.date().isoformat() for d in df.index]
            i = bisect_left(dates, date)
            if i >= len(dates) or dates[i] != date:
                continue
            closes = df["Close"].to_numpy(float)
            for h in HORIZONS:
                v = _fwd(closes, i, h)
                if v is not None:
                    acc[h].append(v)
        if not acc[1]:
            continue
        ctrl_rows.append((date, len(acc[1]),
                          *[float(np.mean(acc[h])) if acc[h] else None for h in HORIZONS],
                          _bench_fwd(ibars, MARKET_INDEX, date, 5),
                          _bench_fwd(ibars, MARKET_INDEX, date, 10), _now()))

    with _connect() as conn:
        conn.executemany(
            "UPDATE ta_detections SET close_eod=?, scan_vs_close_pct=?, vol_eod=?, vol_ratio_eod=?,"
            " state_at_close=?, next_close=?, fwd_1d_pct=?, fwd_3d_pct=?, fwd_5d_pct=?,"
            " fwd_10d_pct=?, fwd_20d_pct=?, delayed_5d_pct=?, delayed_10d_pct=?, mfe_10d_pct=?,"
            " mae_10d_pct=?, bench_symbol=?, bench_1d_pct=?, bench_3d_pct=?, bench_5d_pct=?,"
            " bench_10d_pct=?, bench_20d_pct=?, fwd_bars_available=?, resolved_at=?"
            " WHERE scan_date=? AND ticker=? AND setup_id=? AND state=? AND source='live'", updates)
        conn.executemany(
            "INSERT OR REPLACE INTO ta_control (date, n, ew_fwd_1d_pct, ew_fwd_3d_pct,"
            " ew_fwd_5d_pct, ew_fwd_10d_pct, ew_fwd_20d_pct, mkt_fwd_5d_pct, mkt_fwd_10d_pct,"
            " updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)", ctrl_rows)
        conn.commit()
    return {"resolved": n_res, "pending": len(rows) - n_res, "control_dates": len(ctrl_rows),
            "since": since}


def _state_at_close(ticker: str, df: pd.DataFrame, setup_id: str) -> str:
    """What the completed bar shows for this setup: confirmed / forming / none."""
    try:
        dets = detect_all(ticker, df, vol_scale=1.0)
    except Exception:
        return "unknown"
    states = {d.state for d in dets if d.setup_id == setup_id}
    if "confirmed" in states:
        return "confirmed"
    if "forming" in states:
        return "forming"
    return "none"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _cell_stats(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    """Direction-signed mean, control-excess and a date-clustered t-stat."""
    fk, ck = f"fwd_{horizon}d_pct", f"ew_fwd_{horizon}d_pct"
    per_date: dict[str, list[float]] = {}
    signed: list[float] = []
    excess: list[float] = []
    for r in rows:
        f = r.get(fk)
        if f is None:
            continue
        s = r["direction"] * f
        signed.append(s)
        c = r.get(ck)
        if c is None:
            continue
        e = r["direction"] * (f - c)
        excess.append(e)
        per_date.setdefault(r["scan_date"], []).append(e)
    empty = {"n": 0, "n_dates": 0, "mean": None, "excess": None, "t": None, "hit_rate": None,
             "median": None, "tail_share": None, "mean_ex_tail": None}
    if not signed:
        return empty
    dm = [float(np.mean(v)) for v in per_date.values()]
    t = None
    if len(dm) >= 3:
        sd = float(np.std(dm, ddof=1))
        t = float(np.mean(dm) / (sd / math.sqrt(len(dm)))) if sd > 0 else None
    # Tail dependence: how much of the total gain sits in the best 5% of
    # trades. The bull flag's +0.62% mean was 105% tail -- remove the top 5%
    # and the mean is negative. A mean can survive every gate on a fat right
    # tail; this number and the median say whether it did.
    arr = np.sort(np.asarray(signed, dtype=float))
    k = max(1, int(len(arr) * 0.05))
    tail_share = float(arr[-k:].sum() / arr.sum()) if arr.sum() > 0 and len(arr) >= 20 else None
    mean_ex_tail = float(arr[:-k].mean()) if len(arr) >= 20 else None
    return {"n": len(signed), "n_dates": len(dm), "mean": round(float(np.mean(signed)), 3),
            "median": round(float(np.median(arr)), 3),
            "tail_share": round(tail_share, 3) if tail_share is not None else None,
            "mean_ex_tail": round(mean_ex_tail, 3) if mean_ex_tail is not None else None,
            "excess": round(float(np.mean(excess)), 3) if excess else None,
            "t": round(t, 2) if t is not None else None,
            "hit_rate": round(float(np.mean([e > 0 for e in excess])), 3) if excess else None}


def _badge(live5: dict[str, Any], live10: dict[str, Any], baseline: dict[str, Any] | None) -> str:
    n_dates = live5.get("n_dates") or 0
    t = live5.get("t")
    if n_dates < MIN_DATES_LIVE or t is None:
        return "insufficient"
    if t <= -2:
        return "negative"
    if t < 2:
        return "no edge"
    if n_dates >= MIN_DATES_EDGE and t >= 2.5 and baseline and baseline.get("verdict") == "survived":
        return "edge (unvalidated)"
    if (live10.get("excess") or 0) > 0:
        return "watch"
    return "no edge"


def _rows_with_control(conn: sqlite3.Connection, source: str, setup_id: str | None = None,
                       state: str | None = None) -> list[dict[str, Any]]:
    q = ("SELECT d.*, c.ew_fwd_1d_pct, c.ew_fwd_3d_pct, c.ew_fwd_5d_pct, c.ew_fwd_10d_pct,"
         " c.ew_fwd_20d_pct FROM ta_detections d LEFT JOIN ta_control c ON c.date = d.scan_date"
         " WHERE d.source=? AND d.first_in_window=1 AND d.fwd_bars_available >= ?")
    args: list[Any] = [source, PRIMARY_HORIZON]
    if setup_id:
        q += " AND d.setup_id=?"
        args.append(setup_id)
    if state:
        q += " AND d.state=?"
        args.append(state)
    return [dict(r) for r in conn.execute(q, args).fetchall()]


def scorecard() -> dict[str, Any]:
    with _connect() as conn:
        live = _rows_with_control(conn, "live")
        base = {(r["setup_id"], r["state"]): dict(r)
                for r in conn.execute("SELECT * FROM ta_setup_baseline").fetchall()}
        match = {(r["setup_id"], r["state"]): dict(r) for r in conn.execute(
            "SELECT setup_id, state, COUNT(*) AS n,"
            " SUM(state_at_close = state) AS n_same,"
            " SUM(state_at_close = 'confirmed') AS n_confirmed,"
            " AVG(scan_vs_close_pct) AS mean_drift FROM ta_detections"
            " WHERE source='live' AND state_at_close IS NOT NULL GROUP BY setup_id, state").fetchall()}
        vol_scale, n_scale = provisional_vol_scale()
    cells = []
    for sid, sd in SETUPS.items():
        for st in sd.states:
            rows = [r for r in live if r["setup_id"] == sid and r["state"] == st]
            l5 = _cell_stats(rows, 5)
            l10 = _cell_stats(rows, 10)
            b = base.get((sid, st))
            mt = match.get((sid, st)) or {}
            cells.append({
                "setup_id": sid, "state": st, "label": sd.label, "tier": sd.tier,
                "direction": sd.direction,
                "live": {"n": l5["n"], "n_dates": l5["n_dates"], "mean_5d": l5["mean"],
                         "median_5d": l5["median"], "tail_share_5d": l5["tail_share"],
                         "mean_ex_tail_5d": l5["mean_ex_tail"],
                         "excess_5d": l5["excess"], "t_5d": l5["t"], "hit_rate_5d": l5["hit_rate"],
                         "mean_10d": l10["mean"], "excess_10d": l10["excess"], "t_10d": l10["t"]},
                "baseline": b,
                "close_check": {"n": mt.get("n", 0),
                                "same_rate": (mt["n_same"] / mt["n"]) if mt.get("n") else None,
                                "confirmed_rate": (mt["n_confirmed"] / mt["n"]) if mt.get("n") else None,
                                "mean_drift_pct": mt.get("mean_drift")},
                "badge": _badge(l5, l10, b),
            })
    n_cells = len(cells)
    return {"cells": cells, "n_cells": n_cells,
            "expected_false_positives": round(n_cells * 0.05, 1),
            "vol_scale": vol_scale, "vol_scale_sessions": n_scale,
            "thresholds": {"min_dates_live": MIN_DATES_LIVE, "min_dates_edge": MIN_DATES_EDGE}}


def history(days: int = 30, setup_id: str | None = None) -> dict[str, Any]:
    since = (_sydney_now().date() - timedelta(days=days)).isoformat()
    with _connect() as conn:
        q = ("SELECT d.*, c.ew_fwd_5d_pct, c.ew_fwd_10d_pct FROM ta_detections d"
             " LEFT JOIN ta_control c ON c.date=d.scan_date"
             " WHERE d.source='live' AND d.scan_date >= ?")
        args: list[Any] = [since]
        if setup_id:
            q += " AND d.setup_id=?"
            args.append(setup_id)
        q += " ORDER BY d.scan_date DESC, d.tier, d.setup_id, d.ticker"
        rows = [dict(r) for r in conn.execute(q, args).fetchall()]
    for r in rows:
        r["context"] = json.loads(r.pop("context_json") or "{}")
        r["price"] = r.get("price_at_scan")
    return {"since": since, "rows": _enrich(rows)}


def catalogue() -> dict[str, Any]:
    return {"setups": [{"id": s.id, "tier": s.tier, "direction": s.direction, "label": s.label,
                        "rule": s.rule_text, "level_kind": s.level_kind, "states": list(s.states)}
                       for s in SETUPS.values()]}


# ---------------------------------------------------------------------------
# Backfill + hypothesis bridge
# ---------------------------------------------------------------------------

def backfill(period: str = BACKFILL_PERIOD, limit: int = DEFAULT_LIMIT, store: bool = True,
             run_gates: bool = True, frames: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
    """Historical detections on completed bars for every liquid ticker, with
    outcomes, a date-matched control, and a gate verdict per setup x state.

    Replaces any earlier backfill rows: the baseline is a snapshot of the
    detector code as it stands, not an accumulating log. Survivorship: the
    universe is TODAY's top 300 applied backwards, and the baseline says so."""
    from .sectors import MARKET_INDEX
    from .symbols import company_name

    t0 = time.time()
    if frames is None:
        frames = _load_bars(limit, period)
    tickers = sorted(frames)
    bench = _bench_symbols(tickers)
    ibars = _index_bars(sorted(set(bench.values())), period=period)

    det_rows: list[tuple] = []
    ctrl_acc: dict[str, dict[int, list[float]]] = {}
    scanned_at = _now()
    n_t = 0
    for t in tickers:
        df = frames[t]
        if len(df) <= MIN_BARS:
            continue
        n_t += 1
        try:
            ind = indicator_frame(df)
            dets = detect_history(t, df, MIN_BARS, ind=ind)
        except Exception as exc:
            logger.warning("backfill detect failed for %s: %s", t, exc)
            continue
        dates = [d.date().isoformat() for d in df.index]
        pos = {d: i for i, d in enumerate(dates)}
        closes = df["Close"].to_numpy(float)
        liquid = liquid_mask(ind).fillna(False).to_numpy(bool)
        for i in range(MIN_BARS, len(df)):
            if not liquid[i]:
                continue
            acc = ctrl_acc.setdefault(dates[i], {h: [] for h in HORIZONS})
            for h in HORIZONS:
                v = _fwd(closes, i, h)
                if v is not None:
                    acc[h].append(v)
        last_seen: dict[tuple[str, str], int] = {}
        name = company_name(t)
        sym = bench.get(t, MARKET_INDEX)
        for d in sorted(dets, key=lambda x: x.date):
            i = pos[d.date]
            key = (d.setup_id, d.state)
            first = 1 if (key not in last_seen or i - last_seen[key] > DEDUP_SESSIONS) else 0
            last_seen[key] = i
            m = _forward_metrics(df, i)
            b = {h: _bench_fwd(ibars, sym, d.date, h) for h in HORIZONS}
            det_rows.append((
                d.date, t, d.setup_id, "backfill", d.state, d.tier, d.direction, name, d.level,
                d.level_kind, d.distance_pct, json.dumps(d.context), d.price, 0, "daily_complete",
                d.context.get("vol_ratio"), m["vol_eod"], scanned_at, first,
                m["close_eod"], 0.0, m["vol_eod"], m["vol_ratio_eod"], d.state, m["next_close"],
                m["fwd_1d_pct"], m["fwd_3d_pct"], m["fwd_5d_pct"], m["fwd_10d_pct"], m["fwd_20d_pct"],
                m["delayed_5d_pct"], m["delayed_10d_pct"], m["mfe_10d_pct"], m["mae_10d_pct"],
                sym, b[1], b[3], b[5], b[10], b[20], m["fwd_bars_available"], scanned_at))

    ctrl_rows = []
    for date, acc in ctrl_acc.items():
        if not acc[1]:
            continue
        ctrl_rows.append((date, len(acc[1]),
                          *[float(np.mean(acc[h])) if acc[h] else None for h in HORIZONS],
                          _bench_fwd(ibars, MARKET_INDEX, date, 5),
                          _bench_fwd(ibars, MARKET_INDEX, date, 10), scanned_at))

    out: dict[str, Any] = {"n_tickers": n_t, "n_detections": len(det_rows),
                           "n_control_dates": len(ctrl_rows), "period": period}
    if store:
        with _connect() as conn:
            conn.execute("DELETE FROM ta_detections WHERE source='backfill'")
            conn.executemany(
                "INSERT OR REPLACE INTO ta_detections (scan_date, ticker, setup_id, source, state,"
                " tier, direction, company, level, level_kind, distance_pct, context_json,"
                " price_at_scan, bar_provisional, bar_source, vol_ratio_at_scan, vol_at_scan,"
                " scanned_at, first_in_window, close_eod, scan_vs_close_pct, vol_eod, vol_ratio_eod,"
                " state_at_close, next_close, fwd_1d_pct, fwd_3d_pct, fwd_5d_pct, fwd_10d_pct,"
                " fwd_20d_pct, delayed_5d_pct, delayed_10d_pct, mfe_10d_pct, mae_10d_pct,"
                " bench_symbol, bench_1d_pct, bench_3d_pct, bench_5d_pct, bench_10d_pct,"
                " bench_20d_pct, fwd_bars_available, resolved_at)"
                " VALUES (" + ",".join("?" * 42) + ")", det_rows)
            # Live control rows win where both exist (they are the same
            # quantity; the live one reflects that day's actual liquid list).
            conn.executemany(
                "INSERT OR IGNORE INTO ta_control (date, n, ew_fwd_1d_pct, ew_fwd_3d_pct,"
                " ew_fwd_5d_pct, ew_fwd_10d_pct, ew_fwd_20d_pct, mkt_fwd_5d_pct, mkt_fwd_10d_pct,"
                " updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)", ctrl_rows)
            conn.execute(
                "INSERT OR REPLACE INTO ta_scan_runs (scan_date, source, scanned_at, period,"
                " n_universe, n_with_data, n_liquid, n_detections, bar_provisional, bar_source,"
                " elapsed_s) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (scanned_at[:10], "backfill", scanned_at, period, limit, len(frames), n_t,
                 len(det_rows), 0, "daily_complete", round(time.time() - t0, 1)))
            conn.commit()
        if run_gates:
            out["baseline"] = compute_baselines(store=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def trades_frame(setup_id: str, state: str, horizon: int = PRIMARY_HORIZON,
                 source: str = "backfill") -> pd.DataFrame:
    """The `hypothesis.run()` contract for one setup x state cell."""
    with _connect() as conn:
        rows = _rows_with_control(conn, source, setup_id, state)
    fk, dk, bk = f"fwd_{horizon}d_pct", f"delayed_{horizon}d_pct", f"bench_{horizon}d_pct"
    recs = []
    for r in rows:
        if r.get(fk) is None:
            continue
        side = int(r["direction"])
        recs.append({"date": r["scan_date"], "ticker": r["ticker"], "side": side,
                     "gross_pct": side * r[fk], "price": r.get("close_eod"),
                     "bench_pct": side * r[bk] if r.get(bk) is not None else np.nan,
                     "delayed_pct": side * r[dk] if r.get(dk) is not None else np.nan})
    df = pd.DataFrame(recs, columns=["date", "ticker", "side", "gross_pct", "price",
                                     "bench_pct", "delayed_pct"])
    if df.empty:
        return df
    df = df.sort_values("date").reset_index(drop=True)
    split = df["date"].iloc[int(len(df) * 2 / 3)]
    df["is_train"] = df["date"] < split
    # A gate skips a column it cannot see, but a NaN inside one it can see
    # poisons the mean: drop an empty column, and the few unmatured rows of a
    # populated one.
    for col in ("bench_pct", "delayed_pct"):
        if df[col].isna().all():
            df = df.drop(columns=[col])
        else:
            df = df[~df[col].isna()]
    return df.reset_index(drop=True)


def hypothesis_for(setup_id: str, state: str, n_variants: int = 1, n_passed: int = 1):
    from .hypothesis import Hypothesis
    sd = SETUPS[setup_id]
    return Hypothesis(
        name=f"ta_{setup_id}_{state}",
        description=f"{sd.label} ({state}, tier {sd.tier}): {sd.rule_text}",
        build=lambda: trades_frame(setup_id, state),
        signal_known_at="15:40 Sydney on the signal bar (backfill: the completed bar)",
        entry_at="same-day closing auction",
        entry_is_tradeable=True,
        tradeable_note="The detection precedes the 16:10 auction, so the close is obtainable."
                       " The backfill sees the completed bar, which the live scan does not;"
                       " the live state_at_close rate quantifies that gap.",
        universe="top-300 by market cap as of the backfill date",
        point_in_time=False,
        survivorship_note="today's constituents applied backwards over the backfill period",
        n_variants=n_variants, n_passed=n_passed,
        benchmark_label="sector",
        tags=["ta_setups", f"tier{sd.tier}", state],
    )


def compute_baselines(store: bool = True) -> dict[str, Any]:
    """Per-cell baseline stats from backfill rows, then the gate verdict.

    Two passes: the first learns how many cells survive so the second can
    report the honest multiple-testing denominator."""
    from .hypothesis import run as run_hypothesis

    cells = [(sid, st) for sid, sd in SETUPS.items() for st in sd.states]
    with _connect() as conn:
        rows = _rows_with_control(conn, "backfill")
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    for sid, st in cells:
        sub = [r for r in rows if r["setup_id"] == sid and r["state"] == st]
        s5, s10 = _cell_stats(sub, 5), _cell_stats(sub, 10)
        dates = sorted({r["scan_date"] for r in sub})
        ctrl5 = [r["ew_fwd_5d_pct"] for r in sub if r.get("ew_fwd_5d_pct") is not None]
        ctrl10 = [r["ew_fwd_10d_pct"] for r in sub if r.get("ew_fwd_10d_pct") is not None]
        stats[(sid, st)] = {
            "n": s5["n"], "n_dates": s5["n_dates"], "mean_5d": s5["mean"], "mean_10d": s10["mean"],
            "median_5d": s5["median"], "tail_share_5d": s5["tail_share"],
            "mean_ex_tail_5d": s5["mean_ex_tail"],
            "hit_rate_5d": s5["hit_rate"], "excess_5d": s5["excess"], "excess_10d": s10["excess"],
            "t_5d": s5["t"], "t_10d": s10["t"],
            "ctrl_mean_5d": round(float(np.mean(ctrl5)), 3) if ctrl5 else None,
            "ctrl_mean_10d": round(float(np.mean(ctrl10)), 3) if ctrl10 else None,
            "period_start": dates[0] if dates else None, "period_end": dates[-1] if dates else None,
        }
    evaluable = [c for c in cells if stats[c]["n"] >= MIN_DATES_LIVE]
    first = {c: run_hypothesis(hypothesis_for(*c, n_variants=len(evaluable), n_passed=0), store=False)
             for c in evaluable}
    n_passed = sum(1 for v in first.values() if v.get("verdict") == "survived")
    results = {}
    for c in cells:
        if c in first:
            results[c] = run_hypothesis(hypothesis_for(*c, n_variants=len(evaluable),
                                                       n_passed=n_passed), store=store)
        else:
            results[c] = {"verdict": "insufficient", "killed_by": None, "blocked_by": None,
                          "run_at": None}
    computed_at = _now()
    if store:
        with _connect() as conn:
            for (sid, st), s in stats.items():
                v = results[(sid, st)]
                conn.execute(
                    "INSERT OR REPLACE INTO ta_setup_baseline (setup_id, state, computed_at,"
                    " period_start, period_end, n, n_dates, mean_5d, mean_10d, hit_rate_5d,"
                    " ctrl_mean_5d, ctrl_mean_10d, excess_5d, excess_10d, t_5d, t_10d, verdict,"
                    " killed_by, blocked_by, hypothesis_run_at, median_5d, tail_share_5d,"
                    " mean_ex_tail_5d)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (sid, st, computed_at, s["period_start"], s["period_end"], s["n"],
                     s["n_dates"], s["mean_5d"], s["mean_10d"], s["hit_rate_5d"],
                     s["ctrl_mean_5d"], s["ctrl_mean_10d"], s["excess_5d"], s["excess_10d"],
                     s["t_5d"], s["t_10d"], v.get("verdict"), v.get("killed_by"),
                     v.get("blocked_by"), v.get("run_at"), s["median_5d"], s["tail_share_5d"],
                     s["mean_ex_tail_5d"]))
            conn.commit()
    return {"cells": len(cells), "evaluable": len(evaluable), "survived": n_passed,
            "verdicts": {f"{sid}/{st}": results[(sid, st)].get("verdict") for sid, st in cells}}
