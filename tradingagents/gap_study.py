"""Does a gap up at the open lead anywhere? (2026-08-25, user hypothesis.)

**The rule being tested, exactly as specified**: buy at the open, take profit
at +6%, and if it has not traded there by 15:00 Sydney, close at 15:00.

**Hourly bars, not daily.** A daily bar has no 15:00 price, so a daily-bar
version could only exit at the close -- and worse, it would count a +6% high
that occurred at 15:30 as a fill, crediting the rule with a trade the 15:00
cutoff would have missed. That is a look-ahead error biased toward flattering
the strategy, the same family as the same-bar-exit artifact that invalidated
earlier backtests here. yfinance serves 2 years of hourly bars for ASX names,
so the rule is modelled exactly instead of approximated.

ASX hourly bars are stamped with their START time in Sydney: 10:00 through
16:00, where the 16:00 bar is the closing auction (a single print, O=H=L=C).
So "before 15:00" is the 10:00-14:00 bars, and the 15:00 price is the close of
the 14:00 bar.

**Fills.** One price trigger and a time stop, with no competing stop-loss,
means there is no same-bar ambiguity: if any pre-cutoff bar's high reaches the
target, the limit traded. The one refinement is a session that OPENS above the
target -- impossible here, since the target is defined from that same open.

**Two things without which the result is meaningless**, both included:
  - a NON-GAP baseline over the same names and period, because every stock has
    intraday range and a positive-looking number proves nothing on its own;
  - a SWEEP across gap sizes, because a monotonic gradient is evidence of a
    real relationship while one good bucket among six is noise.

**Survivorship bias is present and not removable.** The universe is asxbrief's
CURRENT top 500 by market cap, so these are companies that survived and grew
into that list. Two years is used rather than ten to limit how far the list
has drifted, but the bias runs toward optimism and the results should be read
with that in mind.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "swing.db"
SYDNEY = "Australia/Sydney"
SESSION_OPEN_HOUR = 10
DEFAULT_EXIT_HOUR = 15          # close the position at 15:00 if unfilled
DEFAULT_TARGET_PCT = 6.0

# Upper bounds, in gap %. The lowest bucket is the no-gap/gap-down baseline.
GAP_BUCKETS = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, float("inf")]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gap_study_sessions (
    ticker      TEXT NOT NULL,
    session     TEXT NOT NULL,
    prev_close  REAL NOT NULL,
    open        REAL NOT NULL,
    gap_pct     REAL NOT NULL,
    hit         INTEGER NOT NULL,     -- reached the target before the cutoff
    exit_price  REAL NOT NULL,
    ret_pct     REAL NOT NULL,        -- gross, before costs
    spread_pct  REAL,                 -- round-trip tick cost estimate
    target_pct  REAL NOT NULL,
    exit_hour   INTEGER NOT NULL,
    PRIMARY KEY (ticker, session, target_pct, exit_hour)
);
CREATE INDEX IF NOT EXISTS idx_gap_sessions_gap ON gap_study_sessions(gap_pct);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _universe(limit: int) -> list[str]:
    from .asx_feed import DB_PATH as ASX_DB
    if not ASX_DB.exists():
        return []
    conn = sqlite3.connect(f"file:{ASX_DB}?mode=ro", uri=True, timeout=10.0)
    try:
        return [r[0] for r in conn.execute(
            "SELECT ticker FROM universe ORDER BY rank LIMIT ?", (limit,))]
    finally:
        conn.close()


def _sessions_for(df, target_pct: float, exit_hour: int) -> list[dict[str, Any]]:
    """One row per session for a single ticker's hourly frame.

    Written against numpy slices rather than repeated boolean masks: hourly
    bars are time-ordered, so each session is a CONTIGUOUS run and the day
    boundaries can be found in a single pass. The obvious
    `df[df["_day"] == day]` per day is O(rows) each time -- across 500 days and
    500 tickers that was slow enough that the first batch had not finished
    after four minutes.
    """
    import numpy as np

    from .backtest import tick_size

    if df.empty:
        return []
    idx = df.index.tz_convert(SYDNEY) if df.index.tz is not None else df.index.tz_localize(SYDNEY)
    days = [t.date() for t in idx]
    hours = np.fromiter((t.hour for t in idx), dtype=int, count=len(idx))
    o = df["Open"].to_numpy(dtype=float)
    hi = df["High"].to_numpy(dtype=float)
    cl = df["Close"].to_numpy(dtype=float)

    starts = [0]
    for i in range(1, len(days)):
        if days[i] != days[i - 1]:
            starts.append(i)
    ends = starts[1:] + [len(days)]

    out: list[dict[str, Any]] = []
    for k in range(1, len(starts)):
        a, b = starts[k], ends[k]
        pa, pb = starts[k - 1], ends[k - 1]
        day, prev_day = days[a], days[pa]
        # A gap only means something against the PREVIOUS session; a multi-week
        # hole in the data is not a gap, it is missing data.
        if (day - prev_day) > timedelta(days=5):
            continue
        sess_hours = hours[a:b]
        open_pos = np.nonzero(sess_hours == SESSION_OPEN_HOUR)[0]
        if not len(open_pos):
            continue                      # no 10:00 bar: half day or data gap
        entry = float(o[a + open_pos[0]])
        prev_close = float(cl[pb - 1])
        if not entry or not prev_close:
            continue

        pre = np.nonzero((sess_hours >= SESSION_OPEN_HOUR) & (sess_hours < exit_hour))[0]
        if not len(pre):
            continue
        lo_i, hi_i = a + pre[0], a + pre[-1]
        target = entry * (1 + target_pct / 100)
        hit = bool(hi[lo_i:hi_i + 1].max() >= target)
        if hit:
            exit_price, ret = target, target_pct
        else:
            exit_price = float(cl[hi_i])          # the 15:00 price
            ret = 100 * (exit_price - entry) / entry

        spread = 100 * tick_size(entry) / entry if entry else None
        out.append({
            "session": day.isoformat(), "prev_close": prev_close, "open": entry,
            "gap_pct": 100 * (entry - prev_close) / prev_close,
            "hit": int(hit), "exit_price": exit_price, "ret_pct": ret,
            # One tick each way is the optimistic floor on a round trip, not a
            # full spread estimate -- enough to show where costs dominate.
            "spread_pct": round(2 * spread, 4) if spread else None,
        })
    return out


def build(universe_limit: int = 500, period: str = "2y", batch: int = 40,
          target_pct: float = DEFAULT_TARGET_PCT,
          exit_hour: int = DEFAULT_EXIT_HOUR) -> dict[str, Any]:
    """Download hourly bars and store one row per ticker-session."""
    import yfinance as yf

    from .yf_lock import YF_LOCK

    tickers = _universe(universe_limit)
    if not tickers:
        return {"error": "no universe available"}
    stored, skipped = 0, 0
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        with YF_LOCK:
            data = yf.download([f"{t}.AX" for t in chunk], period=period, interval="1h",
                               group_by="ticker", auto_adjust=False, threads=True,
                               progress=False)
        rows = []
        for t in chunk:
            try:
                df = data[f"{t}.AX"].dropna(subset=["Close"])
            except (KeyError, TypeError):
                skipped += 1
                continue
            if df.empty:
                skipped += 1
                continue
            for s in _sessions_for(df, target_pct, exit_hour):
                rows.append((t, s["session"], s["prev_close"], s["open"], s["gap_pct"],
                             s["hit"], s["exit_price"], s["ret_pct"], s["spread_pct"],
                             target_pct, exit_hour))
        if rows:
            with _connect() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO gap_study_sessions (ticker, session, prev_close,"
                    " open, gap_pct, hit, exit_price, ret_pct, spread_pct, target_pct, exit_hour)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
                conn.commit()
            stored += len(rows)
        logger.info("gap study: %d/%d tickers, %d sessions stored",
                    min(i + batch, len(tickers)), len(tickers), stored)
    return {"tickers": len(tickers), "sessions": stored, "skipped": skipped,
            "target_pct": target_pct, "exit_hour": exit_hour}


def _bucket_label(lo: float, hi: float) -> str:
    if hi == float("inf"):
        return f">{lo:g}%"
    return f"{lo:g}-{hi:g}%"


def analyse(target_pct: float = DEFAULT_TARGET_PCT,
            exit_hour: int = DEFAULT_EXIT_HOUR,
            min_price: float = 0.0) -> dict[str, Any]:
    """Hit rate and mean return by gap bucket, against the no-gap baseline.

    `min_price` filters out the cheap end where the tick spread swamps
    everything -- the measured finding on this project was that below 50c the
    round trip cost more than the edge.
    """
    import statistics as st

    with _connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM gap_study_sessions WHERE target_pct=? AND exit_hour=? AND open>=?",
            (target_pct, exit_hour, min_price))]
    if not rows:
        return {"error": "no sessions stored; run build() first"}

    edges = [-float("inf")] + GAP_BUCKETS
    buckets: list[dict[str, Any]] = []
    for lo, hi in zip(edges, edges[1:]):
        grp = [r for r in rows if lo < r["gap_pct"] <= hi]
        if not grp:
            buckets.append({"label": _bucket_label(lo, hi), "n": 0})
            continue
        rets = [r["ret_pct"] for r in grp]
        net = [r["ret_pct"] - (r["spread_pct"] or 0) for r in grp]
        buckets.append({
            "label": "gap <=0% (baseline)" if lo == -float("inf") else _bucket_label(lo, hi),
            "lo": None if lo == -float("inf") else lo, "hi": None if hi == float("inf") else hi,
            "n": len(grp),
            "hit_rate_pct": round(100 * sum(r["hit"] for r in grp) / len(grp), 2),
            "mean_ret_pct": round(st.mean(rets), 3),
            "median_ret_pct": round(st.median(rets), 3),
            "mean_net_pct": round(st.mean(net), 3),
            "pct_positive": round(100 * sum(1 for x in rets if x > 0) / len(grp), 1),
            "mean_spread_pct": round(st.mean([r["spread_pct"] or 0 for r in grp]), 3),
        })
    all_rets = [r["ret_pct"] for r in rows]
    return {
        "target_pct": target_pct, "exit_hour": exit_hour, "min_price": min_price,
        "n_sessions": len(rows),
        "n_tickers": len({r["ticker"] for r in rows}),
        "date_range": [min(r["session"] for r in rows), max(r["session"] for r in rows)],
        "overall": {"hit_rate_pct": round(100 * sum(r["hit"] for r in rows) / len(rows), 2),
                    "mean_ret_pct": round(st.mean(all_rets), 3)},
        "buckets": buckets,
    }


def _bucket_label(lo: float, hi: float) -> str:
    if hi == float("inf"):
        return f">{lo:g}%"
    return f"{lo:g}-{hi:g}%"


def analyse(target_pct: float = DEFAULT_TARGET_PCT,
            exit_hour: int = DEFAULT_EXIT_HOUR,
            min_price: float = 0.0) -> dict[str, Any]:
    """Aggregate stored sessions into the gap sweep plus its baseline.

    `min_price` filters out sub-threshold names, because a one-tick round trip
    on a 2c stock is a double-digit percentage and swamps any edge -- the same
    conclusion the earlier momentum work reached.
    """
    import statistics as st

    with _connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM gap_study_sessions WHERE target_pct=? AND exit_hour=? AND open>=?",
            (target_pct, exit_hour, min_price))]
    if not rows:
        return {"error": "no stored sessions -- run build() first"}

    def stats(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"n": 0}
        rets = [r["ret_pct"] for r in group]
        nets = [r["ret_pct"] - (r["spread_pct"] or 0) for r in group]
        return {
            "n": len(group),
            "hit_rate_pct": round(100 * sum(r["hit"] for r in group) / len(group), 2),
            "mean_ret_pct": round(st.mean(rets), 4),
            "median_ret_pct": round(st.median(rets), 4),
            "mean_net_pct": round(st.mean(nets), 4),
            "pct_positive": round(100 * sum(1 for x in rets if x > 0) / len(group), 2),
        }

    buckets = []
    lo = -float("inf")
    for hi in GAP_BUCKETS:
        grp = [r for r in rows if lo < r["gap_pct"] <= hi]
        label = "gap <= 0%" if lo == -float("inf") else _bucket_label(lo, hi)
        buckets.append({"bucket": label, "lo": None if lo == -float("inf") else lo,
                        "hi": None if hi == float("inf") else hi, **stats(grp)})
        lo = hi

    gapped = [r for r in rows if r["gap_pct"] > 0]
    flat_or_down = [r for r in rows if r["gap_pct"] <= 0]
    return {
        "target_pct": target_pct, "exit_hour": exit_hour, "min_price": min_price,
        "n_sessions": len(rows),
        "n_tickers": len({r["ticker"] for r in rows}),
        "date_range": [min(r["session"] for r in rows), max(r["session"] for r in rows)],
        "buckets": buckets,
        "all_sessions": stats(rows),
        "baseline_no_gap": stats(flat_or_down),
        "any_gap_up": stats(gapped),
    }


_VARIANT_SCHEMA = """
CREATE TABLE IF NOT EXISTS gap_study_variants (
    ticker      TEXT NOT NULL,
    session     TEXT NOT NULL,
    direction   TEXT NOT NULL,      -- 'long' | 'short'
    target_pct  REAL NOT NULL,
    stop_pct    REAL,               -- NULL = no stop
    exit_hour   INTEGER NOT NULL,
    gap_pct     REAL NOT NULL,
    open        REAL NOT NULL,
    outcome     TEXT NOT NULL,      -- target | stop | timeout
    ambiguous   INTEGER NOT NULL,   -- target and stop both touched in one bar
    ret_pct     REAL NOT NULL,
    spread_pct  REAL,
    PRIMARY KEY (ticker, session, direction, target_pct, stop_pct, exit_hour)
);
CREATE INDEX IF NOT EXISTS idx_variants_gap ON gap_study_variants(direction, stop_pct, gap_pct);
"""


def _variants_for(df, grid, exit_hour: int) -> list[tuple]:
    """Evaluate every (direction, target, stop) in `grid` over one ticker's
    hourly bars, in a single pass.

    **Same-bar ambiguity is resolved pessimistically.** When one bar's range
    covers both the target and the stop, which came first is unknowable at
    hourly resolution, so the STOP is assumed. Guessing the favourable side
    would manufacture exactly the edge being tested -- the same-bar-exit
    artifact that already invalidated a round of backtests here. Every such
    bar is flagged so the frequency of the assumption stays visible.
    """
    import numpy as np

    from .backtest import tick_size

    if df.empty:
        return []
    idx = df.index.tz_convert(SYDNEY) if df.index.tz is not None else df.index.tz_localize(SYDNEY)
    days = [t.date() for t in idx]
    hours = np.fromiter((t.hour for t in idx), dtype=int, count=len(idx))
    o = df["Open"].to_numpy(dtype=float)
    hi = df["High"].to_numpy(dtype=float)
    lo = df["Low"].to_numpy(dtype=float)
    cl = df["Close"].to_numpy(dtype=float)

    starts = [0]
    for i in range(1, len(days)):
        if days[i] != days[i - 1]:
            starts.append(i)
    ends = starts[1:] + [len(days)]

    out: list[tuple] = []
    for k in range(1, len(starts)):
        a, b = starts[k], ends[k]
        pa, pb = starts[k - 1], ends[k - 1]
        day, prev_day = days[a], days[pa]
        if (day - prev_day) > timedelta(days=5):
            continue
        sess_hours = hours[a:b]
        open_pos = np.nonzero(sess_hours == SESSION_OPEN_HOUR)[0]
        if not len(open_pos):
            continue
        entry = float(o[a + open_pos[0]])
        prev_close = float(cl[pb - 1])
        if not entry or not prev_close:
            continue
        pre = np.nonzero((sess_hours >= SESSION_OPEN_HOUR) & (sess_hours < exit_hour))[0]
        if not len(pre):
            continue
        i0, i1 = a + pre[0], a + pre[-1]
        gap = 100 * (entry - prev_close) / prev_close
        spread = round(2 * 100 * tick_size(entry) / entry, 4) if entry else None
        session_iso = day.isoformat()

        for direction, target_pct, stop_pct in grid:
            if direction == "long":
                target = entry * (1 + target_pct / 100)
                stop = entry * (1 - stop_pct / 100) if stop_pct else None
            else:
                target = entry * (1 - target_pct / 100)
                stop = entry * (1 + stop_pct / 100) if stop_pct else None

            outcome, exit_px, ambiguous = "timeout", float(cl[i1]), 0
            for j in range(i0, i1 + 1):
                if direction == "long":
                    t_hit, s_hit = hi[j] >= target, (stop is not None and lo[j] <= stop)
                else:
                    t_hit, s_hit = lo[j] <= target, (stop is not None and hi[j] >= stop)
                if t_hit and s_hit:
                    outcome, exit_px, ambiguous = "stop", stop, 1
                    break
                if s_hit:
                    outcome, exit_px = "stop", stop
                    break
                if t_hit:
                    outcome, exit_px = "target", target
                    break
            ret = (100 * (exit_px - entry) / entry) if direction == "long" \
                else (100 * (entry - exit_px) / entry)
            out.append((None, session_iso, direction, target_pct, stop_pct, exit_hour,
                        gap, entry, outcome, ambiguous, round(ret, 4), spread))
    return out


def build_variants(universe_limit: int = 500, period: str = "2y", batch: int = 40,
                   target_pct: float = DEFAULT_TARGET_PCT,
                   stops: tuple = (None, 2.0, 3.0, 4.0, 6.0),
                   directions: tuple = ("long", "short"),
                   exit_hour: int = DEFAULT_EXIT_HOUR) -> dict[str, Any]:
    """Download once, evaluate the whole (direction x stop) grid in one pass."""
    import yfinance as yf

    from .yf_lock import YF_LOCK

    grid = [(d, target_pct, s) for d in directions for s in stops]
    tickers = _universe(universe_limit)
    if not tickers:
        return {"error": "no universe available"}
    with _connect() as conn:
        conn.executescript(_VARIANT_SCHEMA)
        conn.commit()

    stored = 0
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        with YF_LOCK:
            data = yf.download([f"{t}.AX" for t in chunk], period=period, interval="1h",
                               group_by="ticker", auto_adjust=False, threads=True,
                               progress=False)
        rows = []
        for t in chunk:
            try:
                df = data[f"{t}.AX"].dropna(subset=["Close"])
            except (KeyError, TypeError):
                continue
            for r in _variants_for(df, grid, exit_hour):
                rows.append((t, *r[1:]))
        if rows:
            with _connect() as conn:
                conn.executescript(_VARIANT_SCHEMA)
                conn.executemany(
                    "INSERT OR REPLACE INTO gap_study_variants (ticker, session, direction,"
                    " target_pct, stop_pct, exit_hour, gap_pct, open, outcome, ambiguous,"
                    " ret_pct, spread_pct) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                conn.commit()
            stored += len(rows)
        logger.info("variants: %d/%d tickers, %d rows", min(i + batch, len(tickers)),
                    len(tickers), stored)
    return {"tickers": len(tickers), "rows": stored, "grid": len(grid)}


def gap_news_study(universe_limit: int = 500, period: str = "2y", batch: int = 40,
                   min_gap: float = 3.0, target_pct: float = 5.5,
                   exit_hour: int = DEFAULT_EXIT_HOUR, vol_window: int = 20):
    """Gap >min_gap at the open: how far does it run, and does a +target_pct
    limit get filled? (2026-08-28, user request.)

    **Conditioned on VOLUME, not on the announcement feed.** The intended
    condition was "day after a price-sensitive announcement", but asxbrief's
    announcement history starts 2026-08-19 -- eight sessions -- so there is
    nothing to backtest that against. `event_momentum.py` hit the same wall and
    established the workaround used here: a move on heavy volume is almost
    certainly repricing news, while the same move on ordinary volume is drift.
    The proxy is imperfect and is labelled as a proxy, but it tests the
    hypothesis on two years instead of eight days.

    Returns one row per qualifying session with the open->high excursion (the
    question asked) and whether the limit filled before the cutoff (the
    tradeable version of it).
    """
    import numpy as np
    import yfinance as yf

    from .yf_lock import YF_LOCK

    tickers = _universe(universe_limit)
    rows = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        with YF_LOCK:
            data = yf.download([f"{t}.AX" for t in chunk], period=period, interval="1h",
                               group_by="ticker", auto_adjust=False, threads=True,
                               progress=False)
        for t in chunk:
            try:
                df = data[f"{t}.AX"].dropna(subset=["Close"])
            except (KeyError, TypeError):
                continue
            if df.empty:
                continue
            idx = df.index.tz_convert(SYDNEY) if df.index.tz is not None \
                else df.index.tz_localize(SYDNEY)
            days = [x.date() for x in idx]
            hours = np.fromiter((x.hour for x in idx), dtype=int, count=len(idx))
            o = df["Open"].to_numpy(float); hi = df["High"].to_numpy(float)
            cl = df["Close"].to_numpy(float)
            vol = df["Volume"].to_numpy(float) if "Volume" in df else np.zeros(len(df))

            starts = [0]
            for j in range(1, len(days)):
                if days[j] != days[j - 1]:
                    starts.append(j)
            ends = starts[1:] + [len(days)]
            sess_vol = [float(np.nansum(vol[starts[k]:ends[k]])) for k in range(len(starts))]

            for k in range(1, len(starts)):
                a, b = starts[k], ends[k]
                pa, pb = starts[k - 1], ends[k - 1]
                if (days[a] - days[pa]).days > 5:
                    continue
                sh = hours[a:b]
                op = np.nonzero(sh == SESSION_OPEN_HOUR)[0]
                if not len(op):
                    continue
                entry = float(o[a + op[0]]); prev_close = float(cl[pb - 1])
                if not entry or not prev_close:
                    continue
                gap = 100 * (entry - prev_close) / prev_close
                if gap < min_gap:
                    continue
                pre = np.nonzero((sh >= SESSION_OPEN_HOUR) & (sh < exit_hour))[0]
                if not len(pre):
                    continue
                i0, i1 = a + pre[0], a + pre[-1]
                high_pre = float(hi[i0:i1 + 1].max())
                # Volume ratio against the median of the prior `vol_window`
                # sessions -- the same "is this heavy?" question the scanner
                # asks live, computed here from the hourly bars themselves.
                base = [v for v in sess_vol[max(0, k - vol_window):k] if v > 0]
                vr = (sess_vol[k] / float(np.median(base))) if base else None
                target = entry * (1 + target_pct / 100)
                hit = bool(high_pre >= target)
                rows.append({
                    "ticker": t, "session": days[a].isoformat(), "gap_pct": gap,
                    "open": entry, "open_to_high_pct": 100 * (high_pre - entry) / entry,
                    "open_to_cutoff_pct": 100 * (float(cl[i1]) - entry) / entry,
                    "hit": hit,
                    "ret_pct": target_pct if hit else 100 * (float(cl[i1]) - entry) / entry,
                    "vol_ratio": vr,
                })
    return rows


def next_day_study(universe_limit: int = 500, period: str = "2y", batch: int = 40,
                   target_pct: float = 5.5, exit_hour: int = DEFAULT_EXIT_HOUR,
                   vol_window: int = 20):
    """Volume spike on day D -> trade day D+1. Strictly no look-ahead.

    **Fixes a look-ahead bug in `gap_news_study`.** That version conditioned on
    the GAP DAY's own full-session volume, which is unknown at 10:00 when the
    entry happens -- so its +0.775% was not obtainable. Here the volume signal
    is day D's, complete and known before D+1 opens, which is also what "the
    day after a price-sensitive announcement" actually means.

    Records both candidate entries so they can be compared on identical events:
      - `overnight_pct`  : buy D's close, sell D+1's open
      - `gap_pct`        : D+1's gap, the thing visible pre-open
      - `open_to_high_pct` / `hit`: buy D+1's open, +target limit by the cutoff
    """
    import numpy as np
    import yfinance as yf

    from .yf_lock import YF_LOCK

    tickers = _universe(universe_limit)
    rows = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        with YF_LOCK:
            data = yf.download([f"{t}.AX" for t in chunk], period=period, interval="1h",
                               group_by="ticker", auto_adjust=False, threads=True,
                               progress=False)
        for t in chunk:
            try:
                df = data[f"{t}.AX"].dropna(subset=["Close"])
            except (KeyError, TypeError):
                continue
            if df.empty:
                continue
            idx = df.index.tz_convert(SYDNEY) if df.index.tz is not None \
                else df.index.tz_localize(SYDNEY)
            days = [x.date() for x in idx]
            hours = np.fromiter((x.hour for x in idx), dtype=int, count=len(idx))
            o = df["Open"].to_numpy(float); hi = df["High"].to_numpy(float)
            cl = df["Close"].to_numpy(float)
            vol = df["Volume"].to_numpy(float) if "Volume" in df else np.zeros(len(df))

            starts = [0]
            for j in range(1, len(days)):
                if days[j] != days[j - 1]:
                    starts.append(j)
            ends = starts[1:] + [len(days)]
            sess_vol = [float(np.nansum(vol[starts[k]:ends[k]])) for k in range(len(starts))]

            for k in range(1, len(starts)):
                a, b = starts[k], ends[k]           # day D+1 (the trading day)
                pa, pb = starts[k - 1], ends[k - 1] # day D  (the signal day)
                if (days[a] - days[pa]).days > 5:
                    continue
                sh = hours[a:b]
                op = np.nonzero(sh == SESSION_OPEN_HOUR)[0]
                if not len(op):
                    continue
                entry = float(o[a + op[0]])
                d_close = float(cl[pb - 1])
                if not entry or not d_close:
                    continue
                # Signal volume is day D's, complete before D+1 opens.
                base = [v for v in sess_vol[max(0, k - 1 - vol_window):k - 1] if v > 0]
                vr = (sess_vol[k - 1] / float(np.median(base))) if base else None
                # And day D's own move, the other thing known in advance.
                ppc = float(cl[starts[k - 2] + (ends[k - 2] - starts[k - 2]) - 1]) \
                    if k >= 2 else None
                d_move = (100 * (d_close - ppc) / ppc) if ppc else None

                pre = np.nonzero((sh >= SESSION_OPEN_HOUR) & (sh < exit_hour))[0]
                if not len(pre):
                    continue
                i0, i1 = a + pre[0], a + pre[-1]
                high_pre = float(hi[i0:i1 + 1].max())
                rows.append({
                    "ticker": t, "session": days[a].isoformat(),
                    "vol_ratio_D": vr, "move_D_pct": d_move,
                    "overnight_pct": 100 * (entry - d_close) / d_close,
                    "gap_pct": 100 * (entry - d_close) / d_close,
                    "open_to_high_pct": 100 * (high_pre - entry) / entry,
                    "open_to_cutoff_pct": 100 * (float(cl[i1]) - entry) / entry,
                    "hit": bool(high_pre >= entry * (1 + target_pct / 100)),
                })
    return rows


def partial_volume_study(universe_limit: int = 500, period: str = "2y", batch: int = 40,
                         signal_hour: int = 15, vol_window: int = 20):
    """The overnight trade using the volume signal ACTUALLY AVAILABLE at scan
    time, not the full day's.

    Only ~43% of an ASX session's volume has traded by 15:00 -- the closing
    auction carries much of the rest -- so a strategy scanned at 15:45 cannot
    use the full-day volume ratio the earlier study conditioned on. Rather than
    scaling a partial figure up by an assumed intraday profile (whose spread is
    28%-70%, far too wide to point-estimate per stock), today's volume through
    `signal_hour` is compared with the prior sessions' volume through the SAME
    hour. Like for like, no profile assumption.
    """
    import numpy as np
    import yfinance as yf

    from .yf_lock import YF_LOCK

    tickers = _universe(universe_limit)
    rows = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        with YF_LOCK:
            data = yf.download([f"{t}.AX" for t in chunk], period=period, interval="1h",
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
            days = [x.date() for x in idx]
            hours = np.fromiter((x.hour for x in idx), dtype=int, count=len(idx))
            o = df["Open"].to_numpy(float); cl = df["Close"].to_numpy(float)
            vol = df["Volume"].to_numpy(float)

            starts = [0]
            for j in range(1, len(days)):
                if days[j] != days[j - 1]:
                    starts.append(j)
            ends = starts[1:] + [len(days)]
            # Volume up to `signal_hour` for every session, plus that session's close.
            partial, closes = [], []
            for k in range(len(starts)):
                a, b = starts[k], ends[k]
                sel = [a + j for j in range(b - a) if hours[a + j] < signal_hour]
                partial.append(float(np.nansum(vol[sel])) if sel else 0.0)
                closes.append(float(cl[b - 1]))

            for k in range(1, len(starts)):
                a = starts[k]
                if (days[a] - days[starts[k - 1]]).days > 5:
                    continue
                sh = hours[a:ends[k]]
                op = np.nonzero(sh == SESSION_OPEN_HOUR)[0]
                if not len(op):
                    continue
                entry = float(o[a + op[0]]); d_close = closes[k - 1]
                if not entry or not d_close:
                    continue
                base = [v for v in partial[max(0, k - 1 - vol_window):k - 1] if v > 0]
                if not base or not partial[k - 1]:
                    continue
                rows.append({
                    "ticker": t, "session": days[a].isoformat(),
                    "vol_ratio_partial": partial[k - 1] / float(np.median(base)),
                    "overnight_pct": 100 * (entry - d_close) / d_close,
                })
    return rows
