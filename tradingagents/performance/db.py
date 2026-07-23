"""Performance tracking database.

Stores every recommendation (ticker, date, signal, score, entry price) in SQLite,
then fills price snapshots at 1m/3m/6m/12m horizons as they become available.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yfinance as yf

logger = logging.getLogger(__name__)

DB_PATH = Path.home() / ".tradingagents" / "performance.db"

# Calendar-day offsets for each horizon label
HORIZONS: Dict[str, int] = {
    "1m":  30,
    "3m":  91,
    "6m":  182,
    "12m": 365,
}

# Maps ticker suffix → benchmark ticker (mirrors default_config.py)
_BENCHMARK_MAP = {
    ".NS":  "^NSEI",
    ".BO":  "^BSESN",
    ".T":   "^N225",
    ".HK":  "^HSI",
    ".L":   "^FTSE",
    ".TO":  "^GSPTSE",
    ".AX":  "^AXJO",
    "":     "SPY",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_benchmark(ticker: str) -> str:
    upper = ticker.upper()
    for suffix, bench in _BENCHMARK_MAP.items():
        if suffix and upper.endswith(suffix):
            return bench
    return _BENCHMARK_MAP[""]


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _parse_score(text: str) -> Optional[int]:
    """Extract total score from scorecard markdown, e.g. '(Score: 77/100)'."""
    m = re.search(r'\(Score:\s*(\d+)\s*/\s*100\)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Fallback: "**Total: 77/100**"
    m = re.search(r'Total[:\s]+\*{0,2}(\d+)\s*/\s*100', text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _fetch_price_history(
    ticker: str,
    start_date: str,
    benchmark: str,
) -> Tuple[Optional[float], Optional[float], Any, Any]:
    """Return (entry_price, bench_entry_price, stock_hist_df, bench_hist_df).

    Fetches from up to 7 days before start_date to today so that weekends and
    public holidays don't produce a NULL entry price. The returned dataframes
    cover the full window needed for all horizon snapshot lookups.
    """
    fetch_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
    try:
        stock_hist = yf.Ticker(ticker).history(start=fetch_start, end=end)
        bench_hist = yf.Ticker(benchmark).history(start=fetch_start, end=end)
    except Exception as exc:
        logger.warning("yfinance error fetching %s / %s: %s", ticker, benchmark, exc)
        return None, None, None, None

    if stock_hist.empty or bench_hist.empty:
        return None, None, None, None

    # Use the last available close on or before start_date as the entry price.
    # This handles weekends and public holidays gracefully.
    from pandas import Timestamp
    start_ts = Timestamp(start_date)
    if stock_hist.index.tz is not None:
        start_ts = start_ts.tz_localize(stock_hist.index.tz)
    on_or_before = stock_hist[stock_hist.index <= start_ts]
    entry_row = on_or_before if not on_or_before.empty else stock_hist
    entry_price = float(entry_row["Close"].iloc[-1])

    on_or_before_b = bench_hist[bench_hist.index <= start_ts]
    bench_row = on_or_before_b if not on_or_before_b.empty else bench_hist
    bench_entry = float(bench_row["Close"].iloc[-1])

    return entry_price, bench_entry, stock_hist, bench_hist


def _due_horizons(rec_date: str, existing: set) -> Dict[str, int]:
    """Horizons not yet captured whose target date has already passed.

    Pure date arithmetic — no network call — so callers can skip the
    yfinance fetch entirely when nothing has actually matured yet.
    """
    start = datetime.strptime(rec_date, "%Y-%m-%d").date()
    today = datetime.now().date()
    return {
        horizon: days
        for horizon, days in HORIZONS.items()
        if horizon not in existing and (start + timedelta(days=days)) < today
    }


def _snapshot_at_horizon(
    rec_date: str,
    horizon_days: int,
    entry_price: float,
    bench_entry: float,
    stock_hist,
    bench_hist,
) -> Optional[Tuple[str, float, float, float, float]]:
    """Find prices at target date and compute returns.

    Returns (snapshot_date, price, raw_return, benchmark_return, alpha) or None.
    """
    from pandas import Timestamp
    target = datetime.strptime(rec_date, "%Y-%m-%d") + timedelta(days=horizon_days)
    if target.date() >= datetime.now().date():
        return None  # not yet reached

    target_ts = Timestamp(target).tz_localize(stock_hist.index.tz)
    future_stock = stock_hist[stock_hist.index >= target_ts]
    future_bench = bench_hist[bench_hist.index >= target_ts]

    if future_stock.empty or future_bench.empty:
        return None

    snap_price = float(future_stock["Close"].iloc[0])
    snap_bench = float(future_bench["Close"].iloc[0])
    snap_date  = str(future_stock.index[0].date())

    raw_return       = (snap_price - entry_price) / entry_price
    benchmark_return = (snap_bench - bench_entry) / bench_entry
    alpha            = raw_return - benchmark_return
    return snap_date, snap_price, raw_return, benchmark_return, alpha


# ── Schema ───────────────────────────────────────────────────────────────────

def init_db() -> None:
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL,
                rec_date    TEXT    NOT NULL,
                signal      TEXT    NOT NULL,
                score       INTEGER,
                price       REAL,
                benchmark   TEXT    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                UNIQUE(ticker, rec_date)
            );

            CREATE TABLE IF NOT EXISTS price_snapshots (
                rec_id           INTEGER NOT NULL REFERENCES recommendations(id) ON DELETE CASCADE,
                horizon          TEXT    NOT NULL,
                snapshot_date    TEXT    NOT NULL,
                price            REAL    NOT NULL,
                raw_return       REAL    NOT NULL,
                benchmark_return REAL    NOT NULL,
                alpha_return     REAL    NOT NULL,
                captured_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (rec_id, horizon)
            );

            CREATE INDEX IF NOT EXISTS idx_rec_ticker   ON recommendations(ticker);
            CREATE INDEX IF NOT EXISTS idx_rec_date     ON recommendations(rec_date);
            CREATE INDEX IF NOT EXISTS idx_rec_signal   ON recommendations(signal);
            CREATE INDEX IF NOT EXISTS idx_snap_rec_id  ON price_snapshots(rec_id);
        """)


# ── Write operations ──────────────────────────────────────────────────────────

def record_recommendation(
    ticker: str,
    rec_date: str,
    signal: str,
    score: Optional[int] = None,
    final_trade_decision: str = "",
) -> None:
    """Insert a recommendation and immediately fill any available snapshots.

    Safe to call multiple times — UNIQUE(ticker, rec_date) prevents duplicates.
    Score is parsed from final_trade_decision if not supplied directly.
    """
    init_db()
    if score is None and final_trade_decision:
        score = _parse_score(final_trade_decision)

    benchmark = _resolve_benchmark(ticker)

    with _connect() as conn:
        existing = conn.execute(
            "SELECT id, price FROM recommendations WHERE ticker=? AND rec_date=?",
            (ticker, rec_date),
        ).fetchone()

        if existing:
            rec_id    = existing["id"]
            rec_price = existing["price"]
        else:
            entry_price, bench_entry, stock_hist, bench_hist = _fetch_price_history(
                ticker, rec_date, benchmark
            )
            conn.execute(
                """INSERT OR IGNORE INTO recommendations
                       (ticker, rec_date, signal, score, price, benchmark)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ticker, rec_date, signal, score, entry_price, benchmark),
            )
            row = conn.execute(
                "SELECT id FROM recommendations WHERE ticker=? AND rec_date=?",
                (ticker, rec_date),
            ).fetchone()
            if not row:
                return
            rec_id    = row["id"]
            rec_price = entry_price

            # Fill snapshots while we have the history in memory
            if entry_price and stock_hist is not None:
                _write_snapshots(conn, rec_id, rec_date, rec_price, bench_entry,
                                 stock_hist, bench_hist, benchmark)
            return

    # Existing record — check for missing snapshots
    if rec_price:
        fill_pending_snapshots(rec_id, ticker, rec_date, rec_price, benchmark)


def _write_snapshots(conn, rec_id, rec_date, entry_price, bench_entry,
                     stock_hist, bench_hist, benchmark):
    """Write all computable snapshots into an open connection."""
    existing = {r["horizon"] for r in conn.execute(
        "SELECT horizon FROM price_snapshots WHERE rec_id=?", (rec_id,)
    )}
    for horizon, days in HORIZONS.items():
        if horizon in existing:
            continue
        result = _snapshot_at_horizon(
            rec_date, days, entry_price, bench_entry, stock_hist, bench_hist
        )
        if result is None:
            continue
        snap_date, price, raw_ret, bench_ret, alpha = result
        conn.execute(
            """INSERT OR REPLACE INTO price_snapshots
                   (rec_id, horizon, snapshot_date, price, raw_return, benchmark_return, alpha_return)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (rec_id, horizon, snap_date, price, raw_ret, bench_ret, alpha),
        )


def fill_pending_snapshots(
    rec_id: int,
    ticker: str,
    rec_date: str,
    rec_price: float,
    benchmark: str,
) -> None:
    """Fill missing snapshots for one recommendation (uses fresh yfinance fetch)."""
    with _connect() as conn:
        existing = {r["horizon"] for r in conn.execute(
            "SELECT horizon FROM price_snapshots WHERE rec_id=?", (rec_id,)
        )}
        if existing == set(HORIZONS):
            return  # all done

    if not _due_horizons(rec_date, existing):
        return  # nothing has matured yet — skip the network fetch entirely

    _, bench_entry, stock_hist, bench_hist = _fetch_price_history(ticker, rec_date, benchmark)
    if stock_hist is None:
        return

    # Recalculate bench_entry from the fresh history
    bench_entry_price = float(bench_hist["Close"].iloc[0]) if bench_hist is not None and not bench_hist.empty else None
    if bench_entry_price is None:
        return

    with _connect() as conn:
        _write_snapshots(conn, rec_id, rec_date, rec_price, bench_entry_price,
                         stock_hist, bench_hist, benchmark)


def _parse_rating_simple(text: str, default: str = "Hold") -> str:
    """Minimal rating extractor — avoids importing the full agents package."""
    import re as _re
    ratings = {"buy", "overweight", "hold", "underweight", "sell"}
    label_re = _re.compile(r"rating.*?[:\-][\s*]*(\w+)", _re.IGNORECASE)
    for line in text.splitlines():
        m = label_re.search(line)
        if m and m.group(1).lower() in ratings:
            return m.group(1).capitalize()
    for line in text.splitlines():
        for word in line.lower().split():
            if word.strip("*:.,") in ratings:
                return word.strip("*:.,").capitalize()
    return default


def ingest_from_logs(logs_dir: Optional[Path] = None) -> Dict[str, int]:
    """Scan the logs directory and import any reports not yet in the DB.

    Returns counts: {"imported": N, "skipped": N, "errors": N}
    """
    import json as _json
    parse_rating = _parse_rating_simple

    if logs_dir is None:
        logs_dir = Path.home() / ".tradingagents" / "logs"
        env_override = __import__("os").getenv("TRADINGAGENTS_RESULTS_DIR")
        if env_override:
            logs_dir = Path(env_override)

    init_db()
    counts = {"imported": 0, "skipped": 0, "errors": 0}

    if not logs_dir.exists():
        return counts

    with _connect() as conn:
        existing = {
            (r["ticker"], r["rec_date"])
            for r in conn.execute("SELECT ticker, rec_date FROM recommendations")
        }

    for ticker_dir in logs_dir.iterdir():
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name

        # New format: logs/{ticker}/{date}/reports/final_trade_decision.md
        for date_dir in ticker_dir.iterdir():
            if not date_dir.is_dir() or date_dir.name == "TradingAgentsStrategy_logs":
                continue
            rec_date = date_dir.name
            md = date_dir / "reports" / "final_trade_decision.md"
            if not md.exists():
                continue
            if (ticker, rec_date) in existing:
                counts["skipped"] += 1
                continue
            try:
                text = md.read_text()
                signal = parse_rating(text)
                record_recommendation(ticker, rec_date, signal, final_trade_decision=text)
                counts["imported"] += 1
            except Exception:
                counts["errors"] += 1

        # Legacy format: logs/{ticker}/TradingAgentsStrategy_logs/full_states_log_{date}.json
        legacy_dir = ticker_dir / "TradingAgentsStrategy_logs"
        if legacy_dir.exists():
            for f in legacy_dir.glob("full_states_log_*.json"):
                rec_date = f.stem.replace("full_states_log_", "")
                if (ticker, rec_date) in existing:
                    counts["skipped"] += 1
                    continue
                try:
                    raw = _json.loads(f.read_text())
                    text = raw.get("final_trade_decision") or ""
                    if not text:
                        counts["skipped"] += 1
                        continue
                    signal = parse_rating(text)
                    record_recommendation(ticker, rec_date, signal, final_trade_decision=text)
                    counts["imported"] += 1
                except Exception:
                    counts["errors"] += 1

    return counts


def refresh_all_snapshots() -> int:
    """Fill missing snapshots for every recommendation. Returns count updated."""
    init_db()
    with _connect() as conn:
        recs = conn.execute(
            """SELECT r.id, r.ticker, r.rec_date, r.price, r.benchmark
               FROM recommendations r
               WHERE r.price IS NOT NULL
                 AND (SELECT COUNT(*) FROM price_snapshots s WHERE s.rec_id = r.id) < ?""",
            (len(HORIZONS),),
        ).fetchall()

    updated = 0
    for rec in recs:
        prev_count = _snapshot_count(rec["id"])
        fill_pending_snapshots(
            rec["id"], rec["ticker"], rec["rec_date"], rec["price"], rec["benchmark"]
        )
        if _snapshot_count(rec["id"]) > prev_count:
            updated += 1
    return updated


def _snapshot_count(rec_id: int) -> int:
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM price_snapshots WHERE rec_id=?", (rec_id,)
        ).fetchone()[0]


# ── Read operations ───────────────────────────────────────────────────────────

def get_performance_data() -> Dict[str, Any]:
    """Return all data needed for the performance UI.

    Returns:
        {
          "records": [...],   # all recommendations with snapshots embedded
          "summary": {...},   # win rates / avg returns by signal × horizon
        }
    """
    init_db()
    with _connect() as conn:
        recs = conn.execute(
            """SELECT id, ticker, rec_date, signal, score, price, benchmark, created_at
               FROM recommendations ORDER BY rec_date DESC, ticker""",
        ).fetchall()

        snaps = conn.execute(
            """SELECT rec_id, horizon, snapshot_date, raw_return, benchmark_return, alpha_return
               FROM price_snapshots""",
        ).fetchall()

    # Index snapshots by rec_id
    snap_map: Dict[int, Dict[str, Dict]] = {}
    for s in snaps:
        snap_map.setdefault(s["rec_id"], {})[s["horizon"]] = {
            "snapshot_date":    s["snapshot_date"],
            "raw_return":       round(s["raw_return"] * 100, 2),
            "benchmark_return": round(s["benchmark_return"] * 100, 2),
            "alpha_return":     round(s["alpha_return"] * 100, 2),
        }

    records = []
    for r in recs:
        records.append({
            "id":         r["id"],
            "ticker":     r["ticker"],
            "rec_date":   r["rec_date"],
            "signal":     r["signal"],
            "score":      r["score"],
            "price":      r["price"],
            "benchmark":  r["benchmark"],
            "snapshots":  snap_map.get(r["id"], {}),
        })

    summary = _build_summary(records)
    return {"records": records, "summary": summary}


def _build_summary(records: List[Dict]) -> Dict:
    """Aggregate win rates and average returns by signal × horizon."""
    from collections import defaultdict

    # bullish signals where "win" = positive return
    bullish = {"Buy", "Overweight"}
    bearish = {"Sell", "Underweight"}

    stats: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        sig = rec["signal"]
        for hz, snap in rec["snapshots"].items():
            stats[sig][hz].append(snap["raw_return"])

    summary = {}
    for sig, horizons in stats.items():
        summary[sig] = {}
        for hz, returns in horizons.items():
            if not returns:
                continue
            if sig in bullish:
                wins = sum(1 for r in returns if r > 0)
            elif sig in bearish:
                wins = sum(1 for r in returns if r < 0)
            else:  # Hold
                wins = sum(1 for r in returns if abs(r) < 5)
            summary[sig][hz] = {
                "count":       len(returns),
                "win_rate":    round(wins / len(returns) * 100, 1),
                "avg_return":  round(sum(returns) / len(returns), 2),
            }

    # Also add alpha stats
    alpha_stats: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        for hz, snap in rec["snapshots"].items():
            alpha_stats[rec["signal"]][hz].append(snap["alpha_return"])

    for sig, horizons in alpha_stats.items():
        for hz, alphas in horizons.items():
            if hz in summary.get(sig, {}):
                summary[sig][hz]["avg_alpha"] = round(sum(alphas) / len(alphas), 2)

    return summary
