"""One-time (and safe-to-rerun) migration: backfill performance.db from all existing JSON logs.

Usage:
    cd /home/geoff/codex/TradingAgents
    .venv/bin/python scripts/migrate_performance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tradingagents.agents.utils.rating import parse_rating
from tradingagents.performance.db import (
    HORIZONS,
    _connect,
    _fetch_price_history,
    _parse_score,
    _resolve_benchmark,
    _write_snapshots,
    init_db,
)

LOGS_DIR = Path.home() / ".tradingagents" / "logs"


def migrate() -> None:
    init_db()

    log_files = sorted(LOGS_DIR.glob("*/TradingAgentsStrategy_logs/full_states_log_*.json"))
    print(f"Found {len(log_files)} log files.")

    inserted = skipped = failed = 0

    for i, path in enumerate(log_files, 1):
        ticker = path.parents[1].name
        date   = path.stem.replace("full_states_log_", "")

        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f"  [{i}/{len(log_files)}] {ticker} {date} — JSON parse error: {exc}")
            failed += 1
            continue

        decision = data.get("final_trade_decision", "")
        if not decision:
            skipped += 1
            continue

        signal    = parse_rating(decision)
        score     = _parse_score(decision)
        benchmark = _resolve_benchmark(ticker)

        # Check if already in DB
        with _connect() as conn:
            existing = conn.execute(
                "SELECT id, price FROM recommendations WHERE ticker=? AND rec_date=?",
                (ticker, date),
            ).fetchone()

        if existing:
            rec_id    = existing["id"]
            rec_price = existing["price"]
            # Still try to fill any missing snapshots
            if rec_price:
                from tradingagents.performance.db import fill_pending_snapshots
                fill_pending_snapshots(rec_id, ticker, date, rec_price, benchmark)
            skipped += 1
            print(f"  [{i}/{len(log_files)}] {ticker} {date} — already exists, refreshed snapshots")
            continue

        print(f"  [{i}/{len(log_files)}] {ticker} {date} — {signal} score={score}  fetching prices...", end=" ", flush=True)
        entry_price, bench_entry, stock_hist, bench_hist = _fetch_price_history(
            ticker, date, benchmark
        )

        if entry_price is None:
            print("no price data")
            with _connect() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO recommendations
                           (ticker, rec_date, signal, score, price, benchmark)
                       VALUES (?, ?, ?, ?, NULL, ?)""",
                    (ticker, date, signal, score, benchmark),
                )
            inserted += 1
            continue

        with _connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO recommendations
                       (ticker, rec_date, signal, score, price, benchmark)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ticker, date, signal, score, entry_price, benchmark),
            )
            row = conn.execute(
                "SELECT id FROM recommendations WHERE ticker=? AND rec_date=?",
                (ticker, date),
            ).fetchone()
            if not row:
                failed += 1
                print("insert failed")
                continue
            rec_id = row["id"]
            _write_snapshots(conn, rec_id, date, entry_price, bench_entry,
                             stock_hist, bench_hist, benchmark)

        snap_count = sum(1 for h in HORIZONS if _has_snapshot(rec_id, h))
        print(f"price={entry_price:.3f}  snapshots={snap_count}/{len(HORIZONS)}")
        inserted += 1

    print(f"\nDone. inserted={inserted}  skipped={skipped}  failed={failed}")


def _has_snapshot(rec_id: int, horizon: str) -> bool:
    with _connect() as conn:
        return bool(conn.execute(
            "SELECT 1 FROM price_snapshots WHERE rec_id=? AND horizon=?",
            (rec_id, horizon),
        ).fetchone())


if __name__ == "__main__":
    migrate()
