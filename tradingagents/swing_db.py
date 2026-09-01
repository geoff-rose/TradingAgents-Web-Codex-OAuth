"""Storage for the simulated Swing Trade page: which focus tickers are in
the swing pool, daily trade proposals, and their lifecycle through to a
closed, realized-P&L row.

Own SQLite db, owned entirely by tradingagents -- unlike research_universe.py,
there's no shared-write exception into asxbrief's db here, so no WAL/ro
concerns; this data has nothing to do with the announcement collector.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

DB_PATH = Path("/opt/tradingagents/data/swing.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS swing_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS swing_universe (
    ticker TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS swing_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    target_price REAL NOT NULL,
    stop_price REAL NOT NULL,
    shares INTEGER NOT NULL,
    dollar_size REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'proposed',
    rationale TEXT,
    ibkr_parent_id INTEGER,
    ibkr_target_id INTEGER,
    ibkr_stop_id INTEGER,
    entered_at TEXT,
    entry_fill_price REAL,
    exited_at TEXT,
    exit_fill_price REAL,
    exit_reason TEXT,
    pnl REAL,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

# Added 2026-08-21 for the range-model daily-refresh mechanic (phase-3 §6):
# `strategy` distinguishes old static-bracket trades ('heuristic') from the
# new day-order-based ones ('range_model'), since both may exist in history.
# `atr_at_entry` is stashed at proposal time so the fixed protective stop can
# be computed consistently once the entry actually fills (the model's ATR
# input at proposal time may differ slightly from ATR at fill time).
# `target_order_date` records which calendar day the *current* target order
# was quoted for -- the daily-refresh job compares this to today's date to
# know whether yesterday's target order needs cancelling and replacing.
_TRADES_MIGRATIONS = [
    "ALTER TABLE swing_trades ADD COLUMN strategy TEXT NOT NULL DEFAULT 'heuristic'",
    "ALTER TABLE swing_trades ADD COLUMN atr_at_entry REAL",
    "ALTER TABLE swing_trades ADD COLUMN target_order_date TEXT",
]

# Statuses that mean "don't propose a new trade for this ticker" -- there's
# already a live proposal or position in flight for it. 'expired' (added for
# range_model day-orders that didn't fill by end of day) is deliberately NOT
# active -- tomorrow's propose job should generate a fresh proposal, not wait
# on an order the exchange has already cancelled.
ACTIVE_STATUSES = ("proposed", "submitted", "open")

DEFAULT_TRADE_DOLLARS = 20_000.0


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    for stmt in _TRADES_MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e):
                raise
    return conn


def get_universe() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ticker, enabled FROM swing_universe ORDER BY ticker"
        ).fetchall()
    return [dict(r) for r in rows]


def set_universe_tickers(tickers: list[str]) -> None:
    """Ensure every given ticker has a swing_universe row (default disabled)
    without touching existing enabled flags -- called to seed new focus
    tickers, not to reset choices already made."""
    now = _utcnow()
    with _connect() as conn:
        for t in tickers:
            conn.execute(
                "INSERT OR IGNORE INTO swing_universe (ticker, enabled, updated_at) VALUES (?, 0, ?)",
                (t, now),
            )
        conn.commit()


def set_enabled(ticker: str, enabled: bool) -> None:
    now = _utcnow()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO swing_universe (ticker, enabled, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(ticker) DO UPDATE SET enabled=excluded.enabled, updated_at=excluded.updated_at",
            (ticker, int(enabled), now),
        )
        conn.commit()


def enabled_tickers() -> list[str]:
    with _connect() as conn:
        rows = conn.execute("SELECT ticker FROM swing_universe WHERE enabled=1").fetchall()
    return [r["ticker"] for r in rows]


def has_active_trade(ticker: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            f"SELECT 1 FROM swing_trades WHERE ticker=? AND status IN "
            f"({','.join('?' * len(ACTIVE_STATUSES))}) LIMIT 1",
            (ticker, *ACTIVE_STATUSES),
        ).fetchone()
    return row is not None


def create_proposal(
    ticker: str, trade_date: str, entry_price: float, target_price: float,
    stop_price: float, shares: int, dollar_size: float, rationale: str,
    strategy: str = "heuristic", atr_at_entry: float | None = None,
) -> int:
    now = _utcnow()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO swing_trades (ticker, trade_date, entry_price, target_price, "
            "stop_price, shares, dollar_size, status, rationale, strategy, atr_at_entry, "
            "created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'proposed', ?, ?, ?, ?, ?)",
            (ticker, trade_date, entry_price, target_price, stop_price, shares,
             dollar_size, rationale, strategy, atr_at_entry, now, now),
        )
        conn.commit()
        return cur.lastrowid


def list_trades(status: str | None = None) -> list[dict[str, Any]]:
    q = "SELECT * FROM swing_trades"
    args: tuple = ()
    if status:
        q += " WHERE status=?"
        args = (status,)
    q += " ORDER BY created_at DESC"
    with _connect() as conn:
        rows = conn.execute(q, args).fetchall()
    return [dict(r) for r in rows]


def get_trade(trade_id: int) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM swing_trades WHERE id=?", (trade_id,)).fetchone()
    return dict(row) if row else None


def update_trade(trade_id: int, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = _utcnow()
    cols = ", ".join(f"{k}=?" for k in fields)
    with _connect() as conn:
        conn.execute(f"UPDATE swing_trades SET {cols} WHERE id=?", (*fields.values(), trade_id))
        conn.commit()


def pnl_summary() -> dict[str, Any]:
    """Realized P&L (closed trades only) for today, this month, and all time,
    Sydney-date bucketed off `exited_at` (already UTC ISO -- fine for
    day/month granularity, a few hours of UTC/AEST skew doesn't change which
    calendar month/day a trade closed in for this purpose)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT exited_at, pnl FROM swing_trades WHERE status='closed' AND pnl IS NOT NULL"
        ).fetchall()
    today = time.strftime("%Y-%m-%d", time.gmtime())
    month = today[:7]
    total = today_total = month_total = 0.0
    for r in rows:
        total += r["pnl"]
        if r["exited_at"]:
            if r["exited_at"][:10] == today:
                today_total += r["pnl"]
            if r["exited_at"][:7] == month:
                month_total += r["pnl"]
    return {"today": today_total, "month": month_total, "all_time": total, "n_closed": len(rows)}


def get_trade_dollars() -> float:
    with _connect() as conn:
        row = conn.execute("SELECT value FROM swing_settings WHERE key='trade_dollars'").fetchone()
    return float(row["value"]) if row else DEFAULT_TRADE_DOLLARS


def set_trade_dollars(amount: float) -> None:
    if amount <= 0:
        raise ValueError("trade_dollars must be positive")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO swing_settings (key, value) VALUES ('trade_dollars', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(amount),),
        )
        conn.commit()


# Range-model entry/target percentiles (phase-3 §6, added 2026-08-21).
# Defaults picked from the first backtest sweep (entry_q=0.1: only the
# deepest ~10% of pullbacks qualify; target_q=0.7) -- see the asx-dashboard
# skill's range_model section for the full sweep and its caveats (small
# out-of-sample count, real multiple-comparisons risk from picking the best
# of several tested combinations). Deliberately user-editable: the user
# expects "a fair bit of refinement" here, not a one-shot final answer.
DEFAULT_ENTRY_Q = 0.1
DEFAULT_TARGET_Q = 0.7


def get_range_model_quantiles() -> tuple[float, float]:
    with _connect() as conn:
        row_e = conn.execute("SELECT value FROM swing_settings WHERE key='range_entry_q'").fetchone()
        row_t = conn.execute("SELECT value FROM swing_settings WHERE key='range_target_q'").fetchone()
    return (
        float(row_e["value"]) if row_e else DEFAULT_ENTRY_Q,
        float(row_t["value"]) if row_t else DEFAULT_TARGET_Q,
    )


def set_range_model_quantiles(entry_q: float, target_q: float) -> None:
    if not (0 < entry_q < 1 and 0 < target_q < 1):
        raise ValueError("quantiles must be between 0 and 1")
    with _connect() as conn:
        for key, value in (("range_entry_q", entry_q), ("range_target_q", target_q)):
            conn.execute(
                "INSERT INTO swing_settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )
        conn.commit()
