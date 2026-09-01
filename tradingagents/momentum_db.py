"""Paper-trade log for the movers scanner (2026-08-22).

**Separate table from `swing_trades`, deliberately.** The project already has
three distinct signal sources sharing one paper account (swing/range model,
announcement AI, and now momentum) and the standing rule is not to conflate
them — mixing them into one table makes per-strategy P&L unrecoverable later,
which is the only thing this log exists to produce.

**Log-based, not broker-routed.** Entries are recorded at a price the user
enters (pre-filled from the scan) and marked to market from daily closes. That
is deliberate for a first cut: the point is to build intuition about which
setups work, and a log cannot half-fill, cannot be rejected, and cannot leave
a stray order on a real book. The IBKR paper path already exists
(`swing_ibkr.place_bracket`) and can be wired in later if realistic fills
become the question — at which point the fill *is* the experiment, and the
cost analysis in `event_momentum.py` says that will matter below 50c.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "swing.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS momentum_trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    opened_at    TEXT NOT NULL,
    entry_price  REAL NOT NULL,
    shares       INTEGER NOT NULL,
    dollars      REAL NOT NULL,
    stop_pct     REAL,
    hold_days    INTEGER,
    status       TEXT NOT NULL DEFAULT 'open',   -- open | closed
    exit_price   REAL,
    exit_at      TEXT,
    exit_reason  TEXT,                            -- manual | stop | time
    pnl_pct      REAL,
    pnl_dollars  REAL,
    note         TEXT,
    setup_json   TEXT                             -- the scanner row that triggered it
);
"""


# Added 2026-08-25. `market` exists because every price path here assumed ASX
# and appended `.AX`, so five US trades (BE, CCJ, OKLO, INTC, UUUU) were being
# marked as `INTC.AX` and returning nothing -- they sat with no last price and
# no P&L forever. Resolved once when the trade is opened and stored, never
# re-derived per mark-to-market: that would mean a network probe per row per
# refresh, and would let a row change exchange if one probe happened to fail.
_EXTRA_COLUMNS = {"market": "TEXT", "company": "TEXT"}


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    have = {r["name"] for r in conn.execute("PRAGMA table_info(momentum_trades)")}
    for col, typ in _EXTRA_COLUMNS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE momentum_trades ADD COLUMN {col} {typ}")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_trade(ticker: str, entry_price: float, dollars: float,
               stop_pct: float | None = None, hold_days: int | None = 10,
               note: str | None = None, setup: dict[str, Any] | None = None,
               market: str | None = None, company: str | None = None) -> dict[str, Any]:
    """`market` is 'AU' or 'US'; omitted, it is detected once here and stored
    with the trade so mark-to-market asks the price provider for the right
    symbol. `company` is likewise resolved once for display."""
    if entry_price <= 0 or dollars <= 0:
        return {"ok": False, "error": "entry price and dollar size must be positive"}
    shares = int(dollars // entry_price)
    if shares < 1:
        return {"ok": False, "error": f"${dollars:,.0f} buys 0 shares at ${entry_price}"}
    ticker = ticker.upper()
    from tradingagents.symbols import company_name, resolve_market
    market = resolve_market(ticker, hint=market)
    if company is None:
        company = company_name(ticker, market)
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO momentum_trades (ticker, opened_at, entry_price, shares, dollars,"
            " stop_pct, hold_days, note, setup_json, market, company)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (ticker, _now(), entry_price, shares, shares * entry_price,
             stop_pct, hold_days, note, json.dumps(setup) if setup else None,
             market, company),
        )
        conn.commit()
        return {"ok": True, "id": cur.lastrowid, "shares": shares,
                "dollars": round(shares * entry_price, 2),
                "market": market, "company": company}


def close_trade(trade_id: int, exit_price: float, reason: str = "manual") -> dict[str, Any]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM momentum_trades WHERE id=?", (trade_id,)).fetchone()
        if not row:
            return {"ok": False, "error": f"no trade {trade_id}"}
        if row["status"] == "closed":
            return {"ok": False, "error": f"trade {trade_id} already closed"}
        pnl_pct = (exit_price - row["entry_price"]) / row["entry_price"] * 100
        conn.execute(
            "UPDATE momentum_trades SET status='closed', exit_price=?, exit_at=?, exit_reason=?,"
            " pnl_pct=?, pnl_dollars=? WHERE id=?",
            (exit_price, _now(), reason, round(pnl_pct, 4),
             round((exit_price - row["entry_price"]) * row["shares"], 2), trade_id),
        )
        conn.commit()
    return {"ok": True, "pnl_pct": round(pnl_pct, 3)}


def list_trades(status: str | None = None) -> list[dict[str, Any]]:
    with _connect() as conn:
        q = "SELECT * FROM momentum_trades"
        args: tuple = ()
        if status:
            q += " WHERE status=?"
            args = (status,)
        rows = [dict(r) for r in conn.execute(q + " ORDER BY id DESC", args)]
    for r in rows:
        if r.get("setup_json"):
            try:
                r["setup"] = json.loads(r["setup_json"])
            except Exception:
                r["setup"] = None
        r.pop("setup_json", None)
    return rows


def mark_to_market() -> list[dict[str, Any]]:
    """Attach a live price and unrealised P&L to each open trade, plus how
    many days it has been held against its planned hold. Read-only — it never
    auto-closes, because the user is running this by hand to build intuition
    and an automatic exit would take away the decision being learned."""
    from tradingagents.backtest import fetch_daily_history

    out = []
    for t in list_trades(status="open"):
        try:
            # Pass the trade's own market. Without it every symbol got `.AX`
            # and a US holding could never mark to market.
            df = fetch_daily_history(t["ticker"], period="1mo",
                                     market=(t.get("market") or "AU"))
            last = float(df["Close"].iloc[-1]) if not df.empty else None
        except Exception:
            last = None
        if last:
            t["last_price"] = round(last, 4)
            t["unrealised_pct"] = round((last - t["entry_price"]) / t["entry_price"] * 100, 3)
            t["unrealised_dollars"] = round((last - t["entry_price"]) * t["shares"], 2)
            if t.get("stop_pct"):
                t["stop_breached"] = bool(t["unrealised_pct"] <= -abs(t["stop_pct"]))
        opened = datetime.fromisoformat(t["opened_at"])
        t["days_held"] = (datetime.now(timezone.utc) - opened).days
        if t.get("hold_days"):
            t["hold_elapsed"] = bool(t["days_held"] >= t["hold_days"])
        out.append(t)
    return out


def summary() -> dict[str, Any]:
    closed = list_trades(status="closed")
    if not closed:
        return {"n_closed": 0, "n_open": len(list_trades(status="open"))}
    pnls = [c["pnl_pct"] for c in closed if c["pnl_pct"] is not None]
    wins = [p for p in pnls if p > 0]
    return {
        "n_closed": len(closed),
        "n_open": len(list_trades(status="open")),
        "win_rate_pct": round(100 * len(wins) / len(pnls), 1) if pnls else None,
        "avg_pnl_pct": round(sum(pnls) / len(pnls), 3) if pnls else None,
        "total_pnl_dollars": round(sum(c["pnl_dollars"] or 0 for c in closed), 2),
        # The measured payoff is lottery-shaped (median 0.00%, 46% positive,
        # p99 +88%), so a median well below the mean is EXPECTED here and is
        # not evidence the approach is failing. Both are shown for that reason.
        "median_pnl_pct": round(sorted(pnls)[len(pnls) // 2], 3) if pnls else None,
    }


def backfill_markets(force: bool = False) -> dict[str, Any]:
    """Resolve `market` and `company` for trades logged before those columns
    existed. Safe to re-run; skips rows already filled unless `force`."""
    from tradingagents.symbols import company_name, resolve_market

    with _connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT id, ticker, market, company FROM momentum_trades")]
    updated = []
    for r in rows:
        need_mkt = force or not r.get("market")
        need_co = force or not r.get("company")
        if not (need_mkt or need_co):
            continue
        market = resolve_market(r["ticker"], hint=None if need_mkt else r.get("market"))
        company = company_name(r["ticker"], market) if need_co else r.get("company")
        with _connect() as conn:
            conn.execute("UPDATE momentum_trades SET market=?, company=? WHERE id=?",
                         (market, company, r["id"]))
            conn.commit()
        updated.append({"ticker": r["ticker"], "market": market, "company": company})
    return {"updated": len(updated), "rows": len(rows), "detail": updated}
