"""Places and syncs simulated swing-trade bracket orders on the paper IBKR
account. Same paper Gateway (127.0.0.1:4002) as the phase-2 bar backfill and
research_universe's ibkr-backfill CLI, but its own clientId (71) so a brief
connect-place-disconnect or connect-sync-disconnect cycle here never clashes
with a concurrent backfill session.

Two order mechanics live here:

- `place_bracket()` -- the original heuristic's three-leg bracket (parent
  limit buy, take-profit limit sell, stop-loss stop sell, GTC, OCA-linked by
  IBKR itself). Kept for reference/backtest comparison; the live heuristic
  strategy is no longer proposed by default after backtest.py showed it loses
  to random entries (see the asx-dashboard skill's backtest.py section).
- `place_day_entry()` / `place_stop()` / `refresh_target_order()` -- the
  range-model daily-refresh mechanic (phase-3 §6, added 2026-08-21): entry is
  a single DAY order that expires automatically if unfilled (a fresh one is
  quoted the next day using updated model output); once filled, a separate
  fixed GTC stop is placed once and left alone; the take-profit side is a DAY
  order that gets cancelled and replaced with a freshly-modelled level every
  trading day the position stays open. This is what "daily refresh" means
  operationally -- risk control (the stop) never moves, only the target does.
"""

from __future__ import annotations

from typing import Any

from ib_async import IB, LimitOrder, Stock, StopOrder

IBKR_HOST = "127.0.0.1"
IBKR_PORT = 4002
IBKR_CLIENT_ID = 71


def _connect() -> IB:
    ib = IB()
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=15)
    return ib


def place_bracket(ticker: str, shares: int, entry: float, target: float, stop: float) -> dict[str, Any]:
    """Submits a GTC bracket (buy limit @entry, sell limit @target, sell stop
    @stop). GTC, not day-only -- this is a swing hold, meant to sit for
    however many days it takes to hit target or stop.

    Contract uses exchange="SMART" with primaryExchange="ASX", not a direct
    "ASX" exchange -- confirmed live 2026-08-21 that direct-to-ASX API orders
    get silently discarded (IBKR error 10311/201: a precautionary setting
    blocks directly-routed orders from the API). SMART routing is IBKR's own
    standard pattern for API stock orders and isn't subject to that block."""
    ib = _connect()
    try:
        contract = Stock(ticker, "SMART", "AUD", primaryExchange="ASX")
        ib.qualifyContracts(contract)
        bracket = ib.bracketOrder("BUY", shares, entry, target, stop)
        for o in bracket:
            o.tif = "GTC"
            ib.placeOrder(contract, o)
        ib.sleep(1.5)
        return {
            "ok": True,
            "parent_id": bracket.parent.orderId,
            "target_id": bracket.takeProfit.orderId,
            "stop_id": bracket.stopLoss.orderId,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        ib.disconnect()


def _contract(ticker: str) -> Stock:
    return Stock(ticker, "SMART", "AUD", primaryExchange="ASX")


def place_day_entry(ticker: str, shares: int, entry_price: float) -> dict[str, Any]:
    """Range-model mechanic (phase-3 §6, added 2026-08-21): a single DAY
    limit buy, not a bracket -- expires automatically at the exchange's end
    of day if unfilled, which is exactly the semantics a daily-refreshed
    entry needs (today's quoted level shouldn't carry over to tomorrow)."""
    ib = _connect()
    try:
        contract = _contract(ticker)
        ib.qualifyContracts(contract)
        order = LimitOrder("BUY", shares, entry_price, tif="DAY")
        ib.placeOrder(contract, order)
        ib.sleep(1.5)
        return {"ok": True, "order_id": order.orderId}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        ib.disconnect()


def place_stop(ticker: str, shares: int, stop_price: float) -> dict[str, Any]:
    """A standalone GTC stop-loss, placed once a position is confirmed
    filled. Fixed for the life of the trade -- risk control shouldn't be a
    moving target, only the take-profit side is daily-refreshed."""
    ib = _connect()
    try:
        contract = _contract(ticker)
        ib.qualifyContracts(contract)
        order = StopOrder("SELL", shares, stop_price, tif="GTC")
        ib.placeOrder(contract, order)
        ib.sleep(1.5)
        return {"ok": True, "order_id": order.orderId}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        ib.disconnect()


def refresh_target_order(ticker: str, shares: int, old_order_id: int | None, new_target_price: float) -> dict[str, Any]:
    """Cancels yesterday's (still-resting) target day-order, if any, and
    submits today's fresh one -- both within one connection, so there's no
    window where the position is briefly unprotected on the target side
    (the fixed GTC stop stays in place throughout regardless)."""
    ib = _connect()
    try:
        if old_order_id is not None:
            ib.reqAllOpenOrders()
            ib.sleep(1)
            for t in ib.trades():
                if t.order.orderId == old_order_id and t.orderStatus.status not in ("Filled", "Cancelled"):
                    ib.cancelOrder(t.order)
            ib.sleep(1)
        contract = _contract(ticker)
        ib.qualifyContracts(contract)
        order = LimitOrder("SELL", shares, new_target_price, tif="DAY")
        ib.placeOrder(contract, order)
        ib.sleep(1.5)
        return {"ok": True, "order_id": order.orderId}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        ib.disconnect()


def cancel_order(order_id: int) -> None:
    ib = _connect()
    try:
        ib.reqAllOpenOrders()
        ib.sleep(1)
        for t in ib.trades():
            if t.order.orderId == order_id:
                ib.cancelOrder(t.order)
        ib.sleep(1)
    finally:
        ib.disconnect()


def fetch_order_statuses(order_ids: list[int]) -> dict[int, dict[str, Any]]:
    """One connect/disconnect cycle covering every tracked order id at once
    -- called by the periodic sync job, not per-trade, to avoid hammering
    the Gateway with a fresh connection per row."""
    if not order_ids:
        return {}
    ib = _connect()
    try:
        ib.reqAllOpenOrders()
        ib.sleep(1)
        ib.reqCompletedOrders(apiOnly=False)
        ib.sleep(1)
        out: dict[int, dict[str, Any]] = {}
        for t in ib.trades():
            oid = t.order.orderId
            if oid not in order_ids:
                continue
            fill_price = t.orderStatus.avgFillPrice or None
            out[oid] = {
                "status": t.orderStatus.status,
                "fill_price": fill_price if fill_price else None,
            }
        return out
    finally:
        ib.disconnect()
