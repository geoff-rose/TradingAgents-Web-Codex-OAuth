"""Paper IBKR orders on client 71.

Both strategies submit staged brackets with linked GTC exits. Range-model
entries expire daily; target prices are modified in place. Cumulative fills,
broker acknowledgements and confirmed cancellations drive reconciliation.
Unknown order state stays visible and never authorizes a duplicate order.
"""

from __future__ import annotations

from typing import Any
import os
import asyncio
import time
from threading import RLock
from functools import wraps
from ib_async import ExecutionFilter

from ib_async import IB, LimitOrder, Stock, StopOrder

IBKR_HOST = os.environ.get("SWING_IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.environ.get("SWING_IBKR_PORT", "4002"))
IBKR_CLIENT_ID = 71
ORDER_LOCK = RLock()


def serialized(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with ORDER_LOCK:
            return fn(*args, **kwargs)
    return wrapped


def _connect() -> IB:
    if IBKR_PORT not in (4002, 7497):
        raise RuntimeError("Swing is restricted to paper Gateway/TWS ports")
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    ib = IB()
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=15)
    accounts = ib.managedAccounts()
    if not accounts or any(not a.startswith("DU") for a in accounts):
        ib.disconnect()
        raise RuntimeError("Swing requires a verified paper account")
    return ib


def _acknowledge(ib, trade):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        state = trade.orderStatus.status
        if state in ("PreSubmitted", "Submitted", "Filled"):
            return
        if state in ("Inactive", "Cancelled", "ApiCancelled"):
            raise RuntimeError(f"Broker rejected/cancelled order {trade.order.orderId}: {state}")
        ib.sleep(0.1)
    raise RuntimeError(f"Broker acknowledgement missing for order {trade.order.orderId}; reconcile before retrying")


@serialized
def place_bracket(ticker: str, shares: int, entry: float, target: float, stop: float,
                  *, day_entry: bool = False, order_ref: str = "", persist_ids=None) -> dict[str, Any]:
    """Submits a GTC bracket (buy limit @entry, sell limit @target, sell stop
    @stop). GTC, not day-only -- this is a swing hold, meant to sit for
    however many days it takes to hit target or stop.

    Contract uses exchange="SMART" with primaryExchange="ASX", not a direct
    "ASX" exchange -- confirmed live 2026-08-21 that direct-to-ASX API orders
    get silently discarded (IBKR error 10311/201: a precautionary setting
    blocks directly-routed orders from the API). SMART routing is IBKR's own
    standard pattern for API stock orders and isn't subject to that block."""
    ib = _connect()
    ids = {}
    try:
        contract = Stock(ticker, "SMART", "AUD", primaryExchange="ASX")
        ib.qualifyContracts(contract)
        bracket = ib.bracketOrder("BUY", shares, entry, target, stop)
        ids = {"parent_id": bracket.parent.orderId, "target_id": bracket.takeProfit.orderId,
               "stop_id": bracket.stopLoss.orderId}
        if persist_ids:
            persist_ids(ids)
        # Explicit partial-exit reduction with overfill protection.
        for child in (bracket.takeProfit, bracket.stopLoss):
            child.ocaGroup = f"swing-{bracket.parent.orderId}"
            child.ocaType = 2
        trades = []
        for o in bracket:
            o.tif = "DAY" if day_entry and o is bracket.parent else "GTC"
            o.orderRef = order_ref
            trades.append(ib.placeOrder(contract, o))
        for trade in trades:
            _acknowledge(ib, trade)
        return {
            "ok": True,
            "parent_id": bracket.parent.orderId,
            "target_id": bracket.takeProfit.orderId,
            "stop_id": bracket.stopLoss.orderId,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), **ids}
    finally:
        ib.disconnect()


@serialized
def ensure_exits(trade, remaining, persist_ids):
    """Repair legacy protection, preserving active IDs and rejecting unknown exposure."""
    ib = _connect()
    try:
        ib.reqOpenOrders()
        active = {t.order.orderId: t for t in ib.openTrades() if t.order.clientId == IBKR_CLIENT_ID}
        contract = _contract(trade["ticker"])
        ib.qualifyContracts(contract)
        held = sum(float(p.position) for p in ib.positions() if p.contract.conId == contract.conId)
        if remaining <= 0 or held < remaining:
            raise RuntimeError("Broker position does not confirm remaining shares; manual reconciliation required")
        group = f"swing-trade-{trade['id']}"
        stop = active.get(trade.get("ibkr_stop_id"))
        target = active.get(trade.get("ibkr_target_id"))
        # Set up the stop first, then the target. Never duplicate an unknown order.
        ids = {}
        for role, current, price in (("stop", stop, trade["stop_price"]), ("target", target, trade["target_price"])):
            if current:
                order = current.order
                if current.contract.conId != contract.conId or order.action != "SELL":
                    raise RuntimeError("Protective order identity mismatch")
                order.totalQuantity = remaining + float(current.orderStatus.filled)
            else:
                order = StopOrder("SELL", remaining, price) if role == "stop" else LimitOrder("SELL", remaining, price)
                order.orderId = ib.client.getReqId()
                order.orderRef = f"swing-{trade['id']}"
                ids[f"ibkr_{role}_id"] = order.orderId
                persist_ids(ids)
            order.tif, order.ocaGroup, order.ocaType = "GTC", group, 2
            _acknowledge(ib, ib.placeOrder(contract, order))
        return {"ok": True, **ids}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        ib.disconnect()


def _contract(ticker: str) -> Stock:
    return Stock(ticker, "SMART", "AUD", primaryExchange="ASX")


@serialized
def refresh_target_order(ticker: str, shares: int, old_order_id: int | None, new_target_price: float) -> dict[str, Any]:
    """Modify the linked target price, preserving quantity and protection."""
    ib = _connect()
    try:
        ib.reqOpenOrders()
        trade = next((t for t in ib.openTrades() if t.order.orderId == old_order_id
                      and t.order.clientId == IBKR_CLIENT_ID), None)
        if trade is None or not trade.order.ocaGroup:
            raise RuntimeError("No active linked target; reconcile protection before refreshing")
        if trade.contract.symbol != ticker or trade.order.action != "SELL":
            raise RuntimeError("Target order identity mismatch")
        # Modify price in place: preserve quantity, OCA linkage, and all fill history.
        trade.order.lmtPrice = new_target_price
        changed = ib.placeOrder(trade.contract, trade.order)
        _acknowledge(ib, changed)
        return {"ok": True, "order_id": trade.order.orderId}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        ib.disconnect()


@serialized
def cancel_order(order_id: int) -> bool:
    ib = _connect()
    try:
        ib.reqOpenOrders()
        completed = ib.reqCompletedOrders(apiOnly=False)
        for t in [*ib.trades(), *completed]:
            if t.order.orderId == order_id and t.order.clientId == IBKR_CLIENT_ID:
                if t.orderStatus.status in ("Cancelled", "ApiCancelled", "Filled"):
                    return True
                ib.cancelOrder(t.order)
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    if t.orderStatus.status in ("Cancelled", "ApiCancelled", "Filled"):
                        return True
                    ib.sleep(0.1)
                return False
        return False
    finally:
        ib.disconnect()


@serialized
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
        completed = ib.reqCompletedOrders(apiOnly=False)
        fills = ib.reqExecutions(ExecutionFilter(clientId=IBKR_CLIENT_ID))
        ib.sleep(1)
        out: dict[int, dict[str, Any]] = {}
        for t in [*completed, *ib.trades()]:
            oid = t.order.orderId
            if oid not in order_ids or t.order.clientId != IBKR_CLIENT_ID:
                continue
            fill_price = t.orderStatus.avgFillPrice or None
            out[oid] = {
                "status": t.orderStatus.status,
                "fill_price": fill_price if fill_price else None,
                "filled": float(t.orderStatus.filled),
                "remaining": float(t.orderStatus.remaining),
                "total_quantity": float(t.order.totalQuantity),
                "oca_group": t.order.ocaGroup,
            }
        for fill in fills:
            ex = fill.execution
            if ex.orderId in order_ids:
                item = out.setdefault(ex.orderId, {"status": "Unknown"})
                if ex.cumQty >= item.get("filled", 0):
                    item.update(filled=float(ex.cumQty), fill_price=float(ex.avgPrice),
                                filled_at=ex.time.isoformat())
        return out
    finally:
        ib.disconnect()
