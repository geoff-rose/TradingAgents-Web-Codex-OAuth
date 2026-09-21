"""Swing proposals, per-trade approval, daily target refresh and reconciliation.

Both strategies use paper brackets; range-model entries are DAY orders and
exits are GTC. The web service reconciles active orders every five seconds
between broker calls. Partial or uncertain execution remains active until
confirmed, preventing a new proposal from duplicating unresolved exposure.
"""

from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

from tradingagents import swing_db, swing_ibkr, swing_signal


def propose_daily() -> dict[str, Any]:
    """Legacy heuristic proposal pass -- kept for reference/backtest parity,
    not called by the live timer by default any more (see module docstring).
    One pass over the enabled swing universe: skip tickers with an
    already-active (proposed/submitted/open) trade, run the heuristic on the
    rest, create a proposal row for anything that qualifies."""
    today = datetime.now(ZoneInfo("Australia/Sydney")).date().isoformat()
    trade_dollars = swing_db.get_trade_dollars()
    created, skipped, no_setup = [], [], []
    for ticker in swing_db.enabled_tickers():
        if swing_db.has_active_trade(ticker):
            skipped.append(ticker)
            continue
        setup = swing_signal.evaluate(ticker)
        if not setup:
            no_setup.append(ticker)
            continue
        shares = int(trade_dollars // setup["entry_price"])
        if shares < 1:
            no_setup.append(ticker)
            continue
        trade_id = swing_db.create_proposal(
            ticker=ticker, trade_date=today,
            entry_price=setup["entry_price"], target_price=setup["target_price"],
            stop_price=setup["stop_price"], shares=shares,
            dollar_size=round(shares * setup["entry_price"], 2),
            rationale=setup["rationale"], strategy="heuristic",
        )
        created.append({"id": trade_id, "ticker": ticker})
    return {"created": created, "skipped_active": skipped, "no_setup": no_setup}


def propose_daily_range_model() -> dict[str, Any]:
    """Live daily proposal pass using the fitted range model (phase-3 §6).
    For each enabled, flat ticker: fit the pooled model on full history (a
    few seconds, cheap enough to refit fresh once a day rather than cache a
    stale one), predict today's entry via swing_db's configured
    (entry_q, target_q), and create a proposal. Doesn't touch tickers that
    already have an active trade."""
    from tradingagents.backtest import STOP_ATR_MULT, fetch_daily_history, round_to_tick
    from tradingagents.range_model import build_features, fit, pool_training_data, predict

    today = datetime.now(ZoneInfo("Australia/Sydney")).date().isoformat()
    trade_dollars = swing_db.get_trade_dollars()
    entry_q, target_q = swing_db.get_range_model_quantiles()

    tickers = [t for t in swing_db.enabled_tickers() if not swing_db.has_active_trade(t)]
    if not tickers:
        return {"created": [], "skipped_active": swing_db.enabled_tickers(), "no_data": []}

    pooled = pool_training_data(swing_db.enabled_tickers())
    if pooled.empty:
        return {"created": [], "error": "no training data available"}
    model = fit(pooled, held_out_months=0)  # live use: train on ALL history, no held-out reservation

    created, no_data = [], []
    for ticker in tickers:
        df = fetch_daily_history(ticker)
        feat = build_features(ticker, df)
        if feat.empty:
            no_data.append(ticker)
            continue
        last_row = feat.iloc[-1]
        entry, target = predict(model, last_row, entry_q, target_q)
        shares = int(trade_dollars // entry)
        if shares < 1:
            no_data.append(ticker)
            continue
        stop_price = round_to_tick(entry - STOP_ATR_MULT * last_row["atr_short"])
        trade_id = swing_db.create_proposal(
            ticker=ticker, trade_date=today, entry_price=entry, target_price=target,
            stop_price=stop_price,
            shares=shares, dollar_size=round(shares * entry, 2),
            rationale=(f"range model: entry_q={entry_q} target_q={target_q}, "
                       f"trained on {model.n_train_rows} rows"),
            strategy="range_model", atr_at_entry=float(last_row["atr_short"]),
        )
        created.append({"id": trade_id, "ticker": ticker, "entry": entry, "target": target})

    return {"created": created, "no_data": no_data, "entry_q": entry_q, "target_q": target_q}


@swing_ibkr.serialized
def approve_trade(trade_id: int) -> dict[str, Any]:
    trade = swing_db.get_trade(trade_id)
    if not trade:
        return {"ok": False, "error": "trade not found"}
    if trade["status"] != "proposed":
        return {"ok": False, "error": f"trade is '{trade['status']}', not 'proposed'"}

    if not (0 < trade["stop_price"] < trade["entry_price"] < trade["target_price"] and trade["shares"] > 0):
        return {"ok": False, "error": "Invalid bracket prices or quantity"}
    if not swing_db.claim_proposal(trade_id):
        return {"ok": False, "error": "Proposal already claimed"}
    def persist(ids):
        swing_db.update_trade(trade_id, **{f"ibkr_{k}": v for k, v in ids.items()})
    try:
        result = swing_ibkr.place_bracket(
            trade["ticker"], trade["shares"], trade["entry_price"],
            trade["target_price"], trade["stop_price"], day_entry=trade["strategy"] == "range_model",
            order_ref=f"swing-{trade_id}", persist_ids=persist,
        )
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    if not result["ok"]:
        swing_db.update_trade(trade_id, status="attention", error=result["error"])
        return result
    swing_db.update_trade(
        trade_id, status="submitted",
        ibkr_parent_id=result["parent_id"], ibkr_target_id=result["target_id"],
        ibkr_stop_id=result["stop_id"],
    )
    return {"ok": True}


def reject_trade(trade_id: int) -> dict[str, Any]:
    trade = swing_db.get_trade(trade_id)
    if not trade:
        return {"ok": False, "error": "trade not found"}
    if trade["status"] != "proposed":
        return {"ok": False, "error": f"trade is '{trade['status']}', not 'proposed'"}
    swing_db.update_trade(trade_id, status="rejected")
    return {"ok": True}


@swing_ibkr.serialized
def refresh_open_targets() -> dict[str, Any]:
    """Daily-refresh half of the range-model mechanic: for every 'open'
    range_model trade whose target order wasn't already quoted today, cancel
    it and place a fresh one at today's freshly-modelled level. The fixed
    GTC stop (placed once, in sync_all(), when the entry filled) is left
    completely alone -- only the take-profit side moves. Call once a day,
    same timer as propose_daily_range_model(); cheap to call more often too
    since it's a no-op for any trade already refreshed today."""
    from tradingagents.backtest import fetch_daily_history
    from tradingagents.range_model import build_features, fit, pool_training_data, predict_pct

    today = datetime.now(ZoneInfo("Australia/Sydney")).date().isoformat()
    open_trades = [t for t in swing_db.list_trades("open") if t["strategy"] == "range_model"]
    open_trades = [t for t in open_trades if t["target_order_date"] != today]
    if not open_trades:
        return {"refreshed": [], "note": "nothing due for refresh"}

    _, target_q = swing_db.get_range_model_quantiles()
    pooled = pool_training_data(swing_db.enabled_tickers())
    model = fit(pooled, held_out_months=0)

    refreshed, errors = [], []
    for t in open_trades:
        df = fetch_daily_history(t["ticker"])
        feat = build_features(t["ticker"], df)
        if feat.empty:
            errors.append({"ticker": t["ticker"], "error": "no feature data"})
            continue
        last_row = feat.iloc[-1]
        high_pct = predict_pct(model, last_row, target_q, "high")
        entry_basis = t["entry_fill_price"] or t["entry_price"]
        from tradingagents.backtest import round_to_tick, tick_size
        tick = tick_size(entry_basis)
        new_target = round_to_tick(entry_basis * (1 + high_pct / 100))
        new_target = max(new_target, round_to_tick(entry_basis + tick))

        result = swing_ibkr.refresh_target_order(t["ticker"], t["shares"], t["ibkr_target_id"], new_target)
        if not result["ok"]:
            errors.append({"ticker": t["ticker"], "error": result["error"]})
            continue
        swing_db.update_trade(
            t["id"], target_price=new_target, ibkr_target_id=result["order_id"],
            target_order_date=today,
        )
        refreshed.append({"ticker": t["ticker"], "new_target": new_target})

    return {"refreshed": refreshed, "errors": errors}


@swing_ibkr.serialized
def sync_all() -> dict[str, Any]:
    """Reconcile cumulative fills; uncertain broker state remains active and visible."""
    trades = [t for t in swing_db.list_trades() if t["status"] in swing_db.ACTIVE_STATUSES
              and t["status"] != "proposed"]
    ids = [t[k] for t in trades for k in ("ibkr_parent_id", "ibkr_target_id", "ibkr_stop_id") if t[k]]
    statuses = swing_ibkr.fetch_order_statuses(ids)
    terminal = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
    updated, errors = 0, []
    for t in trades:
        try:
            parent = statuses.get(t["ibkr_parent_id"], {})
            target = statuses.get(t["ibkr_target_id"], {})
            stop = statuses.get(t["ibkr_stop_id"], {})
            ledger = swing_db.record_order_fills(t, statuses)
            filled = max(t.get("filled_shares") or 0, ledger.get("parent", (0, None))[0])
            if parent.get("status") == "Filled" and not filled:
                raise RuntimeError("Filled entry has no confirmed quantity")
            tf = ledger.get("target", (t.get("target_filled") or 0, None))[0]
            sf = ledger.get("stop", (t.get("stop_filled") or 0, None))[0]
            tp = ledger.get("target", (0, t.get("target_fill_price")))[1]
            sp = ledger.get("stop", (0, t.get("stop_fill_price")))[1]
            entry = parent.get("fill_price") or t.get("entry_fill_price")
            swing_db.update_trade(t["id"], filled_shares=filled, target_filled=tf, stop_filled=sf,
                                  target_fill_price=tp, stop_fill_price=sp, entry_fill_price=entry)
            if not filled:
                if parent.get("status") in terminal:
                    for key in ("ibkr_target_id", "ibkr_stop_id"):
                        if t[key] and not swing_ibkr.cancel_order(t[key]):
                            raise RuntimeError("Entry ended, but child cancellation is unconfirmed")
                    swing_db.update_trade(t["id"], status="expired", error=None)
                elif not parent:
                    raise RuntimeError("Entry order absent from broker response; not assuming cancellation")
                continue
            if not entry:
                raise RuntimeError("Entry fill price missing")
            remaining = filled - tf - sf
            if remaining < 0:
                raise RuntimeError("Exits exceed confirmed entry fills; reconcile broker position")
            if remaining == 0:
                # Confirm every residual order is inactive before closing locally.
                for key in ("ibkr_parent_id", "ibkr_target_id", "ibkr_stop_id"):
                    if t[key] and not swing_ibkr.cancel_order(t[key]):
                        raise RuntimeError("Position exited but residual order cancellation is unconfirmed")
                fresh = swing_ibkr.fetch_order_statuses([t[k] for k in ("ibkr_parent_id", "ibkr_target_id", "ibkr_stop_id") if t[k]])
                latest = swing_db.record_order_fills(t, fresh)
                if (latest.get("parent", (0, None))[0] != filled
                        or latest.get("target", (0, None))[0] != tf
                        or latest.get("stop", (0, None))[0] != sf):
                    raise RuntimeError("Fills changed during cancellation; reconciling again")
                if (tf and not tp) or (sf and not sp):
                    raise RuntimeError("Exit price missing")
                proceeds = tf * (tp or 0) + sf * (sp or 0)
                swing_db.update_trade(t["id"], status="closed", error=None,
                    exited_at=target.get("filled_at") or stop.get("filled_at") or swing_db._utcnow(),
                    exit_fill_price=proceeds / filled,
                    exit_reason="mixed" if tf and sf else ("target" if tf else "stop"),
                    pnl=proceeds - filled * entry)
                updated += 1
                continue
            if parent.get("status") not in terminal:
                # Stop the unfilled remainder before resizing protection. Re-read
                # fills on the next pass so fills racing the cancellation are included.
                if not swing_ibkr.cancel_order(t["ibkr_parent_id"]):
                    raise RuntimeError("Partial entry: cancellation not acknowledged")
                swing_db.update_trade(t["id"], status="partial", error="Reconciling partial entry and protection")
                updated += 1
                continue
            if not parent:
                raise RuntimeError("Cannot verify that entry has stopped filling")
            linked = (target.get("oca_group") and target.get("oca_group") == stop.get("oca_group")
                      and target.get("status") in ("Submitted", "PreSubmitted")
                      and stop.get("status") in ("Submitted", "PreSubmitted")
                      and target.get("remaining") == remaining and stop.get("remaining") == remaining)
            if not linked:
                # Missing IDs are not evidence an order is gone. Only replace known terminal orders.
                for key in ("ibkr_target_id", "ibkr_stop_id"):
                    if t[key] and t[key] not in statuses:
                        raise RuntimeError("Protective order status missing; refusing duplicate protection")
                result = swing_ibkr.ensure_exits(t, remaining, lambda fields: swing_db.update_trade(t["id"], **fields))
                if not result["ok"]:
                    raise RuntimeError(result["error"])
            swing_db.update_trade(t["id"], status="open", error=None,
                                  entered_at=t.get("entered_at") or parent.get("filled_at") or swing_db._utcnow())
            updated += 1
        except Exception as exc:
            swing_db.update_trade(t["id"], status="attention", error=str(exc))
            errors.append({"id": t["id"], "error": str(exc)})
    return {"checked": len(trades), "updated": updated, "errors": errors}
