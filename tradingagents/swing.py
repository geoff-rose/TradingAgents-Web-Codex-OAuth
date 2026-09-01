"""Orchestration for the Swing Trade page: proposal generation, semi-automatic
approval into a real (paper) IBKR order, daily target refresh, and status
sync back from IBKR into swing_db. See swing_signal.py/range_model.py for the
two signal sources and swing_ibkr.py for the IBKR mechanics -- this module
wires them to swing_db and is what web/server.py's /api/swing/* routes call.

Two strategies coexist in swing_trades (`strategy` column):

- 'heuristic' (original): proposed -(approve)-> submitted -(parent fill)->
  open -(target or stop fill)-> closed, via a single GTC bracket order. Kept
  for reference; no longer proposed live by default (backtest.py showed it
  loses to random entries).
- 'range_model' (phase-3 §6, added 2026-08-21): proposed -(approve)-> a
  single DAY entry order -(filled same day, or expires)-> 'open' (stop placed
  once, GTC) or 'expired' (day order lapsed unfilled -- a fresh proposal
  follows tomorrow, not a retry of the same row) -(daily-refreshed target or
  the fixed stop fills)-> closed. `daily_refresh()` is the once-a-day job
  that generates new proposals for flat tickers AND re-quotes the target for
  open ones; `sync_all()` is the frequent (every 15 min) job that just polls
  IBKR for fills and updates local state -- it doesn't touch the model.
"""

from __future__ import annotations

import time
from typing import Any

from tradingagents import swing_db, swing_ibkr, swing_signal


def propose_daily() -> dict[str, Any]:
    """Legacy heuristic proposal pass -- kept for reference/backtest parity,
    not called by the live timer by default any more (see module docstring).
    One pass over the enabled swing universe: skip tickers with an
    already-active (proposed/submitted/open) trade, run the heuristic on the
    rest, create a proposal row for anything that qualifies."""
    today = time.strftime("%Y-%m-%d", time.gmtime())
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

    today = time.strftime("%Y-%m-%d", time.gmtime())
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


def approve_trade(trade_id: int) -> dict[str, Any]:
    trade = swing_db.get_trade(trade_id)
    if not trade:
        return {"ok": False, "error": "trade not found"}
    if trade["status"] != "proposed":
        return {"ok": False, "error": f"trade is '{trade['status']}', not 'proposed'"}

    if trade["strategy"] == "range_model":
        result = swing_ibkr.place_day_entry(trade["ticker"], trade["shares"], trade["entry_price"])
        if not result["ok"]:
            swing_db.update_trade(trade_id, status="error", error=result["error"])
            return result
        swing_db.update_trade(trade_id, status="submitted", ibkr_parent_id=result["order_id"])
        return {"ok": True}

    result = swing_ibkr.place_bracket(
        trade["ticker"], trade["shares"], trade["entry_price"],
        trade["target_price"], trade["stop_price"],
    )
    if not result["ok"]:
        swing_db.update_trade(trade_id, status="error", error=result["error"])
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

    today = time.strftime("%Y-%m-%d", time.gmtime())
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


def sync_all() -> dict[str, Any]:
    """Pull fresh status for every submitted/open trade's IBKR order ids and
    reconcile: parent fill -> open; a bracket/target-or-stop child fill ->
    closed with pnl. For 'range_model' trades specifically: a filled entry
    also triggers placing the (fixed) GTC stop for the first time (the
    heuristic's bracket already includes its stop from approval time, but
    range_model's entry is a standalone DAY order with no children yet); an
    entry day-order that the exchange itself cancelled (unfilled by end of
    day) transitions to 'expired', not left dangling in 'submitted' forever
    -- tomorrow's daily_refresh() will propose a fresh entry, not retry this
    one. This function never touches the model -- that's daily_refresh()'s
    job, run once a day; this just polls IBKR order status, safe to run
    every few minutes."""
    trades = swing_db.list_trades("submitted") + swing_db.list_trades("open")
    if not trades:
        return {"checked": 0, "updated": 0}

    order_ids: list[int] = []
    for t in trades:
        order_ids += [i for i in (t["ibkr_parent_id"], t["ibkr_target_id"], t["ibkr_stop_id"]) if i]
    statuses = swing_ibkr.fetch_order_statuses(order_ids)

    updated = 0
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for t in trades:
        parent = statuses.get(t["ibkr_parent_id"], {})
        target = statuses.get(t["ibkr_target_id"], {})
        stop = statuses.get(t["ibkr_stop_id"], {})

        if t["status"] == "submitted" and parent.get("status") == "Filled":
            if t["strategy"] == "range_model":
                stop_result = swing_ibkr.place_stop(t["ticker"], t["shares"], t["stop_price"])
                if not stop_result["ok"]:
                    swing_db.update_trade(t["id"], status="error", error=stop_result["error"])
                    updated += 1
                    continue
                swing_db.update_trade(
                    t["id"], status="open", entered_at=now,
                    entry_fill_price=parent.get("fill_price"), ibkr_stop_id=stop_result["order_id"],
                )
            else:
                swing_db.update_trade(
                    t["id"], status="open", entered_at=now,
                    entry_fill_price=parent.get("fill_price"),
                )
            updated += 1
            continue

        if t["status"] == "submitted" and t["strategy"] == "range_model" and parent.get("status") == "Cancelled":
            swing_db.update_trade(t["id"], status="expired")
            updated += 1
            continue

        if t["status"] == "open":
            hit, reason = None, None
            if target.get("status") == "Filled":
                hit, reason = target, "target"
            elif stop.get("status") == "Filled":
                hit, reason = stop, "stop"
            if hit:
                exit_price = hit.get("fill_price")
                entry_price = t["entry_fill_price"] or t["entry_price"]
                pnl = (exit_price - entry_price) * t["shares"] if exit_price else None
                swing_db.update_trade(
                    t["id"], status="closed", exited_at=now,
                    exit_fill_price=exit_price, exit_reason=reason, pnl=pnl,
                )
                updated += 1

    return {"checked": len(trades), "updated": updated}
