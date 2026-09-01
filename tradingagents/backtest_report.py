"""The statistical honesty layer for the phase-4 `/backtest` page
(`/root/asxbrief-phase4-backtest-page.md` §6) -- run any strategy from
`strategies.py` (or the special-cased range model) and get back trade count,
confidence interval, an insufficient-sample banner, sub-period breakdown,
both fill models, buy-and-hold benchmark and gate, time-in-market adjustment,
and the randomised-entry test. This is deliberately the single place all of
that lives, so every strategy run gets the same rigor without re-deriving it.

**Central risk this whole module exists to counteract** (spec §1): "one
ticker over three years produces 20-40 trades... an equity curve from 25
trades looks exactly as convincing as one from 2,500." Every function here
is here to make that impossible to miss, not to make the strategy look good.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from tradingagents.backtest import (
    TradeResult, bootstrap_confidence_interval, fetch_daily_history,
    permutation_test, simulate_trade,
)
from tradingagents.strategies import STRATEGIES, fetch_events

INSUFFICIENT_SAMPLE_THRESHOLD = 30
SUBPERIOD_BOUNDARIES_DAYS = [(0, 182, "0-6m"), (182, 365, "6-12m"), (365, 730, "1-2y"),
                             (730, 1095, "2-3y"), (1095, float("inf"), "3y+")]


def _subperiod_label(days_before_end: float) -> str:
    for lo, hi, label in SUBPERIOD_BOUNDARIES_DAYS:
        if lo <= days_before_end < hi:
            return label
    return "3y+"


def sub_period_stats(trades: list[TradeResult], data_end: pd.Timestamp) -> dict[str, dict[str, Any]]:
    """Spec §2: report sub-periods rather than recency-weighting -- a decay
    from 8%/6%/2%/-3% blends to +1.8% under any weighting scheme, and the
    actual finding (decay) disappears. Buckets a trade by how long before the
    dataset's end its entry fell."""
    filled = [t for t in trades if t.filled and t.gross_return_pct is not None]
    buckets: dict[str, list[float]] = {label: [] for _, _, label in SUBPERIOD_BOUNDARIES_DAYS}
    for t in filled:
        days_before_end = (data_end - t.entry_fill_date).days
        buckets[_subperiod_label(days_before_end)].append(t.gross_return_pct)

    out = {}
    for label, returns in buckets.items():
        if not returns:
            out[label] = {"n_trades": 0, "avg_return_pct": None, "win_rate_pct": None}
            continue
        wins = [r for r in returns if r > 0]
        out[label] = {
            "n_trades": len(returns),
            "avg_return_pct": round(sum(returns) / len(returns), 3),
            "win_rate_pct": round(100 * len(wins) / len(returns), 1),
        }
    return out


def max_drawdown(trades: list[TradeResult]) -> dict[str, Any]:
    """Drawdown on the trade-return equity curve (marks only at trade exits,
    not continuously between them -- a known simplification given no
    intraperiod equity is tracked; noted, not hidden). Starts at 100, compounds
    each closed trade's gross return in exit-date order."""
    filled = sorted((t for t in trades if t.filled and t.gross_return_pct is not None),
                    key=lambda t: t.exit_date)
    if not filled:
        return {"max_drawdown_pct": None, "trades_under_water": 0}
    equity = 100.0
    peak = 100.0
    max_dd = 0.0
    under_water = 0
    for t in filled:
        equity *= (1 + t.gross_return_pct / 100)
        peak = max(peak, equity)
        dd = (equity - peak) / peak * 100
        max_dd = min(max_dd, dd)
        if equity < peak:
            under_water += 1
    return {"max_drawdown_pct": round(max_dd, 2), "trades_under_water": under_water, "n_trades": len(filled)}


def buy_and_hold_return(df: pd.DataFrame) -> float:
    # float() cast matters: df["Close"].iloc[...] is numpy.float64, and a
    # comparison downstream (the buy-and-hold gate) between a numpy.float64
    # and a Python float produces numpy.bool_, which FastAPI's JSON encoder
    # can't serialize -- caught live testing the API, not in isolation.
    return float((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100)


def equity_curve(trades: list[TradeResult]) -> list[dict[str, Any]]:
    """Point-per-trade-exit equity series (spec §7's chart) -- same
    compounding as max_drawdown(), but returns the full path, not just the
    min. Starts at 100 on the first trade's entry date so the chart has a
    sensible starting point before any trade has closed."""
    filled = sorted((t for t in trades if t.filled and t.gross_return_pct is not None),
                    key=lambda t: t.exit_date)
    if not filled:
        return []
    equity = 100.0
    points = [{"date": str(filled[0].entry_fill_date.date()), "equity": equity}]
    for t in filled:
        equity *= (1 + t.gross_return_pct / 100)
        points.append({"date": str(t.exit_date.date()), "equity": round(equity, 3)})
    return points


def buy_hold_curve(df: pd.DataFrame, sample_every: int = 5) -> list[dict[str, Any]]:
    """Buy-and-hold overlay series, normalised to start at 100 -- spec §7:
    "buy-and-hold overlaid on the equity curve, always." Sampled every
    `sample_every` trading days (not every single day) to keep the payload
    small over a 10-year daily series; the shape is what matters for an
    overlay, not single-day precision."""
    if df.empty:
        return []
    base = df["Close"].iloc[0]
    sampled = df.iloc[::sample_every]
    return [{"date": str(idx.date()), "equity": round(100 * row["Close"] / base, 3)}
            for idx, row in sampled.iterrows()]


def time_in_market_pct(trades: list[TradeResult], df: pd.DataFrame) -> float:
    filled = [t for t in trades if t.filled and t.exit_date is not None]
    if not filled or len(df) < 2:
        return 0.0
    days_held = sum((t.exit_date - t.entry_fill_date).days for t in filled)
    total_days = (df.index[-1] - df.index[0]).days
    return round(100 * days_held / total_days, 1) if total_days else 0.0


def run_and_report(ticker: str, strategy_key: str, params: dict[str, Any] | None = None,
                    period: str = "10y", market: str = "AU") -> dict[str, Any]:
    """The single entry point the future /backtest page's runner calls.
    Returns everything spec §6/§7 wants displayed: both fill models, CI,
    insufficient-sample banner, sub-periods, drawdown, buy-and-hold + gate,
    time-in-market adjustment, and the randomised-entry percentile."""
    if strategy_key not in STRATEGIES:
        return {"error": f"unknown strategy '{strategy_key}', have: {list(STRATEGIES)}"}

    strategy = STRATEGIES[strategy_key]
    p = {**strategy.default_params, **(params or {})}

    df = fetch_daily_history(ticker, period=period, market=market)
    if df.empty:
        return {"error": f"no price history for {ticker} ({market})"}

    events = fetch_events(ticker, market) if strategy.event_driven else None
    signals = strategy.generate_signals(ticker, df, p, events)
    trades_realistic = [simulate_trade(ticker, df, s, mode="realistic") for s in signals]
    trades_naive = [simulate_trade(ticker, df, s, mode="naive") for s in signals]

    filled = [t for t in trades_realistic if t.filled]
    n_trades = len(filled)

    result: dict[str, Any] = {
        "ticker": ticker, "market": market, "strategy": strategy_key, "params": p,
        "period_start": str(df.index[0].date()), "period_end": str(df.index[-1].date()),
        "price_adjustment": "adjusted",
        "n_trades": n_trades,
        "insufficient_sample": n_trades < INSUFFICIENT_SAMPLE_THRESHOLD,
        "insufficient_sample_note": (
            f"{n_trades} trades is too few to distinguish this from luck (threshold: "
            f"{INSUFFICIENT_SAMPLE_THRESHOLD})" if n_trades < INSUFFICIENT_SAMPLE_THRESHOLD else None
        ),
    }
    if n_trades == 0:
        result["note"] = "no trades generated -- nothing to report"
        return result

    returns_realistic = [t.gross_return_pct for t in filled]
    returns_naive = [t.gross_return_pct for t in trades_naive if t.filled]
    wins = [r for r in returns_realistic if r > 0]

    result.update({
        "win_rate_pct": round(100 * len(wins) / n_trades, 1),
        "realistic": {
            "avg_return_pct": round(sum(returns_realistic) / n_trades, 3),
            "total_return_pct": round(sum(returns_realistic), 2),
        },
        "naive_touch_fill": {
            "n_trades": len(returns_naive),
            "avg_return_pct": round(sum(returns_naive) / len(returns_naive), 3) if returns_naive else None,
            "total_return_pct": round(sum(returns_naive), 2) if returns_naive else None,
        },
        "confidence_interval": bootstrap_confidence_interval(trades_realistic),
        "max_drawdown": max_drawdown(trades_realistic),
        "sub_periods": sub_period_stats(trades_realistic, df.index[-1]),
        "buy_and_hold_return_pct": round(buy_and_hold_return(df), 2),
        "time_in_market_pct": time_in_market_pct(trades_realistic, df),
        "randomised_entry_test": permutation_test(trades_realistic, {ticker: df}),
        "equity_curve_realistic": equity_curve(trades_realistic),
        "equity_curve_naive": equity_curve(trades_naive),
        "buy_hold_curve": buy_hold_curve(df),
        "trade_list": [
            {
                "signal_date": str(t.signal_date.date()), "entry": t.entry,
                "entry_fill_date": str(t.entry_fill_date.date()) if t.entry_fill_date else None,
                "exit_date": str(t.exit_date.date()) if t.exit_date else None,
                "exit_price": t.exit_price, "exit_reason": t.exit_reason,
                "gross_return_pct": t.gross_return_pct,
            }
            for t in sorted((t for t in trades_realistic if t.filled), key=lambda t: t.entry_fill_date)
        ],
    })

    # Exposure-adjusted return: scales the achieved return up to what full-time
    # deployment at the same per-day rate would look like, so it's comparable
    # to buy-and-hold's 100%-exposure figure (spec §6.3b) -- an approximation,
    # not a real capital simulation, and documented as such.
    tim = result["time_in_market_pct"]
    result["exposure_adjusted_return_pct"] = (
        round(result["realistic"]["total_return_pct"] / (tim / 100), 2) if tim > 0 else None
    )

    # Buy-and-hold gate (spec §6.3b): a strategy must beat BOTH the
    # randomised-entry null AND buy-and-hold to be reported as an
    # improvement -- the null alone can't catch "just captured drift".
    perm_pct = result["randomised_entry_test"].get("actual_percentile_in_null")
    beats_random = bool(perm_pct is not None and perm_pct >= 95)
    beats_buy_hold = bool(result["exposure_adjusted_return_pct"] is not None and
                          result["exposure_adjusted_return_pct"] > result["buy_and_hold_return_pct"])
    result["passes_both_gates"] = bool(beats_random and beats_buy_hold)
    result["gate_detail"] = {"beats_randomised_entry_null": beats_random, "beats_buy_and_hold": beats_buy_hold}

    return result


def run_and_report_range_model(tickers: list[str], entry_q: float, target_q: float) -> dict[str, Any]:
    """Range model special case (spec §3's table lists it, but its
    daily-refresh mechanic -- a fresh entry/target quoted every day, not one
    static Signal per setup -- doesn't fit strategies.py's shared shape; see
    strategies.py's bottom-of-file note). Normalises range_model.py's own
    evaluation into a result shape close enough to run_and_report()'s for
    the future page to display uniformly, reusing this module's sub-period/
    drawdown/buy-and-hold helpers rather than duplicating them."""
    from tradingagents.range_model import (
        _to_trade_results, fit, pool_training_data, simulate_daily_refresh,
    )

    pooled = pool_training_data(tickers)
    if pooled.empty:
        return {"error": "no training data"}
    model = fit(pooled)

    all_trades: list[TradeResult] = []
    histories: dict[str, pd.DataFrame] = {}
    buy_hold_by_ticker: dict[str, float] = {}
    for ticker in tickers:
        df = fetch_daily_history(ticker)
        if df.empty:
            continue
        histories[ticker] = df
        trades, _ = simulate_daily_refresh(ticker, df, model, entry_q, target_q, mode="realistic")
        all_trades += _to_trade_results(trades)
        buy_hold_by_ticker[ticker] = buy_and_hold_return(df[df.index >= model.train_cutoff])

    filled = [t for t in all_trades if t.filled]
    if not filled:
        return {"n_trades": 0, "note": "no fills at this (entry_q, target_q)"}

    returns = [t.gross_return_pct for t in filled]
    avg_buy_hold = sum(buy_hold_by_ticker.values()) / len(buy_hold_by_ticker) if buy_hold_by_ticker else None

    return {
        "strategy": "range_model", "tickers": tickers, "entry_q": entry_q, "target_q": target_q,
        "held_out_cutoff": str(model.train_cutoff.date()),
        "n_trades": len(filled),
        "insufficient_sample": len(filled) < INSUFFICIENT_SAMPLE_THRESHOLD,
        "win_rate_pct": round(100 * len([r for r in returns if r > 0]) / len(returns), 1),
        "realistic": {"avg_return_pct": round(sum(returns) / len(returns), 3), "total_return_pct": round(sum(returns), 2)},
        "confidence_interval": bootstrap_confidence_interval(all_trades),
        "max_drawdown": max_drawdown(all_trades),
        "sub_periods": sub_period_stats(all_trades, max(df.index[-1] for df in histories.values())),
        "buy_and_hold_return_pct_avg_across_tickers": round(avg_buy_hold, 2) if avg_buy_hold is not None else None,
        "randomised_entry_test": permutation_test(all_trades, histories),
        "note": "run selection_permutation_test() separately if this result came from a grid search "
                "over multiple (entry_q, target_q) -- see backtest.py; not repeated here per-call "
                "since it needs the whole grid's trades, not just one cell's.",
    }


def run_and_report_overnight(tickers: list[str], condition: str = "after_big_down_day",
                             period: str = "3y") -> dict[str, Any]:
    """Overnight-hold special case (added 2026-08-22 at the user's request).

    Like the range model, this doesn't fit `strategies.py`'s one-Signal-with-
    target-and-stop shape: there is no target and no stop, the entry is the
    closing auction and the exit is the next opening auction, and the outcome
    is whatever the gap turns out to be. Normalised here into the same result
    shape the page already renders.

    Two things this reports that `run_and_report()` does not, because they are
    the questions that decide an overnight strategy:
      - a **spread sweep**, since the per-night edge (~0.13-0.32%) is the same
        order of magnitude as the round-trip cost, so a single cost assumption
        would decide the answer by itself
      - a **condition null** that holds each ticker's night COUNT fixed and
        randomises WHICH nights, so the unconditional overnight anomaly and
        the ticker mix are both inside the null and cannot be mistaken for
        conditional skill
    """
    from tradingagents.overnight import (
        CONDITIONS, SPREAD_SWEEP_PCT, _day_clustered_ci, condition_null_test,
        simulate_overnight,
    )

    if condition not in CONDITIONS:
        return {"error": f"unknown condition '{condition}', have: {list(CONDITIONS)}"}
    cond = CONDITIONS[condition]

    per_ticker, frames, buy_hold = {}, [], []
    for ticker in tickers:
        df = fetch_daily_history(ticker, period=period)
        if len(df) < 100:
            continue
        per_ticker[ticker] = df
        sub = simulate_overnight(df, cond(df))
        sub["ticker"] = ticker
        frames.append(sub)
        buy_hold.append(buy_and_hold_return(df))

    if not frames:
        return {"error": "no price data"}
    frame = pd.concat(frames)
    if frame.empty:
        return {"n_trades": 0, "note": f"no nights matched condition '{condition}'"}

    returns = frame["gross_return_pct"]
    gross = float(returns.mean())
    n = len(frame)

    result: dict[str, Any] = {
        "strategy": "overnight_hold", "condition": condition,
        "tickers": list(per_ticker), "period": period,
        "n_trades": n,
        "insufficient_sample": n < INSUFFICIENT_SAMPLE_THRESHOLD,
        "win_rate_pct": round(100 * float((returns > 0).mean()), 1),
        "realistic": {"avg_return_pct": round(gross, 4),
                      "total_return_pct": round(float(returns.sum()), 2)},
        "day_clustered_ci": _day_clustered_ci(frame),
        "buy_and_hold_return_pct_avg_across_tickers": round(float(sum(buy_hold) / len(buy_hold)), 2),
        "spread_sweep": {
            f"half_spread_{sp}pct": round(gross - (2 * 0.0005 * 100 + 2 * sp), 4)
            for sp in SPREAD_SWEEP_PCT
        },
        "condition_null": condition_null_test(per_ticker, cond) if condition != "all_nights" else None,
        "note": ("Per-night edge is the same order as the round-trip cost -- read the spread sweep, "
                 "not the gross figure. The day-clustered CI is the correct one: tickers on the same "
                 "night share that night's market move and are not independent observations."),
    }
    return result
