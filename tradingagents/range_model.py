"""Fitted range model for Swing Trade (phase-3 spec §6): predicts a
distribution of *tomorrow's* low and high (as a % move from today's close),
pooled across the swing universe, so entry/target can be placed at a chosen
percentile of that distribution and refreshed daily -- instead of the
existing heuristic's one-shot ATR snapshot held static until it fills or
resolves (shown by `backtest.py` to lose to random entries).

**Gate, per the spec**: this model only ships if it beats the existing
heuristic out-of-sample. `evaluate_daily_refresh()` runs that comparison on
the same held-out period `backtest.py` uses. Don't wire this into the live
`swing.py`/`swing_ibkr.py` order mechanic until that comparison is favourable
-- rewiring live (even paper) order placement around an unvalidated model is
exactly the mistake the whole backtest exists to prevent.

Run as a script for a report:
    .venv/bin/python -m tradingagents.range_model
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg

from tradingagents.backtest import (
    HELD_OUT_MONTHS, MAX_HOLD_DAYS, STOP_ATR_MULT, _resolve_same_bar_ambiguity,
    fetch_daily_history, round_to_tick, tick_size,
)

ATR_SHORT, ATR_LONG = 14, 50
VOL_WINDOW_SHORT = (5, 10, 20)
GAP_MIN_HISTORY = 60  # days of expanding history before trusting a ticker's own gap-propensity stat

FEATURE_COLUMNS = [
    "ret_std_5", "ret_std_10", "ret_std_20",
    "atr_ratio_short_long",
    "range_vs_avg20", "close_pos_in_range",
    "volume_ratio_20", "gap_propensity",
]

# Full pooling (shared coefficients across all tickers, no per-ticker terms)
# is the deliberate starting point, not a shortcut -- spec §2.6: "per-stock
# coefficients only once a stock has earned them." With ~5 tickers and a few
# thousand pooled rows, full pooling is the correctly-shrunk end of the
# spectrum; a per-ticker refinement is a legitimate future step once there's
# evidence a specific stock's dynamics genuinely diverge from the pooled fit,
# not something to add pre-emptively.
# Finer grid than the first sweep used, so a user-set (swing_db
# set_range_model_quantiles) entry_q/target_q away from the handful tested
# in that sweep still hits a genuinely-fitted quantile rather than an error.
LOW_QUANTILES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
HIGH_QUANTILES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def build_features(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Causal features at the close of day i (predicting day i+1's low/high)
    plus the (i+1) targets for training rows where day i+1 exists. Every
    feature uses only data available up to and including day i's close --
    no lookahead."""
    if len(df) < ATR_LONG + 10:
        return pd.DataFrame()

    close, high, low, open_, vol = (
        df["Close"], df["High"], df["Low"], df["Open"], df["Volume"],
    )
    ret = close.pct_change()

    tr = pd.concat([
        high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_short = tr.rolling(ATR_SHORT).mean()
    atr_long = tr.rolling(ATR_LONG).mean()

    range_today = high - low
    avg_range_20 = range_today.rolling(20).mean()
    close_pos = ((close - low) / (high - low)).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    avg_vol_20 = vol.rolling(20).mean()

    gap_pct = (open_ - close.shift(1)).abs() / close.shift(1)
    gap_propensity = gap_pct.expanding(min_periods=GAP_MIN_HISTORY).mean()

    feat = pd.DataFrame({
        "ret_std_5": ret.rolling(5).std() * 100,
        "ret_std_10": ret.rolling(10).std() * 100,
        "ret_std_20": ret.rolling(20).std() * 100,
        "atr_ratio_short_long": atr_short / atr_long,
        "range_vs_avg20": range_today / avg_range_20,
        "close_pos_in_range": close_pos,
        "volume_ratio_20": vol / avg_vol_20,
        "gap_propensity": gap_propensity * 100,
    }, index=df.index)

    next_low_pct = (low.shift(-1) - close) / close * 100
    next_high_pct = (high.shift(-1) - close) / close * 100
    feat["next_low_pct"] = next_low_pct
    feat["next_high_pct"] = next_high_pct
    feat["ticker"] = ticker
    feat["close"] = close
    feat["atr_short"] = atr_short

    return feat.dropna(subset=FEATURE_COLUMNS + ["next_low_pct", "next_high_pct"])


def pool_training_data(tickers: list[str], period: str = "10y") -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        df = fetch_daily_history(ticker, period=period)
        if df.empty:
            continue
        feat = build_features(ticker, df)
        if not feat.empty:
            frames.append(feat)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


@dataclass
class FittedRangeModel:
    low_models: dict[float, Any]
    high_models: dict[float, Any]
    train_cutoff: pd.Timestamp
    n_train_rows: int
    coefficients: pd.DataFrame  # for reporting per-predictor contribution


def fit(pooled: pd.DataFrame, held_out_months: int = HELD_OUT_MONTHS) -> FittedRangeModel:
    """Fits pooled quantile regressions on the in-sample portion only (same
    held-out-N-months split as backtest.py, so evaluate_daily_refresh() below
    tests on genuinely unseen data)."""
    cutoff = pooled.index.max() - pd.DateOffset(months=held_out_months)
    train = pooled[pooled.index < cutoff]

    X = train[FEATURE_COLUMNS].copy()
    X.insert(0, "const", 1.0)

    low_models, high_models = {}, {}
    coef_rows = []
    for q in LOW_QUANTILES:
        m = QuantReg(train["next_low_pct"], X).fit(q=q, max_iter=2000)
        low_models[q] = m
        coef_rows.append({"target": "low", "quantile": q, **m.params.to_dict()})
    for q in HIGH_QUANTILES:
        m = QuantReg(train["next_high_pct"], X).fit(q=q, max_iter=2000)
        high_models[q] = m
        coef_rows.append({"target": "high", "quantile": q, **m.params.to_dict()})

    return FittedRangeModel(
        low_models=low_models, high_models=high_models, train_cutoff=cutoff,
        n_train_rows=len(train), coefficients=pd.DataFrame(coef_rows),
    )


def predict_pct(model: FittedRangeModel, features_row: pd.Series, quantile: float, kind: str) -> float:
    """Raw predicted % move (not yet converted to a price) -- the model's
    native output is "next-day low/high as a % of THIS row's close". Callers
    decide what to anchor that % to (see predict() vs the exit-target logic
    in simulate_daily_refresh() -- those are different reference frames and
    conflating them was a real bug caught while building this, see below)."""
    x = np.concatenate([[1.0], features_row[FEATURE_COLUMNS].values.astype(float)])
    models = model.low_models if kind == "low" else model.high_models
    if quantile not in models:
        quantile = min(models, key=lambda q: abs(q - quantile))
    return float(models[quantile].params.values @ x)


def predict(model: FittedRangeModel, features_row: pd.Series, entry_q: float, target_q: float) -> tuple[float, float]:
    """Returns (entry_price, target_price) for a FRESH, flat-position day
    order: both anchored to THIS row's own close, which is correct here --
    a new order's price should be relative to where the market is today."""
    low_pct = predict_pct(model, features_row, entry_q, "low")
    high_pct = predict_pct(model, features_row, target_q, "high")
    close = features_row["close"]
    entry = round_to_tick(close * (1 + low_pct / 100))
    target = round_to_tick(close * (1 + high_pct / 100))
    return entry, target


def check_calibration(model: FittedRangeModel, pooled: pd.DataFrame) -> dict[str, Any]:
    """Out-of-sample coverage check: for a well-calibrated quantile model, the
    q-th quantile prediction should be exceeded by the actual outcome roughly
    (1-q) of the time (for lows, "exceeded" means actual < predicted, since
    lows are typically negative -- checked directionally below)."""
    held_out = pooled[pooled.index >= model.train_cutoff]
    if held_out.empty:
        return {"error": "no held-out rows"}
    X = held_out[FEATURE_COLUMNS].copy()
    X.insert(0, "const", 1.0)
    out = {}
    for q, m in model.low_models.items():
        pred = X.values @ m.params.values
        actual_below = (held_out["next_low_pct"].values < pred).mean()
        out[f"low_q{q}"] = {"expected_below_frac": round(q, 2), "actual_below_frac": round(float(actual_below), 3)}
    for q, m in model.high_models.items():
        pred = X.values @ m.params.values
        actual_above = (held_out["next_high_pct"].values > pred).mean()
        out[f"high_q{q}"] = {"expected_above_frac": round(1 - q, 2), "actual_above_frac": round(float(actual_above), 3)}
    return out


def simulate_daily_refresh(
    ticker: str, df: pd.DataFrame, model: FittedRangeModel,
    entry_q: float, target_q: float, mode: str = "realistic",
    direction: str = "mean_reversion",
    same_bar_exit: str = "conservative",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Walk-forward simulation of the daily-refresh mechanic on the held-out
    period only: while flat, a fresh entry is quoted each day (day-order,
    discarded if unfilled by that day's close) from that day's model
    prediction. Once filled, the STOP is fixed at entry time (ATR-based, not
    refreshed -- risk control shouldn't be a moving target); the TARGET is
    refreshed daily the same way the entry was, until stop or a refreshed
    target fills, or MAX_HOLD_DAYS elapses.

    `mode` matches backtest.py's convention: 'naive' fills on mere touch
    (the number to distrust), 'realistic' requires trading through the level
    by one tick. Default is 'realistic' -- unlike the first version of this
    function, which used naive touch-fill unconditionally and produced a
    suspiciously good result that didn't survive this fix (see range_model
    section of the asx-dashboard skill).

    `direction` (added 2026-08-22, per the user's request for a mirror-image
    test before fully trusting the mean-reversion result): 'mean_reversion'
    (default) is the original mechanic -- buy a dip (LOW quantile model,
    entry_q small = deep/rare pullback), fill on price trading DOWN through
    the entry. 'momentum' buys strength instead: entry uses the HIGH quantile
    model at `1 - entry_q` (kept as `1-entry_q` so entry_q's "how rare/extreme"
    meaning is preserved -- small entry_q still means a rare, extreme trigger,
    just on the upside), fills on price trading UP through the entry (a
    breakout/strength trigger, not a dip). Target/stop mechanics are
    otherwise identical (target still a further HIGH-quantile extension
    anchored to entry, stop still ATR-based below entry) -- only how the
    entry itself triggers differs.

    Returns (trades, diagnostics) -- diagnostics tracks quote_attempts (every
    flat-day a fresh entry was quoted, filled or not) and fills, so the fill
    rate can be checked directly (a low fill rate at a deep entry_q is
    expected and desired -- it means the harsher realistic fill model is
    actually biting, not passing everything through)."""
    feat = build_features(ticker, df)
    feat = feat[feat.index >= model.train_cutoff]
    if feat.empty:
        return [], {"quote_attempts": 0, "fills": 0}

    trades = []
    n_quote_attempts = 0
    n_fills = 0
    i = 0
    rows = feat.index.tolist()
    while i < len(rows) - 1:
        day = rows[i]
        row = feat.loc[day]
        if direction == "momentum":
            high_pct = predict_pct(model, row, 1 - entry_q, "high")
            entry = round_to_tick(row["close"] * (1 + high_pct / 100))
        else:
            entry, _ = predict(model, row, entry_q, target_q)
        next_day = rows[i + 1]
        next_bar = df.loc[next_day]
        tick = tick_size(entry)
        n_quote_attempts += 1

        if direction == "momentum":
            entry_threshold = entry if mode == "naive" else entry + tick
            if next_bar["High"] < entry_threshold:
                i += 1  # unfilled today's quote, refresh tomorrow
                continue
        else:
            entry_threshold = entry if mode == "naive" else entry - tick
            if next_bar["Low"] > entry_threshold:
                i += 1  # unfilled today's quote, refresh tomorrow
                continue
        n_fills += 1

        # filled -- set fixed stop from this entry's ATR, then daily-refresh the target
        stop = round_to_tick(entry - STOP_ATR_MULT * row["atr_short"])
        entry_fill_date = next_day
        j = i + 1
        exit_price = exit_date = exit_reason = None
        for _ in range(MAX_HOLD_DAYS):
            if j >= len(rows) - 1:
                break
            day_j = rows[j]
            bar_j = df.loc[day_j]
            # Anchored to the fixed entry price, not day_j's own close: the
            # model's native prediction is "next-day high as a % of THIS
            # row's close", which is the right frame for a fresh entry quote
            # but the wrong one for an open position's profit target -- if
            # price has drifted down since entry, "a modest rally from
            # today's close" can still be a loss versus the actual cost
            # basis. Refreshing daily still updates the % assumption itself
            # (today's volatility/range features), it just applies that %
            # to the entry, not to wherever price has wandered since.
            high_pct = predict_pct(model, feat.loc[day_j], target_q, "high")
            target = round_to_tick(entry * (1 + high_pct / 100))
            target = max(target, round_to_tick(entry + tick))
            target_threshold = target if mode == "naive" else target + tick
            stop_threshold = stop if mode == "naive" else stop - tick
            # See backtest.simulate_trade's `same_bar_exit` note: a daily bar
            # carries no path information, so crediting the target on the bar
            # that filled the entry assumes the low preceded the high. Under
            # 'conservative' (default) only the stop may resolve on that bar.
            # This was measured to be the dominant effect on the expanded
            # universe, not a marginal one -- 82% of trades were same-bar.
            on_entry_bar = (day_j == entry_fill_date)
            hit_target = (bar_j["High"] >= target_threshold
                          and not (on_entry_bar and same_bar_exit == "conservative"))
            hit_stop = bar_j["Low"] <= stop_threshold
            ambiguous_same_bar = False
            resolved_by = None
            if hit_target and hit_stop:
                ambiguous_same_bar = True
                winner, resolved_by = _resolve_same_bar_ambiguity(ticker, day_j, target, stop)
                exit_price = target if winner == "target" else stop
                exit_reason = winner
                exit_date = day_j
                break
            if hit_target:
                exit_price, exit_reason, exit_date = target, "target", day_j
                break
            if hit_stop:
                exit_price, exit_reason, exit_date = stop, "stop", day_j
                break
            j += 1
        else:
            pass
        if exit_price is None:
            last_j = min(j, len(rows) - 1)
            exit_date = rows[last_j]
            exit_price = df.loc[exit_date, "Close"]
            exit_reason = "timeout"
            ambiguous_same_bar, resolved_by = False, None

        ret_pct = (exit_price - entry) / entry * 100
        trades.append({
            "ticker": ticker, "signal_date": day, "entry": entry, "stop": stop,
            "entry_fill_date": entry_fill_date, "exit_date": exit_date,
            "exit_price": exit_price, "exit_reason": exit_reason, "gross_return_pct": ret_pct,
            "ambiguous_same_bar": ambiguous_same_bar, "ambiguous_resolved_by": resolved_by,
        })
        i = max(j, i + 1) + 1

    return trades, {"quote_attempts": n_quote_attempts, "fills": n_fills}


def _to_trade_results(trades: list[dict[str, Any]]) -> list["TradeResult"]:
    """Adapts range_model's plain trade dicts to backtest.py's TradeResult
    shape so the SAME permutation_test() machinery built for the heuristic
    can be reused here -- no second implementation to keep in sync."""
    from tradingagents.backtest import TradeResult

    out = []
    for t in trades:
        holding_days = (t["exit_date"] - t["entry_fill_date"]).days
        out.append(TradeResult(
            ticker=t["ticker"], signal_date=t["signal_date"], entry=t["entry"],
            target=t["entry"], stop=t["stop"],  # 'target' unused by permutation_test, entry as harmless filler
            filled=True, entry_fill_date=t["entry_fill_date"], exit_date=t["exit_date"],
            exit_price=t["exit_price"], exit_reason=t["exit_reason"],
            holding_days=holding_days, gross_return_pct=t["gross_return_pct"],
        ))
    return out


def evaluate_daily_refresh(tickers: list[str], entry_q: float = 0.4, target_q: float = 0.6,
                           mode: str = "realistic") -> dict[str, Any]:
    from tradingagents.backtest import DEFAULT_BROKERAGE_PCT, DEFAULT_SPREAD_PCT_SWEEP, permutation_test

    pooled = pool_training_data(tickers)
    if pooled.empty:
        return {"error": "no training data"}
    model = fit(pooled)
    calibration = check_calibration(model, pooled)

    all_trades = []
    histories = {}
    fill_diagnostics = {}
    for ticker in tickers:
        df = fetch_daily_history(ticker)
        if df.empty:
            continue
        histories[ticker] = df
        ticker_trades, diag = simulate_daily_refresh(ticker, df, model, entry_q, target_q, mode=mode)
        all_trades += ticker_trades
        fill_diagnostics[ticker] = {
            **diag,
            "fill_rate_pct": round(100 * diag["fills"] / diag["quote_attempts"], 1) if diag["quote_attempts"] else None,
        }

    if not all_trades:
        return {"model_n_train_rows": model.n_train_rows, "calibration": calibration,
                "fill_diagnostics": fill_diagnostics,
                "held_out_trades": 0, "note": "no fills in held-out period at this (entry_q, target_q)"}

    n_ambiguous = sum(1 for t in all_trades if t["ambiguous_same_bar"])
    n_ambiguous_conservative = sum(1 for t in all_trades if t["ambiguous_resolved_by"] == "conservative_default")

    returns = [t["gross_return_pct"] for t in all_trades]
    wins = [r for r in returns if r > 0]

    # Cost sweep, same assumptions as backtest.py's baseline evaluation, so
    # the two are comparable -- at this trade frequency (a fresh quote every
    # day, not a rare setup), costs compound fast; a mid-sweep spread
    # assumption is used as the headline net number.
    mid_spread = DEFAULT_SPREAD_PCT_SWEEP[2]
    cost_pct = (2 * DEFAULT_BROKERAGE_PCT + 2 * mid_spread) * 100
    net_returns = [r - cost_pct for r in returns]

    # Same randomised-entry null the heuristic was judged against (spec
    # §3.3) -- "beats zero" is the wrong null for a timing strategy, and a
    # positive gross/net return here says nothing about timing value on its
    # own until it's actually compared against this null. Reusing
    # backtest.py's permutation_test(), not a second implementation.
    perm = permutation_test(_to_trade_results(all_trades), histories)

    return {
        "model_n_train_rows": model.n_train_rows,
        "held_out_cutoff": str(model.train_cutoff.date()),
        "calibration": calibration,
        "entry_q": entry_q, "target_q": target_q, "fill_mode": mode,
        "held_out_trades": len(all_trades),
        "win_rate_pct": round(100 * len(wins) / len(returns), 1),
        "avg_gross_return_pct": round(sum(returns) / len(returns), 3),
        "total_gross_return_pct": round(sum(returns), 2),
        f"avg_net_return_pct_at_{mid_spread*100:.1f}pct_spread": round(sum(net_returns) / len(net_returns), 3),
        f"total_net_return_pct_at_{mid_spread*100:.1f}pct_spread": round(sum(net_returns), 2),
        "permutation_test": perm,
        "fill_diagnostics": fill_diagnostics,
        "pct_ambiguous_same_bar": round(100 * n_ambiguous / len(all_trades), 1),
        "pct_ambiguous_resolved_conservatively": round(100 * n_ambiguous_conservative / len(all_trades), 1),
    }


def sweep(tickers: list[str], entry_qs: list[float] | None = None, target_qs: list[float] | None = None) -> pd.DataFrame:
    """Grid over (entry_q, target_q) -- a first pass at spec §6.3's
    percentile sweep. Deliberately modest (a handful of combinations, one
    held-out period) rather than the full fill-probability/capture/cost
    optimisation the spec describes -- that's a reasonable next refinement,
    not attempted in this first cut."""
    entry_qs = entry_qs or [0.3, 0.4, 0.5]
    target_qs = target_qs or [0.5, 0.6, 0.7]
    rows = []
    for eq in entry_qs:
        for tq in target_qs:
            result = evaluate_daily_refresh(tickers, eq, tq)
            skip = {"model_n_train_rows", "held_out_cutoff", "calibration", "entry_q", "target_q", "fill_mode"}
            rows.append({"entry_q": eq, "target_q": tq,
                        **{k: v for k, v in result.items() if k not in skip}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from tradingagents.swing_db import enabled_tickers
    tickers = enabled_tickers()
    if not tickers:
        print("No swing-enabled tickers -- toggle some on /swing first.")
    else:
        t0 = time.time()
        print(f"Fitting pooled range model on: {', '.join(tickers)}\n")
        result = evaluate_daily_refresh(tickers)
        for k, v in result.items():
            print(f"{k}: {v}")
        print(f"\n(fit + eval took {time.time() - t0:.1f}s)")
        print("\n--- percentile sweep ---")
        print(sweep(tickers).to_string(index=False))
