"""Phase-4 §16.6: re-fit the pooled range model on the expanded universe and
put every configuration through the FULL statistical gate.

The spec is blunt about why this is mandatory rather than optional: "The
`(0.1, 0.7)` finding was established on 5 tickers and does not transfer."
Nothing from the old universe carries over, so this re-runs the whole grid
from a fresh fit and judges it against every gate the project has
accumulated:

1. **Randomised-entry permutation** per cell -- would a random entry held the
   same number of days have done as well? ("beats zero" is the wrong null for
   a timing strategy.)
2. **Bootstrap CI** on average return per trade -- does it cross zero?
3. **Buy-and-hold gate** -- an exposure-adjusted return must beat simply
   owning the universe over the same window, or the "edge" is just drift.
4. **Selection permutation (White's Reality Check)** across the whole grid --
   the one that matters. Picking the best of N cells and quoting its
   percentile is the multiple-comparisons mistake; this asks whether the best
   cell beats what a blind grid search over pure noise typically produces.

**Why this module exists rather than calling `range_model.sweep()`**:
`sweep()` calls `evaluate_daily_refresh()` per cell, which re-pools and
re-fits the model every time. At 5 tickers that was merely wasteful; at 26
tickers (~58k pooled rows, 20 quantile regressions per fit) it is
unusable. Here the fit happens ONCE and every cell reuses it, which is also
strictly more correct -- every cell is then evaluated against an identical
model, so differences between cells are attributable to (entry_q, target_q)
and nothing else.

**Null-sampling window**: all cells share one `min_date` (the earliest real
entry across the whole grid) and pre-sliced histories, so cells are
comparable to each other and the null never samples outside the window the
real trades came from (the asymmetry fixed on 2026-08-22 -- see
`backtest._draw_null_avg`). Pre-slicing is a pure speed measure: with
`min_date` equal to the slice point the draw is identical, just over a
255-row frame instead of a 2,500-row one.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from tradingagents.backtest import (
    DEFAULT_BROKERAGE_PCT, DEFAULT_SPREAD_PCT_SWEEP,
    bootstrap_confidence_interval, fetch_daily_history, permutation_test,
    selection_permutation_test,
)
from tradingagents.range_model import (
    _to_trade_results, fit, pool_training_data, simulate_daily_refresh,
)

DEFAULT_ENTRY_QS = [0.05, 0.1, 0.15, 0.2, 0.3]
DEFAULT_TARGET_QS = [0.5, 0.6, 0.7, 0.8, 0.9]
PERM_N_PER_CELL = 10_000     # cheap again since _null_draw_batch vectorised the draw
SELECTION_PERM_N = 10_000
BOOTSTRAP_N = 2_000        # still pure-python resampling; 2k is ample for a CI


def _universe_buy_and_hold(histories: dict[str, pd.DataFrame], start: pd.Timestamp) -> float:
    """Equal-weight buy-and-hold across the universe over the held-out
    window, in percent. The benchmark a timing strategy has to beat to have
    earned its complexity."""
    rets = []
    for df in histories.values():
        sub = df[df.index >= start]
        if len(sub) < 2:
            continue
        rets.append((sub["Close"].iloc[-1] - sub["Close"].iloc[0]) / sub["Close"].iloc[0] * 100)
    return float(np.mean(rets)) if rets else float("nan")


def _log(msg: str) -> None:
    """Phase logging, flushed. A run this long that prints nothing until the
    end is undiagnosable if it dies -- which is exactly what happened on the
    first attempt at this (2026-08-22)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_full_gate(tickers: list[str],
                  entry_qs: list[float] | None = None,
                  target_qs: list[float] | None = None,
                  perm_n: int = PERM_N_PER_CELL,
                  same_bar_exit: str = "conservative") -> dict[str, Any]:
    entry_qs = entry_qs or DEFAULT_ENTRY_QS
    target_qs = target_qs or DEFAULT_TARGET_QS

    t0 = time.time()
    _log(f"pooling training data for {len(tickers)} tickers...")
    pooled = pool_training_data(tickers)
    if pooled.empty:
        return {"error": "no training data"}
    _log(f"pooled {len(pooled):,} rows; fitting quantile models...")
    model = fit(pooled)
    fit_secs = time.time() - t0
    _log(f"fit done in {fit_secs:.1f}s ({model.n_train_rows:,} train rows)")

    _log("fetching daily histories...")
    histories = {}
    for ticker in tickers:
        df = fetch_daily_history(ticker)
        if not df.empty:
            histories[ticker] = df

    # --- simulate every cell against the one shared fit ---
    cell_trades: dict[tuple[float, float], list[dict[str, Any]]] = {}
    cell_fills: dict[tuple[float, float], dict[str, int]] = {}
    n_cells = len(entry_qs) * len(target_qs)
    for eq in entry_qs:
        for tq in target_qs:
            _log(f"simulating cell {len(cell_trades) + 1}/{n_cells} (entry_q={eq}, target_q={tq})")
            trades, quotes, fills = [], 0, 0
            for ticker, df in histories.items():
                tt, diag = simulate_daily_refresh(ticker, df, model, eq, tq, mode="realistic",
                                                  same_bar_exit=same_bar_exit)
                trades += tt
                quotes += diag["quote_attempts"]
                fills += diag["fills"]
            cell_trades[(eq, tq)] = trades
            cell_fills[(eq, tq)] = {"quote_attempts": quotes, "fills": fills}

    all_entry_dates = [t["entry_fill_date"] for tr in cell_trades.values() for t in tr]
    if not all_entry_dates:
        return {"error": "no fills anywhere in the grid"}
    min_date = min(all_entry_dates)
    sliced = {k: v[v.index >= min_date] for k, v in histories.items()}

    mid_spread = DEFAULT_SPREAD_PCT_SWEEP[2]
    cost_pct = (2 * DEFAULT_BROKERAGE_PCT + 2 * mid_spread) * 100
    buy_hold = _universe_buy_and_hold(histories, min_date)
    held_out_days = max(len(v) for v in sliced.values())
    capacity_days = held_out_days * len(sliced)

    _log(f"simulation done; grading {n_cells} cells (perm n={perm_n})")
    rows = []
    tr_results: dict[tuple[float, float], list[Any]] = {}
    for key, trades in cell_trades.items():
        eq, tq = key
        results = _to_trade_results(trades)
        tr_results[key] = results
        filled = [t for t in results if t.filled]
        base = {"entry_q": eq, "target_q": tq, "n_trades": len(filled),
                "fill_rate_pct": round(100 * cell_fills[key]["fills"] / cell_fills[key]["quote_attempts"], 2)
                if cell_fills[key]["quote_attempts"] else None}
        if not filled:
            rows.append({**base, "avg_gross_pct": None})
            continue

        returns = [t.gross_return_pct for t in filled]
        avg = float(np.mean(returns))
        total = float(np.sum(returns))
        days_in_market = sum(t.holding_days or 0 for t in filled)
        time_in_market = days_in_market / capacity_days if capacity_days else 0.0
        perm = permutation_test(results, sliced, n=perm_n, min_date=min_date)
        ci = bootstrap_confidence_interval(results, n=BOOTSTRAP_N)

        rows.append({
            **base,
            "win_rate_pct": round(100 * sum(1 for r in returns if r > 0) / len(returns), 1),
            "avg_gross_pct": round(avg, 3),
            "total_gross_pct": round(total, 2),
            "avg_net_pct": round(avg - cost_pct, 3),
            "ci_lo": ci.get("ci_low_pct"), "ci_hi": ci.get("ci_high_pct"),
            "ci_excludes_zero": bool(ci.get("ci_low_pct") is not None and ci["ci_low_pct"] > 0),
            "perm_pct": perm.get("actual_percentile_in_null"),
            "beats_random": bool((perm.get("actual_percentile_in_null") or 0) >= 95),
            "time_in_market_pct": round(100 * time_in_market, 2),
            "exposure_adj_total_pct": round(total / time_in_market, 2) if time_in_market > 0 else None,
            "beats_buy_hold": bool(time_in_market > 0 and (total / time_in_market) > buy_hold),
        })

    grid = pd.DataFrame(rows)
    graded = grid[grid.avg_gross_pct.notna()].copy()
    if not graded.empty:
        graded["passes_all_gates"] = (
            graded.ci_excludes_zero & graded.beats_random & graded.beats_buy_hold)

    _log(f"running selection permutation (White's Reality Check, n={SELECTION_PERM_N})...")
    selection = selection_permutation_test(
        {k: v for k, v in tr_results.items() if any(t.filled for t in v)},
        sliced, n=SELECTION_PERM_N, min_date=min_date,
    )

    _log("done")
    return {
        "tickers": tickers,
        "n_train_rows": model.n_train_rows,
        "train_cutoff": str(model.train_cutoff.date()),
        "held_out_start": str(min_date.date()),
        "held_out_trading_days": held_out_days,
        "fit_seconds": round(fit_secs, 1),
        "same_bar_exit": same_bar_exit,
        "cost_assumption_pct_round_trip": round(cost_pct, 3),
        "universe_buy_and_hold_pct": round(buy_hold, 2),
        "grid": graded if not graded.empty else grid,
        "selection_permutation": selection,
    }


def per_ticker_for_cell(tickers: list[str], entry_q: float, target_q: float,
                        perm_n: int = PERM_N_PER_CELL) -> pd.DataFrame:
    """Break one grid cell down per ticker.

    **Read this as description, not as a shortlist.** Picking the best-looking
    tickers out of this table and calling them "the ones that work" is the
    selection problem the whole gate exists to catch, one level down -- with
    ~10-20 trades per ticker, the spread between best and worst name is mostly
    noise, and `selection_permutation_test()` is not applied here because a
    per-ticker null would need the same best-of-N correction to mean anything.
    It is here to answer "is the aggregate result carried by one or two names
    or spread across the universe", which is a different and answerable
    question.
    """
    pooled = pool_training_data(tickers)
    model = fit(pooled)
    rows = []
    for ticker in tickers:
        df = fetch_daily_history(ticker)
        if df.empty:
            continue
        trades, diag = simulate_daily_refresh(ticker, df, model, entry_q, target_q, mode="realistic")
        results = _to_trade_results(trades)
        filled = [t for t in results if t.filled]
        if not filled:
            rows.append({"ticker": ticker, "n_trades": 0})
            continue
        returns = [t.gross_return_pct for t in filled]
        sliced = {ticker: df[df.index >= min(t.entry_fill_date for t in filled)]}
        perm = permutation_test(results, sliced, n=perm_n)
        ci = bootstrap_confidence_interval(results, n=BOOTSTRAP_N)
        rows.append({
            "ticker": ticker, "n_trades": len(filled),
            "fill_rate_pct": round(100 * diag["fills"] / diag["quote_attempts"], 2)
            if diag["quote_attempts"] else None,
            "win_rate_pct": round(100 * sum(1 for r in returns if r > 0) / len(returns), 1),
            "avg_gross_pct": round(float(np.mean(returns)), 3),
            "total_gross_pct": round(float(np.sum(returns)), 2),
            "ci_lo": ci.get("ci_low_pct"), "ci_hi": ci.get("ci_high_pct"),
            "perm_pct": perm.get("actual_percentile_in_null"),
        })
    return pd.DataFrame(rows).sort_values("total_gross_pct", ascending=False, na_position="last")


def print_gate(res: dict[str, Any]) -> None:
    if "error" in res:
        print("ERROR:", res["error"])
        return
    print("=" * 110)
    print("RANGE MODEL RE-FIT + FULL GATE (phase-4 spec 16.6)")
    print("=" * 110)
    print(f"tickers ({len(res['tickers'])}): {', '.join(res['tickers'])}")
    print(f"train rows      : {res['n_train_rows']:,} (to {res['train_cutoff']})")
    print(f"held-out        : {res['held_out_start']} onward, {res['held_out_trading_days']} trading days")
    print(f"fit time        : {res['fit_seconds']}s")
    print(f"same-bar policy : {res['same_bar_exit']}  "
          "(target may NOT resolve on the entry bar; stop may -- see backtest.simulate_trade)")
    print(f"cost assumption : {res['cost_assumption_pct_round_trip']}% round trip")
    print(f"universe buy&hold over held-out window: {res['universe_buy_and_hold_pct']:+.2f}%")
    print()
    g = res["grid"]
    cols = ["entry_q", "target_q", "n_trades", "fill_rate_pct", "win_rate_pct", "avg_gross_pct",
            "avg_net_pct", "ci_lo", "ci_hi", "perm_pct", "exposure_adj_total_pct"]
    cols = [c for c in cols if c in g.columns]
    print(g[cols].to_string(index=False))
    print()
    if "passes_all_gates" in g.columns:
        winners = g[g.passes_all_gates]
        print(f"cells clearing CI + randomised-entry + buy-and-hold: "
              f"{len(winners)} of {len(g)}")
        if not winners.empty:
            print(winners[cols].to_string(index=False))
    print()
    sp = res["selection_permutation"]
    print("--- SELECTION PERMUTATION (White's Reality Check) " + "-" * 40)
    for k in ("n_grid_cells", "real_best_cell", "real_best_avg_return_pct",
              "null_best_of_grid_median_pct", "real_best_percentile_in_null"):
        print(f"  {k}: {sp.get(k)}")
    print(f"  -> {sp.get('interpretation')}")


if __name__ == "__main__":
    from tradingagents.screener import get_model_universe
    tickers = [r["ticker"] for r in get_model_universe(tradeable_only=True)]
    print_gate(run_full_gate(tickers))
