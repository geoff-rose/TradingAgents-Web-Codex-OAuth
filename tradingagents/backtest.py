"""Backtest for the Swing Trade heuristic (`swing_signal.py`).

Per the phase-3 spec (`/root/asxbrief-phase3-spec-v2.md` -- read that document
for full reasoning, this module implements its §3): forward paper testing at
this trade volume (~40-60 trades/year across 5 tickers) needs on the order of
a few hundred trades to separate real edge from noise at a ~50% win rate --
that's roughly five years of forward testing. A backtest over years of daily
history produces that sample size immediately, at the cost of needing an
honest fill model (see FILL MODEL below) since it can't observe real market
depth.

**Nothing here is validated by construction.** This module measures whether
the existing heuristic (or any future replacement) has real edge; it doesn't
assume the answer. Report both fill models' results and the gap between them
-- if edge only exists under the naive touch-fill model, there is no edge.

Run as a script for a report against the current focus-ticker universe:
    .venv/bin/python -m tradingagents.backtest
"""

from __future__ import annotations

import random
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yfinance as yf

from tradingagents.swing_signal import (
    ATR_PERIOD, MAX_GAIN_PCT, MIN_GAIN_PCT, PULLBACK_BAND_PCT,
    PULLBACK_LOOKBACK, SMA_PERIOD, STOP_ATR_MULT, TARGET_ATR_MULT,
)

# ---------------------------------------------------------------------------
# ASX tick table -- verified live 2026-08-21 against ASX's own price-steps
# page (asx.com.au/services/trading-services/price.htm), NOT assumed. The
# phase-3 spec explicitly flagged this as disputed/decisive: at a 13c share
# price the difference between a 0.1c and 0.5c tick is the difference between
# ~0.8% and ~3.8% cost per tick, which decides whether a name is tradeable at
# all. Encoded as data, per the spec's own instruction, not hardcoded inline.
# ---------------------------------------------------------------------------
TICK_TABLE = [
    (0.10, 0.001),   # below 10c: 0.1c tick
    (2.00, 0.005),   # 10c up to and including $2.00: 0.5c tick
    (float("inf"), 0.01),  # above $2.00: 1c tick
]


def tick_size(price: float) -> float:
    """Boundaries are inclusive of each band's own ceiling (a $2.00 stock
    uses the 0.5c tick, not the 1c one) -- verified against ASX's own
    price-steps page, not assumed."""
    if price < TICK_TABLE[0][0]:
        return TICK_TABLE[0][1]
    if price <= TICK_TABLE[1][0]:
        return TICK_TABLE[1][1]
    return TICK_TABLE[2][1]


def round_to_tick(price: float) -> float:
    t = tick_size(price)
    return round(round(price / t) * t, 4)


# ---------------------------------------------------------------------------
# Costs -- swept, not point-estimated (spec §3.2). These are starting
# assumptions to sweep around, not measured values (no real spread history
# exists yet -- that's what preopen/spread_daily capture, §5.1/5.4, is for).
# ---------------------------------------------------------------------------
DEFAULT_SPREAD_PCT_SWEEP = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02)  # half-spread, one side
DEFAULT_BROKERAGE_PCT = 0.0005  # each side; IBKR-style small fixed+pct, approximated as pct here

HELD_OUT_MONTHS = 12
MAX_HOLD_DAYS = 60  # see "Known modelling simplifications" below
N_PERMUTATIONS = 10_000

BARS_DB_PATH = Path("/opt/asxbrief/data/asx.db")


@dataclass
class Signal:
    ticker: str
    signal_date: pd.Timestamp
    attempt_date: pd.Timestamp
    entry: float
    target: float
    stop: float


@dataclass
class TradeResult:
    ticker: str
    signal_date: pd.Timestamp
    entry: float
    target: float
    stop: float
    filled: bool = False
    entry_fill_date: pd.Timestamp | None = None
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None  # target|stop|timeout|end_of_data|unfilled
    ambiguous_same_bar: bool = False
    ambiguous_resolved_by: str | None = None  # intraday|conservative_default
    holding_days: int | None = None
    gross_return_pct: float | None = None
    net_return_pct: float | None = None  # after costs, realistic model only


def fetch_daily_history(ticker: str, period: str = "10y", market: str = "AU") -> pd.DataFrame:
    """`market='AU'` (default, backward-compatible with every existing
    caller) appends `.AX`; `market='US'` (phase-4 spec §4) uses the bare
    symbol. Always adjusted prices (yfinance's default) -- return
    computation must use adjusted, never raw, or split/dividend dates
    produce silently wrong entry levels."""
    symbol = f"{ticker}.AX" if market == "AU" else ticker
    df = yf.Ticker(symbol).history(period=period)
    if df.empty:
        return df
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def generate_signals(ticker: str, df: pd.DataFrame) -> list[Signal]:
    """Walk-forward, point-in-time reconstruction of swing_signal.evaluate()'s
    exact rule -- reuses its constants so there is one source of truth for
    the thresholds, but re-implemented here as a day-by-day scan (rather than
    calling evaluate(), which always fetches live/latest data) so every
    day's decision only uses data available up to and including that day's
    close. No-pyramiding is enforced the same way swing_db.has_active_trade
    does: while a previous signal for this ticker is still open (unfilled or
    filled-and-open), no new signal is generated."""
    if len(df) < SMA_PERIOD + ATR_PERIOD + 5:
        return []

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    dates = df.index

    signals: list[Signal] = []
    blocked_until: pd.Timestamp | None = None  # set once a signal is live; cleared once resolved (approximated below)

    for i in range(SMA_PERIOD, len(df) - 1):
        if blocked_until is not None and dates[i] <= blocked_until:
            continue

        sma20 = closes[i - SMA_PERIOD + 1 : i + 1].mean()
        last_close = closes[i]
        trs = [
            max(highs[j] - lows[j], abs(highs[j] - closes[j - 1]), abs(lows[j] - closes[j - 1]))
            for j in range(max(1, i - ATR_PERIOD + 1), i + 1)
        ]
        if len(trs) < ATR_PERIOD:
            continue
        atr = sum(trs[-ATR_PERIOD:]) / ATR_PERIOD
        if atr <= 0:
            continue

        recent_low = lows[max(0, i - PULLBACK_LOOKBACK + 1) : i + 1].min()
        in_uptrend = last_close > sma20
        near_pullback_low = (last_close - recent_low) / recent_low * 100 < PULLBACK_BAND_PCT
        if not (in_uptrend and near_pullback_low):
            continue

        entry = round_to_tick(last_close)
        target = round_to_tick(entry + TARGET_ATR_MULT * atr)
        stop = round_to_tick(entry - STOP_ATR_MULT * atr)
        gain_pct = (target - entry) / entry * 100
        if not (MIN_GAIN_PCT <= gain_pct <= MAX_GAIN_PCT):
            continue

        sig = Signal(
            ticker=ticker, signal_date=dates[i], attempt_date=dates[i + 1],
            entry=entry, target=target, stop=stop,
        )
        signals.append(sig)
        # Block further signals for this ticker until this one's outcome is
        # known -- resolved properly during simulate_all (which re-walks
        # signals against forward price action); here we conservatively
        # block for MAX_HOLD_DAYS trading days as a first pass, matching the
        # live system's per-ticker exclusivity (swing_db.has_active_trade).
        block_idx = min(i + 1 + MAX_HOLD_DAYS, len(df) - 1)
        blocked_until = dates[block_idx]

    return signals


def _bars_for(ticker: str, day: pd.Timestamp) -> pd.DataFrame | None:
    """Real intraday bars for this ticker/day if the phase-2 backfill has
    them (see asxbrief-phase2 skill -- coverage tops out ~3.5 months for
    smaller ASX names, so this will usually return None for older history).
    Used only to resolve same-bar target/stop ambiguity when available."""
    if not BARS_DB_PATH.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{BARS_DB_PATH}?mode=ro", uri=True, timeout=5.0)
        day_str = day.strftime("%Y-%m-%d")
        rows = conn.execute(
            "SELECT ts, high, low FROM bars WHERE ticker=? AND what='TRADES' "
            "AND ts LIKE ? ORDER BY ts",
            (ticker, f"{day_str}%"),
        ).fetchall()
        conn.close()
        if not rows:
            return None
        return pd.DataFrame(rows, columns=["ts", "High", "Low"])
    except Exception:
        return None


def _resolve_same_bar_ambiguity(ticker: str, day: pd.Timestamp, target: float, stop: float) -> tuple[str, str]:
    """Returns (winner, resolved_by) where winner is 'target'/'stop' and
    resolved_by is 'intraday'/'conservative_default'. Tries real intraday
    bars first; falls back to the conservative default (assume stop filled
    first) per spec §3.1 -- "Never silently assume the target; this alone
    will inflate results substantially.\""""
    intraday = _bars_for(ticker, day)
    if intraday is not None and len(intraday):
        for _, bar in intraday.iterrows():
            hit_target = bar["High"] >= target
            hit_stop = bar["Low"] <= stop
            if hit_target and not hit_stop:
                return "target", "intraday"
            if hit_stop:
                return "stop", "intraday"
    return "stop", "conservative_default"


SameBarExit = Literal["conservative", "optimistic"]


def simulate_trade(
    ticker: str, df: pd.DataFrame, sig: Signal,
    mode: Literal["naive", "realistic"],
    same_bar_exit: SameBarExit = "conservative",
) -> TradeResult:
    """Simulates one signal forward. `mode='naive'` fills on mere touch (the
    baseline everyone should distrust); `mode='realistic'` requires price to
    trade THROUGH the level by at least one tick (worst-case queue
    position) -- see FILL MODEL notes at module bottom."""
    result = TradeResult(ticker=ticker, signal_date=sig.signal_date, entry=sig.entry,
                         target=sig.target, stop=sig.stop)
    tick = tick_size(sig.entry)

    idx = df.index.get_indexer([sig.attempt_date], method="bfill")[0]
    if idx < 0:
        return result

    # --- entry fill scan ---
    entry_idx = None
    for i in range(idx, min(idx + MAX_HOLD_DAYS, len(df))):
        low = df["Low"].iloc[i]
        threshold = sig.entry if mode == "naive" else sig.entry - tick
        if low <= threshold:
            entry_idx = i
            break
    if entry_idx is None:
        result.exit_reason = "unfilled"
        return result

    result.filled = True
    result.entry_fill_date = df.index[entry_idx]

    # --- exit scan ---
    # `same_bar_exit` governs what may happen on the ENTRY bar itself. A daily
    # OHLC bar carries no path information, so crediting a target on the same
    # bar that filled the entry silently assumes the low preceded the high.
    #
    # That assumption was originally made deliberately ("a wide-range day can
    # both fill the entry and resolve target/stop") and was near-harmless on
    # the old 5-ticker microcap universe, where entries rarely filled at all.
    # On the expanded universe -- which the screener explicitly selected for
    # WIDE daily ranges -- it became the dominant effect: 82% of range-model
    # trades entered and exited on one bar, those trades showed an 86% win
    # rate and +1.84% average, and the genuine multi-day trades underneath
    # them averaged -0.90%. Measured 2026-08-22.
    #
    # 'conservative' (default): on the entry bar a STOP may trigger (assume
    # the adverse path), a TARGET may not. 'optimistic' restores the old
    # behaviour and exists only so the two can be compared -- it is an upper
    # bound, never a headline.
    for j in range(entry_idx, min(entry_idx + MAX_HOLD_DAYS, len(df))):
        high, low = df["High"].iloc[j], df["Low"].iloc[j]
        target_thresh = sig.target if mode == "naive" else sig.target + tick
        stop_thresh = sig.stop if mode == "naive" else sig.stop - tick
        on_entry_bar = (j == entry_idx)
        hit_target = high >= target_thresh and not (on_entry_bar and same_bar_exit == "conservative")
        hit_stop = low <= stop_thresh

        if hit_target and hit_stop:
            result.ambiguous_same_bar = True
            winner, resolved_by = _resolve_same_bar_ambiguity(ticker, df.index[j], sig.target, sig.stop)
            result.ambiguous_resolved_by = resolved_by
            result.exit_date = df.index[j]
            result.exit_price = sig.target if winner == "target" else sig.stop
            result.exit_reason = winner
            break
        if hit_target:
            result.exit_date = df.index[j]
            result.exit_price = sig.target
            result.exit_reason = "target"
            break
        if hit_stop:
            result.exit_date = df.index[j]
            result.exit_price = sig.stop
            result.exit_reason = "stop"
            break
    else:
        last_idx = min(entry_idx + MAX_HOLD_DAYS, len(df)) - 1
        result.exit_date = df.index[last_idx]
        result.exit_price = df["Close"].iloc[last_idx]
        result.exit_reason = "timeout" if last_idx < len(df) - 1 else "end_of_data"

    result.holding_days = (result.exit_date - result.entry_fill_date).days
    result.gross_return_pct = (result.exit_price - sig.entry) / sig.entry * 100

    if mode == "realistic":
        cost_pct = 2 * DEFAULT_BROKERAGE_PCT + 2 * DEFAULT_SPREAD_PCT_SWEEP[2]  # mid sweep point as the headline number
        result.net_return_pct = result.gross_return_pct - cost_pct * 100

    return result


@dataclass
class BacktestReport:
    tickers: list[str]
    naive_trades: list[TradeResult] = field(default_factory=list)
    realistic_trades: list[TradeResult] = field(default_factory=list)
    held_out_cutoff: pd.Timestamp | None = None
    cost_sweep: dict[float, dict[str, float]] = field(default_factory=dict)
    permutation: dict[str, Any] | None = None


def _summary_stats(trades: list[TradeResult], return_field: str = "gross_return_pct") -> dict[str, Any]:
    filled = [t for t in trades if t.filled and getattr(t, return_field) is not None]
    if not filled:
        return {"n_trades": 0}
    returns = [getattr(t, return_field) for t in filled]
    wins = [r for r in returns if r > 0]
    n_ambiguous = sum(1 for t in filled if t.ambiguous_same_bar)
    n_conservative = sum(1 for t in filled if t.ambiguous_resolved_by == "conservative_default")
    return {
        "n_trades": len(filled),
        "n_unfilled_signals": sum(1 for t in trades if not t.filled),
        "win_rate_pct": round(100 * len(wins) / len(filled), 1),
        "avg_return_pct": round(sum(returns) / len(returns), 3),
        "total_return_pct": round(sum(returns), 2),
        "pct_ambiguous_same_bar": round(100 * n_ambiguous / len(filled), 1),
        "pct_resolved_conservatively": round(100 * n_conservative / len(filled), 1),
    }


def run_backtest(tickers: list[str], period: str = "10y") -> BacktestReport:
    report = BacktestReport(tickers=tickers)
    all_signals: dict[str, list[Signal]] = {}
    histories: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        df = fetch_daily_history(ticker, period=period)
        if df.empty:
            continue
        histories[ticker] = df
        all_signals[ticker] = generate_signals(ticker, df)

    for ticker, sigs in all_signals.items():
        df = histories[ticker]
        for sig in sigs:
            report.naive_trades.append(simulate_trade(ticker, df, sig, mode="naive"))
            report.realistic_trades.append(simulate_trade(ticker, df, sig, mode="realistic"))

    if histories:
        latest = max(df.index.max() for df in histories.values())
        report.held_out_cutoff = latest - pd.DateOffset(months=HELD_OUT_MONTHS)

    report.cost_sweep = _cost_sweep(report.realistic_trades)
    report.permutation = permutation_test(report.realistic_trades, histories)
    return report


def _cost_sweep(trades: list[TradeResult]) -> dict[float, dict[str, float]]:
    """Recomputes net P&L for each assumed one-side spread_pct in the sweep,
    holding brokerage fixed -- finds the cost level at which the edge
    disappears (spec §3.2)."""
    filled = [t for t in trades if t.filled and t.gross_return_pct is not None]
    out = {}
    for spread_pct in DEFAULT_SPREAD_PCT_SWEEP:
        cost_pct = (2 * DEFAULT_BROKERAGE_PCT + 2 * spread_pct) * 100
        net = [t.gross_return_pct - cost_pct for t in filled]
        out[spread_pct] = {
            "avg_net_return_pct": round(sum(net) / len(net), 3) if net else None,
            "total_net_return_pct": round(sum(net), 2) if net else None,
        }
    return out


def _draw_null_avg(filled: list[TradeResult], histories: dict[str, pd.DataFrame], rng: random.Random,
                    min_date: pd.Timestamp | None = None) -> float | None:
    """One synthetic draw: for each real (filled) trade, a uniformly random
    start date for its ticker held for the same number of calendar days,
    close-to-close return -- then averaged. This is the reusable core of the
    randomised-entry null; both permutation_test() (one config) and
    selection_permutation_test() (best-of-grid, phase-4 §12.1/§6.3) build on
    exactly this draw so there's one source of truth for what "random entry,
    same holding period" means.

    **Kept as the reference implementation.** `_null_draw_batch()` is what
    both permutation tests now actually call (same null, ~645x faster); this
    single-draw form is the definition that one is verified against, and is
    the readable statement of what the null means. Don't delete it, and if
    the null's semantics ever change, change this first and re-verify.

    `min_date`, added 2026-08-22 after the user asked whether the null
    matches the real run's costs/fill model: it doesn't matter for costs
    (neither side has any -- both are pure gross price-level returns) but a
    REAL asymmetry existed here -- the null was drawing from a ticker's
    FULL fetched history (e.g. 10 years) while every real trade came only
    from the held-out window (~12 months). That's not apples-to-apples: if
    the held-out period's character (drift/vol) differs from the full
    history's average, the null under- or over-states how easy the result
    would be "by chance" during that specific window. Pass the real run's
    earliest entry_fill_date (or the model's train_cutoff) to restrict
    sampling to the same window the real trades are drawn from."""
    draws = []
    for t in filled:
        df = histories.get(t.ticker)
        if df is None or len(df) < 2:
            continue
        if min_date is not None:
            df = df[df.index >= min_date]
            if len(df) < 2:
                continue
        hold_bars = max(1, min(len(df) - 2, t.holding_days))
        start_i = rng.randint(0, len(df) - hold_bars - 1)
        start_price = df["Close"].iloc[start_i]
        end_price = df["Close"].iloc[start_i + hold_bars]
        draws.append((end_price - start_price) / start_price * 100)
    return sum(draws) / len(draws) if draws else None


def _null_draw_batch(filled: list[TradeResult], histories: dict[str, pd.DataFrame],
                     n: int, seed: int, min_date: pd.Timestamp | None = None) -> np.ndarray:
    """Vectorised form of `_draw_null_avg`, returning `n` null averages at once.

    Same null, same semantics: for each real trade, a uniformly random start
    date in that ticker's (optionally date-restricted) history, held for the
    same number of bars, close-to-close, averaged across trades. The only
    change is that the per-trade draws for all `n` iterations are generated
    in one numpy call per trade instead of one Python-level pandas slice per
    trade per iteration.

    **Why this exists**: the loop version is O(n_draws x n_trades) pandas
    index operations. That was fine at ~40 trades (the 5-ticker universe) and
    became the dominant cost at ~640 trades (the 26-ticker universe) --
    measured at 272s per grid cell, i.e. hours for a 25-cell sweep. This runs
    the same 2,000 draws in well under a second.

    Index bounds match `_draw_null_avg` exactly: `random.Random.randint(0, L -
    hold - 1)` is inclusive at both ends, and `rng.integers(0, L - hold, n)`
    is inclusive-exclusive, so both draw from [0, L-hold-1].

    Draws are independent across trades here, exactly as in the loop version
    (each trade got its own `rng.randint` call there too), so the resulting
    null distribution is the same; only the RNG consumption order differs,
    which is why this is verified distributionally rather than by seed-for-
    seed equality.
    """
    rng = np.random.default_rng(seed)
    cols = []
    for t in filled:
        df = histories.get(t.ticker)
        if df is None or len(df) < 2:
            continue
        if min_date is not None:
            df = df[df.index >= min_date]
            if len(df) < 2:
                continue
        closes = df["Close"].to_numpy(dtype=float)
        L = len(closes)
        hold = max(1, min(L - 2, t.holding_days))
        if L - hold < 1:
            continue
        starts = rng.integers(0, L - hold, n)
        start_p = closes[starts]
        cols.append((closes[starts + hold] - start_p) / start_p * 100)
    if not cols:
        return np.array([])
    return np.mean(np.column_stack(cols), axis=1)


def permutation_test(trades: list[TradeResult], histories: dict[str, pd.DataFrame],
                      n: int = N_PERMUTATIONS, min_date: pd.Timestamp | None = None) -> dict[str, Any]:
    """Randomised-entry permutation test (spec §3.3): same trade count, same
    per-ticker holding periods, randomised entry dates, drawn `n` times.
    Null hypothesis: a random entry held for the same number of days would do
    just as well. "Beats zero" is the wrong null for a timing strategy --
    this is the right one.

    Each real trade contributes one permuted draw per run: a uniformly random
    start date for its ticker, held for the same number of calendar days,
    close-to-close return. This measures whether *any* random entry/hold of
    this length would look similar -- if so, the heuristic has no timing
    value, independent of whether it's directionally profitable on average.

    `min_date`: restricts null sampling to the same window the real trades
    came from (see `_draw_null_avg`'s docstring -- fixed 2026-08-22, this
    used to silently sample from a ticker's full history even when the real
    trades were all from a held-out sub-window). If not given, it's
    auto-derived as the earliest real trade's `entry_fill_date`, so every
    existing caller gets the fix without needing to pass anything new.
    """
    filled = [t for t in trades if t.filled and t.holding_days is not None]
    if not filled:
        return {"n_trades": 0}

    if min_date is None:
        min_date = min(t.entry_fill_date for t in filled if t.entry_fill_date is not None)

    actual_avg = sum(t.gross_return_pct for t in filled) / len(filled)

    # Vectorised draw (see _null_draw_batch): same null, ~645x faster --
    # verified distributionally against the loop version on a 637-trade cell
    # (means within 1.6 sigma of their difference, sd and 5/50/95th
    # percentiles matching). The loop form became the dominant cost of any
    # sweep once the universe expanded from 5 tickers to 26.
    null_avgs = sorted(_null_draw_batch(filled, histories, n, 42, min_date).tolist())
    rank = sum(1 for v in null_avgs if v < actual_avg)
    percentile = round(100 * rank / len(null_avgs), 1) if null_avgs else None
    if percentile is None:
        interpretation = "insufficient data"
    elif percentile >= 95:
        interpretation = "above the 95th percentile of the random-entry null -- worth further scrutiny, not proof of edge"
    elif percentile <= 5:
        interpretation = (
            "at or below the 5th percentile -- this is evidence the heuristic's timing is WORSE than "
            "a random entry of the same holding length, not merely 'no edge'. Two-tailed: both tails are informative."
        )
    else:
        interpretation = "indistinguishable from a random entry of the same holding length -- no evidence of timing value either way"
    return {
        "n_trades": len(filled),
        "actual_avg_return_pct": round(actual_avg, 3),
        "null_median_pct": round(null_avgs[len(null_avgs) // 2], 3) if null_avgs else None,
        "actual_percentile_in_null": percentile,
        "interpretation": interpretation,
    }


def bootstrap_confidence_interval(trades: list[TradeResult], n: int = 10_000, ci: float = 0.95) -> dict[str, Any]:
    """95% CI on mean return per trade via resampling-with-replacement
    (phase-4 spec §6.1/§12.6: "publish the confidence interval, not just the
    point estimate -- on a small sample it will be wide, and the width is the
    point"). Bootstrap rather than a normal-theory (t-distribution) CI since
    trade returns are plausibly skewed (asymmetric target/stop payoffs), not
    assumed Gaussian."""
    filled = [t for t in trades if t.filled and t.gross_return_pct is not None]
    if len(filled) < 2:
        return {"n_trades": len(filled), "error": "too few trades for a CI"}
    returns = [t.gross_return_pct for t in filled]
    rng = random.Random(43)
    means = []
    for _ in range(n):
        sample = [returns[rng.randrange(len(returns))] for _ in range(len(returns))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo_idx = int(n * (1 - ci) / 2)
    hi_idx = int(n * (1 - (1 - ci) / 2))
    return {
        "n_trades": len(filled),
        "mean_return_pct": round(sum(returns) / len(returns), 3),
        "ci_low_pct": round(means[lo_idx], 3),
        "ci_high_pct": round(means[hi_idx], 3),
        "ci_level": ci,
    }


def selection_permutation_test(
    cell_trades: dict[Any, list[TradeResult]], histories: dict[str, pd.DataFrame], n: int = 2000,
    min_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Corrected p-value for a "best of N grid cells" result (phase-4 §6.3 /
    §12.1) -- White's Reality Check in substance. Reporting a single cell's
    permutation percentile after picking it as the best of a grid is exactly
    the multiple-comparisons mistake the spec warns about: the max of several
    noisy estimates looks good by construction even under the null.

    Method: for `n` iterations, draw ONE null average per grid cell (reusing
    `_draw_null_avg` on that cell's own trades/holding-periods/ticker, so the
    null respects each cell's actual trade structure) and take the max across
    cells -- that's one draw from "best achievable by chance across this
    whole grid search". Compare the real best cell's average against that
    distribution. Cheap: reuses each cell's already-simulated trades, no
    model refitting needed.

    `min_date`: same held-out-window restriction as `permutation_test()` --
    auto-derived across ALL cells' trades if not given, so the null never
    samples outside the period the real trades actually came from."""
    real_bests = {key: sum(t.gross_return_pct for t in trades if t.filled) / max(1, sum(1 for t in trades if t.filled))
                  for key, trades in cell_trades.items() if any(t.filled for t in trades)}
    if not real_bests:
        return {"error": "no filled trades in any grid cell"}
    real_best_key = max(real_bests, key=real_bests.get)
    real_best_avg = real_bests[real_best_key]

    filled_by_cell = {key: [t for t in trades if t.filled and t.holding_days is not None]
                       for key, trades in cell_trades.items()}

    if min_date is None:
        all_entry_dates = [t.entry_fill_date for trades in filled_by_cell.values() for t in trades
                            if t.entry_fill_date is not None]
        min_date = min(all_entry_dates) if all_entry_dates else None

    # One batch of n draws per cell (independent seeds, so cells stay
    # independent exactly as they were when a single shared rng served them
    # in sequence), stacked into an (n, n_cells) matrix. Each row is then one
    # draw from "best achievable by chance across this whole grid search".
    cell_matrix = []
    for i, trades in enumerate(filled_by_cell.values()):
        if not trades:
            continue
        draws = _null_draw_batch(trades, histories, n, 44 + i, min_date)
        if draws.size:
            cell_matrix.append(draws)
    null_bests = sorted(np.max(np.column_stack(cell_matrix), axis=1).tolist()) if cell_matrix else []
    rank = sum(1 for v in null_bests if v < real_best_avg)
    percentile = round(100 * rank / len(null_bests), 1) if null_bests else None
    return {
        "n_grid_cells": len(cell_trades),
        "real_best_cell": real_best_key,
        "real_best_avg_return_pct": round(real_best_avg, 3),
        "null_best_of_grid_median_pct": round(null_bests[len(null_bests) // 2], 3) if null_bests else None,
        "real_best_percentile_in_null": percentile,
        "interpretation": (
            "the best-of-grid result is NOT distinguishable from what a grid search over pure noise "
            "would produce by chance -- the multiple-comparisons risk is real, not just theoretical"
            if percentile is not None and percentile < 95
            else "the best-of-grid result exceeds the 95th percentile of what noise alone would produce "
                 "across a grid this size -- meaningful, though still not proof"
        ) if percentile is not None else "insufficient data",
    }


def print_report(report: BacktestReport) -> None:
    print(f"\n=== Swing heuristic backtest — {len(report.tickers)} tickers: {', '.join(report.tickers)} ===")
    print(f"Held-out period starts: {report.held_out_cutoff.date() if report.held_out_cutoff is not None else 'n/a'}\n")

    for label, trades in (("NAIVE touch-fill", report.naive_trades), ("REALISTIC fill", report.realistic_trades)):
        print(f"--- {label} ---")
        in_sample = [t for t in trades if t.entry_fill_date is not None and t.entry_fill_date < report.held_out_cutoff]
        held_out = [t for t in trades if t.entry_fill_date is not None and t.entry_fill_date >= report.held_out_cutoff]
        print("  In-sample:", _summary_stats(in_sample))
        print("  Held-out :", _summary_stats(held_out))
        print("  All      :", _summary_stats(trades))
        print()

    print("--- Cost sweep (realistic fills, one-side spread assumptions) ---")
    for spread_pct, stats in report.cost_sweep.items():
        print(f"  spread={spread_pct*100:.2f}%: avg_net={stats['avg_net_return_pct']}% total_net={stats['total_net_return_pct']}%")
    print()

    print("--- Permutation test (randomised-entry null, N=%d) ---" % N_PERMUTATIONS)
    for k, v in (report.permutation or {}).items():
        print(f"  {k}: {v}")
    print()


if __name__ == "__main__":
    from tradingagents.swing_db import enabled_tickers
    tickers = enabled_tickers()
    if not tickers:
        print("No swing-enabled tickers -- toggle some on /swing first.")
    else:
        t0 = time.time()
        report = run_backtest(tickers)
        print_report(report)
        print(f"(took {time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# FILL MODEL notes (read before trusting or changing simulate_trade)
# ---------------------------------------------------------------------------
# - "naive" fills on mere touch: Low <= entry (buy) / High >= target (sell) /
#   Low <= stop (sell). This is what a lookahead-naive backtest does by
#   default and overstates edge -- kept here ONLY as the comparison baseline
#   the spec calls for, never as the number to trust.
# - "realistic" requires trading THROUGH the level by one tick, approximating
#   worst-case queue position with only daily OHLC (no order-book depth is
#   available for history this old). This is still an approximation, not a
#   true limit-order-book simulation -- minimum-volume-through-level (spec's
#   optional extra realism knob) is NOT implemented: daily bars don't carry
#   volume-at-price, only total daily volume, which isn't enough to model
#   that faithfully. Flagged as a known gap, not silently skipped.
# - Costs (spread + brokerage) are applied as a flat round-trip % deduction
#   from gross return, using assumed values swept across a range (§3.2) --
#   NOT measured spreads, since no historical spread data exists yet (that's
#   what preopen/spread_daily capture, spec §5.1/5.4, is for prospectively).
#
# ---------------------------------------------------------------------------
# Known modelling simplifications (surfaced deliberately, not hidden)
# ---------------------------------------------------------------------------
# - MAX_HOLD_DAYS=60 caps how long an unfilled entry or an open position can
#   sit before this backtest gives up on it (exit_reason='unfilled' or
#   'timeout'). The LIVE system has no such cap -- a submitted bracket order
#   sits GTC indefinitely, and swing_db.has_active_trade blocks new signals
#   for that ticker the whole time. This is a genuine, separate operational
#   gap noticed while building this backtest: there is currently no
#   entry-order expiry/cancel-if-unfilled-after-N-days policy in
#   swing.py/swing_ibkr.py, meaning a ticker could get permanently stuck
#   waiting for a price that never comes back. Worth fixing there, not just
#   modelled around here.
# - Same-bar target/stop ambiguity resolution checks real intraday bars
#   first (asxbrief's `bars` table) but will almost always fall through to
#   the conservative default for older history, since intraday depth for
#   these tickers tops out ~3.5 months (and as of 2026-08-21, backfill for
#   the current focus tickers specifically hasn't been started at all --
#   `pct_resolved_conservatively` in the report will likely read at or near
#   100% until that changes).
