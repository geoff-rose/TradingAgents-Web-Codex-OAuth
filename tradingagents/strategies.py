"""Strategy library for the phase-4 `/backtest` page
(`/root/asxbrief-phase4-backtest-page.md` §3) -- transparent, named-constant
strategies, each declaring parameters with sensible defaults.

Every strategy (except buy-and-hold, which is a benchmark, not a signal
generator) implements
`generate_signals(ticker, df, params, events=None) -> list[Signal]`,
reusing `backtest.py`'s `Signal`/`TradeResult`/`simulate_trade()` machinery
UNCHANGED -- `simulate_trade()` already generically handles fill/exit/
tick-rounding/same-bar-ambiguity/costs for any (entry, target, stop) proposal
at a given date, regardless of which rule produced it. Adding a strategy
here should never require touching the fill model.

**`events` is the event-driven hook** (added 2026-08-22, per explicit user
design requirement: "make sure the strategy interface can express
event-driven strategies, not just price-pattern ones... retrofitting it
later would be painful"). It's an optional `pd.DataFrame` of
`{event_date, event_type, detail}` rows for that ticker -- e.g. announcement
publication dates from the phase-1 collector (see `fetch_events()` below,
which already wires this to real data for AU tickers, not a hollow stub).
Every strategy in this file today is price-pattern-only and ignores it
(default `None`, backward compatible) -- but the seam exists now: a future
strategy like "enter on a price-sensitive announcement, ATR target/stop"
just reads `events` instead of (or alongside) `df`'s indicators, and still
returns the same `list[Signal]`, so it flows through `run_and_report()`,
the fill model, the honesty layer, and the page UI with zero changes to any
of them. This is the actual point of the hook: don't make event-driven
strategies a second, parallel system later.

No-pyramiding is enforced the same way for every strategy: once a signal
fires, no new signal is generated for that ticker until MAX_HOLD_DAYS have
passed (matching how the live system's `swing_db.has_active_trade` blocks
overlapping proposals) -- each `generate_signals` implementation below
follows this `blocked_until` pattern; keep it if adding a new strategy.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from tradingagents.backtest import (
    ATR_PERIOD, MAX_HOLD_DAYS, MAX_GAIN_PCT, MIN_GAIN_PCT, PULLBACK_BAND_PCT,
    PULLBACK_LOOKBACK, SMA_PERIOD, STOP_ATR_MULT, TARGET_ATR_MULT,
    Signal, generate_signals as _pullback_generate_signals, round_to_tick,
)

_ANNOUNCEMENTS_DB_PATH = Path("/opt/asxbrief/data/asx.db")


def fetch_events(ticker: str, market: str = "AU") -> pd.DataFrame:
    """Real event data for AU tickers -- announcement publication dates from
    the phase-1 collector (`asx-collector`/`asxbrief`), read-only, same DB
    `asx_feed.py` reads. `event_type` is 'halt', 'price_sensitive', or
    'announcement' (in that priority order per day -- a day can have more
    than one announcement; this collapses to the most notable type per day,
    which is enough for an event-driven strategy's entry trigger). Empty
    DataFrame for US tickers (no source wired yet) or if the announcements
    DB isn't reachable -- callers must handle the empty case, not assume
    events exist."""
    columns = ["event_date", "event_type", "detail"]
    if market != "AU" or not _ANNOUNCEMENTS_DB_PATH.exists():
        return pd.DataFrame(columns=columns)
    try:
        conn = sqlite3.connect(f"file:{_ANNOUNCEMENTS_DB_PATH}?mode=ro", uri=True, timeout=5.0)
        rows = conn.execute(
            "SELECT released_at, headline, is_halt, price_sensitive FROM announcements "
            "WHERE ticker=? ORDER BY released_at", (ticker,),
        ).fetchall()
        conn.close()
    except Exception:
        return pd.DataFrame(columns=columns)
    if not rows:
        return pd.DataFrame(columns=columns)
    out = []
    for released_at, headline, is_halt, price_sensitive in rows:
        event_type = "halt" if is_halt else ("price_sensitive" if price_sensitive else "announcement")
        # tz-naive to match fetch_daily_history()'s df.index (backtest.py
        # strips tz info) -- a tz-aware/naive mismatch here silently fails
        # every `day in trigger_dates` membership check with no error at
        # all, which is exactly what happened testing this live: real events
        # existed, price data existed, and it still produced zero trades.
        out.append({"event_date": pd.Timestamp(released_at).tz_localize(None).normalize(),
                    "event_type": event_type, "detail": headline})
    return pd.DataFrame(out, columns=columns)


def _atr_series(df: pd.DataFrame, period: int) -> pd.Series:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _rsi_series(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


# ---------------------------------------------------------------------------
# Strategy 1: pullback in uptrend (the existing swing_signal.py rule)
# ---------------------------------------------------------------------------

def pullback_in_uptrend_signals(ticker: str, df: pd.DataFrame, params: dict[str, Any], events: pd.DataFrame | None = None) -> list[Signal]:
    """Delegates straight to backtest.py's own generate_signals() -- this
    strategy already has a point-in-time, no-lookahead implementation there,
    with parameters currently as swing_signal.py's module constants, not yet
    parameterized. `params` is accepted for interface consistency but unused
    until swing_signal.py's constants are made configurable -- flagged, not
    silently ignored."""
    return _pullback_generate_signals(ticker, df)


# ---------------------------------------------------------------------------
# Strategy 2: SMA crossover (long-only -- this project has deliberately
# deferred shorting, per user's own "ignore shorting for now")
# ---------------------------------------------------------------------------

def sma_crossover_signals(ticker: str, df: pd.DataFrame, params: dict[str, Any], events: pd.DataFrame | None = None) -> list[Signal]:
    fast_n = params.get("fast", 20)
    slow_n = params.get("slow", 50)
    target_mult = params.get("target_atr_mult", TARGET_ATR_MULT)
    stop_mult = params.get("stop_atr_mult", STOP_ATR_MULT)

    if len(df) < slow_n + ATR_PERIOD + 5:
        return []
    close = df["Close"]
    fast = close.rolling(fast_n).mean()
    slow = close.rolling(slow_n).mean()
    atr = _atr_series(df, ATR_PERIOD)
    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))

    signals = []
    blocked_until = None
    for i in range(slow_n, len(df) - 1):
        day = df.index[i]
        if blocked_until is not None and day <= blocked_until:
            continue
        if not cross_up.iloc[i] or pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0:
            continue
        entry = round_to_tick(close.iloc[i])
        target = round_to_tick(entry + target_mult * atr.iloc[i])
        stop = round_to_tick(entry - stop_mult * atr.iloc[i])
        signals.append(Signal(ticker=ticker, signal_date=day, attempt_date=df.index[i + 1],
                               entry=entry, target=target, stop=stop))
        blocked_until = df.index[min(i + 1 + MAX_HOLD_DAYS, len(df) - 1)]
    return signals


# ---------------------------------------------------------------------------
# Strategy 3: RSI mean reversion
# ---------------------------------------------------------------------------

def rsi_mean_reversion_signals(ticker: str, df: pd.DataFrame, params: dict[str, Any], events: pd.DataFrame | None = None) -> list[Signal]:
    period = params.get("rsi_period", 14)
    buy_below = params.get("buy_below", 30)
    target_mult = params.get("target_atr_mult", TARGET_ATR_MULT)
    stop_mult = params.get("stop_atr_mult", STOP_ATR_MULT)

    if len(df) < period + ATR_PERIOD + 5:
        return []
    close = df["Close"]
    rsi = _rsi_series(close, period)
    atr = _atr_series(df, ATR_PERIOD)

    signals = []
    blocked_until = None
    for i in range(period + ATR_PERIOD, len(df) - 1):
        day = df.index[i]
        if blocked_until is not None and day <= blocked_until:
            continue
        if pd.isna(rsi.iloc[i]) or rsi.iloc[i] >= buy_below or pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0:
            continue
        entry = round_to_tick(close.iloc[i])
        target = round_to_tick(entry + target_mult * atr.iloc[i])
        stop = round_to_tick(entry - stop_mult * atr.iloc[i])
        signals.append(Signal(ticker=ticker, signal_date=day, attempt_date=df.index[i + 1],
                               entry=entry, target=target, stop=stop))
        blocked_until = df.index[min(i + 1 + MAX_HOLD_DAYS, len(df) - 1)]
    return signals


# ---------------------------------------------------------------------------
# Strategy 4: N-day breakout
# ---------------------------------------------------------------------------

def n_day_breakout_signals(ticker: str, df: pd.DataFrame, params: dict[str, Any], events: pd.DataFrame | None = None) -> list[Signal]:
    lookback = params.get("lookback_days", 20)
    target_mult = params.get("target_atr_mult", TARGET_ATR_MULT)
    stop_mult = params.get("stop_atr_mult", STOP_ATR_MULT)

    if len(df) < lookback + ATR_PERIOD + 5:
        return []
    close = df["Close"]
    high = df["High"]
    rolling_high = high.rolling(lookback).max()
    atr = _atr_series(df, ATR_PERIOD)
    new_high = close > rolling_high.shift(1)

    signals = []
    blocked_until = None
    for i in range(lookback + ATR_PERIOD, len(df) - 1):
        day = df.index[i]
        if blocked_until is not None and day <= blocked_until:
            continue
        if not new_high.iloc[i] or pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0:
            continue
        entry = round_to_tick(close.iloc[i])
        target = round_to_tick(entry + target_mult * atr.iloc[i])
        stop = round_to_tick(entry - stop_mult * atr.iloc[i])
        signals.append(Signal(ticker=ticker, signal_date=day, attempt_date=df.index[i + 1],
                               entry=entry, target=target, stop=stop))
        blocked_until = df.index[min(i + 1 + MAX_HOLD_DAYS, len(df) - 1)]
    return signals


# ---------------------------------------------------------------------------
# Strategy 5: announcement reaction -- EVENT-DRIVEN, not price-pattern.
# Proves the `events` hook actually works end-to-end (real data, real
# signals) rather than being an interface that's never exercised.
# ---------------------------------------------------------------------------

def announcement_reaction_signals(ticker: str, df: pd.DataFrame, params: dict[str, Any],
                                   events: pd.DataFrame | None = None) -> list[Signal]:
    """Enter the day after a price-sensitive announcement (or a halt lifting,
    per `event_types` param), ATR target/stop -- the simplest possible
    event-driven strategy, deliberately: the point right now is proving the
    interface, not finding an edge. `df` is still needed for the ATR/entry
    price levels even though the trigger itself comes from `events`."""
    event_types = params.get("event_types", ["price_sensitive", "halt"])
    target_mult = params.get("target_atr_mult", TARGET_ATR_MULT)
    stop_mult = params.get("stop_atr_mult", STOP_ATR_MULT)

    if events is None or events.empty or len(df) < ATR_PERIOD + 5:
        return []
    atr = _atr_series(df, ATR_PERIOD)
    close = df["Close"]
    trigger_dates = set(events.loc[events["event_type"].isin(event_types), "event_date"])

    signals = []
    blocked_until = None
    for i in range(ATR_PERIOD, len(df) - 1):
        day = df.index[i]
        if blocked_until is not None and day <= blocked_until:
            continue
        if day not in trigger_dates or pd.isna(atr.iloc[i]) or atr.iloc[i] <= 0:
            continue
        entry = round_to_tick(close.iloc[i])
        target = round_to_tick(entry + target_mult * atr.iloc[i])
        stop = round_to_tick(entry - stop_mult * atr.iloc[i])
        signals.append(Signal(ticker=ticker, signal_date=day, attempt_date=df.index[i + 1],
                               entry=entry, target=target, stop=stop))
        blocked_until = df.index[min(i + 1 + MAX_HOLD_DAYS, len(df) - 1)]
    return signals


@dataclass
class StrategyDef:
    name: str
    default_params: dict[str, Any]
    generate_signals: Callable[[str, pd.DataFrame, dict[str, Any], pd.DataFrame | None], list[Signal]]
    description: str
    event_driven: bool = False


STRATEGIES: dict[str, StrategyDef] = {
    "pullback_in_uptrend": StrategyDef(
        name="Pullback in uptrend", default_params={},
        generate_signals=pullback_in_uptrend_signals,
        description="close > SMA20, within 2% of 10-day low, ATR target/stop (swing_signal.py's live rule).",
    ),
    "sma_crossover": StrategyDef(
        name="SMA crossover", default_params={"fast": 20, "slow": 50, "target_atr_mult": TARGET_ATR_MULT, "stop_atr_mult": STOP_ATR_MULT},
        generate_signals=sma_crossover_signals,
        description="Long on fast SMA crossing above slow SMA, ATR target/stop.",
    ),
    "rsi_mean_reversion": StrategyDef(
        name="RSI mean reversion", default_params={"rsi_period": 14, "buy_below": 30, "target_atr_mult": TARGET_ATR_MULT, "stop_atr_mult": STOP_ATR_MULT},
        generate_signals=rsi_mean_reversion_signals,
        description="Long when RSI drops below a threshold, ATR target/stop.",
    ),
    "n_day_breakout": StrategyDef(
        name="N-day breakout", default_params={"lookback_days": 20, "target_atr_mult": TARGET_ATR_MULT, "stop_atr_mult": STOP_ATR_MULT},
        generate_signals=n_day_breakout_signals,
        description="Long on a new N-day closing high, ATR target/stop.",
    ),
    "announcement_reaction": StrategyDef(
        name="Announcement reaction", default_params={"event_types": ["price_sensitive", "halt"], "target_atr_mult": TARGET_ATR_MULT, "stop_atr_mult": STOP_ATR_MULT},
        generate_signals=announcement_reaction_signals,
        description="Event-driven (AU only): enter the day after a price-sensitive announcement or halt, ATR target/stop.",
        event_driven=True,
    ),
}

# NOT YET IMPLEMENTED (spec §3), following the same pattern when added:
# Bollinger reversion, Buy the dip, Gap fade/continuation. Range model
# (spec table) is intentionally NOT in this registry -- its daily-refresh
# mechanic doesn't fit the static-Signal shape these strategies share; it's
# evaluated via range_model.py's own evaluate_daily_refresh() instead. The
# /backtest page's runner (backtest_report.py) special-cases it for that
# reason -- don't force it into this registry without redesigning the
# daily-refresh mechanic to fit, which isn't a small change.
