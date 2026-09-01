"""Daily swing-trade proposal heuristic.

Deliberately simple and transparent: an ATR-based pullback-in-uptrend setup
on daily OHLC (yfinance), not the day-shape/clustering engine from the
phase-2 spec. That engine needs per-ticker intraday bar history to find
recurring patterns, and IBKR's own historical 1-min bar depth for smaller
ASX names tops out at roughly 3.5 months (confirmed live during the phase-2
backfill, see the asxbrief-phase2 skill) -- nowhere near enough to trust a
clustered pattern without serious overfitting risk on 5 brand-new tickers.
This heuristic has no proven edge either -- that's what the Swing page's own
P&L table is for judging, the same honest standard the (still unproven) AI
announcement-signal score is held to.
"""

from __future__ import annotations

from typing import Any

import yfinance as yf

ATR_PERIOD = 14
SMA_PERIOD = 20
PULLBACK_LOOKBACK = 10
PULLBACK_BAND_PCT = 2.0  # must be within this % of the recent low
TARGET_ATR_MULT = 1.5
STOP_ATR_MULT = 1.0
MIN_GAIN_PCT = 1.0  # "small gain" band on the target, not a moonshot
MAX_GAIN_PCT = 6.0


def _atr(highs: list[float], lows: list[float], closes: list[float], period: int = ATR_PERIOD) -> float | None:
    if len(closes) < period + 1:
        return None
    trs = [
        max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        for i in range(1, len(closes))
    ]
    return sum(trs[-period:]) / period


def evaluate(ticker: str) -> dict[str, Any] | None:
    """Long-only pullback-in-uptrend setup, or None if today doesn't qualify.
    `ticker` is the bare ASX code (e.g. "CBA"); ".AX" is appended for yfinance."""
    try:
        hist = yf.Ticker(f"{ticker}.AX").history(period="4mo")
    except Exception:
        return None
    if len(hist) < SMA_PERIOD + ATR_PERIOD:
        return None

    closes = hist["Close"].tolist()
    highs = hist["High"].tolist()
    lows = hist["Low"].tolist()

    sma20 = sum(closes[-SMA_PERIOD:]) / SMA_PERIOD
    last_close = closes[-1]
    atr = _atr(highs, lows, closes)
    if not atr or atr <= 0:
        return None

    recent_low = min(lows[-PULLBACK_LOOKBACK:])
    in_uptrend = last_close > sma20
    near_pullback_low = (last_close - recent_low) / recent_low * 100 < PULLBACK_BAND_PCT

    if not (in_uptrend and near_pullback_low):
        return None

    entry = last_close
    target = entry + TARGET_ATR_MULT * atr
    stop = entry - STOP_ATR_MULT * atr
    gain_pct = (target - entry) / entry * 100
    if not (MIN_GAIN_PCT <= gain_pct <= MAX_GAIN_PCT):
        return None

    return {
        "ticker": ticker,
        "entry_price": round(entry, 3),
        "target_price": round(target, 3),
        "stop_price": round(stop, 3),
        "rationale": (
            f"close ({last_close:.3f}) > {SMA_PERIOD}d SMA ({sma20:.3f}), "
            f"within {PULLBACK_BAND_PCT:.0f}% of {PULLBACK_LOOKBACK}d low ({recent_low:.3f}); "
            f"ATR{ATR_PERIOD}={atr:.3f}, target +{gain_pct:.1f}%"
        ),
    }
