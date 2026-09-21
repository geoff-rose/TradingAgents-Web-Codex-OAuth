"""Pure-pandas indicators for the `/setups` scanner. No I/O, no state.

Conventions, chosen to match what a chart-watching user will see on
TradingView so a detection here agrees with what they would call by eye:
EMA seeded with the SMA of its first `n` closes then recursed with
`adjust=False`; RSI with Wilder smoothing (`strategies._rsi_series` is
SMA-based and deliberately left alone -- the backtest strategies were
measured with it); Bollinger with population stdev (ddof=0).

`indicator_frame()` is the one entry point the detectors use. Every column
is causal at its own bar; anything that must only be known AFTER the bar --
`hh20`, `ll20`, `vol_med20` -- is shifted by one so a breakout compares
today against the PRIOR twenty sessions, never against itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    """SMA-seeded EMA. Bars before the seed are NaN."""
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if len(s) < n:
        return out
    vals = s.to_numpy(dtype=float)
    seed_at = n - 1
    k = 2.0 / (n + 1)
    prev = float(np.mean(vals[:n]))
    res = np.full(len(vals), np.nan)
    res[seed_at] = prev
    for i in range(n, len(vals)):
        prev = vals[i] * k + prev * (1 - k)
        res[i] = prev
    out[:] = res
    return out


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # First average is a plain mean of the first n changes, then Wilder's
    # recursion -- identical to ewm(alpha=1/n, adjust=False) seeded that way.
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # All-gain windows: avg_loss == 0 -> RSI 100, not NaN.
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return rsi


def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    line = ema(close, fast) - ema(close, slow)
    valid = line.dropna()
    signal = pd.Series(np.nan, index=close.index, dtype=float)
    if len(valid):
        signal.loc[valid.index] = ema(valid, sig)
    hist = line - signal
    return line, signal, hist


def trix(close: pd.Series, n: int = 15, sig: int = 9):
    e1 = ema(close, n).dropna()
    e2 = ema(e1, n).dropna()
    e3 = ema(e2, n).dropna()
    t = e3.pct_change() * 100.0
    t = t.dropna()
    s = ema(t, sig)
    out_t = pd.Series(np.nan, index=close.index, dtype=float)
    out_s = pd.Series(np.nan, index=close.index, dtype=float)
    out_t.loc[t.index] = t
    out_s.loc[s.index] = s
    return out_t, out_s


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    upper = mid + k * sd
    lower = mid - k * sd
    width = upper - lower
    pct_b = (close - lower) / width.replace(0.0, np.nan)
    bandwidth = width / mid.replace(0.0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
                         "bb_pct_b": pct_b, "bb_bandwidth": bandwidth})


def indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All the series the detectors read, aligned to `df`'s index.

    `df` needs Open/High/Low/Close/Volume. Returns a new frame; `df` is not
    modified.
    """
    from .strategies import _atr_series

    close = df["Close"].astype(float)
    ind = pd.DataFrame(index=df.index)
    ind["open"] = df["Open"].astype(float)
    ind["high"] = df["High"].astype(float)
    ind["low"] = df["Low"].astype(float)
    ind["close"] = close
    ind["volume"] = df["Volume"].astype(float)
    ind["ema20"] = ema(close, 20)
    ind["ema50"] = ema(close, 50)
    ind["rsi14"] = rsi_wilder(close, 14)
    _, _, ind["macd_hist"] = macd(close)
    ind["trix"], ind["trix_sig"] = trix(close)
    ind = ind.join(bollinger(close))
    ind["atr14"] = _atr_series(df, 14)
    ind["atr_pct"] = ind["atr14"] / close * 100.0
    # Prior-window extremes: shifted so today's bar is never its own level.
    ind["hh20"] = ind["high"].rolling(20).max().shift(1)
    ind["ll20"] = ind["low"].rolling(20).min().shift(1)
    ind["vol_med20"] = ind["volume"].rolling(20).median().shift(1)
    ind["vol_ratio"] = ind["volume"] / ind["vol_med20"].replace(0.0, np.nan)
    ind["turnover_med20"] = (close * ind["volume"]).rolling(20).median()
    ind["ret_3"] = close.pct_change(3) * 100.0
    ind["ret_60"] = close.pct_change(60) * 100.0
    # Six-month bandwidth floor, used by the squeeze detector. min_periods
    # lets a 2y frame answer from bar ~130 rather than waiting for a full
    # window.
    ind["bw_min_126"] = ind["bb_bandwidth"].rolling(126, min_periods=60).min()
    return ind


def swing_pivots(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Swing highs/lows: a bar whose High is the max (Low the min) of the
    2k+1 bars centred on it. `known_at = idx + k` is the first bar at which
    the pivot is knowable -- the k bars after it must have printed. Every
    consumer must respect that, or the geometric detectors look ahead.

    Ties keep the LAST index (a flat top is pivoted at its last bar), which
    is deterministic and errs towards the later, i.e. safer, known_at.
    """
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    n = len(df)
    rows = []
    for i in range(k, n - k):
        hw = high[i - k:i + k + 1]
        lw = low[i - k:i + k + 1]
        hmax = hw.max()
        lmin = lw.min()
        if high[i] == hmax and not np.any(hw[k + 1:] == hmax):
            rows.append((i, 1, float(high[i]), i + k))
        if low[i] == lmin and not np.any(lw[k + 1:] == lmin):
            rows.append((i, -1, float(low[i]), i + k))
    return pd.DataFrame(rows, columns=["idx", "kind", "price", "known_at"])
