"""Concrete hypotheses, defined against real data and run through the harness.

Each one is a claim that looked plausible during the 2026-08 research. They
live here rather than in scratch scripts so that re-running them is one
command and the verdicts accumulate in the registry.
"""

from __future__ import annotations

import functools
from typing import Any

from .hypothesis import Hypothesis, run

COVID = ("2020-02-01", "2020-12-31")
SPLIT = "2021-01-01"


@functools.lru_cache(maxsize=4)
def _daily(limit: int = 300, period: str = "10y"):
    import pandas as pd
    from .gap_study import _universe
    from .screener import bulk_daily
    from .session_split import MIN_PRICE, MIN_TURNOVER, _legs
    from .yf_lock import YF_LOCK
    tickers = _universe(limit)
    with YF_LOCK:
        hist = bulk_daily(tickers, period=period, batch=40)
    out = {}
    for i, t in enumerate(tickers):
        df = hist.get(t)
        if df is None or len(df) < 500:
            continue
        if float(df["Close"].median()) < MIN_PRICE:
            continue
        if float((df["Close"] * df["Volume"]).median()) < MIN_TURNOVER:
            continue
        legs = _legs(df).dropna()
        legs = legs[~((legs.index >= COVID[0]) & (legs.index <= COVID[1]))]
        out[t] = {"rank": i + 1, "legs": legs, "close": df["Close"]}
    return out


def _gap_trades(lo: int, hi: int, thresh: float, side: int):
    """Gap-conditioned day trades, entered at the open and exited at the close."""
    import pandas as pd
    data = _daily()
    rows = []
    for t, d in data.items():
        if not (lo <= d["rank"] <= hi):
            continue
        legs = d["legs"]
        gap, day = legs["overnight"] * 100, legs["daytime"] * 100
        m = (gap < -thresh) if side > 0 else (gap > thresh)
        px = d["close"].reindex(legs.index)
        for dt in legs.index[m]:
            rows.append({"date": dt.date().isoformat(), "ticker": t, "side": side,
                         "gross_pct": float(day.loc[dt]) * side,
                         "price": float(px.loc[dt]) if px.loc[dt] == px.loc[dt] else None,
                         "is_train": dt.date().isoformat() < SPLIT})
    return pd.DataFrame(rows)


def _both_sides(lo: int, hi: int, thresh: float):
    import pandas as pd
    return pd.concat([_gap_trades(lo, hi, thresh, +1),
                      _gap_trades(lo, hi, thresh, -1)], ignore_index=True)


def _overnight_etf(ticker: str, period: str = "5y"):
    """Buy the ETF at each close, sell at the next open."""
    import pandas as pd
    import yfinance as yf
    from .yf_lock import YF_LOCK
    with YF_LOCK:
        d = yf.download(f"{ticker}.AX", period=period, interval="1d",
                        auto_adjust=True, progress=False)
    if hasattr(d.columns, "levels"):
        d.columns = d.columns.droplevel(1)
    d = d.dropna(subset=["Close"])
    ov = (d["Open"] / d["Close"].shift(1) - 1).dropna() * 100
    return pd.DataFrame({
        "date": [x.date().isoformat() for x in ov.index],
        "ticker": ticker, "side": 1, "gross_pct": ov.to_numpy(),
        "price": d["Close"].reindex(ov.index).to_numpy(),
        "is_train": [x.date().isoformat() < "2024-01-01" for x in ov.index]})


def _daytime_band(lo: int, hi: int):
    """Hold every name in the band from open to close -- the intraday leg."""
    import pandas as pd
    rows = []
    for t, d in _daily().items():
        if not (lo <= d["rank"] <= hi):
            continue
        legs = d["legs"]
        px = d["close"].reindex(legs.index)
        for dt in legs.index:
            rows.append({"date": dt.date().isoformat(), "ticker": t, "side": 1,
                         "gross_pct": float(legs["daytime"].loc[dt]) * 100,
                         "price": float(px.loc[dt]) if px.loc[dt] == px.loc[dt] else None,
                         "is_train": dt.date().isoformat() < SPLIT})
    return pd.DataFrame(rows)


HYPOTHESES: list[Hypothesis] = [
    Hypothesis(
        name="gap_reversion_asx50_open",
        description="Buy ASX 50 names gapping down >3% at the OPEN, exit at the close.",
        build=lambda: _gap_trades(1, 50, 3.0, +1),
        signal_known_at="after the opening auction has printed",
        entry_at="the opening auction price",
        entry_is_tradeable=False,
        tradeable_note="the auction clears at one price and orders must be in "
                       "before it clears, so the gap is not observable in time",
        universe="today's top 50 by market cap", point_in_time=False,
        survivorship_note="buying gap-downs is exactly what delisted names would have done",
        realistic_cost_pct=0.05, n_variants=12, n_passed=1, tags=["gap", "asx50"]),
    Hypothesis(
        name="gap_reversion_smallcap_both_sides",
        description="Rank 101-300, gap >3% either way, entered at the open.",
        build=lambda: _both_sides(101, 300, 3.0),
        signal_known_at="after the opening auction has printed",
        entry_at="the opening auction price",
        entry_is_tradeable=False,
        universe="today's rank 101-300", point_in_time=False,
        realistic_cost_pct=0.20, n_variants=12, n_passed=2, tags=["gap", "smallcap"]),
    Hypothesis(
        name="overnight_hold_vas",
        description="Buy VAS at each close, sell at the next open. The ASX overnight "
                    "anomaly, on the cheapest instrument that expresses it.",
        build=lambda: _overnight_etf("VAS"),
        signal_known_at="no signal -- unconditional, every night",
        entry_at="the closing auction", entry_is_tradeable=True,
        tradeable_note="both legs are auction prints, genuinely executable",
        universe="VAS (ASX 300 ETF)", point_in_time=True,
        realistic_cost_pct=0.029, n_variants=3, n_passed=2,
        tags=["overnight", "etf"]),
    Hypothesis(
        name="overnight_hold_stw",
        description="Same trade on STW, as an independent read of the same index.",
        build=lambda: _overnight_etf("STW"),
        signal_known_at="no signal -- unconditional, every night",
        entry_at="the closing auction", entry_is_tradeable=True,
        universe="STW (ASX 200 ETF)", point_in_time=True,
        realistic_cost_pct=0.033, n_variants=3, n_passed=2,
        tags=["overnight", "etf"]),
    Hypothesis(
        name="daytime_hold_asx50",
        description="Hold the ASX 50 from open to close every session -- the other "
                    "half of the clock, as the control on the overnight trade.",
        build=lambda: _daytime_band(1, 50),
        signal_known_at="no signal -- unconditional, every session",
        entry_at="the opening auction", entry_is_tradeable=True,
        universe="today's top 50 by market cap", point_in_time=False,
        n_variants=1, n_passed=0, tags=["intraday", "asx50"]),
]


def run_all(store: bool = True) -> list[dict[str, Any]]:
    return [run(h, store=store) for h in HYPOTHESES]
