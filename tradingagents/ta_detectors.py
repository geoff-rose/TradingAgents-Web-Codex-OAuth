"""Setup definitions for the `/setups` scanner: what counts as a pattern
*starting to form*, and what counts as confirmed.

Two tiers. Tier 1 is indicator-defined and vectorised -- every rule is a
boolean Series over the whole frame, so the historical backfill is a single
pass per ticker and, more importantly, the live scan and the backfill run the
IDENTICAL code on a bar. Tier 2 is geometric (swing-pivot shapes) and is
flagged unvalidated on the page by construction: the tolerances are fixed by
hand, not fitted, and there is no published evidence worth citing for them.

Each detection is one of two `state`s. "forming" is the earlier, riskier
call -- price is approaching the level with the supporting condition in
place. "confirmed" is the textbook trigger (a close through the level). The
scorecard keeps them apart because they are different bets: forming gets in
before the crowd and is wrong more often.

Rules for the RSI/MACD/EMA20/TRIX family follow the exhaustion/rebound flags
in github.com/Oft3r/agentic-trading-desk (MIT, `score.py::_flags`),
re-expressed as Series arithmetic.

**No lookahead.** Every input is causal at its bar (see `ta_indicators`),
and pivots are only consulted once `known_at <= t`. `test_ta_setups.py`
checks that `detect_all(df[:t+1])` equals `detect_history(df)` at `t`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .session_split import MIN_PRICE, MIN_TURNOVER
from .ta_indicators import indicator_frame, swing_pivots

MIN_BARS = 260            # a year of history before a ticker is scanned
PIVOT_K = 5
TIER2_LOOKBACK = 60
STATES = ("forming", "confirmed")


@dataclass(frozen=True)
class SetupDef:
    id: str
    tier: int
    direction: int            # +1 long, -1 short
    label: str
    rule_text: str
    level_kind: str
    states: tuple[str, ...] = STATES


@dataclass
class Detection:
    setup_id: str
    tier: int
    ticker: str
    date: str
    direction: int
    state: str
    level: float
    level_kind: str
    price: float
    distance_pct: float
    context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "setup_id": self.setup_id, "tier": self.tier, "ticker": self.ticker,
            "date": self.date, "direction": self.direction, "state": self.state,
            "level": self.level, "level_kind": self.level_kind, "price": self.price,
            "distance_pct": self.distance_pct, "context": self.context,
        }


SETUPS: dict[str, SetupDef] = {s.id: s for s in [
    SetupDef("rsi_turn_up", 1, +1, "RSI turn from oversold",
             "RSI-14 (Wilder) was below 35 yesterday, is higher today and still below 45. Forming only -- the 'confirmation' (RSI back above 50) is what "
             "the forward return measures.", "20-day low", states=("forming",)),
    SetupDef("rsi_exhaust", 1, -1, "RSI exhaustion",
             "RSI-14 was at or above 70 yesterday and is falling today, with price either "
             "at/above the upper Bollinger band (%B >= 1) or stretched >= 10% over EMA20. "
             "Forming only.", "EMA20", states=("forming",)),
    SetupDef("macd_hist_up", 1, +1, "MACD histogram turning up",
             "Histogram negative, rising two bars in a row, and its magnitude at most half of "
             "two bars ago (forming). Confirmed: histogram crosses above zero.", "EMA20"),
    SetupDef("macd_hist_down", 1, -1, "MACD histogram turning down",
             "Histogram positive, falling two bars in a row, magnitude at most half of two "
             "bars ago (forming). Confirmed: histogram crosses below zero.", "EMA20"),
    SetupDef("ema20_reclaim", 1, +1, "EMA20 reclaim after a dip",
             "Closed below EMA20 within the last 5 bars, EMA20 rising over 10 bars. Forming: "
             "close within 1% below EMA20 on an up bar. Confirmed: the first close back above EMA20.",
             "EMA20"),
    SetupDef("ema20_loss", 1, -1, "EMA20 loss after a rally",
             "Closed above EMA20 within the last 5 bars, EMA20 falling over 10 bars. Forming: "
             "close within 1% above EMA20 on a down bar. Confirmed: the first close below EMA20.",
             "EMA20"),
    SetupDef("trix_cross_up", 1, +1, "TRIX bullish cross below zero",
             "TRIX-15 below zero and under its 9-signal; the gap narrowed two bars running to "
             "at most 30% of its size three bars ago (forming). Confirmed: TRIX crosses above "
             "the signal while still below zero.", "EMA20"),
    SetupDef("donchian20_up", 1, +1, "20-day high breakout",
             "Today's high within 1% of the prior 20-day high on volume >= 1.2x the 20-day "
             "median (forming). Confirmed: close above the prior 20-day high on that volume.",
             "prior 20-day high"),
    SetupDef("donchian20_down", 1, -1, "20-day low breakdown",
             "Today's low within 1% of the prior 20-day low on volume >= 1.2x the 20-day "
             "median (forming). Confirmed: close below the prior 20-day low on that volume.",
             "prior 20-day low"),
    SetupDef("bb_squeeze_up", 1, +1, "Bollinger squeeze release (up)",
             "Bandwidth hit its 6-month low within the last 5 bars and is now expanding, with "
             "price in the upper quarter of the bands (forming). Confirmed: close above the "
             "upper band.", "upper band"),
    SetupDef("bb_squeeze_down", 1, -1, "Bollinger squeeze release (down)",
             "Bandwidth hit its 6-month low within the last 5 bars and is now expanding, with "
             "price in the lower quarter of the bands (forming). Confirmed: close below the "
             "lower band.", "lower band"),
    SetupDef("pullback_ema20", 1, +1, "Pullback to EMA20 in an uptrend",
             "EMA20 > EMA50, EMA50 rising over 10 bars, 60-day return > 10%. Forming: 3-day "
             "return negative and today's low touched EMA20 (within 1.5%) with the close held "
             "within 1.5% of it. Confirmed: a touch within the last 3 bars and today's close "
             "above yesterday's high.", "EMA20"),
    # ---- tier 2: geometric, unvalidated ----
    SetupDef("double_bottom", 2, +1, "Double bottom",
             "Two swing lows (k=5) 15-60 bars apart within 3% of each other, the intervening "
             "high (neckline) at least 5% above them, second low known and at most 40 bars "
             "ago. Forming: close between the second low and the neckline and rising over 2 "
             "bars. Confirmed: first close above the neckline.", "neckline"),
    SetupDef("double_top", 2, -1, "Double top",
             "Mirror of the double bottom: two swing highs within 3%, the intervening low "
             "(neckline) at least 5% below. Forming: close between neckline and second high "
             "and falling over 2 bars. Confirmed: first close below the neckline.", "neckline"),
    SetupDef("asc_triangle", 2, +1, "Ascending triangle",
             "At least two swing highs within 2.5% of a common resistance R over 20-60 bars, "
             "at least two strictly rising swing lows since the first of them, and the last "
             "low's distance to R at most 70% of the first's. Forming: close within 3% below "
             "R. Confirmed: first close above R.", "resistance"),
    SetupDef("bull_flag", 2, +1, "Bull flag",
             "A pole of >= 8% in at most 15 bars to a swing high, then 5-12 bars of "
             "consolidation retracing at most half the pole on average volume at most 0.8x "
             "the pole's. Forming: close within 3% below the flag high. Confirmed: first "
             "close above it.", "flag high"),
]}


def liquid_mask(ind: pd.DataFrame) -> pd.Series:
    return (ind["close"] >= MIN_PRICE) & (ind["turnover_med20"] >= MIN_TURNOVER)


def tier1_masks(ind: pd.DataFrame, vol_scale: float = 1.0) -> dict[str, dict[str, Any]]:
    """Boolean Series per setup and state, plus the level Series.

    `vol_scale` corrects a provisional bar's volume: at 15:40 only part of the
    day has traded, so the ratio to full prior sessions is divided by the
    measured fraction (see `ta_setups.provisional_vol_scale`). 1.0 for
    completed bars.
    """
    c, o, h, lo = ind["close"], ind["open"], ind["high"], ind["low"]
    rsi, hist = ind["rsi14"], ind["macd_hist"]
    e20, e50 = ind["ema20"], ind["ema50"]
    va = ind["vol_ratio"] / vol_scale
    out: dict[str, dict[str, Any]] = {}

    # RSI
    rsi_prev = rsi.shift(1)
    out["rsi_turn_up"] = {
        "forming": (rsi_prev < 35) & (rsi > rsi_prev) & (rsi < 45),
        "confirmed": pd.Series(False, index=ind.index),
        "level": ind["ll20"],
    }
    out["rsi_exhaust"] = {
        "forming": (rsi_prev >= 70) & (rsi < rsi_prev)
                   & ((ind["bb_pct_b"] >= 1.0) | (c >= 1.10 * e20)),
        "confirmed": pd.Series(False, index=ind.index),
        "level": e20,
    }

    # MACD histogram
    h1, h2 = hist.shift(1), hist.shift(2)
    out["macd_hist_up"] = {
        "forming": (hist < 0) & (hist > h1) & (h1 > h2) & (hist.abs() <= 0.5 * h2.abs()),
        "confirmed": (h1 < 0) & (hist >= 0),
        "level": e20,
    }
    out["macd_hist_down"] = {
        "forming": (hist > 0) & (hist < h1) & (h1 < h2) & (hist.abs() <= 0.5 * h2.abs()),
        "confirmed": (h1 > 0) & (hist <= 0),
        "level": e20,
    }

    # EMA20 reclaim / loss
    below_recent = (c < e20).astype(float).shift(1).rolling(5).max() == 1
    above_recent = (c > e20).astype(float).shift(1).rolling(5).max() == 1
    # A ten-bar slope reads the trend the dip interrupts; a three-bar slope
    # is dragged negative by the dip itself and the rule could never fire.
    slope_up = e20 > e20.shift(10)
    slope_down = e20 < e20.shift(10)
    out["ema20_reclaim"] = {
        "forming": below_recent & slope_up & (c >= 0.99 * e20) & (c < e20) & (c > o),
        "confirmed": below_recent & slope_up & (c > e20) & (c.shift(1) <= e20.shift(1)),
        "level": e20,
    }
    out["ema20_loss"] = {
        "forming": above_recent & slope_down & (c <= 1.01 * e20) & (c > e20) & (c < o),
        "confirmed": above_recent & slope_down & (c < e20) & (c.shift(1) >= e20.shift(1)),
        "level": e20,
    }

    # TRIX
    gap = ind["trix"] - ind["trix_sig"]
    g1, g2, g3 = gap.shift(1), gap.shift(2), gap.shift(3)
    out["trix_cross_up"] = {
        "forming": (ind["trix"] < 0) & (gap < 0) & (gap > g1) & (g1 > g2)
                   & (gap.abs() <= 0.3 * g3.abs()),
        "confirmed": (g1 < 0) & (gap > 0) & (ind["trix"] <= 0),
        "level": e20,
    }

    # Donchian
    hh, ll = ind["hh20"], ind["ll20"]
    out["donchian20_up"] = {
        "forming": (h >= 0.99 * hh) & (c < hh) & (va >= 1.2),
        "confirmed": (c > hh) & (va >= 1.2),
        "level": hh,
    }
    out["donchian20_down"] = {
        "forming": (lo <= 1.01 * ll) & (c > ll) & (va >= 1.2),
        "confirmed": (c < ll) & (va >= 1.2),
        "level": ll,
    }

    # Bollinger squeeze release
    bw = ind["bb_bandwidth"]
    squeeze = (bw.rolling(6).min() <= ind["bw_min_126"]) & (bw > bw.shift(1))
    pb = ind["bb_pct_b"]
    out["bb_squeeze_up"] = {
        "forming": squeeze & (pb >= 0.75) & (pb <= 1.0),
        "confirmed": squeeze & (pb > 1.0),
        "level": ind["bb_upper"],
    }
    out["bb_squeeze_down"] = {
        "forming": squeeze & (pb <= 0.25) & (pb >= 0.0),
        "confirmed": squeeze & (pb < 0.0),
        "level": ind["bb_lower"],
    }

    # Pullback in uptrend
    base = (e20 > e50) & (e50 > e50.shift(10)) & (ind["ret_60"] > 10)
    touch = (lo <= 1.015 * e20) & (c >= 0.985 * e20)
    touched_recent = touch.astype(float).shift(1).rolling(3).max() == 1
    out["pullback_ema20"] = {
        "forming": base & (ind["ret_3"] < 0) & touch,
        "confirmed": base & touched_recent & (c > h.shift(1)),
        "level": e20,
    }

    liquid = liquid_mask(ind)
    for sid, m in out.items():
        for st in STATES:
            m[st] = (m[st] & liquid).fillna(False).astype(bool)
    return out


# ---------------------------------------------------------------------------
# Tier 2: geometry on swing pivots
# ---------------------------------------------------------------------------

class Prepared:
    """Numpy views of everything the per-bar code reads, built once per
    ticker. Per-bar `iloc` across 300 tickers x 500 bars is what made the
    first cut of the backfill take minutes."""

    def __init__(self, ind: pd.DataFrame, pivots: pd.DataFrame, vol_scale: float = 1.0):
        self.ind = ind
        self.index = ind.index
        self.close = ind["close"].to_numpy(float)
        self.high = ind["high"].to_numpy(float)
        self.low = ind["low"].to_numpy(float)
        self.vol = ind["volume"].to_numpy(float).copy()
        if vol_scale != 1.0 and len(self.vol):
            # A provisional last bar carries only part of the day; without
            # this the bull flag's quiet-volume test reads it as quieter than
            # it is and leans towards firing at 15:40.
            self.vol[-1] = self.vol[-1] / vol_scale
        self.rsi = ind["rsi14"].to_numpy(float)
        self.vol_ratio = ind["vol_ratio"].to_numpy(float) / vol_scale
        self.atr_pct = ind["atr_pct"].to_numpy(float)
        self.ema20 = ind["ema20"].to_numpy(float)
        self.pct_b = ind["bb_pct_b"].to_numpy(float)
        self.liquid = liquid_mask(ind).fillna(False).to_numpy(bool)
        self.vol_scale = vol_scale
        masks = tier1_masks(ind, vol_scale)
        self.masks = {sid: {"forming": m["forming"].to_numpy(bool),
                            "confirmed": m["confirmed"].to_numpy(bool),
                            "level": m["level"].to_numpy(float)}
                      for sid, m in masks.items()}
        self.any_t1 = np.zeros(len(ind), dtype=bool)
        for m in self.masks.values():
            self.any_t1 |= m["forming"] | m["confirmed"]
        if pivots is None or pivots.empty:
            self.pv_idx = np.zeros(0, int); self.pv_kind = np.zeros(0, int)
            self.pv_price = np.zeros(0, float); self.pv_known = np.zeros(0, int)
        else:
            self.pv_idx = pivots["idx"].to_numpy(int)
            self.pv_kind = pivots["kind"].to_numpy(int)
            self.pv_price = pivots["price"].to_numpy(float)
            self.pv_known = pivots["known_at"].to_numpy(int)

    def date(self, i: int) -> str:
        return str(pd.Timestamp(self.index[i]).date())


def tier2_at(P: Prepared, at: int) -> list[dict[str, Any]]:
    """Geometric detections at bar index `at`, using only pivots known by then.

    Returns partial dicts (setup_id, state, level, context); the caller fills
    ticker/date/price. Each pattern reports at most one detection per bar.
    """
    if at < TIER2_LOOKBACK or not len(P.pv_idx):
        return []
    sel = (P.pv_known <= at) & (P.pv_idx >= at - TIER2_LOOKBACK)
    if not sel.any():
        return []
    idx, kind, price = P.pv_idx[sel], P.pv_kind[sel], P.pv_price[sel]
    close, high, low, vol = P.close, P.high, P.low, P.vol
    c, c1, c2 = close[at], close[at - 1], close[at - 2]
    d = P.date
    out: list[dict[str, Any]] = []

    lows_i = idx[kind == -1]
    lows_p = price[kind == -1]
    highs_i = idx[kind == 1]
    highs_p = price[kind == 1]

    # Double bottom
    if len(lows_i) >= 2:
        i1, i2 = lows_i[-2], lows_i[-1]
        p1, p2 = lows_p[-2], lows_p[-1]
        gap = i2 - i1
        if 15 <= gap <= 60 and abs(p2 - p1) / p1 <= 0.03 and at - i2 <= 40:
            neck = float(high[i1:i2 + 1].max())
            if neck >= 1.05 * min(p1, p2):
                ctx = {"low1": d(i1), "low2": d(i2), "low_px": round(float(min(p1, p2)), 4)}
                if p2 < c < neck and c > c2:
                    out.append({"setup_id": "double_bottom", "state": "forming",
                                "level": neck, "context": ctx})
                elif c > neck >= c1:
                    out.append({"setup_id": "double_bottom", "state": "confirmed",
                                "level": neck, "context": ctx})

    # Double top
    if len(highs_i) >= 2:
        i1, i2 = highs_i[-2], highs_i[-1]
        p1, p2 = highs_p[-2], highs_p[-1]
        gap = i2 - i1
        if 15 <= gap <= 60 and abs(p2 - p1) / p1 <= 0.03 and at - i2 <= 40:
            neck = float(low[i1:i2 + 1].min())
            if neck <= 0.95 * max(p1, p2):
                ctx = {"high1": d(i1), "high2": d(i2), "high_px": round(float(max(p1, p2)), 4)}
                if neck < c < p2 and c < c2:
                    out.append({"setup_id": "double_top", "state": "forming",
                                "level": neck, "context": ctx})
                elif c < neck <= c1:
                    out.append({"setup_id": "double_top", "state": "confirmed",
                                "level": neck, "context": ctx})

    # Ascending triangle
    if len(highs_i) >= 2:
        # Take the longest run of trailing highs that agree within 1.5%.
        used = None
        for n_h in (4, 3, 2):
            if len(highs_i) < n_h:
                continue
            hp = highs_p[-n_h:]
            r = float(hp.mean())
            if (hp.max() - hp.min()) / r <= 0.025:
                used = n_h
                break
        if used:
            first_h = highs_i[-used]
            span = at - first_h
            r = float(highs_p[-used:].mean())
            lo_sel = lows_i >= first_h
            li, lp = lows_i[lo_sel], lows_p[lo_sel]
            if 20 <= span <= 60 and len(li) >= 2 and np.all(np.diff(lp) > 0) \
                    and (r - lp[-1]) <= 0.7 * (r - lp[0]) and r > lp[-1]:
                ctx = {"resistance": round(r, 4), "n_highs": int(used),
                       "first_high": d(first_h), "last_low": d(li[-1])}
                if 0.97 * r <= c < r:
                    out.append({"setup_id": "asc_triangle", "state": "forming",
                                "level": r, "context": ctx})
                elif c > r >= c1:
                    out.append({"setup_id": "asc_triangle", "state": "confirmed",
                                "level": r, "context": ctx})

    # Bull flag
    if len(highs_i) >= 1:
        pi, pp = highs_i[-1], highs_p[-1]
        flag_len = at - pi
        if 3 <= flag_len <= 12 and pi >= 15:
            pole_lo = float(low[pi - 15:pi + 1].min())
            pole_h = pp - pole_lo
            if pole_lo > 0 and pp / pole_lo - 1 >= 0.08:
                flag_low = float(low[pi + 1:at + 1].min())
                flag_high = float(high[pi:at].max())
                pole_vol = float(np.nanmean(vol[pi - 15:pi + 1]))
                flag_vol = float(np.nanmean(vol[pi + 1:at + 1]))
                if (pp - flag_low) <= 0.5 * pole_h and pole_vol > 0 \
                        and flag_vol <= 0.8 * pole_vol:
                    ctx = {"pole_top": d(pi), "pole_pct": round((pp / pole_lo - 1) * 100, 2),
                           "flag_bars": int(flag_len),
                           "flag_vol_ratio": round(flag_vol / pole_vol, 2)}
                    if 0.97 * flag_high <= c < flag_high:
                        out.append({"setup_id": "bull_flag", "state": "forming",
                                    "level": flag_high, "context": ctx})
                    elif c > flag_high >= c1:
                        out.append({"setup_id": "bull_flag", "state": "confirmed",
                                    "level": flag_high, "context": ctx})
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _f(v, nd=2):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def _row_context(P: Prepared, at: int) -> dict[str, Any]:
    return {"rsi": _f(P.rsi[at], 1), "vol_ratio": _f(P.vol_ratio[at]),
            "atr_pct": _f(P.atr_pct[at]), "ema20": _f(P.ema20[at], 4),
            "bb_pct_b": _f(P.pct_b[at])}


def _make(ticker: str, P: Prepared, at: int, sid: str, state: str,
          level: float, ctx: dict[str, Any]) -> Detection | None:
    sd = SETUPS[sid]
    price = float(P.close[at])
    if not np.isfinite(level) or not np.isfinite(price) or price <= 0:
        return None
    return Detection(
        setup_id=sid, tier=sd.tier, ticker=ticker, date=P.date(at),
        direction=sd.direction, state=state, level=round(float(level), 4),
        level_kind=sd.level_kind, price=round(price, 4),
        distance_pct=round((level - price) / price * 100.0, 2), context=ctx)


def prepare(df: pd.DataFrame, vol_scale: float = 1.0, ind=None, pivots=None) -> Prepared:
    ind = indicator_frame(df) if ind is None else ind
    pivots = swing_pivots(df, PIVOT_K) if pivots is None else pivots
    return Prepared(ind, pivots, vol_scale)


def detect_at(ticker: str, P: Prepared, at: int) -> list[Detection]:
    out: list[Detection] = []
    base_ctx: dict[str, Any] | None = None
    if P.any_t1[at]:
        base_ctx = _row_context(P, at)
        for sid, m in P.masks.items():
            for st in SETUPS[sid].states:
                if m[st][at]:
                    det = _make(ticker, P, at, sid, st, float(m["level"][at]), dict(base_ctx))
                    if det:
                        out.append(det)
    if P.liquid[at]:
        for part in tier2_at(P, at):
            if base_ctx is None:
                base_ctx = _row_context(P, at)
            ctx = dict(base_ctx)
            ctx.update(part["context"])
            det = _make(ticker, P, at, part["setup_id"], part["state"], part["level"], ctx)
            if det:
                out.append(det)
    return out


def detect_all(ticker: str, df: pd.DataFrame, *, vol_scale: float = 1.0,
               ind=None, pivots=None) -> list[Detection]:
    """Detections on the LAST bar of `df` (the live scan)."""
    if len(df) < MIN_BARS:
        return []
    P = prepare(df, vol_scale, ind, pivots)
    return detect_at(ticker, P, len(df) - 1)


def detect_history(ticker: str, df: pd.DataFrame, start_idx: int = MIN_BARS,
                   *, ind=None, pivots=None) -> list[Detection]:
    """Detections on every bar from `start_idx` (the backfill). Completed-bar
    semantics: no volume scaling."""
    if len(df) <= start_idx:
        return []
    P = prepare(df, 1.0, ind, pivots)
    out: list[Detection] = []
    for at in range(start_idx, len(df)):
        if P.any_t1[at] or P.liquid[at]:
            out.extend(detect_at(ticker, P, at))
    return out
