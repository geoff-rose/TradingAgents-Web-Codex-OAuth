"""Open-entry intraday range trading -- the user's actual proposal, which is
NOT what `range_model.py` tests (phase-4 spec §13.2c).

**The distinction that matters.** `range_model.py` buys a low quantile BELOW
the previous close: it is a mean-reversion engine that requires a dip. The
user's description is different -- "previous close was $10.00, you have reason
to think the market does well today, so you set the range at $10.01-$10.50 and
buy at $10.20, out at $10.45." No dip required; the entry is inside the day's
expected range, not below it.

**Why this is measurable where the other was contaminated.** The same-bar
artifact (see the asx-dashboard skill) exists because a resting limit order
below the market has an UNKNOWN fill time within the day, so crediting a
same-bar target assumes the low preceded the high. An **open entry has a known
fill time**: you are in at the start of the bar, so everything in that bar
happens after you. Same-day target fills are then legitimate, and the only
genuine ambiguity left is target-vs-stop ordering -- which a stop bounds
conservatively (assume the stop went first). That was the user's own point and
it is correct.

Resolution rules per day, in order:
  - high >= target and low  > stop   -> TARGET (unambiguous)
  - low  <= stop   and high < target -> STOP   (unambiguous)
  - both touched                     -> STOP   (conservative; no intraday path)
  - neither                          -> exit at the close

**The number that decides this is the base rate, not the win rate** (spec
§13.2c): what fraction of days reach the target before the stop *anyway*,
versus on the days your signal fires. 30% vs 25% is nothing; 30% vs 8% is an
effect. A win rate quoted without its base rate is meaningless, and so is one
quoted without the break-even rate the payoff ratio implies.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import yfinance as yf

from tradingagents.backtest import DEFAULT_BROKERAGE_PCT, fetch_daily_history

US_INDEX_SYMBOL = "^GSPC"


def break_even_hit_rate(target_pct: float, stop_pct: float, cost_pct: float = 0.0) -> float:
    """Win rate required to break even: p*T - (1-p)*S - cost = 0.
    Quote this next to any observed hit rate or the hit rate means nothing."""
    return (stop_pct + cost_pct) / (target_pct + stop_pct)


def simulate_open_entry(df: pd.DataFrame, target_pct: float, stop_pct: float,
                        mask: pd.Series | None = None) -> pd.DataFrame:
    """One trade per qualifying day: buy at the open, exit at target, stop, or
    the close. `mask` (optional, aligned to df.index) restricts to signal days;
    without it every day trades, which is exactly the base-rate control."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    target = o * (1 + target_pct / 100)
    stop = o * (1 - stop_pct / 100)

    hit_t = h >= target
    hit_s = l <= stop
    outcome = pd.Series("close", index=df.index, dtype=object)
    outcome[hit_t & ~hit_s] = "target"
    outcome[hit_s & ~hit_t] = "stop"
    outcome[hit_t & hit_s] = "stop_ambiguous"   # conservative

    ret = pd.Series(np.nan, index=df.index, dtype=float)
    ret[outcome == "target"] = target_pct
    ret[outcome.isin(["stop", "stop_ambiguous"])] = -stop_pct
    at_close = outcome == "close"
    ret[at_close] = (c[at_close] - o[at_close]) / o[at_close] * 100

    out = pd.DataFrame({"outcome": outcome, "gross_return_pct": ret,
                        "open": o, "high": h, "low": l, "close": c})
    if mask is not None:
        out = out[mask.reindex(out.index).fillna(False)]
    return out.dropna(subset=["gross_return_pct"])


def us_overnight_signal(period: str = "3y") -> pd.Series:
    """Prior US session's S&P 500 close-to-close return, z-scored, aligned to
    the FOLLOWING ASX date.

    **Replaces an earlier `^AXJO` open-gap signal that was invalid**: Yahoo
    reports the ASX index's Open as the previous Close on 88% of days, so that
    "gap" was 0.000 almost always and the handful of non-zero days were feed
    quirks, not hot opens. `^GSPC` has no such problem (0% exactly-zero gaps).

    Causally clean: the US session closes around 06:00 AEST, hours before the
    ASX opens, so a US date `d` return is known before ASX date `d+1` opens.
    The index is shifted forward one day and forward-filled across weekends
    and holidays, which never lets a return be used on or before the session
    that produced it.
    """
    us = yf.Ticker(US_INDEX_SYMBOL).history(period=period)
    if us.empty:
        return pd.Series(dtype=float)
    ret = us["Close"].pct_change() * 100
    ret.index = pd.to_datetime(ret.index).tz_localize(None).normalize() + pd.Timedelta(days=1)
    z = ret / ret.rolling(20).std()
    return z.dropna()


def cross_sectional_open_signal(opens: dict[str, pd.Series]) -> pd.Series:
    """Median open-gap across the universe -- spec §13.4's 'cross-sectional
    agreement' measure: are all names gapping the same way (a market regime)
    or is it stock-specific news?

    Known at the moment of entry, since entry IS the open, so using the
    day's own opens is not lookahead. It says nothing about what happens
    after the open, which is the whole point of §13.3's filter.
    """
    frame = pd.DataFrame(opens)
    return frame.median(axis=1).dropna()


def _day_clustered_bootstrap(frame: pd.DataFrame, n: int = 5_000, seed: int = 7) -> dict[str, float]:
    """95% CI on mean return, resampling **whole days**, not ticker-days.

    26 tickers on one day are nowhere near 26 independent observations --
    they share the market's move that day, and treating them as independent
    would shrink the interval by roughly sqrt(26) for no reason. This is the
    same 'effective independent positions' concern the screener reports
    (16.05 of 30 across time), and it bites much harder within a single day.
    """
    if frame.empty:
        return {}
    by_day = frame.groupby(frame.index)["gross_return_pct"].mean()
    vals = by_day.to_numpy()
    if len(vals) < 3:
        return {}
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), (n, len(vals)))].mean(axis=1)
    return {"ci_lo": round(float(np.percentile(means, 2.5)), 4),
            "ci_hi": round(float(np.percentile(means, 97.5)), 4),
            "n_days": int(len(vals))}


def base_rate_study(tickers: list[str], target_pct: float = 2.5, stop_pct: float = 1.5,
                    hot_z: float = 1.0, period: str = "3y",
                    signal: str = "us_overnight") -> dict[str, Any]:
    """The §13.2c test: target-before-stop rate and average return on ALL days
    versus on signal days, with day-clustered CIs.

    `signal`: 'us_overnight' (prior US session, known pre-open) or
    'cross_sectional' (median open gap across the universe).
    """
    all_rows, opens = [], {}
    for t in tickers:
        df = fetch_daily_history(t, period=period)
        if df.empty:
            continue
        res = simulate_open_entry(df, target_pct, stop_pct)
        res["ticker"] = t
        all_rows.append(res)
        opens[t] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100
    if not all_rows:
        return {"error": "no price data"}
    all_t = pd.concat(all_rows)

    if signal == "cross_sectional":
        raw = cross_sectional_open_signal(opens)
        sig = (raw / raw.rolling(60).std()).dropna()
    else:
        sig = us_overnight_signal(period)
    if sig.empty:
        return {"error": f"no signal data for '{signal}'"}
    hot_days = set(sig.index[sig >= hot_z])
    hot_t = all_t[all_t.index.isin(hot_days)]

    cost_pct = 2 * DEFAULT_BROKERAGE_PCT * 100

    def stats(frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {"n": 0}
        n = len(frame)
        resolved = frame[frame.outcome != "close"]
        out = {
            "n": n,
            "n_days": int(frame.index.nunique()),
            "target_rate_pct": round(100 * (frame.outcome == "target").sum() / n, 2),
            "stop_rate_pct": round(100 * frame.outcome.isin(["stop", "stop_ambiguous"]).sum() / n, 2),
            "ambiguous_pct": round(100 * (frame.outcome == "stop_ambiguous").sum() / n, 2),
            "closed_out_pct": round(100 * (frame.outcome == "close").sum() / n, 2),
            # Break-even applies ONLY to trades that actually resolved at a
            # target or stop. Quoting it against the full sample (as an earlier
            # version of this module did) is wrong whenever a large share of
            # days exit at the close instead -- which is most of them.
            "target_share_of_resolved_pct": round(
                100 * (frame.outcome == "target").sum() / len(resolved), 2) if len(resolved) else None,
            "avg_gross_pct": round(float(frame.gross_return_pct.mean()), 4),
            "avg_net_pct": round(float(frame.gross_return_pct.mean()) - cost_pct, 4),
        }
        out.update(_day_clustered_bootstrap(frame))
        return out

    return {
        "signal": signal, "target_pct": target_pct, "stop_pct": stop_pct, "hot_z": hot_z,
        "break_even_of_resolved_pct": round(100 * break_even_hit_rate(target_pct, stop_pct, cost_pct), 2),
        "cost_pct_round_trip": round(cost_pct, 3),
        "all_days": stats(all_t),
        "hot_open_days": stats(hot_t),
        "n_signal_days": len(hot_days),
    }


def print_study(r: dict[str, Any]) -> None:
    if "error" in r:
        print("ERROR:", r["error"]); return
    print("=" * 100)
    print(f"OPEN-ENTRY INTRADAY  target +{r['target_pct']}%  stop -{r['stop_pct']}%  "
          f"signal={r['signal']} (z >= {r['hot_z']})")
    print("=" * 100)
    print(f"break-even share OF RESOLVED trades (incl {r['cost_pct_round_trip']}% costs): "
          f"{r['break_even_of_resolved_pct']}%   |   {r['n_signal_days']} signal days")
    print()
    print(f"  {'':<14}{'n':>7}{'days':>6}{'target%':>9}{'stop%':>8}{'close%':>8}"
          f"{'tgt/resolved':>13}{'avgGross':>10}{'avgNet':>9}{'95% CI (day-clustered)':>26}")
    for label, key in (("ALL DAYS", "all_days"), ("SIGNAL DAYS", "hot_open_days")):
        s = r[key]
        if not s.get("n"):
            print(f"  {label:<14}{0:>7}"); continue
        ci = (f"[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]" if "ci_lo" in s else "n/a")
        print(f"  {label:<14}{s['n']:>7}{s['n_days']:>6}{s['target_rate_pct']:>9.2f}"
              f"{s['stop_rate_pct']:>8.2f}{s['closed_out_pct']:>8.2f}"
              f"{(s['target_share_of_resolved_pct'] or 0):>13.2f}"
              f"{s['avg_gross_pct']:>+10.4f}{s['avg_net_pct']:>+9.4f}{ci:>26}")
    a, h = r["all_days"], r["hot_open_days"]
    if a.get("n") and h.get("n"):
        print()
        print(f"  lift in target rate: {h['target_rate_pct'] - a['target_rate_pct']:+.2f}pp   "
              f"lift in avg net: {h['avg_net_pct'] - a['avg_net_pct']:+.4f}pp")
        if "ci_lo" in h:
            verdict = ("CI excludes zero" if h["ci_lo"] > 0 else "CI INCLUDES zero -- not distinguishable from no effect")
            print(f"  signal-day avg net CI: {verdict}  (on {h['n_days']} independent days)")


if __name__ == "__main__":
    from tradingagents.screener import get_model_universe
    tickers = [r["ticker"] for r in get_model_universe(tradeable_only=True)]
    for sig in ("us_overnight", "cross_sectional"):
        for tgt, stp in ((2.5, 1.5), (5.0, 3.0)):
            print_study(base_rate_study(tickers, target_pct=tgt, stop_pct=stp, signal=sig))
            print()
