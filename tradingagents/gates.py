"""The rejection battery: the checks that killed a candidate that looked real.

**Why this file exists.** Every strategy module in this package describes an
idea. Until now, the checks that REJECTED ideas lived in throwaway scripts and
were rewritten from memory each time -- so whether a hypothesis got tested
properly depended on whether anyone remembered the relevant lesson. Each gate
below is named for the thing it actually caught:

  cost_sweep        PNV: the whole edge was the spread, visible only once cost
                    was swept rather than point-estimated at brokerage.
  symmetry          small-cap gap: +0.498%/trade long AND +0.495% short. Two
                    opposite trades cannot both have an edge; that is bounce.
  price_gradient    top-500 overnight: +42bp/night under $0.50 decaying
                    monotonically to +9.8bp above $10. An effect that scales
                    with relative tick size is tick size.
  delayed_entry     small-cap Q1 short: +1.568% entering at the open,
                    -0.366% entering at 11:00. A real edge decays; it does
                    not change sign.
  out_of_sample     per-stock daytime returns: spearman +0.11 train->test,
                    i.e. the ranking did not survive at all.
  leave_one_out     the v4 prompt: advantage +0.103 overall but +0.034 with
                    BC8 removed. Survived, but only just, and now visibly.
  benchmark_split   BC8: -6.19% raw reads as a rejected announcement, -1.2%
                    against its own sector reads as noise.

Three further gates are DECLARATIONS, not computations -- `tradeable_window`,
`survivorship` and `multiple_testing` cannot be derived from a returns frame
and must be asserted by whoever writes the hypothesis. They are gates anyway,
because the two most expensive errors this project has made (entering at an
opening auction price that is not knowable until after the auction clears, and
buying gap-downs in a universe defined by today's index membership) were both
undetectable in the numbers and obvious in the description.

**A gate returns a verdict, not a p-value.** `passed=False` on a fatal gate
means the result is an artifact, not that it is weak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

# Round-trip costs to sweep, in percent. Corrected 2026-08-31 once the real
# brokerage was known ($2 per $20k = 2bp round trip, not the 5bp assumed
# before) and spreads were measured: the brokerage is the SMALL half and the
# spread varies twentyfold across the universe -- 0.026% for CBA, 0.640% for
# LRV. Prefer passing `per_trade_cost` from `costs.cost_vector()`; this sweep
# is the fallback when a per-instrument estimate is unavailable, and it spans
# a liquid ETF through to a sub-$2 small cap on purpose.
DEFAULT_COSTS = (0.0, 0.03, 0.10, 0.30, 0.60)
REALISTIC_COST = 0.10

# Above this share of floor-derived costs, a cost verdict is provisional.
FLOOR_WARN = 0.5


@dataclass
class GateResult:
    name: str
    passed: bool | None            # None = not enough data to judge
    verdict: str
    detail: dict[str, Any] = field(default_factory=dict)
    fatal: bool = True             # a failed fatal gate condemns the result
    # Evaluated, but on an input that is a bound rather than a measurement.
    # Distinct from `passed is None` with provisional=False, which means the
    # data simply did not support the check. The difference matters: a gate
    # that COULD NOT run is not evidence either way, while a gate running on a
    # lower bound is actively withholding a verdict it might yet fail.
    provisional: bool = False

    @property
    def status(self) -> str:
        if self.provisional:
            return "prov"
        if self.passed is None:
            return "n/a"
        return "pass" if self.passed else ("FAIL" if self.fatal else "warn")


def _stats(x) -> dict[str, float]:
    import numpy as np
    a = np.asarray([v for v in x if v == v], dtype=float)
    if len(a) < 2:
        return {"n": len(a)}
    sd = float(a.std(ddof=1))
    return {"n": int(len(a)), "mean": float(a.mean()), "sd": sd,
            "t": float(a.mean() / (sd / math.sqrt(len(a)))) if sd > 0 else float("nan"),
            "win": float((a > 0).mean() * 100)}


# --------------------------------------------------------------------------
# Computed gates
# --------------------------------------------------------------------------

def cost_sweep(gross_pct: Sequence[float], costs: Sequence[float] = DEFAULT_COSTS,
               realistic: float = REALISTIC_COST,
               per_trade_cost: Sequence[float] | None = None,
               floor_fraction: float = 0.0) -> GateResult:
    """Net return after cost. Swept, never point-estimated.

    `per_trade_cost` is the honest version: one cost per trade from
    `costs.cost_vector()`, since a strategy trading CBA (0.026% round trip)
    and one trading LRV (0.640%) face hurdles twentyfold apart. Without it the
    scalar `realistic` applies to every trade, which is wrong for any
    strategy whose universe spans liquidity bands.

    `floor_fraction` is the share of those costs that came from the ONE-TICK
    FLOOR rather than a measurement. A floor is a lower bound, so clearing a
    hurdle built from floors is not evidence of clearing the real one -- the
    VAS overnight trade nets +0.036%/night at its floor cost and turns
    NEGATIVE against buy-and-hold at two ticks. Above `FLOOR_WARN` the gate
    reports `warn` instead of `pass`, and stops being fatal, because the
    honest verdict is "not yet measured", not "passed".
    """
    import numpy as np
    s = _stats(gross_pct)
    if s.get("n", 0) < 30:
        return GateResult("cost_sweep", None, f"only {s.get('n', 0)} trades", s)
    table = {f"{c:.2f}%": round(s["mean"] - c, 4) for c in costs}
    if per_trade_cost is not None and len(per_trade_cost) == len(list(gross_pct)):
        g = np.asarray(list(gross_pct), dtype=float)
        c = np.asarray(list(per_trade_cost), dtype=float)
        net_each = g - c
        n = _stats(net_each)
        provisional = floor_fraction >= FLOOR_WARN
        note = (f" -- but {floor_fraction:.0%} of these costs are the one-tick FLOOR, "
                f"a lower bound, so this is not yet measured" if provisional else "")
        return GateResult(
            "cost_sweep", None if provisional else n["mean"] > 0,
            f"gross {s['mean']:+.3f}%/trade -> net {n['mean']:+.3f}% "
            f"(t {n.get('t', 0):+.1f}) at a per-instrument cost averaging "
            f"{c.mean():.3f}%{note}",
            {**s, "by_cost": table, "net": n, "mean_cost": round(float(c.mean()), 4),
             "cost_range": [round(float(c.min()), 4), round(float(c.max()), 4)],
             "per_instrument": True, "floor_fraction": floor_fraction},
            fatal=True, provisional=provisional)
    net = s["mean"] - realistic
    return GateResult(
        "cost_sweep", net > 0,
        f"gross {s['mean']:+.3f}%/trade -> net {net:+.3f}% at a FLAT {realistic:.2f}% cost",
        {**s, "by_cost": table, "realistic_cost": realistic, "per_instrument": False})


def symmetry(long_gross: Sequence[float], short_gross: Sequence[float],
             tol: float = 0.5) -> GateResult:
    """Both directions profitable at similar size means the spread, not an edge.

    `tol` is the fraction by which the two means may differ before the result
    is considered suspiciously symmetric.
    """
    L, S = _stats(long_gross), _stats(short_gross)
    if L.get("n", 0) < 30 or S.get("n", 0) < 30:
        return GateResult("symmetry", None, "one side has too few trades",
                          {"long": L, "short": S})
    lm, sm = L["mean"], S["mean"]
    both_positive = lm > 0 and sm > 0
    close = abs(lm - sm) <= tol * max(abs(lm), abs(sm))
    bad = both_positive and close
    return GateResult(
        "symmetry", not bad,
        (f"long {lm:+.3f}% and short {sm:+.3f}% BOTH positive and within "
         f"{tol:.0%} -- bid-ask bounce" if bad
         else f"long {lm:+.3f}% vs short {sm:+.3f}% -- not symmetric"),
        {"long": L, "short": S})


def price_gradient(gross_pct: Sequence[float], price: Sequence[float],
                   buckets: Sequence[float] = (0.5, 2.0, 10.0)) -> GateResult:
    """An effect that grows as price falls is measuring relative tick size."""
    import numpy as np
    g = np.asarray(list(gross_pct), dtype=float)
    p = np.asarray(list(price), dtype=float)
    ok = (g == g) & (p == p) & (p > 0)
    g, p = g[ok], p[ok]
    if len(g) < 60:
        return GateResult("price_gradient", None, f"only {len(g)} priced trades", {})
    edges = [0.0, *buckets, float("inf")]
    rows, means = [], []
    for lo, hi in zip(edges, edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() < 20:
            rows.append({"bucket": f"${lo:g}-{hi:g}", "n": int(m.sum()), "mean": None})
            continue
        mu = float(g[m].mean())
        rows.append({"bucket": f"${lo:g}-{hi:g}", "n": int(m.sum()), "mean": round(mu, 4)})
        means.append(mu)
    if len(means) < 3:
        return GateResult("price_gradient", None, "too few populated buckets",
                          {"buckets": rows})
    # A RANK trend, not strict monotonicity. Requiring every adjacent pair to
    # decrease made this gate miss the case it was built for: the top-500
    # overnight buckets are 42.2 / 19.2 / 17.6 / 9.8bp, and the middle two sit
    # closer together than one standard error, so they invert on resampling
    # and a strict test lets the artifact through. Spearman over the bucket
    # means tolerates that inversion while still catching the slope.
    from scipy.stats import spearmanr
    rho = float(spearmanr(range(len(means)), means).statistic)
    blowup = abs(means[0]) > 2.5 * abs(means[-1]) if means[-1] else True
    bad = rho <= -0.6 and blowup
    return GateResult(
        "price_gradient", not bad,
        (f"effect falls with price (rho {rho:+.2f}, {means[0]:+.3f}% -> "
         f"{means[-1]:+.3f}%) -- tick-size artifact" if bad
         else f"no price gradient (rho {rho:+.2f})"),
        {"buckets": rows, "rho": round(rho, 3)})


def delayed_entry(at_signal: Sequence[float], delayed: Sequence[float],
                  expected_retained: float | None = None) -> GateResult:
    """A real edge decays when you enter later. An artifact changes sign.

    `expected_retained` is the fraction the measured intraday shape predicts
    should survive the delay; without it the gate only rejects sign flips.
    """
    A, B = _stats(at_signal), _stats(delayed)
    if A.get("n", 0) < 25 or B.get("n", 0) < 25:
        return GateResult("delayed_entry", None, "too few trades", {"at_signal": A, "delayed": B})
    am, bm = A["mean"], B["mean"]
    flipped = am > 0 > bm or am < 0 < bm
    retained = bm / am if am else float("nan")
    detail = {"at_signal": A, "delayed": B, "retained_frac": round(retained, 3)}
    if flipped:
        return GateResult("delayed_entry", False,
                          f"sign flips on delay ({am:+.3f}% -> {bm:+.3f}%) -- the edge is "
                          f"in the entry print, not the market", detail)
    if expected_retained is not None and retained < 0.4 * expected_retained:
        return GateResult("delayed_entry", False,
                          f"only {retained:.0%} retained against ~{expected_retained:.0%} "
                          f"expected from the intraday shape", detail)
    return GateResult("delayed_entry", True,
                      f"{retained:.0%} retained on delay ({am:+.3f}% -> {bm:+.3f}%)", detail)


def out_of_sample(train_gross: Sequence[float], test_gross: Sequence[float]) -> GateResult:
    """Fit on the first period, measure on the second. Decay is expected;
    disappearance is not."""
    A, B = _stats(train_gross), _stats(test_gross)
    if A.get("n", 0) < 30 or B.get("n", 0) < 30:
        return GateResult("out_of_sample", None, "too few trades in one period",
                          {"train": A, "test": B})
    ok = B["mean"] > 0 and B.get("t", 0) > 1.0
    return GateResult("out_of_sample", ok,
                      f"train {A['mean']:+.3f}% (t {A.get('t', 0):.1f}) -> "
                      f"test {B['mean']:+.3f}% (t {B.get('t', 0):.1f})",
                      {"train": A, "test": B})


def leave_one_out(gross_pct: Sequence[float], group: Sequence[Any],
                  min_groups: int = 5) -> GateResult:
    """Does one ticker (or one day) carry the whole result?"""
    import numpy as np
    g = np.asarray(list(gross_pct), dtype=float)
    k = np.asarray(list(group), dtype=object)
    ok = g == g
    g, k = g[ok], k[ok]
    groups = sorted(set(k.tolist()))
    if len(groups) < min_groups or len(g) < 30:
        return GateResult("leave_one_out", None,
                          f"only {len(groups)} groups / {len(g)} trades", {})
    full = float(g.mean())
    worst, worst_key = full, None
    for key in groups:
        m = k != key
        if m.sum() < 20:
            continue
        mu = float(g[m].mean())
        if (full > 0 and mu < worst) or (full <= 0 and mu > worst):
            worst, worst_key = mu, key
    survives = (full > 0 and worst > 0) or (full <= 0 and worst <= 0)
    return GateResult("leave_one_out", survives,
                      f"{full:+.3f}% overall, {worst:+.3f}% excluding {worst_key}",
                      {"full_mean": round(full, 4), "worst_mean": round(worst, 4),
                       "worst_group": str(worst_key), "n_groups": len(groups)})


def benchmark_split(gross_pct: Sequence[float], bench_pct: Sequence[float],
                    label: str = "sector") -> GateResult:
    """Raw vs benchmark-relative. Not fatal -- it reframes, it does not reject."""
    import numpy as np
    g = np.asarray(list(gross_pct), dtype=float)
    b = np.asarray(list(bench_pct), dtype=float)
    ok = (g == g) & (b == b)
    g, b = g[ok], b[ok]
    if len(g) < 30:
        return GateResult("benchmark_split", None, "too few paired trades", {}, fatal=False)
    R, X = _stats(g), _stats(g - b)
    return GateResult("benchmark_split", True,
                      f"raw {R['mean']:+.3f}% -> vs {label} {X['mean']:+.3f}% "
                      f"(t {X.get('t', 0):.1f})",
                      {"raw": R, "relative": X, "benchmark": label}, fatal=False)


# --------------------------------------------------------------------------
# Declared gates -- asserted by the hypothesis, not derived from returns
# --------------------------------------------------------------------------

def tradeable_window(signal_known_at: str, entry_at: str, ok: bool,
                     note: str = "") -> GateResult:
    """Is the entry price knowable BEFORE you must commit?

    The gap-reversion strategy failed exactly here: the ASX opening auction
    clears at one price, orders must be in before it clears, and the gap is
    not observable until after. Entering 'at the open' on a gap signal is
    look-ahead, and no amount of return data reveals it.
    """
    return GateResult("tradeable_window", ok,
                      f"signal known {signal_known_at}, entry {entry_at}"
                      + (f" -- {note}" if note else ""),
                      {"signal_known_at": signal_known_at, "entry_at": entry_at})


def survivorship(universe: str, point_in_time: bool, note: str = "") -> GateResult:
    """Is the universe today's membership applied backwards through history?"""
    return GateResult("survivorship", point_in_time,
                      (f"{universe}: point-in-time" if point_in_time
                       else f"{universe}: TODAY's membership applied to history"
                            + (f" -- {note}" if note else "")),
                      {"universe": universe, "point_in_time": point_in_time},
                      fatal=False)


def multiple_testing(n_variants: int, n_passed: int, alpha: float = 0.05) -> GateResult:
    """How many of the passing variants would chance alone have produced?"""
    expected = n_variants * alpha
    ok = n_passed > 2 * expected if n_variants else None
    return GateResult("multiple_testing", ok,
                      f"{n_passed} of {n_variants} variants passed; "
                      f"~{expected:.1f} expected by chance at alpha={alpha}",
                      {"n_variants": n_variants, "n_passed": n_passed,
                       "expected_by_chance": round(expected, 2)},
                      fatal=False)


ALL_GATES: dict[str, Callable[..., GateResult]] = {
    f.__name__: f for f in (
        cost_sweep, symmetry, price_gradient, delayed_entry, out_of_sample,
        leave_one_out, benchmark_split, tradeable_window, survivorship,
        multiple_testing)
}
