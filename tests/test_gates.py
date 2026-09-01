"""Each gate must still reject the case it was derived from.

These are not invented examples. Every case reproduces the statistics of a
candidate that looked real during the 2026-08-24..31 research and was killed
by exactly this check; the two marked "should PASS" are the controls that stop
a gate from rejecting everything. Run after touching `gates.py`:

    .venv/bin/python tests/test_gates.py

Exits non-zero on any incorrect verdict. The price_gradient case is the reason
that gate uses a rank trend rather than strict monotonicity -- the first
version passed the top-500 overnight artifact straight through, because its
two middle buckets sit closer together than one standard error and invert on
resampling.
"""
import numpy as np
from tradingagents.gates import (cost_sweep, symmetry, price_gradient, delayed_entry,
                                 out_of_sample, leave_one_out, tradeable_window)
rng = np.random.default_rng(11)
def sample(mean, sd, n): return rng.normal(mean, sd, n)

cases = []

# 1. small-cap gap: +0.498% long AND +0.495% short (10y daily). Bounce.
cases.append(("symmetry / small-cap gap", symmetry(
    sample(0.498, 6.0, 4382), sample(0.495, 6.0, 5702))))

# 2. a genuine asymmetry, for the false-positive check
cases.append(("symmetry / ASX50 gap (should PASS)", symmetry(
    sample(0.200, 3.0, 1689), sample(0.092, 3.0, 1769))))

# 3. top-500 overnight by price bucket: 42.15 / 19.16 / 17.58 / 9.84 bp
g, p = [], []
for mean, px, n in ((0.4215, 0.30, 400), (0.1916, 1.2, 400),
                    (0.1758, 5.0, 400), (0.0984, 25.0, 400)):
    g += list(sample(mean, 2.0, n)); p += [px] * n
cases.append(("price_gradient / top-500 overnight", price_gradient(g, p)))

# 4. small-cap Q1 short: +1.568% at open -> -0.366% at 11:00. Sign flip.
cases.append(("delayed_entry / small-cap Q1 short", delayed_entry(
    sample(1.568, 5.0, 137), sample(-0.366, 5.0, 137))))

# 5. ASX50 Q1 long: +0.901% -> +0.228%, shape predicted ~32% retained.
cases.append(("delayed_entry / ASX50 Q1 long (should PASS)", delayed_entry(
    sample(0.901, 1.2, 400), sample(0.228, 1.2, 400), expected_retained=0.32)))

# 6. per-stock daytime: ranked on train, +19.5bp -> -0.03bp out of sample.
cases.append(("out_of_sample / per-stock daytime", out_of_sample(
    sample(0.195, 2.0, 500), sample(-0.0003, 2.0, 500))))

# 7. PNV round trip: gross +0.478%, realistic cost 0.46% for a $1.23 stock.
cases.append(("cost_sweep / PNV round trip", cost_sweep(
    sample(0.478, 3.19, 480), realistic=0.46)))

# 8. leave-one-out where one group carries everything
g, k = [], []
for name in ("A", "B", "C", "D", "E", "F"):
    g += list(sample(-0.05, 1.0, 80)); k += [name] * 80
g += list(sample(3.0, 1.0, 80)); k += ["CARRIER"] * 80
cases.append(("leave_one_out / single carrier", leave_one_out(g, k)))

# 9. the look-ahead that no return data reveals
cases.append(("tradeable_window / gap at the open", tradeable_window(
    "after the opening auction clears", "the opening auction price", ok=False,
    note="the gap is not observable until the auction has already printed")))

print(f"{'case':<44}{'status':>8}   verdict")
print("-" * 118)
fails = 0
for name, r in cases:
    should_pass = "should PASS" in name
    correct = (r.passed is True) == should_pass
    if not correct and r.passed is not None:
        fails += 1
    mark = "" if correct or r.passed is None else "   <-- WRONG"
    print(f"{name:<44}{r.status:>8}   {r.verdict[:64]}{mark}")
print(f"\n{len(cases)} cases, {fails} incorrect verdicts")

import sys
sys.exit(1 if fails else 0)
