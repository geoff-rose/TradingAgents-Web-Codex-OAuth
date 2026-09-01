"""Phase-4 §13.1 diagnostic: does the range model predict the *width* of
tomorrow's range, its *centre*, or both?

**Why this exists.** `range_model.py` produces next-day low/high quantiles,
and every trading configuration built on it has failed the randomised-entry
and selection-permutation gates (see the `asx-dashboard` skill). The spec's
diagnosis-before-features rule (§13.1) says to find out *which half* of the
prediction is working before adding any feature:

    "Range trading needs the centre of the distribution to be stable. A model
    that predicts the *width* of tomorrow's range but nothing about its
    *centre* can size bands correctly and still have no timing edge."

If width is well predicted and centre is not, that is a complete explanation
of the null trading results, and it redirects the work to regime
classification (§13.2) rather than more regressors.

**What is measured.** For each held-out day, using only the pooled model's
median (q=0.5) low and high predictions:

    predicted_width  = pred_high_q50 - pred_low_q50
    predicted_centre = (pred_high_q50 + pred_low_q50) / 2
    realised_width   = next_high_pct - next_low_pct
    realised_centre  = (next_high_pct + next_low_pct) / 2

(all as % of the anchoring day's close, which is the model's native frame).

**Skill is always relative to a baseline, never absolute.** A raw MAE or R^2
on width would look impressive purely because volatility clusters. Two
baselines per target, both causal:

  width  -- `constant`:    the optimal constant predictor (training-set median)
            `persistence`: today's own high-low range as % of close. This is
                           the honest baseline; beating a constant is trivial,
                           beating yesterday's range is not.
  centre -- `constant`:    the optimal constant predictor (training-set median),
                           which absorbs whatever unconditional drift exists
            `zero`:        tomorrow's range centres on today's close. The
                           random-walk null, i.e. the efficient-markets
                           position, and the one that matters.

**Every predictor is level-calibrated on the training split before scoring,
and this is not a cosmetic step.** The model's native width prediction is
`median(next_high) - median(next_low)`, which is biased *low* by construction:
next-day highs are right-skewed and next-day lows left-skewed, so each
median sits closer to zero than its mean. Uncalibrated, the model predicted a
mean width of 4.74% against a realised 6.62% and lost to a constant purely on
that level bias, saying nothing about whether it carries conditional
information. Each predictor (model and persistence alike) therefore gets a
single additive offset fitted on the training split -- `median(actual_train) -
median(pred_train)` -- which is causal and puts every candidate on equal
footing, so the MAE comparison measures conditional information only. The
constant baseline is the training median itself, which is already the
MAE-optimal constant, so it needs no offset. MAE (not RMSE) throughout,
because the underlying models are q=0.5 quantile regressions and scoring them
on squared error would penalise them on a loss they were never fit for.

Skill score = 1 - MAE_model / MAE_baseline. Positive means the model beats
that baseline; <= 0 means it adds nothing over it. Bootstrap CIs are reported
because ~1,250 held-out rows across 5 tickers is not a large sample, and a
skill score without an interval invites exactly the over-reading this whole
module exists to prevent.

**Read the width-vs-persistence number with care.** Persistence turned out to
be a *worse* baseline than a constant here (MAE 2.649 vs 2.375) -- on names
whose whole daily range is 2-3 ticks wide, a single day's range is mostly
quantisation noise, so an unconditional average beats yesterday's reading.
That makes "+0.159 skill vs persistence" the flattering comparison and
"+0.062 vs constant" the honest headline. Lead with the constant.

**A shuffle null runs on top of the CI**, because this project has been
burned before by a result that looked significant under its own assumptions
(see the `asx-dashboard` skill's range-model section). It re-scores the model
after randomly permuting its predictions across held-out rows, destroying the
feature-to-outcome pairing while preserving both marginal distributions
exactly. A real conditional signal should sit far above that null; a skill
score that a shuffled model reproduces is a distributional artefact, not
information.

**Result on the live 5-ticker universe (run 2026-08-22, held-out year to
2026-08-20, 1,275 rows).** The spec's hypothesis was "width predicted, centre
not." What the data says is narrower than that: *both* carry a little
conditional information -- width skill +0.062 vs a constant (95% CI
[+0.039, +0.086]), centre +0.035 ([+0.019, +0.050]), both at the 100th
percentile of the shuffle null -- and *neither* is remotely tradeable. Width's
edge over a constant is 0.42pp of price, centre's is 0.09pp, against a tick
that is 2.2-3.4% of price on four of the five names.

The structural finding matters more than either skill score: **the average
day on BRN/INR/MEI/TTT spans 2.0-3.0 ticks end to end.** Buying the low band
and selling the high band of a 2-tick day is not a forecasting problem, it is
an arithmetic impossibility once a spread is crossed. PNV, at $1.06 with a
10.6-tick day, is the only name in the universe where range trading is even
structurally expressible -- and it is the one the earlier heuristic backtest
found to be 0-for-7. This is a universe-composition problem before it is a
modelling problem, and it should redirect work to spec §14/§16 (expand to 30
tickers, explicitly not 30 speculatives) ahead of §13.2's regime classifier.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tradingagents.range_model import (
    FEATURE_COLUMNS, fit, pool_training_data,
)
from tradingagents.backtest import HELD_OUT_MONTHS, fetch_daily_history, tick_size

MEDIAN_Q = 0.5
N_BOOTSTRAP = 5_000
BOOTSTRAP_SEED = 20260822  # fixed: this is a diagnostic, its numbers should reproduce


def _today_range_pct(ticker: str, period: str = "10y") -> pd.Series:
    """Today's own high-low range as % of today's close -- the persistence
    baseline for width. Indexed by date, so it aligns with the pooled
    feature frame's index for this ticker."""
    df = fetch_daily_history(ticker, period=period)
    if df.empty:
        return pd.Series(dtype=float)
    return ((df["High"] - df["Low"]) / df["Close"] * 100).rename("today_range_pct")


def _skill(actual: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> float:
    """1 - MAE(model)/MAE(baseline). MAE rather than RMSE because the
    underlying models are quantile regressions at q=0.5, which minimise
    absolute error -- scoring them on squared error would penalise them on a
    loss they were never fit for."""
    mae_model = float(np.mean(np.abs(actual - pred)))
    mae_base = float(np.mean(np.abs(actual - baseline)))
    if mae_base == 0:
        return float("nan")
    return 1.0 - mae_model / mae_base


def _skill_ci(actual: np.ndarray, pred: np.ndarray, baseline: np.ndarray,
              n: int = N_BOOTSTRAP, ci: float = 0.95) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx_max = len(actual)
    scores = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, idx_max, idx_max)
        scores[i] = _skill(actual[idx], pred[idx], baseline[idx])
    lo, hi = np.percentile(scores, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def _calibrate(pred_train: np.ndarray, actual_train: np.ndarray,
               pred_held: np.ndarray) -> tuple[np.ndarray, float]:
    """Add a single additive offset fitted on the TRAINING split only, so a
    predictor is judged on conditional information rather than on a level
    bias it never had a chance to correct. The median (not the mean) is used
    because MAE is the scoring loss and the median is its optimal constant."""
    offset = float(np.median(actual_train) - np.median(pred_train))
    return pred_held + offset, offset


def _shuffle_null(actual: np.ndarray, pred: np.ndarray, baseline: np.ndarray,
                  n: int = 1_000) -> dict[str, Any]:
    """Permute the model's predictions across held-out rows, destroying the
    feature-to-outcome pairing while leaving both marginal distributions
    untouched, and re-score. Reports where the real skill sits in that null.
    A genuine conditional signal should be far above it."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    real = _skill(actual, pred, baseline)
    null = np.empty(n)
    for i in range(n):
        null[i] = _skill(actual, rng.permutation(pred), baseline)
    pct = float((null < real).mean() * 100)
    return {
        "real_skill": round(real, 4),
        "null_median_skill": round(float(np.median(null)), 4),
        "null_p95_skill": round(float(np.percentile(null, 95)), 4),
        "actual_percentile_in_null": round(pct, 1),
        "clears_95th": bool(pct >= 95),
    }


def economic_significance(held: pd.DataFrame, mae_gain_pp: dict[str, float],
                          realised_width_pct: np.ndarray | None = None) -> dict[str, Any]:
    """Translate an MAE improvement (in percentage points of price) into the
    only unit that decides whether it is tradeable: **ticks**.

    Statistical significance and economic significance are different
    questions, and on this universe they diverge violently. Four of the five
    tickers sit in ASX's 10c-$2.00 band, where one tick is 0.5c -- which on a
    16c stock is 3.1% of price. A forecast improvement of 0.09pp is real and
    reproducible and still roughly a thirtieth of the smallest price increment
    that exists. Nothing can be traded on it. Reporting a skill score without
    this conversion is how a null strategy gets called promising.

    Reported per ticker as well as pooled, because PNV ($1.06, one tick =
    0.47%) is an order of magnitude finer-grained than the microcaps and the
    pooled median hides that.

    Also reports **the whole realised daily range in ticks**, which turned out
    to be the structurally decisive number: if an average day on a 16c stock
    spans only ~2 price increments, then placing a buy band and a sell band
    inside that range is not a forecasting problem at all -- there is almost
    no room between them, and any modelled precision finer than one tick is
    unrepresentable in the market. That constraint binds regardless of how
    good the forecast is, so it must be checked before any further modelling.
    """
    out: dict[str, Any] = {"per_ticker": {}}
    tick_pcts = []
    tickers_col = held["ticker"].values
    for ticker in sorted(held["ticker"].unique()):
        price = float(held.loc[held["ticker"] == ticker, "close"].median())
        tick_pct = tick_size(price) / price * 100
        tick_pcts.append(tick_pct)
        mean_width = (float(np.mean(realised_width_pct[tickers_col == ticker]))
                      if realised_width_pct is not None else None)
        out["per_ticker"][ticker] = {
            "median_close": round(price, 4),
            "tick_size": tick_size(price),
            "one_tick_pct_of_price": round(tick_pct, 4),
            "gain_in_ticks": {
                target: round(gain / tick_pct, 3) for target, gain in mae_gain_pp.items()
            },
            "mean_daily_range_pct": round(mean_width, 3) if mean_width is not None else None,
            "mean_daily_range_in_ticks": (
                round(mean_width / tick_pct, 2) if mean_width is not None else None
            ),
        }
    median_tick_pct = float(np.median(tick_pcts))
    out["median_one_tick_pct_of_price"] = round(median_tick_pct, 4)
    out["gains_in_ticks"] = {
        target: {
            "mae_gain_pp": round(gain, 4),
            "gain_in_ticks": round(gain / median_tick_pct, 3),
            "tradeable": bool(gain / median_tick_pct >= 1.0),
        }
        for target, gain in mae_gain_pp.items()
    }
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without pulling in scipy (not currently a dependency
    of this module's import path)."""
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _evaluate_block(label: str, actual: np.ndarray, pred: np.ndarray,
                    baselines: dict[str, np.ndarray], with_ci: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {
        "target": label,
        "n": int(len(actual)),
        "mae_model": round(float(np.mean(np.abs(actual - pred))), 4),
        "mean_actual": round(float(np.mean(actual)), 4),
        "mean_predicted": round(float(np.mean(pred)), 4),
        "spearman_pred_vs_actual": round(_spearman(actual, pred), 4),
        "baselines": {},
    }
    for name, base in baselines.items():
        entry = {
            "mae_baseline": round(float(np.mean(np.abs(actual - base))), 4),
            "skill_score": round(_skill(actual, pred, base), 4),
        }
        if with_ci:
            entry["skill_score_ci95"] = _skill_ci(actual, pred, base)
            entry["beats_baseline"] = bool(entry["skill_score_ci95"]["lo"] > 0)
        out["baselines"][name] = entry
    return out


def diagnose(tickers: list[str], period: str = "10y",
             held_out_months: int = HELD_OUT_MONTHS) -> dict[str, Any]:
    """Run the §13.1 width-vs-centre decomposition on the held-out window.

    Returns pooled results plus a per-ticker breakdown. The per-ticker figures
    are diagnostic colour only -- with ~250 held-out rows each they are far
    too noisy to act on individually, and are reported without CIs or nulls to
    make that obvious rather than inviting a per-ticker decision.
    """
    pooled = pool_training_data(tickers, period=period)
    if pooled.empty:
        return {"error": "no pooled training data"}

    model = fit(pooled, held_out_months=held_out_months)

    # Persistence baseline for width: today's own range, joined per ticker
    # across the WHOLE pooled frame -- the training half is needed to fit the
    # baseline's own calibration offset, not just the held-out half.
    pooled = pooled.copy()
    pooled["today_range_pct"] = np.nan
    for ticker in pooled["ticker"].unique():
        rng_series = _today_range_pct(ticker, period=period)
        mask = pooled["ticker"] == ticker
        pooled.loc[mask, "today_range_pct"] = rng_series.reindex(pooled.index[mask]).values
    pooled = pooled.dropna(subset=["today_range_pct"])

    train = pooled[pooled.index < model.train_cutoff]
    held = pooled[pooled.index >= model.train_cutoff]
    if held.empty or train.empty:
        return {"error": "empty train or held-out split"}

    def _design(frame: pd.DataFrame) -> np.ndarray:
        X = frame[FEATURE_COLUMNS].copy()
        X.insert(0, "const", 1.0)
        return X.values

    lo_params = model.low_models[MEDIAN_Q].params.values
    hi_params = model.high_models[MEDIAN_Q].params.values

    def _targets(frame: pd.DataFrame, design: np.ndarray):
        pred_low, pred_high = design @ lo_params, design @ hi_params
        act_low = frame["next_low_pct"].values
        act_high = frame["next_high_pct"].values
        return {
            "width": (act_high - act_low, pred_high - pred_low),
            "centre": ((act_high + act_low) / 2, (pred_high + pred_low) / 2),
        }

    tr = _targets(train, _design(train))
    hd = _targets(held, _design(held))

    persistence_train = train["today_range_pct"].values
    persistence_held = held["today_range_pct"].values

    results: dict[str, Any] = {}
    for key in ("width", "centre"):
        act_train, pred_train = tr[key]
        act_held, pred_held = hd[key]

        pred_cal, offset = _calibrate(pred_train, act_train, pred_held)
        const_baseline = np.full_like(act_held, float(np.median(act_train)))

        baselines = {"constant": const_baseline}
        if key == "width":
            pers_cal, pers_offset = _calibrate(persistence_train, act_train, persistence_held)
            baselines["persistence"] = pers_cal
        else:
            baselines["zero"] = np.zeros_like(act_held)

        block = _evaluate_block(key, act_held, pred_cal, baselines)
        block["calibration_offset_pct"] = round(offset, 4)
        block["mean_predicted_uncalibrated"] = round(float(np.mean(pred_held)), 4)
        # The shuffle null is run against the constant baseline: that is the
        # comparison where a distributional artefact would be easiest to
        # mistake for skill.
        block["shuffle_null_vs_constant"] = _shuffle_null(act_held, pred_cal, const_baseline)

        if key == "centre":
            block["direction_accuracy"] = round(
                float(np.mean(np.sign(pred_cal) == np.sign(act_held))), 4
            )
            block["direction_base_rate"] = round(
                float(max(np.mean(act_held > 0), np.mean(act_held <= 0))), 4
            )
        results[key] = block

    per_ticker = {}
    tickers_held = held["ticker"].values
    for ticker in sorted(held["ticker"].unique()):
        m = tickers_held == ticker
        if m.sum() < 30:
            continue
        entry: dict[str, Any] = {"n": int(m.sum())}
        for key in ("width", "centre"):
            act_train, pred_train = tr[key]
            act_held, pred_held = hd[key]
            pred_cal, _ = _calibrate(pred_train, act_train, pred_held)
            const_baseline = np.full(int(m.sum()), float(np.median(act_train)))
            baselines = {"constant": const_baseline}
            if key == "width":
                pers_cal, _ = _calibrate(persistence_train, act_train, persistence_held)
                baselines["persistence"] = pers_cal[m]
            else:
                baselines["zero"] = np.zeros(int(m.sum()))
            sub = _evaluate_block(key, act_held[m], pred_cal[m], baselines, with_ci=False)
            if key == "centre":
                sub["direction_accuracy"] = round(
                    float(np.mean(np.sign(pred_cal[m]) == np.sign(act_held[m]))), 4
                )
            entry[key] = sub
        per_ticker[ticker] = entry

    # Economic significance: the MAE improvement over the *hardest* baseline
    # for each target, converted into ticks.
    mae_gain_pp = {
        "width": (results["width"]["baselines"]["persistence"]["mae_baseline"]
                  - results["width"]["mae_model"]),
        "centre": (results["centre"]["baselines"]["zero"]["mae_baseline"]
                   - results["centre"]["mae_model"]),
    }

    return {
        "tickers": tickers,
        "economic_significance": economic_significance(
            held, mae_gain_pp, realised_width_pct=hd["width"][0]),
        "n_train_rows": int(len(train)),
        "n_held_out_rows": int(len(held)),
        "train_cutoff": str(model.train_cutoff.date()),
        "held_out_end": str(held.index.max().date()),
        "width": results["width"],
        "centre": results["centre"],
        "per_ticker": per_ticker,
    }


def print_diagnosis(result: dict[str, Any]) -> None:
    if "error" in result:
        print("ERROR:", result["error"])
        return
    print("=" * 74)
    print("RANGE MODEL DIAGNOSIS -- width vs centre (phase-4 spec 13.1)")
    print("=" * 74)
    print(f"tickers        : {', '.join(result['tickers'])}")
    print(f"train rows     : {result['n_train_rows']:,} (up to {result['train_cutoff']})")
    print(f"held-out rows  : {result['n_held_out_rows']:,} (to {result['held_out_end']})")

    for key in ("width", "centre"):
        b = result[key]
        print()
        print(f"--- {key.upper()} " + "-" * (68 - len(key)))
        print(f"  mean actual {b['mean_actual']:+.3f}%   mean predicted {b['mean_predicted']:+.3f}% "
              f"(uncalibrated {b['mean_predicted_uncalibrated']:+.3f}%, "
              f"offset {b['calibration_offset_pct']:+.3f})")
        print(f"  MAE(model)  {b['mae_model']:.3f}")
        print(f"  Spearman(pred, actual) = {b['spearman_pred_vs_actual']:+.4f}")
        if key == "centre":
            print(f"  direction accuracy {b['direction_accuracy']:.1%} "
                  f"(always-guess-majority base rate {b['direction_base_rate']:.1%})")
        for name, e in b["baselines"].items():
            verdict = "BEATS baseline" if e.get("beats_baseline") else "no better than baseline"
            ci = e["skill_score_ci95"]
            print(f"  vs {name:<12} MAE {e['mae_baseline']:.3f}  "
                  f"skill {e['skill_score']:+.4f}  95% CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]  -> {verdict}")
        sn = b["shuffle_null_vs_constant"]
        print(f"  shuffle null (vs constant): real {sn['real_skill']:+.4f}  "
              f"null median {sn['null_median_skill']:+.4f}  null p95 {sn['null_p95_skill']:+.4f}")
        print(f"    -> real sits at the {sn['actual_percentile_in_null']:.1f}th percentile of the null"
              f"  [{'CLEARS 95th' if sn['clears_95th'] else 'does NOT clear 95th'}]")

    econ = result["economic_significance"]
    print()
    print("--- ECONOMIC SIGNIFICANCE (the question that actually decides it) ---")
    print(f"  pooled median: one tick = {econ['median_one_tick_pct_of_price']:.3f}% of price")
    for target, g in econ["gains_in_ticks"].items():
        mark = "TRADEABLE" if g["tradeable"] else "BELOW ONE TICK -- not tradeable"
        print(f"  {target:<7} forecast gain {g['mae_gain_pp']:+.3f}pp "
              f"= {g['gain_in_ticks']:.2f} ticks  -> {mark}")
    print(f"  {'ticker':<8}{'med close':>11}{'1 tick %':>10}{'width ticks':>13}"
          f"{'centre ticks':>14}{'day range':>11}{'in ticks':>10}")
    for ticker, e in econ["per_ticker"].items():
        print(f"  {ticker:<8}{e['median_close']:>11.3f}{e['one_tick_pct_of_price']:>10.2f}"
              f"{e['gain_in_ticks']['width']:>13.2f}{e['gain_in_ticks']['centre']:>14.2f}"
              f"{e['mean_daily_range_pct']:>10.2f}%{e['mean_daily_range_in_ticks']:>10.1f}")

    print()
    print("--- PER TICKER (no CIs: too few rows each to act on) " + "-" * 20)
    hdr = f"  {'ticker':<8}{'n':>6}{'width/persist':>16}{'centre/zero':>14}{'dir acc':>10}"
    print(hdr)
    for ticker, pt in result["per_ticker"].items():
        w = pt["width"]["baselines"]["persistence"]["skill_score"]
        c = pt["centre"]["baselines"]["zero"]["skill_score"]
        d = pt["centre"]["direction_accuracy"]
        print(f"  {ticker:<8}{pt['n']:>6}{w:>+16.4f}{c:>+14.4f}{d:>10.1%}")

    narrow = [tk for tk, e in econ["per_ticker"].items()
              if (e["mean_daily_range_in_ticks"] or 99) < 5]
    print()
    print("--- VERDICT " + "-" * 62)
    for target in ("width", "centre"):
        b, g = result[target], econ["gains_in_ticks"][target]
        ci = b["baselines"]["persistence" if target == "width" else "zero"]["skill_score_ci95"]
        has_skill = ci["lo"] > 0
        print(f"  {target:<7}: {'has' if has_skill else 'no'} conditional skill over its hardest "
              f"baseline (CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]); "
              f"gain {g['gain_in_ticks']:.2f} ticks -> "
              f"{'economically meaningful' if g['tradeable'] else 'below one tick, not tradeable'}")
    if narrow:
        print(f"  STRUCTURAL BLOCKER: {', '.join(narrow)} average under 5 ticks of range")
        print("  per day -- there is no room to put a buy band and a sell band inside")
        print("  that, whatever the forecast says. Fix the universe (spec 14/16.1)")
        print("  before fitting anything further.")
    print("  NB: the shuffle null is a LOW bar -- permuting predictions scores worse")
    print("  than a constant, so near-zero skill still 'clears' it. Read the CI and")
    print("  the tick conversion, not the null, to decide anything.")


if __name__ == "__main__":
    from tradingagents.swing_db import enabled_tickers
    print_diagnosis(diagnose(enabled_tickers()))
