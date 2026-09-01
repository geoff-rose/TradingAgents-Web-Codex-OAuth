"""Overnight hold: buy at the close, sell at the next open (phase-4 follow-up,
added 2026-08-22 at the user's request).

**Why this strategy and not another.** Splitting three years of the 26-ticker
universe showed roughly 94% of the total return accruing between the close and
the next open (+94.0% raw / +97.9% dividend-adjusted, against +6.6% intraday).
Two further facts point the same way: opening gaps FADE rather than continue
(gap >+3% is followed by a mean -0.141% rest-of-day, only 41% of such days
positive), and days after a big up day are positive only 44.6% of the time. So
the daytime session is where the losses live and the overnight window is where
the gains are. This module tests capturing that window directly.

**The cost question is the whole question.** The overnight edge is a
well-documented anomaly that survives partly because it is small per
occurrence -- roughly 0.12% per night here -- while capturing it *daily*
means a round trip every day. A round trip costs brokerage plus (at minimum)
the tick/spread on both sides. If the per-night drift is smaller than the
per-night cost, the anomaly is real and untradeable at that frequency, and
the honest alternative is simply holding, which captures the same drift with
one transaction instead of 500. `evaluate()` reports gross and net side by
side, and the buy-and-hold comparison, so this cannot be reported ambiguously.

Execution assumption: buying at the close means the closing auction, selling
at the open means the opening auction. Both are genuinely executable on ASX,
which is why this is worth testing at all -- unlike an intraday limit whose
fill time is unknown, an auction fill is at a published price.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from tradingagents.backtest import DEFAULT_BROKERAGE_PCT, fetch_daily_history, tick_size

# Half-spread paid on each side, as a fraction of price. One tick is the
# optimistic floor for a liquid name; wider for anything illiquid. Swept
# rather than point-estimated, same convention as backtest.py.
SPREAD_SWEEP_PCT = (0.0, 0.05, 0.10, 0.25)


def simulate_overnight(df: pd.DataFrame, mask: pd.Series | None = None) -> pd.DataFrame:
    """Buy at each close, sell at the next open. `mask` restricts to
    qualifying days (aligned to the BUY date); without it every night is
    traded, which is the base-rate control."""
    close, nxt_open = df["Close"], df["Open"].shift(-1)
    ret = (nxt_open - close) / close * 100
    out = pd.DataFrame({"gross_return_pct": ret, "close": close, "next_open": nxt_open})
    if mask is not None:
        out = out[mask.reindex(out.index).fillna(False)]
    return out.dropna(subset=["gross_return_pct"])


def _day_clustered_ci(frame: pd.DataFrame, n: int = 5_000, seed: int = 11) -> dict[str, Any]:
    """95% CI resampling whole DAYS -- tickers on the same night share that
    night's market move and are not independent observations."""
    if frame.empty:
        return {}
    by_day = frame.groupby(frame.index)["gross_return_pct"].mean().to_numpy()
    if len(by_day) < 3:
        return {}
    rng = np.random.default_rng(seed)
    means = by_day[rng.integers(0, len(by_day), (n, len(by_day)))].mean(axis=1)
    return {"ci_lo": round(float(np.percentile(means, 2.5)), 4),
            "ci_hi": round(float(np.percentile(means, 97.5)), 4),
            "n_days": int(len(by_day))}


CONDITIONS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # Every night -- the control.
    "all_nights": lambda df: pd.Series(True, index=df.index),
    # After a down day. Motivated by the measured reversal: down days are
    # followed by strength, and gaps down are bought back.
    "after_down_day": lambda df: df["Close"].pct_change() < 0,
    # After a big down day (<= -3%).
    "after_big_down_day": lambda df: df["Close"].pct_change() <= -0.03,
    # After an up day -- included precisely so the comparison is symmetric
    # and the reversal claim can be falsified rather than assumed.
    "after_up_day": lambda df: df["Close"].pct_change() > 0,
    # Closed strong within its own day's range (close in the top 20%).
    "closed_near_high": lambda df: (
        (df["Close"] - df["Low"]) / (df["High"] - df["Low"]).replace(0, np.nan) >= 0.8),
}


def condition_null_test(per_ticker: dict[str, pd.DataFrame],
                        cond: Callable[[pd.DataFrame], pd.Series],
                        n: int = 5_000, seed: int = 13) -> dict[str, Any]:
    """Does the CONDITION carry information, or is it just the base rate?

    Null: keep each ticker's number of selected nights identical, but choose
    WHICH nights at random from that ticker's own history. So the comparison
    isolates "does picking these particular nights help" from "does holding
    overnight help at all" -- the unconditional overnight drift is present in
    the null by construction and cannot be mistaken for conditional skill.

    This is the same logic as `backtest.permutation_test()`'s randomised
    entry, specialised to a night-selection mask, and it is the check that
    matters here: an overnight edge that any random selection of nights would
    reproduce is not a signal, it is just the anomaly.
    """
    rng = np.random.default_rng(seed)
    real_vals, pools = [], []
    for df in per_ticker.values():
        ret = ((df["Open"].shift(-1) - df["Close"]) / df["Close"] * 100).to_numpy()[:-1]
        mask = cond(df).to_numpy()[:len(ret)]
        good = ~np.isnan(ret)
        ret, mask = ret[good], mask[good]
        k = int(mask.sum())
        if k == 0 or len(ret) == 0:
            continue
        real_vals.append(ret[mask])
        pools.append((ret, k))
    if not real_vals:
        return {}
    real = float(np.concatenate(real_vals).mean())

    null = np.empty(n)
    for i in range(n):
        picks = [r[rng.choice(len(r), k, replace=False)] for r, k in pools]
        null[i] = float(np.concatenate(picks).mean())
    pct = float((null < real).mean() * 100)
    return {
        "real_avg_pct": round(real, 4),
        "null_median_pct": round(float(np.median(null)), 4),
        "null_p95_pct": round(float(np.percentile(null, 95)), 4),
        "percentile_in_null": round(pct, 1),
        "clears_95th": bool(pct >= 95),
    }


def evaluate(tickers: list[str], period: str = "3y") -> dict[str, Any]:
    """Every condition, gross and net across the spread sweep, with a
    buy-and-hold comparison over the same window."""
    per_ticker: dict[str, pd.DataFrame] = {}
    buy_hold = []
    for t in tickers:
        df = fetch_daily_history(t, period=period)
        if len(df) < 100:
            continue
        per_ticker[t] = df
        buy_hold.append((df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100)

    if not per_ticker:
        return {"error": "no price data"}

    results = {}
    for name, cond in CONDITIONS.items():
        frames = []
        for t, df in per_ticker.items():
            sub = simulate_overnight(df, cond(df))
            sub["ticker"] = t
            frames.append(sub)
        frame = pd.concat(frames) if frames else pd.DataFrame()
        if frame.empty:
            results[name] = {"n": 0}
            continue
        gross = float(frame.gross_return_pct.mean())
        entry = {
            "n_trades": len(frame),
            "avg_gross_pct": round(gross, 4),
            "total_gross_pct": round(float(frame.gross_return_pct.sum()), 1),
            "pct_positive": round(100 * float((frame.gross_return_pct > 0).mean()), 2),
            **_day_clustered_ci(frame),
        }
        if name != "all_nights":
            entry["condition_null"] = condition_null_test(per_ticker, cond)
        # Net across the spread sweep: brokerage both sides + half-spread both sides.
        for sp in SPREAD_SWEEP_PCT:
            cost = 2 * DEFAULT_BROKERAGE_PCT * 100 + 2 * sp
            entry[f"avg_net_at_spread_{sp}pct"] = round(gross - cost, 4)
        results[name] = entry

    return {
        "tickers": list(per_ticker),
        "period": period,
        "buy_and_hold_pct": round(float(np.mean(buy_hold)), 1),
        "conditions": results,
    }


def print_evaluation(r: dict[str, Any]) -> None:
    if "error" in r:
        print("ERROR:", r["error"]); return
    print("=" * 108)
    print(f"OVERNIGHT HOLD (buy at close, sell at next open) -- {len(r['tickers'])} tickers, {r['period']}")
    print("=" * 108)
    print(f"equal-weight BUY AND HOLD over the same window: {r['buy_and_hold_pct']:+.1f}%")
    print()
    hdr = (f"  {'condition':<20}{'n':>7}{'days':>6}{'avgGross':>10}{'%pos':>7}"
           f"{'net@0':>9}{'net@.05':>9}{'net@.10':>9}{'net@.25':>9}{'95% CI (gross)':>24}")
    print(hdr)
    for name, s in r["conditions"].items():
        if not s.get("n_trades"):
            print(f"  {name:<20}{0:>7}"); continue
        ci = f"[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]" if "ci_lo" in s else "n/a"
        print(f"  {name:<20}{s['n_trades']:>7}{s.get('n_days',0):>6}{s['avg_gross_pct']:>+10.4f}"
              f"{s['pct_positive']:>7.1f}"
              f"{s['avg_net_at_spread_0.0pct']:>+9.4f}{s['avg_net_at_spread_0.05pct']:>+9.4f}"
              f"{s['avg_net_at_spread_0.1pct']:>+9.4f}{s['avg_net_at_spread_0.25pct']:>+9.4f}{ci:>24}")
    print()
    print("  net@X = average per night after 0.1% brokerage round trip + X% half-spread each side.")
    print("  A condition is only interesting if it stays positive at a spread you could actually trade.")
    print()
    print("  --- does the CONDITION carry information, or is it just the overnight anomaly? ---")
    print("  (null keeps each ticker's night COUNT and randomises WHICH nights, so the")
    print("   unconditional drift is inside the null and cannot be mistaken for skill)")
    for name, s in r["conditions"].items():
        cn = s.get("condition_null")
        if not cn:
            continue
        verdict = "CLEARS 95th -- the selection adds information" if cn["clears_95th"] else "does NOT clear 95th -- no better than picking nights at random"
        print(f"    {name:<20} real {cn['real_avg_pct']:+.4f}  null median {cn['null_median_pct']:+.4f}  "
              f"p95 {cn['null_p95_pct']:+.4f}  -> {cn['percentile_in_null']:.1f}th  [{verdict}]")


if __name__ == "__main__":
    from tradingagents.screener import get_model_universe
    tickers = [r["ticker"] for r in get_model_universe(tradeable_only=True)]
    print_evaluation(evaluate(tickers))


def walk_forward(tickers: list[str], condition: str = "after_big_down_day",
                 period: str = "10y", window_months: int = 12) -> dict[str, Any]:
    """Does the condition hold in periods it was NOT chosen on?

    **Why this is the test that matters here.** There is nothing to *fit* in
    this strategy -- the rule is a fixed threshold -- so the risk isn't
    overfitting parameters, it's that the condition was **selected after
    looking at the gap table** on a 3-year window. The honest check is
    whether it survives in data that selection never touched. Fetching 10
    years and reporting every 12-month window gives roughly seven such
    windows before the discovery period, plus the discovery period itself for
    comparison.

    **Survivorship caveat, and why the null absorbs it.** The 26 tickers were
    chosen by a screener run on *today's* data (liquidity, history, range),
    so going back 10 years they are all survivors and their raw overnight
    drift is biased upward. That inflates the absolute numbers in every
    window. It does **not** inflate the conditional result, because
    `condition_null_test()` compares these nights against random nights *for
    the same tickers over the same window* -- survivorship is present on both
    sides and cancels. Read the null percentile as the finding; read the
    absolute return as an upper bound.
    """
    if condition not in CONDITIONS:
        return {"error": f"unknown condition '{condition}'"}
    cond = CONDITIONS[condition]

    per_ticker = {}
    for t in tickers:
        df = fetch_daily_history(t, period=period)
        if len(df) >= 100:
            per_ticker[t] = df
    if not per_ticker:
        return {"error": "no price data"}

    end = max(df.index[-1] for df in per_ticker.values())
    start = min(df.index[0] for df in per_ticker.values())

    windows, cursor = [], end
    while cursor > start:
        w_start = cursor - pd.DateOffset(months=window_months)
        windows.append((w_start, cursor))
        cursor = w_start
    windows.reverse()

    rows = []
    for w_start, w_end in windows:
        sliced = {t: df[(df.index >= w_start) & (df.index < w_end)]
                  for t, df in per_ticker.items()}
        sliced = {t: d for t, d in sliced.items() if len(d) >= 30}
        if len(sliced) < 3:
            continue
        frames = []
        for t, d in sliced.items():
            sub = simulate_overnight(d, cond(d))
            sub["ticker"] = t
            frames.append(sub)
        frame = pd.concat(frames) if frames else pd.DataFrame()
        if frame.empty or len(frame) < 20:
            continue
        gross = float(frame.gross_return_pct.mean())
        # Trade-weighted vs day-weighted are DIFFERENT quantities and both are
        # reported, because the gap between them is itself a finding. In a
        # broad selloff many tickers qualify on the same night, so the losing
        # nights carry far more trades than the winning ones -- the
        # trade-weighted mean then falls well below the day-weighted mean.
        # The day-clustered CI is built on day means, so it brackets the
        # DAY-weighted figure; quoting it beside the trade-weighted one (as an
        # earlier version did) produces a CI that doesn't contain its own
        # point estimate. **Trade-weighted is the deployment-relevant number**
        # -- it is what capital spread across every qualifying name actually
        # earns -- so read that, and read the spread between them as
        # concentration risk.
        day_means = frame.groupby(frame.index)["gross_return_pct"].mean()
        null = condition_null_test(sliced, cond, n=2_000)
        rows.append({
            "window_start": str(w_start.date()), "window_end": str(w_end.date()),
            "n_tickers": len(sliced), "n_trades": len(frame),
            "avg_gross_pct": round(gross, 4),
            "avg_gross_day_wtd": round(float(day_means.mean()), 4),
            "trades_per_day": round(len(frame) / max(1, frame.index.nunique()), 2),
            "net_at_0.05": round(gross - (2 * DEFAULT_BROKERAGE_PCT * 100 + 0.10), 4),
            "net_at_0.10": round(gross - (2 * DEFAULT_BROKERAGE_PCT * 100 + 0.20), 4),
            **{k: null.get(k) for k in ("null_median_pct", "percentile_in_null", "clears_95th")},
            **_day_clustered_ci(frame),
        })
    return {"condition": condition, "windows": pd.DataFrame(rows)}


def print_walk_forward(r: dict[str, Any]) -> None:
    if "error" in r:
        print("ERROR:", r["error"]); return
    w = r["windows"]
    print("=" * 116)
    print(f"WALK-FORWARD -- condition '{r['condition']}' by 12-month window")
    print("=" * 116)
    print("  The condition was CHOSEN on the most recent ~3 years. Every earlier window is")
    print("  data that selection never touched. Read `pctile` (survivorship-robust), not `avgGross`.")
    print("  avg_gross_pct is TRADE-weighted (what capital across all qualifying names earns);")
    print("  avg_gross_day_wtd is DAY-weighted and is what ci_lo/ci_hi bracket. A large gap")
    print("  between them means losses concentrate on nights when many tickers qualify at once.")
    print()
    cols = ["window_start", "window_end", "n_trades", "trades_per_day", "avg_gross_pct",
            "avg_gross_day_wtd", "net_at_0.05", "null_median_pct", "percentile_in_null",
            "ci_lo", "ci_hi"]
    cols = [c for c in cols if c in w.columns]
    print(w[cols].to_string(index=False))
    if "percentile_in_null" in w.columns:
        clears = int((w.percentile_in_null >= 95).sum())
        pos = int((w.avg_gross_pct > 0).sum())
        print()
        print(f"  windows clearing the 95th percentile of their own null: {clears} of {len(w)}")
        print(f"  windows with positive gross return:                     {pos} of {len(w)}")
        print(f"  median percentile across windows:                       {w.percentile_in_null.median():.1f}")


# ---------------------------------------------------------------------------
# Exit-hour sweep (2026-08-27, user request): the strategy above always sells
# at the opening auction. This asks whether holding INTO the session does
# better -- "buy before close, sell in the first few hours".
#
# Needs hourly bars, because a daily bar has no 11:00 price. yfinance serves
# ~2 years of hourly ASX data, which is why this is a separate function rather
# than a parameter on `evaluate()`: that one runs on 3y of daily bars and
# cannot answer an intraday-exit question at all.
#
# Prior from the gap study (same codebase, 2026-08-25): on gap-up days the
# open->15:00 window is negative in every gap bucket. If the overnight hold
# delivers a gap up, holding past the open is holding through the fade. This
# measures whether that prior survives across ALL overnight holds, not just
# the gapping ones.
# ---------------------------------------------------------------------------

SYDNEY_TZ = "Australia/Sydney"
OPEN_HOUR = 10
CLOSE_HOUR = 16


def _hourly_sessions(df):
    """[(date, open_px, {hour: close}, last_close)] for one ticker."""
    import numpy as np

    idx = df.index.tz_convert(SYDNEY_TZ) if df.index.tz is not None \
        else df.index.tz_localize(SYDNEY_TZ)
    days = [t.date() for t in idx]
    hours = np.fromiter((t.hour for t in idx), dtype=int, count=len(idx))
    o = df["Open"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)

    starts = [0]
    for i in range(1, len(days)):
        if days[i] != days[i - 1]:
            starts.append(i)
    ends = starts[1:] + [len(days)]

    out = []
    for k, a in enumerate(starts):
        b = ends[k]
        sess_hours = hours[a:b]
        op = np.nonzero(sess_hours == OPEN_HOUR)[0]
        if not len(op):
            continue
        out.append((days[a], float(o[a + op[0]]),
                    {int(sess_hours[j]): float(c[a + j]) for j in range(b - a)},
                    float(c[b - 1])))
    return out


def exit_hour_sweep(universe_limit: int = 500, period: str = "2y", batch: int = 40,
                    exit_hours: tuple = ("open", 11, 12, 13, 14, 15, "close")):
    """Buy the previous session's close, sell at each candidate exit.

    'open' is the opening auction (the existing strategy's exit); an integer H
    is the price at H:00, which is the CLOSE of the H-1 bar; 'close' is that
    session's own closing auction. Returns per-exit gross stats.
    """
    import numpy as np
    import yfinance as yf

    from .gap_study import _universe
    from .yf_lock import YF_LOCK

    tickers = _universe(universe_limit)
    acc = {h: [] for h in exit_hours}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        with YF_LOCK:
            data = yf.download([f"{t}.AX" for t in chunk], period=period, interval="1h",
                               group_by="ticker", auto_adjust=False, threads=True,
                               progress=False)
        for t in chunk:
            try:
                df = data[f"{t}.AX"].dropna(subset=["Close"])
            except (KeyError, TypeError):
                continue
            sess = _hourly_sessions(df)
            for j in range(1, len(sess)):
                d_prev, _, _, prev_close = sess[j - 1]
                d_now, open_px, closes, day_close = sess[j]
                if (d_now - d_prev).days > 5 or not prev_close:
                    continue
                for h in exit_hours:
                    if h == "open":
                        px = open_px
                    elif h == "close":
                        px = day_close
                    else:
                        px = closes.get(h - 1)      # price AT h:00 = close of h-1 bar
                    if px:
                        acc[h].append(100 * (px - prev_close) / prev_close)
    return acc
