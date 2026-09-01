"""Split each session's return into its overnight and daytime halves, by
market-cap band (2026-08-29, user request).

**The question.** Earlier work on the top-500 basket found the daytime
session (open -> close) was flat-to-negative while essentially all of the
return accrued overnight (close -> next open). That is a well-documented
anomaly, but the top-500 list is dominated by small caps, where the daytime
loss could just be the spread/impact signature of illiquid names rather than
anything about the trading day. So: does the daytime session go positive as
you move up the size ladder -- ASX 50, ASX 100 -- the way it does in QQQ/SPY,
where BOTH halves are positive?

**Bands are by market-cap rank, not official index membership.** The
`universe` table is asxbrief's Yahoo-screener top-500 ranked by market cap.
Top-50 and top-100 slices of it overlap heavily with the real ASX 50/100 but
are not the same list, and -- more importantly -- they are TODAY's ranking
applied to history. A company that has fallen out of the top 100 over the
window is absent, so these numbers are survivorship-biased UPWARD. That bias
hits the two legs unequally is unlikely (a survivor's return is split between
them by the same clock), so the LEG COMPARISON is more trustworthy than the
absolute level of either leg.

**Adjusted prices.** `auto_adjust=True`, so both legs are dividend-adjusted.
The adjustment factor steps at the ex-date, which falls between one close and
the next open -- i.e. inside the overnight leg -- so this credits dividends to
the overnight half. That is the correct total-return treatment (the holder
does receive them) but it is worth naming: a few basis points a year of the
overnight edge in a high-yield market like the ASX is dividend, not drift.

Neither leg here is a costed strategy. Both are one round trip per day, so
whichever wins gross still has to clear ~5bp of round-trip cost before it
beats simply holding, which pays that cost once. `overnight.py` is where the
costed version lives; this module only decomposes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .gap_study import _universe
from .screener import bulk_daily
from .yf_lock import YF_LOCK

TRADING_DAYS = 252

# (label, rank_lo, rank_hi) -- inclusive, 1-based. The disjoint slices are
# included so a top-100 result cannot be read as "large caps are fine" when
# it is really the top 50 carrying it.
DEFAULT_BANDS = (
    ("ASX 50 (rank 1-50)", 1, 50),
    ("ASX 100 (rank 1-100)", 1, 100),
    ("rank 51-100", 51, 100),
    ("top 500 (rank 1-500)", 1, 500),
    ("rank 101-500", 101, 500),
)


def _legs(df: pd.DataFrame) -> pd.DataFrame:
    """Overnight / daytime / full-session returns as fractions, one row per
    session. Overnight is stamped on the SELL date (the open that ends it)
    so both legs of the same calendar day line up in one row."""
    open_, close = df["Open"], df["Close"]
    prev_close = close.shift(1)
    return pd.DataFrame({
        "overnight": open_ / prev_close - 1.0,
        "daytime": close / open_ - 1.0,
        "full": close / prev_close - 1.0,
    }).replace([np.inf, -np.inf], np.nan)


def _equal_weight(per_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean across every ticker trading that day. Equal-weight rather than
    cap-weight because the question is about the typical stock in the band,
    and because cap-weighting a top-500 band would just re-report BHP."""
    frames = {t: _legs(df) for t, df in per_ticker.items() if len(df) > 2}
    if not frames:
        return pd.DataFrame()
    out = {}
    for leg in ("overnight", "daytime", "full"):
        wide = pd.DataFrame({t: f[leg] for t, f in frames.items()})
        out[leg] = wide.mean(axis=1, skipna=True)
        if leg == "overnight":
            out["n_tickers"] = wide.notna().sum(axis=1)
    return pd.DataFrame(out).dropna(subset=["overnight", "daytime"])


def _leg_stats(series: pd.Series) -> dict[str, Any]:
    """Compounded, not summed -- a daily strategy compounds, and on a series
    this noisy the difference is not cosmetic."""
    s = series.dropna()
    if len(s) < 30:
        return {"n_days": len(s)}
    total = float((1.0 + s).prod() - 1.0)
    years = len(s) / TRADING_DAYS
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else float("nan")
    mean = float(s.mean())
    sd = float(s.std(ddof=1))
    return {
        "n_days": int(len(s)),
        "mean_bp": round(mean * 10_000, 2),
        "total_pct": round(total * 100, 1),
        "cagr_pct": round(cagr * 100, 2),
        "pct_positive": round(float((s > 0).mean()) * 100, 1),
        "t_stat": round(mean / (sd / math.sqrt(len(s))), 2) if sd > 0 else None,
        "ann_vol_pct": round(sd * math.sqrt(TRADING_DAYS) * 100, 1),
    }


def _by_year(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for year, chunk in frame.groupby(frame.index.year):
        if len(chunk) < 40:
            continue
        rows.append({
            "year": int(year),
            "n_days": len(chunk),
            "overnight_pct": round(float((1 + chunk["overnight"]).prod() - 1) * 100, 1),
            "daytime_pct": round(float((1 + chunk["daytime"]).prod() - 1) * 100, 1),
            "full_pct": round(float((1 + chunk["full"]).prod() - 1) * 100, 1),
        })
    return rows


def run(bands: tuple = DEFAULT_BANDS, period: str = "5y",
        batch: int = 40) -> dict[str, Any]:
    """One fetch of the top-500 daily history, sliced into bands. Fetching
    once and slicing keeps every band on identical price data, so a
    difference between bands cannot be a difference in what was downloaded."""
    max_rank = max(hi for _, _, hi in bands)
    tickers = _universe(max_rank)
    if not tickers:
        return {"error": "no universe available"}

    with YF_LOCK:
        history = bulk_daily(tickers, period=period, batch=batch)
    if not history:
        return {"error": "no price history returned"}

    results = []
    for label, lo, hi in bands:
        members = [t for t in tickers[lo - 1:hi] if t in history]
        frame = _equal_weight({t: history[t] for t in members})
        if frame.empty:
            results.append({"band": label, "error": "no usable data"})
            continue
        results.append({
            "band": label,
            "rank_lo": lo, "rank_hi": hi,
            "n_tickers": len(members),
            "start": str(frame.index[0].date()),
            "end": str(frame.index[-1].date()),
            "overnight": _leg_stats(frame["overnight"]),
            "daytime": _leg_stats(frame["daytime"]),
            "full": _leg_stats(frame["full"]),
            "by_year": _by_year(frame),
        })
    return {
        "period": period,
        "universe_size": len(tickers),
        "fetched": len(history),
        "bands": results,
    }


def print_run(r: dict[str, Any]) -> None:
    if "error" in r:
        print(f"error: {r['error']}")
        return
    print(f"\nSession decomposition -- {r['period']}, equal-weight, "
          f"dividend-adjusted ({r['fetched']}/{r['universe_size']} tickers fetched)")
    print("Overnight = prev close -> open.  Daytime = open -> close.  Gross of costs.\n")
    head = f"{'band':<24}{'n':>4}  {'leg':<10}{'mean bp':>9}{'CAGR':>9}{'total':>10}{'%pos':>7}{'t':>7}{'vol':>7}"
    print(head)
    print("-" * len(head))
    for b in r["bands"]:
        if "error" in b:
            print(f"{b['band']:<24}  {b['error']}")
            continue
        for i, leg in enumerate(("overnight", "daytime", "full")):
            s = b[leg]
            if "mean_bp" not in s:
                continue
            name = b["band"] if i == 0 else ""
            n = str(b["n_tickers"]) if i == 0 else ""
            print(f"{name:<24}{n:>4}  {leg:<10}{s['mean_bp']:>9.2f}"
                  f"{s['cagr_pct']:>8.1f}%{s['total_pct']:>9.1f}%"
                  f"{s['pct_positive']:>6.1f}%{s['t_stat']:>7.1f}{s['ann_vol_pct']:>6.1f}%")
        print()
    print("Per-year totals (%):")
    for b in r["bands"]:
        if "error" in b or not b.get("by_year"):
            continue
        print(f"  {b['band']}")
        print(f"    {'year':<8}{'overnight':>11}{'daytime':>10}{'full':>10}")
        for y in b["by_year"]:
            print(f"    {y['year']:<8}{y['overnight_pct']:>10.1f}%"
                  f"{y['daytime_pct']:>9.1f}%{y['full_pct']:>9.1f}%")


if __name__ == "__main__":
    import sys
    period = sys.argv[1] if len(sys.argv) > 1 else "5y"
    print_run(run(period=period))


# ---------------------------------------------------------------------------
# Per-stock daytime performance (2026-08-29, user request).
#
# "Are there individual ASX 300 names that regularly perform well during the
# daytime session?" With 300 candidates the naive version of this question
# answers itself: roughly 7-8 will clear t=2 on pure noise, and picking those
# is data mining, not a finding. So the test that matters is PERSISTENCE --
# fit on the first half, measure on the second, and see whether the ranking
# survives. If it does not, there are no such stocks, only lucky ones.
#
# Two further controls:
#   * Market-relative. Every ASX stock shares the same daytime market factor,
#     so raw daytime means are heavily cross-correlated and a naive t-stat
#     overstates significance. Demeaning by the cross-sectional daily average
#     asks the sharper question: does this stock beat the AVERAGE stock during
#     the day? That is also the only version that is tradeable long/short.
#   * Price and turnover floors. The band study showed low-priced names carry
#     a large mechanical daytime bias from bid-ask bounce; without a floor the
#     ranking is a list of penny stocks at whichever end of the bounce.
# ---------------------------------------------------------------------------

MIN_PRICE = 1.0            # median close, dollars
MIN_TURNOVER = 500_000     # median daily dollar volume


def per_stock(universe_limit: int = 300, period: str = "10y", batch: int = 40,
              split: str = "2021-01-01") -> dict[str, Any]:
    tickers = _universe(universe_limit)
    if not tickers:
        return {"error": "no universe available"}
    with YF_LOCK:
        history = bulk_daily(tickers, period=period, batch=batch)

    legs, meta = {}, {}
    for t, df in history.items():
        if len(df) < 500:
            continue
        med_px = float(df["Close"].median())
        med_to = float((df["Close"] * df["Volume"]).median())
        if med_px < MIN_PRICE or med_to < MIN_TURNOVER:
            continue
        legs[t] = _legs(df)
        meta[t] = {"median_price": round(med_px, 2),
                   "median_turnover": int(med_to)}
    if len(legs) < 20:
        return {"error": f"only {len(legs)} tickers cleared the liquidity floor"}

    day = pd.DataFrame({t: f["daytime"] for t, f in legs.items()})
    night = pd.DataFrame({t: f["overnight"] for t, f in legs.items()})
    # Market-relative: subtract the equal-weight daytime move of that day.
    rel = day.sub(day.mean(axis=1, skipna=True), axis=0)

    def col_stats(s: pd.Series) -> tuple[float, float, int]:
        s = s.dropna()
        if len(s) < 100:
            return float("nan"), float("nan"), len(s)
        sd = s.std(ddof=1)
        t_ = s.mean() / (sd / math.sqrt(len(s))) if sd > 0 else float("nan")
        return float(s.mean() * 10_000), float(t_), len(s)

    rows = []
    for t in day.columns:
        d_bp, d_t, n = col_stats(day[t])
        r_bp, r_t, _ = col_stats(rel[t])
        n_bp, _, _ = col_stats(night[t])
        full = _leg_stats(legs[t]["full"])
        dayc = _leg_stats(legs[t]["daytime"])
        rows.append({
            "ticker": t, "n_days": n,
            "daytime_bp": round(d_bp, 2), "daytime_t": round(d_t, 2),
            "rel_daytime_bp": round(r_bp, 2), "rel_daytime_t": round(r_t, 2),
            "overnight_bp": round(n_bp, 2),
            "daytime_cagr_pct": dayc.get("cagr_pct"),
            "full_cagr_pct": full.get("cagr_pct"),
            **meta[t],
        })
    rows.sort(key=lambda r: -r["daytime_bp"])

    # --- persistence: rank in the first half, score in the second ----------
    train, test = day.loc[:split], day.loc[split:]
    rel_tr, rel_te = rel.loc[:split], rel.loc[split:]
    ok = [t for t in day.columns
          if train[t].notna().sum() > 200 and test[t].notna().sum() > 200]
    tr = pd.Series({t: train[t].mean() * 10_000 for t in ok})
    te = pd.Series({t: test[t].mean() * 10_000 for t in ok})
    rtr = pd.Series({t: rel_tr[t].mean() * 10_000 for t in ok})
    rte = pd.Series({t: rel_te[t].mean() * 10_000 for t in ok})

    def decile(rank_on: pd.Series, score_on: pd.Series, frac: float = 0.1) -> dict[str, Any]:
        k = max(3, int(len(rank_on) * frac))
        top = rank_on.nlargest(k).index
        bot = rank_on.nsmallest(k).index
        return {"n": k,
                "top_train_bp": round(float(rank_on[top].mean()), 2),
                "top_test_bp": round(float(score_on[top].mean()), 2),
                "bottom_train_bp": round(float(rank_on[bot].mean()), 2),
                "bottom_test_bp": round(float(score_on[bot].mean()), 2),
                "all_test_bp": round(float(score_on.mean()), 2)}

    persistence = {
        "split": split,
        "n_tickers": len(ok),
        "train_days": int(len(train)), "test_days": int(len(test)),
        "raw": {"spearman": round(float(tr.corr(te, method="spearman")), 3),
                "pearson": round(float(tr.corr(te)), 3),
                "decile": decile(tr, te)},
        "relative": {"spearman": round(float(rtr.corr(rte, method="spearman")), 3),
                     "pearson": round(float(rtr.corr(rte)), 3),
                     "decile": decile(rtr, rte)},
    }

    n_sig = sum(1 for r in rows if r["rel_daytime_t"] and r["rel_daytime_t"] > 1.96)
    n_sig_neg = sum(1 for r in rows if r["rel_daytime_t"] and r["rel_daytime_t"] < -1.96)
    return {
        "period": period, "universe_limit": universe_limit,
        "n_screened": len(history), "n_kept": len(legs),
        "min_price": MIN_PRICE, "min_turnover": MIN_TURNOVER,
        "rows": rows,
        "n_rel_sig_positive": n_sig, "n_rel_sig_negative": n_sig_neg,
        "expected_by_chance": round(len(rows) * 0.025, 1),
        "persistence": persistence,
    }


def print_per_stock(r: dict[str, Any], top: int = 15) -> None:
    if "error" in r:
        print(f"error: {r['error']}")
        return
    print(f"\nPer-stock daytime session -- top {r['universe_limit']} by market cap, "
          f"{r['period']}, dividend-adjusted")
    print(f"{r['n_kept']}/{r['n_screened']} kept (median price >= ${r['min_price']:.0f}, "
          f"median turnover >= ${r['min_turnover']:,})\n")
    hdr = (f"{'ticker':<8}{'day bp':>8}{'t':>6}{'rel bp':>8}{'rel t':>7}"
           f"{'night bp':>10}{'day CAGR':>10}{'full CAGR':>11}{'med $':>8}")
    print("BEST daytime sessions:")
    print(hdr); print("-" * len(hdr))
    for row in r["rows"][:top]:
        print(f"{row['ticker']:<8}{row['daytime_bp']:>8.2f}{row['daytime_t']:>6.2f}"
              f"{row['rel_daytime_bp']:>8.2f}{row['rel_daytime_t']:>7.2f}"
              f"{row['overnight_bp']:>10.2f}{row['daytime_cagr_pct']:>9.1f}%"
              f"{row['full_cagr_pct']:>10.1f}%{row['median_price']:>8.2f}")
    print("\nWORST daytime sessions:")
    print(hdr); print("-" * len(hdr))
    for row in r["rows"][-5:]:
        print(f"{row['ticker']:<8}{row['daytime_bp']:>8.2f}{row['daytime_t']:>6.2f}"
              f"{row['rel_daytime_bp']:>8.2f}{row['rel_daytime_t']:>7.2f}"
              f"{row['overnight_bp']:>10.2f}{row['daytime_cagr_pct']:>9.1f}%"
              f"{row['full_cagr_pct']:>10.1f}%{row['median_price']:>8.2f}")
    print(f"\nMarket-relative significance: {r['n_rel_sig_positive']} positive / "
          f"{r['n_rel_sig_negative']} negative at |t|>1.96; "
          f"~{r['expected_by_chance']} expected each way by chance.")
    p = r["persistence"]
    print(f"\nPersistence -- rank on data before {p['split']}, score after "
          f"({p['train_days']} train / {p['test_days']} test days, {p['n_tickers']} tickers):")
    for key in ("raw", "relative"):
        b = p[key]; d = b["decile"]
        print(f"  {key:<9} spearman {b['spearman']:+.3f}  pearson {b['pearson']:+.3f}")
        print(f"            top decile  {d['top_train_bp']:+7.2f}bp in train -> "
              f"{d['top_test_bp']:+7.2f}bp in test")
        print(f"            bot decile  {d['bottom_train_bp']:+7.2f}bp in train -> "
              f"{d['bottom_test_bp']:+7.2f}bp in test")
        print(f"            all tickers                    -> {d['all_test_bp']:+7.2f}bp in test")


# ---------------------------------------------------------------------------
# Does a big overnight gap predict the daytime session? (2026-08-29, user
# request: "should I sell after an overnight gap because it gets dragged down
# during the day?")
#
# **The confound this is built to defeat.** Conditioning on a positive gap
# selects days whose OPEN printed high in the spread. If the open prints at
# the ask and the close at the bid, open->close is mechanically negative by
# roughly the spread -- with no information in it whatsoever. That artifact
# grows as price falls and liquidity thins, which is exactly the gradient the
# band study found. So this reports by market-cap band: if the effect is real
# it survives in the ASX 50, where the spread is a few basis points and cannot
# manufacture a 50bp result. If it only exists below rank 100, it is the
# spread talking.
#
# **Why the cost bar is different here.** This is not a strategy that trades
# daily -- it is a question about WHICH SIDE OF ONE DAY to exit a position the
# holder is selling anyway. Brokerage is paid either way, so only the
# difference in spread between the auctions counts. An effect far too small to
# trade can still be worth acting on for an exit already decided.
# ---------------------------------------------------------------------------

GAP_BUCKETS = ((-99, -3), (-3, -1), (-1, -0.25), (-0.25, 0.25),
               (0.25, 1), (1, 3), (3, 99))


def gap_conditioned_daytime(universe_limit: int = 300, period: str = "10y",
                            batch: int = 40,
                            exclude: tuple[str, str] | None = None) -> dict[str, Any]:
    """`exclude` drops a date range (e.g. the COVID window). Excluding a
    period is itself a choice that can flatter a result, so the caller
    names the window and it is echoed back in the output rather than being
    silently baked in."""
    tickers = _universe(universe_limit)
    if not tickers:
        return {"error": "no universe available"}
    with YF_LOCK:
        history = bulk_daily(tickers, period=period, batch=batch)

    bands = (("ASX 50", 1, 50), ("rank 51-100", 51, 100), ("rank 101-300", 101, 300))
    out = []
    for label, lo, hi in bands:
        gaps, days, kept = [], [], 0
        for t in tickers[lo - 1:hi]:
            df = history.get(t)
            if df is None or len(df) < 250:
                continue
            if float(df["Close"].median()) < MIN_PRICE:
                continue
            if float((df["Close"] * df["Volume"]).median()) < MIN_TURNOVER:
                continue
            kept += 1
            legs = _legs(df).dropna()
            if exclude:
                lo_d, hi_d = exclude
                legs = legs[~((legs.index >= lo_d) & (legs.index <= hi_d))]
            if legs.empty:
                continue
            gaps.append(legs["overnight"] * 100)
            days.append(legs["daytime"] * 100)
        if not gaps:
            continue
        g = pd.concat(gaps).to_numpy()
        d = pd.concat(days).to_numpy()
        rows = []
        for lo_b, hi_b in GAP_BUCKETS:
            m = (g >= lo_b) & (g < hi_b)
            if m.sum() < 50:
                continue
            sub = d[m]
            sd = sub.std(ddof=1)
            rows.append({
                "gap_bucket": _bucket_label(lo_b, hi_b),
                "n": int(m.sum()),
                "mean_daytime_pct": round(float(sub.mean()), 3),
                "median_daytime_pct": round(float(np.median(sub)), 3),
                "pct_positive": round(float((sub > 0).mean()) * 100, 1),
                "t_stat": round(float(sub.mean() / (sd / math.sqrt(len(sub)))), 1) if sd > 0 else None,
            })
        out.append({"band": label, "n_tickers": kept, "n_obs": int(len(d)),
                    "overall_daytime_pct": round(float(d.mean()), 3),
                    "buckets": rows})
    return {"period": period, "excluded": exclude, "bands": out}


def _bucket_label(lo: float, hi: float) -> str:
    if lo <= -99:
        return f"gap < {hi:g}%"
    if hi >= 99:
        return f"gap > +{lo:g}%"
    return f"{lo:+g}% to {hi:+g}%"


def print_gap_conditioned(r: dict[str, Any]) -> None:
    if "error" in r:
        print(f"error: {r['error']}")
        return
    print(f"\nDaytime (open->close) return conditioned on the overnight gap "
          f"-- {r['period']}, dividend-adjusted")
    if r.get("excluded"):
        print(f"EXCLUDING {r['excluded'][0]} .. {r['excluded'][1]}")
    print("Each row: given a gap of this size, what did the following session do?\n")
    for b in r["bands"]:
        print(f"{b['band']}  ({b['n_tickers']} tickers, {b['n_obs']:,} stock-days, "
              f"unconditional daytime mean {b['overall_daytime_pct']:+.3f}%)")
        hdr = f"  {'gap bucket':<16}{'n':>9}{'mean day':>11}{'median':>10}{'%pos':>8}{'t':>8}"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for row in b["buckets"]:
            print(f"  {row['gap_bucket']:<16}{row['n']:>9,}{row['mean_daytime_pct']:>10.3f}%"
                  f"{row['median_daytime_pct']:>9.3f}%{row['pct_positive']:>7.1f}%"
                  f"{row['t_stat']:>8.1f}")
        print()


# ---------------------------------------------------------------------------
# WHEN during the session does the gap reversion happen? (2026-08-29.)
#
# The daily study measures open->close and so cannot say whether a holder must
# sell into the opening auction to capture the reversion or can wait until the
# afternoon. Only hourly bars answer that, and yfinance caps ASX intraday
# history at ~2 years -- so this is a much smaller sample than the daily work
# and is a shape estimate, not a significance test.
#
# ASX hourly bars are stamped at the START of the hour and the session runs
# 10:00-16:00 Sydney, giving 7 bars. The 16:00 bar is the CLOSING AUCTION and
# prints O=H=L=C, frequently away from the last continuous trade at 15:59 --
# so the auction is reported as its own leg rather than folded into "the
# close", because a holder who sells at 15:55 does not get the auction price.
# ---------------------------------------------------------------------------

SYDNEY_TZ = "Australia/Sydney"


def _bulk_hourly(tickers: list[str], period: str = "2y",
                 batch: int = 30) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        data = yf.download([f"{t}.AX" for t in chunk], period=period, interval="1h",
                           group_by="ticker", auto_adjust=True, threads=True,
                           progress=False)
        for t in chunk:
            try:
                df = data[f"{t}.AX"].dropna(subset=["Close"])
            except KeyError:
                continue
            if not df.empty:
                out[t] = df.tz_convert(SYDNEY_TZ)
    return out


def intraday_shape(universe_limit: int = 300, period: str = "2y", batch: int = 30,
                   gap_cuts: tuple = (3.0, 2.0, 1.0)) -> dict[str, Any]:
    tickers = _universe(universe_limit)
    if not tickers:
        return {"error": "no universe available"}
    with YF_LOCK:
        hourly = _bulk_hourly(tickers, period=period, batch=batch)
    if not hourly:
        return {"error": "no hourly history returned"}

    # One row per stock-session: the gap, then cumulative open->X for each hour.
    checkpoints = [10, 11, 12, 13, 14, 15]   # bar START hours, continuous
    records: dict[str, list[pd.DataFrame]] = {}
    bands = (("ASX 50", 1, 50), ("rank 51-100", 51, 100), ("rank 101-300", 101, 300))
    band_of = {}
    for label, lo, hi in bands:
        for t in tickers[lo - 1:hi]:
            band_of[t] = label
        records[label] = []

    for t, df in hourly.items():
        if t not in band_of or len(df) < 200:
            continue
        if float(df["Close"].median()) < MIN_PRICE:
            continue
        sess = df.groupby(df.index.date)
        rows = []
        prev_auction = None
        for day, chunk in sess:
            by_hour = {h: chunk[chunk.index.hour == h] for h in checkpoints + [16]}
            first = by_hour[10]
            if first.empty or by_hour[16].empty:
                prev_auction = float(by_hour[16]["Close"].iloc[0]) if not by_hour[16].empty else prev_auction
                continue
            o = float(first["Open"].iloc[0])
            auction = float(by_hour[16]["Close"].iloc[0])
            if prev_auction and o > 0:
                row = {"gap": (o / prev_auction - 1) * 100}
                for h in checkpoints:
                    b = by_hour[h]
                    row[f"by_{h + 1}"] = (float(b["Close"].iloc[0]) / o - 1) * 100 if not b.empty else np.nan
                row["auction"] = (auction / o - 1) * 100
                rows.append(row)
            prev_auction = auction
        if rows:
            records[band_of[t]].append(pd.DataFrame(rows))

    labels = [f"by {h + 1}:00" for h in checkpoints] + ["close (auction)"]
    cols = [f"by_{h + 1}" for h in checkpoints] + ["auction"]
    out = []
    for label, _, _ in bands:
        if not records[label]:
            continue
        f = pd.concat(records[label], ignore_index=True)
        band = {"band": label, "n_sessions": int(len(f)), "cuts": []}
        for cut in gap_cuts:
            for side, m in (("up", f["gap"] > cut), ("down", f["gap"] < -cut)):
                sub = f[m]
                if len(sub) < 60:
                    continue
                path = [round(float(sub[c].mean()), 3) for c in cols]
                total = path[-1]
                band["cuts"].append({
                    "condition": f"gap {'>' if side == 'up' else '<'} "
                                 f"{'+' if side == 'up' else '-'}{cut:g}%",
                    "n": int(len(sub)),
                    "path": path,
                    "share_done": [round(p / total * 100, 0) if total else None for p in path],
                })
        out.append(band)
    return {"period": period, "labels": labels, "bands": out}


def print_intraday_shape(r: dict[str, Any]) -> None:
    if "error" in r:
        print(f"error: {r['error']}")
        return
    print(f"\nWhere in the session does the gap reversion happen? "
          f"({r['period']} of hourly bars, Sydney time)")
    print("Cumulative mean return from the OPEN, %. Last column is the closing auction.\n")
    for b in r["bands"]:
        print(f"{b['band']}  ({b['n_sessions']:,} stock-sessions)")
        hdr = f"  {'condition':<14}{'n':>7}" + "".join(f"{l:>16}" for l in r["labels"])
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for c in b["cuts"]:
            print(f"  {c['condition']:<14}{c['n']:>7,}"
                  + "".join(f"{v:>15.3f}%" for v in c["path"]))
            print(f"  {'':<14}{'% of move':>7}"
                  + "".join(f"{s:>15.0f}%" if s is not None else f"{'-':>16}"
                            for s in c["share_done"]))
        print()
