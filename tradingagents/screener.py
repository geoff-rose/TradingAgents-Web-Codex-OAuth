"""Universe screener (phase-3 §4, phase-4 §14/§16) -- pick the ~30 tickers the
range model and the swing strategy are actually allowed to look at.

**Why this exists, and why the obvious screen is wrong.** The intuitive filter
is "keep stocks that move at least 5% in a day." Every one of the five tickers
in the current universe already passes it -- BRN/INR/MEI/TTT average 6.2-8.0%
daily range -- and range trading is still structurally impossible on them.
Percentage range is the wrong unit.

The unit that matters is **ticks**. ASX prices move on a fixed grid (0.1c below
10c, 0.5c from 10c to $2.00, 1c above $2.00), so a 16c stock moves in 3.1%
jumps. Its "6% daily range" is *two price steps*. You cannot buy the low of a
two-step day and sell the high of it, because the bid-ask spread is itself
about one step -- entering and exiting consumes the entire move. The same 6%
on a $30 stock is 180 one-cent steps, which is all the room anyone needs.

So the headline screen here is `median_range_ticks`, not `median_range_pct`,
and `pct_days_range_ge_5pct` is computed only so the two can be compared side
by side and the difference seen directly. See `range_diagnostics.py` for the
measurement that forced this conclusion.

**Price used for the tick band is the db's current unadjusted price, not the
adjusted history.** yfinance returns split/dividend-adjusted OHLC, and a tick
band is a property of the price a stock actually trades at today -- computing
it from an adjusted 2023 close would put a stock in the wrong band entirely.
Range *percentages* are scale-invariant so they can come from adjusted data
safely; the tick conversion cannot.

Spread is estimated as **one tick, which is optimistic** -- no real spread
history exists yet (phase-3 §5.1's preopen/spread capture is what would fix
that). Every liquidity- and cost-sensitive number here is therefore a
best case, and a candidate that fails on an optimistic spread would fail
harder on a real one.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from tradingagents.backtest import DEFAULT_BROKERAGE_PCT, tick_size

ASX_DB = "/opt/asxbrief/data/asx.db"

# Screen thresholds. Deliberately named constants rather than inline magic --
# these are the knobs a future session will want to move, and moving them
# silently is how a universe drifts.
MIN_HISTORY_DAYS = 700          # ~3 years of trading days (spec §16.3)
MIN_TURNOVER_AUD = 300_000      # median daily $ turnover (spec §16.3)
MID_BUCKET_TURNOVER_AUD = 1_000_000   # "raise it for the mid bucket"
MIN_RANGE_EFFICIENCY = 3.0      # ATR14 / (spread + 2*brokerage) (spec §16.3)
MIN_RANGE_TICKS = 8.0           # the corrected version of "moves 5% a day"
TURNOVER_WINDOW_DAYS = 120      # recent liquidity, not a 3-year average

LARGE_CAP_MIN = 10e9
MID_CAP_MIN = 1e9

EXISTING_UNIVERSE = ["BRN", "INR", "MEI", "PNV", "TTT"]


@dataclass
class Candidate:
    ticker: str
    company: str
    market_cap: float
    price: float
    metrics: dict[str, Any] = field(default_factory=dict)
    sector: str | None = None

    @property
    def cap_bucket(self) -> str:
        if self.market_cap >= LARGE_CAP_MIN:
            return "large"
        if self.market_cap >= MID_CAP_MIN:
            return "mid"
        return "small"


def load_candidates(limit: int = 500, db_path: str = ASX_DB) -> list[Candidate]:
    """Top-N ASX names by market cap, from the collector's own universe table
    (built via Yahoo's screener -- see the asxbrief-collector skill for why
    not ASX's own CSV)."""
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT ticker, company, price, market_cap FROM universe "
            "WHERE price IS NOT NULL AND market_cap IS NOT NULL "
            "ORDER BY rank LIMIT ?", (limit,),
        ).fetchall()
    finally:
        con.close()
    return [Candidate(ticker=t, company=c, price=p, market_cap=m) for t, c, p, m in rows]


def bulk_daily(tickers: list[str], period: str = "3y", batch: int = 60) -> dict[str, pd.DataFrame]:
    """Batched yfinance download. Batched rather than one-shot because a
    single 500-symbol request is one failure away from losing everything,
    and rather than per-ticker because that is 500 round trips."""
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        data = yf.download(
            [f"{t}.AX" for t in chunk], period=period, interval="1d",
            group_by="ticker", auto_adjust=True, threads=True, progress=False,
        )
        for t in chunk:
            key = f"{t}.AX"
            try:
                df = data[key].dropna(subset=["Close"])
            except KeyError:
                continue
            if not df.empty:
                out[t] = df
    return out


def variance_ratio(close: pd.Series, lag: int = 5) -> float:
    """Var(lag-day returns) / (lag * Var(1-day returns)). Below 1 means
    mean-reverting, above 1 trending. Phase-3's own screener metric, reused
    here rather than reimplemented so the two agree."""
    r1 = np.log(close).diff().dropna()
    rl = np.log(close).diff(lag).dropna()
    if len(r1) < lag * 10 or r1.var() == 0:
        return float("nan")
    return float(rl.var() / (lag * r1.var()))


def compute_metrics(cand: Candidate, df: pd.DataFrame) -> dict[str, Any]:
    """All screen metrics for one candidate. `tick_pct` comes from the
    candidate's CURRENT price (see module docstring); everything else from
    adjusted history."""
    if len(df) < 60:
        return {"n_days": len(df), "usable": False}

    high, low, close, vol = df["High"], df["Low"], df["Close"], df["Volume"]
    range_pct = ((high - low) / close * 100).dropna()

    tick_pct = tick_size(cand.price) / cand.price * 100
    range_ticks = range_pct / tick_pct

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14_pct = float((tr.rolling(14).mean() / close * 100).dropna().iloc[-1]) if len(tr) > 14 else float("nan")

    recent = df.tail(TURNOVER_WINDOW_DAYS)
    turnover = float((recent["Close"] * recent["Volume"]).median())

    # Optimistic: assumes the spread is exactly one tick. Real spreads on
    # illiquid names are wider, so this flatters every candidate equally.
    round_trip_cost_pct = tick_pct + 2 * DEFAULT_BROKERAGE_PCT * 100
    range_efficiency = atr14_pct / round_trip_cost_pct if round_trip_cost_pct else float("nan")

    return {
        "n_days": len(df),
        "usable": True,
        "tick_size": tick_size(cand.price),
        "tick_pct": round(tick_pct, 4),
        "median_range_pct": round(float(range_pct.median()), 3),
        "median_range_ticks": round(float(range_ticks.median()), 2),
        "pct_days_range_ge_5pct": round(float((range_pct >= 5).mean() * 100), 1),
        "pct_days_range_ge_min_ticks": round(float((range_ticks >= MIN_RANGE_TICKS).mean() * 100), 1),
        "atr14_pct": round(atr14_pct, 3),
        "range_efficiency": round(range_efficiency, 2),
        "median_turnover_aud": round(turnover),
        "variance_ratio": round(variance_ratio(close), 3),
    }


def screen(limit: int = 500, period: str = "3y") -> list[Candidate]:
    """Load candidates and attach metrics. No filtering yet -- filtering is
    separate so the full distribution can be inspected before thresholds are
    chosen, rather than thresholds being picked first and justified after."""
    cands = load_candidates(limit)
    hist = bulk_daily([c.ticker for c in cands], period=period)
    for c in cands:
        df = hist.get(c.ticker)
        c.metrics = compute_metrics(c, df) if df is not None else {"n_days": 0, "usable": False}
    return cands


def to_frame(cands: list[Candidate]) -> pd.DataFrame:
    rows = []
    for c in cands:
        if not c.metrics.get("usable"):
            continue
        rows.append({
            "ticker": c.ticker, "company": c.company, "price": c.price,
            "market_cap": c.market_cap, "cap_bucket": c.cap_bucket,
            "sector": c.sector, **c.metrics,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Selection (spec §16.2-§16.4)
# ---------------------------------------------------------------------------

MAX_PER_SECTOR = {"large": 2, "mid": 3, "small": 3}
TARGET_COUNTS = {"large": 8, "mid": 9, "small": 8}


def fetch_sectors(tickers: list[str]) -> dict[str, str]:
    """Sector per ticker via yfinance. Called only for names that already
    passed the numeric screen -- the collector's `universe.industry` column
    is entirely NULL, so there is no cheaper local source, and 500 lookups
    to sector-tag names that will be filtered out anyway is waste."""
    out: dict[str, str] = {}
    for t in tickers:
        try:
            out[t] = yf.Ticker(f"{t}.AX").info.get("sector") or "Unknown"
        except Exception:
            out[t] = "Unknown"
    return out


def passes(row: pd.Series, bucket: str) -> tuple[bool, list[str]]:
    """Spec §16.3's selection rules. Returns (pass, reasons_failed) rather
    than a bare bool so a rejected candidate can be explained -- silent
    filtering is how a universe ends up unexplainable six weeks later.

    Note what is NOT screened here because the data does not exist locally:
    live scheme/takeover/suspension status, and cash runway for the
    speculative end. Both are in the spec's list. They must be eyeballed on
    the shortlist before anything is traded -- BET's acquisition is the
    standing reminder that names go stale.
    """
    fails = []
    if row["n_days"] < MIN_HISTORY_DAYS:
        fails.append(f"history {int(row['n_days'])}d < {MIN_HISTORY_DAYS}")
    floor = MID_BUCKET_TURNOVER_AUD if bucket == "mid" else MIN_TURNOVER_AUD
    if row["median_turnover_aud"] < floor:
        fails.append(f"turnover ${row['median_turnover_aud']:,.0f} < ${floor:,}")
    if row["range_efficiency"] < MIN_RANGE_EFFICIENCY:
        fails.append(f"range_efficiency {row['range_efficiency']:.2f} < {MIN_RANGE_EFFICIENCY}")
    if row["median_range_ticks"] < MIN_RANGE_TICKS:
        fails.append(f"median_range_ticks {row['median_range_ticks']:.2f} < {MIN_RANGE_TICKS}")
    # Variance ratio < 1 (mean-reverting) is required only for the small
    # bucket, where the strategy is actually aimed. The anchors are in the
    # pool to make the model estimable, not to be traded, so a trending
    # large cap is fine (spec §16.3).
    if bucket == "small" and not (row["variance_ratio"] < 1):
        fails.append(f"variance_ratio {row['variance_ratio']:.3f} >= 1 (not mean-reverting)")
    return (not fails), fails


def select_bucket(df: pd.DataFrame, bucket: str, exclude: set[str]) -> pd.DataFrame:
    """Rank a bucket's qualifying names and pick the target count, capping
    per-sector concentration so 'nine miners' can't happen (spec §16.2).

    Ranked by `range_efficiency` -- ATR relative to what a round trip costs.
    That is the quantity the strategy actually consumes, and it already
    blends volatility with the tick/spread constraint, so ranking on it
    avoids the trap of ranking on raw volatility and re-importing the exact
    problem this screener exists to prevent.
    """
    sub = df[(df.cap_bucket == bucket) & (~df.ticker.isin(exclude))].copy()
    keep = sub.apply(lambda r: passes(r, bucket)[0], axis=1)
    sub = sub[keep].sort_values("range_efficiency", ascending=False)

    picked, per_sector = [], {}
    cap = MAX_PER_SECTOR[bucket]
    for _, row in sub.iterrows():
        sector = row["sector"] or "Unknown"
        if per_sector.get(sector, 0) >= cap:
            continue
        picked.append(row)
        per_sector[sector] = per_sector.get(sector, 0) + 1
        if len(picked) >= TARGET_COUNTS[bucket]:
            break
    return pd.DataFrame(picked)


def effective_independent_positions(returns: pd.DataFrame) -> dict[str, Any]:
    """Spec §16.4: 30 tickers that move together are not 30 bets.

    Effective N = 1 / sum(normalised eigenvalue^2) of the correlation matrix
    (the participation-ratio form the spec names). Reported alongside the
    universe so the number of *independent* positions stays visible rather
    than the headline count flattering the cross-sectional power.
    """
    corr = returns.corr().dropna(how="all").dropna(axis=1, how="all")
    if corr.empty or corr.isna().any().any():
        return {"error": "insufficient overlapping history"}
    eig = np.linalg.eigvalsh(corr.values)
    eig = eig[eig > 0]
    w = eig / eig.sum()
    return {
        "n_tickers": int(corr.shape[0]),
        "effective_independent_positions": round(float(1.0 / np.sum(w ** 2)), 2),
        "mean_pairwise_corr": round(float(
            corr.values[np.triu_indices_from(corr.values, k=1)].mean()), 3),
    }


def build_universe(limit: int = 500, period: str = "3y") -> dict[str, Any]:
    """The full §14/§16 pipeline: screen -> sector-tag survivors -> pick
    8 large / 9 mid / 8 small around the 5 existing names -> report
    correlation structure."""
    cands = screen(limit=limit, period=period)

    # Carry the existing universe even if it has dropped out of the top-500
    # (TTT has). Spec §16.2 says keep all five for continuity; they must
    # still be measured, or the comparison that motivated this whole exercise
    # can't be shown.
    have = {c.ticker for c in cands}
    missing = [t for t in EXISTING_UNIVERSE if t not in have]
    if missing:
        extra_hist = bulk_daily(missing, period=period)
        con = sqlite3.connect(ASX_DB)
        try:
            for t in missing:
                df = extra_hist.get(t)
                if df is None or df.empty:
                    continue
                price = float(df["Close"].iloc[-1])
                c = Candidate(ticker=t, company=f"{t} (outside top-500)",
                              price=price, market_cap=0.0)
                c.metrics = compute_metrics(c, df)
                cands.append(c)
        finally:
            con.close()

    df = to_frame(cands)

    # Sector-tag only names that could plausibly be selected.
    shortlist = df[
        (df.median_range_ticks >= MIN_RANGE_TICKS)
        & (df.range_efficiency >= MIN_RANGE_EFFICIENCY)
        & (df.median_turnover_aud >= MIN_TURNOVER_AUD)
        & (df.n_days >= MIN_HISTORY_DAYS)
    ].copy()
    # Sector-tag the carried-over existing names too, even though most fail
    # the screen -- they still appear in the final table and an untagged row
    # reads as an oversight rather than a deliberate carry-over.
    sectors = fetch_sectors(sorted(set(shortlist.ticker.tolist()) | set(EXISTING_UNIVERSE)))
    df["sector"] = df.ticker.map(sectors)

    existing = df[df.ticker.isin(EXISTING_UNIVERSE)].copy()
    exclude = set(EXISTING_UNIVERSE)
    buckets = {}
    for b in ("large", "mid", "small"):
        picked = select_bucket(df, b, exclude)
        buckets[b] = picked
        exclude |= set(picked.ticker) if not picked.empty else set()

    selected = pd.concat(
        [existing.assign(bucket="existing")]
        + [p.assign(bucket=b) for b, p in buckets.items() if not p.empty],
        ignore_index=True,
    )

    hist = bulk_daily(selected.ticker.tolist(), period=period)
    rets = pd.DataFrame({t: np.log(d["Close"]).diff() for t, d in hist.items()}).dropna()

    return {
        "all_metrics": df,
        "selected": selected,
        "correlation": effective_independent_positions(rets),
        "screen_comparison": {
            "passes_5pct_screen": int((df.pct_days_range_ge_5pct >= 30).sum()),
            "passes_tick_screen": int((df.median_range_ticks >= MIN_RANGE_TICKS).sum()),
            "passes_5pct_but_fails_ticks": int((
                (df.pct_days_range_ge_5pct >= 30) & (df.median_range_ticks < MIN_RANGE_TICKS)).sum()),
            "passes_ticks_but_fails_5pct": int((
                (df.pct_days_range_ge_5pct < 30) & (df.median_range_ticks >= MIN_RANGE_TICKS)).sum()),
        },
    }


# ---------------------------------------------------------------------------
# Persistence + CLI
# ---------------------------------------------------------------------------

# Own table rather than reusing `research_universe`, deliberately. That table
# is written from BOTH sides -- this app's /research page and asxbrief's own
# `research recommend` command, which does `DELETE FROM research_universe
# WHERE kind=?` -- and `bar_coverage()` reads every row in it regardless of
# kind, so adding rows there would silently change what the /research page
# displays. A separate table keeps ownership unambiguous.
#
# Spec §16.5: the research universe and the TRADED universe are different
# things. This writes the research/model universe only. `swing_universe`
# (what actually gets orders placed against it) is untouched, and expanding
# one must never silently expand the other.
MODEL_UNIVERSE_DDL = """
CREATE TABLE IF NOT EXISTS model_universe (
    ticker TEXT PRIMARY KEY,
    bucket TEXT NOT NULL,
    sector TEXT,
    company TEXT,
    price REAL,
    market_cap REAL,
    median_range_ticks REAL,
    median_range_pct REAL,
    range_efficiency REAL,
    median_turnover_aud REAL,
    variance_ratio REAL,
    tradeable INTEGER NOT NULL DEFAULT 1,
    added_at TEXT NOT NULL
)
"""


def persist_universe(selected: pd.DataFrame, db_path: str = ASX_DB) -> dict[str, Any]:
    """Replace the stored model universe with `selected`.

    `tradeable` is 0 for any name that fails the tick screen -- the carried-
    over existing tickers mostly do. They stay in the pool because pooled
    fitting benefits from the extra rows and because the spec asks for
    continuity, but they are flagged so nothing downstream mistakes
    "in the model universe" for "worth placing an order against".
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con = sqlite3.connect(db_path)
    try:
        con.execute(MODEL_UNIVERSE_DDL)
        con.execute("DELETE FROM model_universe")
        for _, r in selected.iterrows():
            tradeable = int(
                (r["median_range_ticks"] >= MIN_RANGE_TICKS)
                and (r["range_efficiency"] >= MIN_RANGE_EFFICIENCY)
            )
            con.execute(
                "INSERT INTO model_universe (ticker, bucket, sector, company, price, market_cap,"
                " median_range_ticks, median_range_pct, range_efficiency, median_turnover_aud,"
                " variance_ratio, tradeable, added_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (r["ticker"], r["bucket"], r.get("sector"), r.get("company"),
                 float(r["price"]), float(r["market_cap"]),
                 float(r["median_range_ticks"]), float(r["median_range_pct"]),
                 float(r["range_efficiency"]), float(r["median_turnover_aud"]),
                 float(r["variance_ratio"]), tradeable, now),
            )
        con.commit()
    finally:
        con.close()
    return {"ok": True, "n": len(selected),
            "n_tradeable": int(((selected.median_range_ticks >= MIN_RANGE_TICKS)
                                & (selected.range_efficiency >= MIN_RANGE_EFFICIENCY)).sum())}


def get_model_universe(db_path: str = ASX_DB, tradeable_only: bool = False) -> list[dict[str, Any]]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        con.execute(MODEL_UNIVERSE_DDL)
        sql = "SELECT * FROM model_universe"
        if tradeable_only:
            sql += " WHERE tradeable=1"
        return [dict(r) for r in con.execute(sql + " ORDER BY bucket, ticker")]
    finally:
        con.close()


def print_universe(result: dict[str, Any]) -> None:
    sel, comp, corr = result["selected"], result["screen_comparison"], result["correlation"]
    print("=" * 118)
    print("UNIVERSE EXPANSION (phase-4 spec 14/16)")
    print("=" * 118)
    print()
    print("--- WHY PERCENT IS THE WRONG SCREEN " + "-" * 60)
    print(f"  names where >=30% of days move >=5%          : {comp['passes_5pct_screen']}")
    print(f"  names with median daily range >= {MIN_RANGE_TICKS:.0f} ticks     : {comp['passes_tick_screen']}")
    print(f"  pass the 5% screen but FAIL the tick screen  : {comp['passes_5pct_but_fails_ticks']}"
          "   <- volatile but untradeable (the current universe)")
    print(f"  pass the tick screen but FAIL the 5% screen  : {comp['passes_ticks_but_fails_5pct']}"
          "   <- would be wrongly discarded by a 5% screen")
    print()
    cols = ["ticker", "bucket", "sector", "price", "median_range_pct", "median_range_ticks",
            "pct_days_range_ge_5pct", "range_efficiency", "median_turnover_aud", "variance_ratio"]
    print(sel[cols].to_string(index=False))
    print()
    print(f"  effective independent positions: {corr.get('effective_independent_positions')} "
          f"of {corr.get('n_tickers')} tickers  (mean pairwise corr {corr.get('mean_pairwise_corr')})")
    untradeable = sel[(sel.median_range_ticks < MIN_RANGE_TICKS)
                      | (sel.range_efficiency < MIN_RANGE_EFFICIENCY)]
    if not untradeable.empty:
        print(f"  carried for continuity but NOT tradeable: {', '.join(untradeable.ticker)}")


if __name__ == "__main__":
    import sys
    res = build_universe()
    print_universe(res)
    if "--persist" in sys.argv:
        print()
        print("persist:", persist_universe(res["selected"]))
