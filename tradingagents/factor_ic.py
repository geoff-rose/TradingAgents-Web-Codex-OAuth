"""Information Coefficient for the announcement classifier (2026-08-31).

**What IC is.** On one trading day, rank every scored name by its score and
rank the same names by what they actually did. The correlation between those
two rankings is that day's Information Coefficient. Average it across days and
you have the IC; divide that mean by its own standard deviation across days
and you have the ICIR.

**Why per-date and then averaged, rather than pooling every observation.**
Pooling lets one big market day dominate: if everything rose 4%, a pooled
correlation partly measures the market, not the score. A per-date correlation
only ever asks "did the higher-scored names beat the lower-scored ones ON THAT
DAY", so the market move cancels by construction. That is the same disease the
sector benchmark was added to cure, fixed structurally instead of by
subtraction -- which is why IC is the standard measure for a signal whose job
is RANKING rather than predicting a level.

**Rank IC is the one to read.** Pearson IC is dominated by outliers. Measured
on 2026-08-26, this classifier's Pearson IC was -0.1169 while its Rank IC was
+0.1030 -- opposite signs, from two names (KLS scored 85 and fell 10.4%, HMC
scored 18 and rose 9.4%). Both are reported; Rank IC is the robust one.

**Why this beats the ad-hoc correlations used before.**
  * It puts the classifier on a published scale. From AlphaSeek (arXiv
    2608.13913, CSI300, ~950 test dates x 300 names): Alpha158 library 0.0131,
    LightGBM 0.0247, LSTM 0.0331, TRA 0.0421, RD-Agent 0.0401, AlphaSeek
    0.0454. An IC of 0.04 is a GOOD factor, not a disappointing one.
  * It converts "how long do I run this" into arithmetic. The t-statistic on
    mean IC is `ICIR * sqrt(n_dates)`, so at a published-typical ICIR of 0.25
    reaching t=2 takes ~64 trading days. `dates_needed` reports that per row.

**Measured on `signal_outcomes`, not `mover_log`.** `mover_log` holds only
names that passed the scanner's move/volume screen -- 6 to 19 a session out of
the 130-166 actually scored. That is both too narrow to rank (simulated at a
true rank-IC of 0.04, 15 names/day needs ~154 trading days to reach t=2 while
100 needs ~27) and, worse, conditioned on the OUTCOME: a high-scoring
announcement whose stock did nothing never entered the sample, so the
classifier's most important failure mode was invisible to its own scorecard.
`signal_outcomes` covers every scored ticker-session.

**The honest caveat that remains.** The paper's cross-sections are 300 names;
these are ~130. A single day's IC on 130 names is still noisy, which is why
the date count and `dates_needed` sit next to every figure.
"""

from __future__ import annotations

import math
from typing import Any

MIN_NAMES = 5          # below this a day's ranking is not worth computing
TARGET_T = 2.0         # t-stat treated as "settled"

# ICIR is a mean divided by a standard deviation estimated from the SAME few
# dates, so at small date counts it is not merely noisy, it is unbounded: the
# 3-day-forward row reported RankICIR -8.99 and t -15.58 off three dates on
# 2026-08-31. Below this many dates the consistency figures are marked
# provisional and must not be read as significance.
MIN_DATES_FOR_T = 10

# (label, factor column, return column, benchmark column or None)
VARIANTS: tuple[tuple[str, str, str, str | None], ...] = (
    ("same-day open->close", "score", "open_close_pct", None),
    ("same-day, sector-relative", "score", "open_close_pct", "sect_open_close_pct"),
    ("full day (prev close)", "score", "full_day_pct", "sect_full_day_pct"),
    ("1-day forward", "score", "fwd_1d_pct", None),
    ("1-day, sector-relative", "score", "fwd_1d_pct", "sect_1d_pct"),
    ("3-day, sector-relative", "score", "fwd_3d_pct", "sect_3d_pct"),
    ("5-day, sector-relative", "score", "fwd_5d_pct", "sect_5d_pct"),
    ("10-day, sector-relative", "score", "fwd_10d_pct", "sect_10d_pct"),
)

BENCHMARKS = {
    "Alpha158 library": 0.0131, "LightGBM": 0.0247, "LSTM": 0.0331,
    "RD-Agent (LLM)": 0.0401, "TRA": 0.0421, "AlphaSeek": 0.0454,
}


def _frame():
    import pandas as pd
    from .signal_outcomes import _connect
    cols = {"as_of", "ticker", "score"}
    for _, f, r, b in VARIANTS:
        cols |= {f, r} | ({b} if b else set())
    with _connect() as conn:
        have = {row[1] for row in conn.execute("PRAGMA table_info(signal_outcomes)")}
        sel = sorted(c for c in cols if c in have)
        df = pd.read_sql(
            f"SELECT {', '.join(sel)} FROM signal_outcomes WHERE score IS NOT NULL",
            conn)
    return df.rename(columns={"as_of": "trade_date"})


def compute(min_names: int = MIN_NAMES) -> dict[str, Any]:
    import pandas as pd
    df = _frame()
    if df.empty:
        return {"error": "no scored mover-days yet", "rows": []}

    out = []
    for label, fcol, rcol, bcol in VARIANTS:
        if fcol not in df or rcol not in df:
            continue
        d = df.dropna(subset=[fcol, rcol]).copy()
        if bcol and bcol in d:
            d = d.dropna(subset=[bcol])
            d["_r"] = d[rcol] - d[bcol]
        else:
            d["_r"] = d[rcol]
        per_day = []
        for day, g in d.groupby("trade_date"):
            if len(g) < min_names:
                continue
            per_day.append((day, len(g), g[fcol].corr(g["_r"]),
                            g[fcol].corr(g["_r"], method="spearman")))
        row: dict[str, Any] = {"variant": label, "n_dates": len(per_day),
                               "n_obs": int(len(d))}
        if per_day:
            p = pd.DataFrame(per_day, columns=["d", "n", "ic", "ric"]).dropna()
            row["n_dates"] = int(len(p))
            row["median_names"] = int(p["n"].median()) if len(p) else None
            for key, col in (("ic", "ic"), ("rank_ic", "ric")):
                if p.empty:
                    continue
                mean = float(p[col].mean())
                sd = float(p[col].std(ddof=1)) if len(p) > 1 else float("nan")
                ir = mean / sd if sd and not math.isnan(sd) and sd > 0 else None
                row[key] = round(mean, 4)
                row[f"{key}ir"] = round(ir, 3) if ir is not None else None
                row[f"{key}_t"] = (round(ir * math.sqrt(len(p)), 2)
                                   if ir is not None else None)
                row["provisional"] = len(p) < MIN_DATES_FOR_T
                # Days still needed for the OBSERVED consistency to reach t=2.
                row[f"{key}_dates_needed"] = (
                    int(math.ceil((TARGET_T / abs(ir)) ** 2))
                    if ir and abs(ir) > 1e-9 else None)
        out.append(row)
    return {"rows": out, "benchmarks": BENCHMARKS, "min_names": min_names,
            "target_t": TARGET_T, "min_dates_for_t": MIN_DATES_FOR_T,
            "total_scored_rows": int(len(df)),
            "sessions": int(df["trade_date"].nunique())}


def print_panel(r: dict[str, Any]) -> None:
    if r.get("error"):
        print(r["error"]); return
    print(f"\nClassifier IC -- {r['total_scored_rows']} scored mover-days "
          f"across {r['sessions']} sessions (min {r['min_names']} names/day)\n")
    hdr = (f"  {'variant':<28}{'dates':>6}{'n/day':>7}{'IC':>9}{'ICIR':>8}"
           f"{'RankIC':>9}{'RankICIR':>10}{'t':>7}{'need':>7}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for row in r["rows"]:
        if not row.get("n_dates"):
            print(f"  {row['variant']:<28}{'--':>6}   (no day has enough names)")
            continue
        print(f"  {row['variant']:<28}{row['n_dates']:>6}{row.get('median_names') or 0:>7}"
              f"{row.get('ic') if row.get('ic') is not None else float('nan'):>9.4f}"
              f"{row.get('icir') if row.get('icir') is not None else float('nan'):>8.3f}"
              f"{row.get('rank_ic') if row.get('rank_ic') is not None else float('nan'):>9.4f}"
              f"{row.get('rank_icir') if row.get('rank_icir') is not None else float('nan'):>10.3f}"
              f"{row.get('rank_ic_t') if row.get('rank_ic_t') is not None else float('nan'):>7.2f}"
              f"{row.get('rank_ic_dates_needed') or 0:>7}"
              f"{'  provisional' if row.get('provisional') else ''}")
    print("\n  published IC for scale: " +
          "  ".join(f"{k} {v:.4f}" for k, v in r["benchmarks"].items()))


if __name__ == "__main__":
    print_panel(compute())
