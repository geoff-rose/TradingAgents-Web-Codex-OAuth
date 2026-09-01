"""Rank the classifier's worst calls automatically, instead of waiting to
trip over one (2026-09-01).

**Why this exists.** Every classifier failure examined so far was found the
same way: the user noticed a stock moving and asked about it. BC8, MVF. That
finds real bugs but it samples whatever happened to be looked at, and it
missed the actual pattern -- which was sitting in `signal_outcomes` the whole
time. Ranked systematically, the twelve worst calls are ALL high scores that
fell, none are low scores that rose, and every one is a multi-document results
release. The single case that prompted the most analysis (MVF, scored 30 and
rose) turned out to be the rare inverse.

**A miss is not a big move.** A 10% move on a score of 50 is not a mistake --
the model made no claim. `miss` weights the move by how far the score
committed, so it ranks by how WRONG the call was, not how large the move.

**Directional accuracy is reported separately for Buy and Sell.** Pooling them
hides the finding that matters: on strong Buys the model is directionally right
about half the time, on strong Sells about two thirds. Those are different
problems and one of them may not be worth fixing.
"""

from __future__ import annotations

from typing import Any

STRONG_TILT = 15          # |score - 50| at or above which the model made a call


def _frame():
    import pandas as pd
    from .signal_outcomes import _connect
    with _connect() as conn:
        df = pd.read_sql(
            "SELECT ticker, as_of, score, signal, n_announcements, prompt_version,"
            " open_close_pct, sect_open_close_pct, fwd_1d_pct, sect_1d_pct"
            " FROM signal_outcomes WHERE score IS NOT NULL", conn)
    if df.empty:
        return df
    df["rel"] = df["open_close_pct"] - df["sect_open_close_pct"]
    df["rel_1d"] = df["fwd_1d_pct"] - df["sect_1d_pct"]
    df = df.dropna(subset=["rel"]).copy()
    df["tilt"] = df["score"] - 50
    # Positive when the score and the move disagreed, scaled by conviction.
    df["miss"] = -(df["tilt"] / 50) * df["rel"]
    return df


def report(top: int = 15) -> dict[str, Any]:
    df = _frame()
    if df is None or df.empty:
        return {"error": "no scored outcomes yet", "worst": [], "best": []}
    strong = df[df["tilt"].abs() >= STRONG_TILT]

    def rows(frame):
        return [{"ticker": r.ticker, "as_of": r.as_of, "score": int(r.score),
                 "vs_sector_pct": round(float(r.rel), 2),
                 "next_day_pct": (round(float(r.rel_1d), 2)
                                  if r.rel_1d == r.rel_1d else None),
                 "n_announcements": int(r.n_announcements or 0),
                 "prompt_version": r.prompt_version}
                for r in frame.itertuples()]

    accuracy = {}
    for lo, hi, lbl in ((65, 101, "buy"), (0, 36, "sell")):
        g = df[(df.score >= lo) & (df.score < hi)]
        if len(g) < 5:
            continue
        accuracy[lbl] = {
            "n": int(len(g)), "mean_vs_sector_pct": round(float(g["rel"].mean()), 3),
            "median_vs_sector_pct": round(float(g["rel"].median()), 3),
            "right_direction_pct": round(
                float(((g["rel"] > 0) == (lo >= 65)).mean()) * 100, 1)}

    return {"n_scored": int(len(df)), "n_strong_calls": int(len(strong)),
            "worst": rows(strong.nlargest(top, "miss")),
            "best": rows(strong.nsmallest(6, "miss")),
            "accuracy": accuracy, "strong_tilt": STRONG_TILT}


def print_report(r: dict[str, Any]) -> None:
    if r.get("error"):
        print(r["error"]); return
    print(f"\n{r['n_scored']} scored ticker-sessions, {r['n_strong_calls']} carrying a real "
          f"call (|score-50| >= {r['strong_tilt']})\n")
    print("WORST CALLS -- the score said one thing and the stock did the other")
    print(f"  {'ticker':<7}{'date':<12}{'score':>6}{'vs sector':>11}{'next day':>10}"
          f"{'docs':>6}  version")
    for x in r["worst"]:
        nd = f"{x['next_day_pct']:+.1f}%" if x["next_day_pct"] is not None else "    --"
        print(f"  {x['ticker']:<7}{x['as_of']:<12}{x['score']:>6}"
              f"{x['vs_sector_pct']:>10.1f}%{nd:>10}{x['n_announcements']:>6}  "
              f"{x['prompt_version']}")
    print("\nDIRECTIONAL ACCURACY ON STRONG CALLS")
    for k, v in r["accuracy"].items():
        print(f"  {k:<5} n={v['n']:>3}  mean vs sector {v['mean_vs_sector_pct']:+.2f}%  "
              f"right direction {v['right_direction_pct']:.0f}%")


if __name__ == "__main__":
    print_report(report())
