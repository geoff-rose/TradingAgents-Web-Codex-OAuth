"""The harness: one runner that puts a hypothesis through every gate.

**The problem it solves.** Testing an idea properly took a 200-line throwaway
script, and whether the right checks got run depended on remembering the
relevant lesson. Three candidates reached "promising" during the 2026-08
research before a check that had already been written -- for a different
strategy, in a script since deleted -- was rewritten and killed them.

**A hypothesis is a spec plus a build function.** The spec carries the things
no returns frame can reveal (when the signal is known, when you enter, whether
the universe is point-in-time); `build()` returns one row per trade. `run()`
applies every applicable gate and returns a verdict.

**Trades frame contract** -- one row per trade, columns:
    date         str    the session the trade belongs to
    ticker       str
    side         int    +1 long, -1 short
    gross_pct    float  return in percent, sign already applied to `side`
  optional:
    price        float  entry price, enables the price-gradient gate
    bench_pct    float  benchmark return over the identical window
    delayed_pct  float  the same trade entered one bar later
    is_train     bool   in-sample flag, enables the out-of-sample gate

Omitted columns skip their gate rather than failing it -- a gate that cannot
be evaluated reports `skipped`, and the scorecard says so, because "not
checked" and "checked and passed" must never look the same.

**Verdict rules.** Any failed FATAL gate makes the verdict `rejected`. All
applicable fatal gates passing makes it `survived` -- which means "not yet
shown to be an artifact", not "tradeable". Anything else is `inconclusive`.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from . import gates as G

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hypothesis_runs (
    name          TEXT NOT NULL,
    run_at        TEXT NOT NULL,
    description   TEXT,
    universe      TEXT,
    n_trades      INTEGER,
    gross_pct     REAL,
    net_pct       REAL,
    verdict       TEXT NOT NULL,
    killed_by     TEXT,
    n_variants    INTEGER,
    gates_json    TEXT NOT NULL,
    PRIMARY KEY (name, run_at)
);
"""


def _connect() -> sqlite3.Connection:
    from .mover_log import DB_PATH
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


@dataclass
class Hypothesis:
    name: str
    description: str
    build: Callable[[], Any]          # -> DataFrame matching the contract above
    # Declared facts. These cannot be derived from returns and are the two
    # places the most expensive mistakes were made.
    signal_known_at: str = "unspecified"
    entry_at: str = "unspecified"
    entry_is_tradeable: bool = True
    tradeable_note: str = ""
    universe: str = "unspecified"
    point_in_time: bool = False
    survivorship_note: str = ""
    realistic_cost_pct: float = G.REALISTIC_COST
    parcel_dollars: float = 20_000.0
    # How many variants were tried to arrive at this one. Honesty about the
    # search, not about the result.
    n_variants: int = 1
    n_passed: int = 1
    expected_retained: float | None = None
    benchmark_label: str = "sector"
    tags: list[str] = field(default_factory=list)


def _costs_for(df, h: Hypothesis):
    """Per-trade round-trip cost from the measured spread table, when the
    frame carries tickers. Falls back to the hypothesis's flat estimate."""
    if "ticker" not in df:
        return None
    try:
        from .costs import cost_vector
        prices = df["price"] if "price" in df else None
        return cost_vector(df["ticker"], prices, dollars=h.parcel_dollars)
    except Exception:
        return None


def _floor_frac(df) -> float:
    if "ticker" not in df:
        return 0.0
    try:
        from .costs import floor_fraction
        return floor_fraction(set(df["ticker"]))
    except Exception:
        return 0.0


def run(h: Hypothesis, store: bool = True) -> dict[str, Any]:
    df = h.build()
    if df is None or not len(df):
        return {"name": h.name, "verdict": "inconclusive",
                "error": "build() produced no trades", "gates": []}

    long_g = df.loc[df["side"] > 0, "gross_pct"] if "side" in df else df["gross_pct"]
    short_g = df.loc[df["side"] < 0, "gross_pct"] if "side" in df else []
    results: list[G.GateResult] = [
        G.tradeable_window(h.signal_known_at, h.entry_at, h.entry_is_tradeable,
                           h.tradeable_note),
        G.survivorship(h.universe, h.point_in_time, h.survivorship_note),
        G.cost_sweep(df["gross_pct"], realistic=h.realistic_cost_pct,
                     per_trade_cost=_costs_for(df, h),
                     floor_fraction=_floor_frac(df)),
    ]
    if len(short_g) and len(long_g):
        results.append(G.symmetry(long_g, short_g))
    if "price" in df:
        results.append(G.price_gradient(df["gross_pct"], df["price"]))
    if "delayed_pct" in df:
        results.append(G.delayed_entry(df["gross_pct"], df["delayed_pct"],
                                       h.expected_retained))
    if "is_train" in df and df["is_train"].nunique() == 2:
        results.append(G.out_of_sample(df.loc[df["is_train"], "gross_pct"],
                                       df.loc[~df["is_train"], "gross_pct"]))
    if "ticker" in df:
        results.append(G.leave_one_out(df["gross_pct"], df["ticker"]))
    if "bench_pct" in df:
        results.append(G.benchmark_split(df["gross_pct"], df["bench_pct"],
                                         h.benchmark_label))
    results.append(G.multiple_testing(h.n_variants, h.n_passed))

    failed_fatal = [r for r in results if r.fatal and r.passed is False]
    provisional = [r for r in results if r.provisional]
    applicable = [r for r in results if r.fatal and r.passed is not None
                  and not r.provisional]
    # A provisional fatal gate BLOCKS `survived`. Without this the VAS
    # overnight trade read "survived" while its deciding gate had quietly
    # become unevaluated -- the cost hurdle it cleared was a one-tick floor,
    # a lower bound, and at two ticks the trade loses to buy-and-hold. "Not
    # measured" must never render as "passed".
    if failed_fatal:
        verdict = "rejected"
    elif provisional:
        verdict = "inconclusive"
    elif applicable and all(r.passed for r in applicable):
        verdict = "survived"
    else:
        verdict = "inconclusive"

    gross = float(df["gross_pct"].mean())
    out = {
        "name": h.name, "description": h.description, "universe": h.universe,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_trades": int(len(df)), "gross_pct": round(gross, 4),
        "net_pct": round(gross - h.realistic_cost_pct, 4),
        "realistic_cost_pct": h.realistic_cost_pct,
        "verdict": verdict,
        "killed_by": ", ".join(r.name for r in failed_fatal) or None,
        "n_variants": h.n_variants, "tags": h.tags,
        "blocked_by": ", ".join(r.name for r in provisional) or None,
        "gates": [{"name": r.name, "status": r.status, "passed": r.passed,
                   "fatal": r.fatal, "provisional": r.provisional,
                   "verdict": r.verdict, "detail": r.detail}
                  for r in results],
    }
    if store:
        with _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO hypothesis_runs (name, run_at, description,"
                " universe, n_trades, gross_pct, net_pct, verdict, killed_by,"
                " blocked_by, n_variants, gates_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (h.name, out["run_at"], h.description, h.universe, out["n_trades"],
                 out["gross_pct"], out["net_pct"], verdict, out["killed_by"],
                 out["blocked_by"], h.n_variants, json.dumps(out["gates"], default=str)))
            conn.commit()
    return out


def history(limit: int = 100, latest_only: bool = True) -> dict[str, Any]:
    """The multiple-testing ledger: the denominator for "one survivor out of a
    dozen" has to be a query, not a recollection.

    `latest_only` shows the CURRENT verdict per hypothesis. Every run is kept
    -- a verdict that changed because a gate got stricter is worth being able
    to look up -- but the headline must be what the hypothesis stands at now,
    or an old `survived` row keeps advertising a conclusion that has since
    been withdrawn.
    """
    with _connect() as conn:
        if latest_only:
            rows = [dict(r) for r in conn.execute(
                "SELECT h.* FROM hypothesis_runs h JOIN ("
                "  SELECT name, MAX(run_at) AS run_at FROM hypothesis_runs GROUP BY name"
                ") m ON h.name = m.name AND h.run_at = m.run_at"
                " ORDER BY h.run_at DESC LIMIT ?", (limit,))]
        else:
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM hypothesis_runs ORDER BY run_at DESC LIMIT ?", (limit,))]
    for r in rows:
        try:
            r["gates"] = json.loads(r.pop("gates_json"))
        except Exception:
            r["gates"] = []
    tally: dict[str, int] = {}
    for r in rows:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    killers: dict[str, int] = {}
    for r in rows:
        for k in (r.get("killed_by") or "").split(", "):
            if k:
                killers[k] = killers.get(k, 0) + 1
    with _connect() as conn:
        n_runs = conn.execute("SELECT COUNT(*) FROM hypothesis_runs").fetchone()[0]
    return {"runs": rows, "tally": tally, "killed_by": killers,
            "total_variants": sum(r.get("n_variants") or 1 for r in rows),
            "n_hypotheses": len(rows), "n_runs_total": n_runs}


def print_scorecard(out: dict[str, Any]) -> None:
    print(f"\n{out['name']} -- {out['verdict'].upper()}")
    print(f"  {out.get('description', '')}")
    print(f"  {out['n_trades']} trades, gross {out['gross_pct']:+.3f}%/trade, "
          f"net {out['net_pct']:+.3f}% at {out['realistic_cost_pct']:.2f}% cost")
    if out.get("killed_by"):
        print(f"  killed by: {out['killed_by']}")
    if out.get("blocked_by"):
        print(f"  awaiting measurement: {out['blocked_by']}")
    print()
    for g in out["gates"]:
        mark = {"pass": " ok ", "FAIL": "FAIL", "warn": "warn",
                "prov": "PROV", "n/a": " -- "}[g["status"]]
        print(f"  [{mark}] {g['name']:<18} {g['verdict']}")
