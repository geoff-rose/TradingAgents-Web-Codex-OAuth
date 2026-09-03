"""A second classifier lane that gets tuned, beside a frozen one that does not
(2026-09-03, user request).

**The idea.** Keep the live classifier untouched so its scores accumulate into
a clean record, and clone it into a lane that gets adjusted whenever it makes a
mistake. In a month, compare. If the frozen one turns out to have been right
all along and merely needed ten days to prove it, the tuned lane is deleted.

**The trap this module exists to prevent.** "Tune it on the mistakes, then
compare the scores it gave" is a rigged comparison. If a version is adjusted
using outcomes it is later measured on, it is being fitted to its own test set,
and it will win by construction -- the more you tune, the bigger the fake
margin. The month-end number would be meaningless and, worse, would look
convincing.

**So every tune mints a version, and a version is only ever judged on
announcements scored AFTER it was minted.** `frozen_at` is the boundary.
Anything before it is in-sample: useful for diagnosis, never for the verdict.
That makes the lane a walk-forward test -- each variant is graded on its own
genuinely out-of-sample window, and those windows are short, which is the
honest cost of tuning often.

**Retrospective re-scoring is still allowed, and still not evidence.** Running
v6.3 over last month tells you what it WOULD have said, which is how you find
out whether a change did what you meant. It cannot tell you whether the change
helps, because the change was made knowing those outcomes.

**Variants are counted.** Ten tunes and a winner is not a discovery if the
denominator is hidden; `report()` returns `n_versions` so the multiple-testing
cost stays visible, the same reason the hypothesis ledger exists.

**Comparison is on RANKING, never the average score.** A version can improve
its mean by shifting everything down; v5 did exactly that and passed a
one-sided test until the control caught it.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

LANE_FROZEN = "frozen"      # the production classifier, left alone
LANE_TUNED = "tuned"        # the clone that gets adjusted

# A tuned version must score a full block before the next tune, or its
# out-of-sample window closes before the outcomes it would be judged on have
# matured -- fwd_10d_pct needs ten trading days to exist at all. Tuning sooner
# does not produce a faster answer, it produces a version with no verdict.
MIN_BLOCK_DAYS = 10
# Both are reported so the "could 5 days be enough" question is answerable from
# data rather than settled by assumption. 10d is the decision horizon until it
# has been shown that 5d says the same thing.
HORIZONS = ("fwd_5d_pct", "fwd_10d_pct")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lane_versions (
    version    TEXT PRIMARY KEY,
    lane       TEXT NOT NULL,
    frozen_at  TEXT NOT NULL,   -- the session boundary: scores AFTER this are out-of-sample
    parent     TEXT,
    note       TEXT,
    created_at TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    from .mover_log import DB_PATH
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.executescript(_SCHEMA)
    return conn


def register(version: str, lane: str, frozen_at: str = "", parent: str = "",
             note: str = "") -> dict[str, Any]:
    """Record a version and the session boundary its out-of-sample window starts
    after. The prompt text itself lives in `_frozen_prompts.json`; this records
    WHEN it started scoring, which is the only thing that makes a later
    comparison honest."""
    from .prompt_ab import FROZEN_PATH
    data = json.loads(FROZEN_PATH.read_text())
    if version not in data:
        raise KeyError(f"{version} is not in _frozen_prompts.json; freeze the "
                       f"prompt text first. Have: {sorted(data)}")
    frozen_at = frozen_at or datetime.now(timezone.utc).date().isoformat()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as db:
        db.execute("INSERT OR REPLACE INTO lane_versions"
                   " (version, lane, frozen_at, parent, note, created_at)"
                   " VALUES (?,?,?,?,?,?)",
                   (version, lane, frozen_at, parent or None, note or None, now))
    return {"version": version, "lane": lane, "frozen_at": frozen_at}


def mint(version: str, prompt: str, parent: str = "", note: str = "",
         force: bool = False) -> dict[str, Any]:
    """Freeze a newly tuned prompt and register it as a tuned-lane version.

    Refuses to overwrite an existing version. v5 overwriting v4 in place is why
    v4 briefly existed nowhere, and a lane whose history can be rewritten
    cannot be audited afterwards.

    Also refuses to mint within `MIN_BLOCK_DAYS` of the previous tune. The
    block has to run long enough for its own outcomes to mature -- cutting it
    short does not get you an answer sooner, it gets you a version that never
    had one. `force=True` for a deliberate exception, e.g. a plain bug rather
    than a tuning judgement.
    """
    from .prompt_ab import FROZEN_PATH
    data = json.loads(FROZEN_PATH.read_text())
    if version in data:
        raise ValueError(f"{version} already frozen -- pick a new name; "
                         f"overwriting destroys the audit trail")
    if not prompt.strip():
        raise ValueError("empty prompt")
    prev = _last_tuned()
    if prev and not force:
        from datetime import date
        age = (date.today() - date.fromisoformat(prev[1])).days
        if age < MIN_BLOCK_DAYS:
            raise ValueError(
                f"{prev[0]} was minted {age} days ago; its block needs "
                f"{MIN_BLOCK_DAYS} to mature. Wait {MIN_BLOCK_DAYS - age} more "
                f"day(s), or pass force=True if this is a bug fix rather than a "
                f"tuning judgement.")
    data[version] = prompt
    FROZEN_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return register(version, LANE_TUNED, parent=parent, note=note)


def _last_tuned() -> tuple[str, str] | None:
    """The most recent tuned version and its frozen_at, for the block guard."""
    with _connect() as db:
        row = db.execute(
            "SELECT version, frozen_at FROM lane_versions WHERE lane=?"
            " ORDER BY frozen_at DESC, created_at DESC LIMIT 1",
            (LANE_TUNED,)).fetchone()
    return (row[0], row[1]) if row else None


def block_status() -> dict[str, Any]:
    """Where the current block stands: how long the active tuned version has
    been scoring, and whether the next tune is due."""
    from datetime import date
    prev = _last_tuned()
    if not prev:
        return {"active_version": None, "tunable": True,
                "note": "no tuned version yet; the first mint starts block 1"}
    version, frozen_at = prev
    age = (date.today() - date.fromisoformat(frozen_at)).days
    return {"active_version": version, "frozen_at": frozen_at,
            "days_elapsed": age, "block_days": MIN_BLOCK_DAYS,
            "tunable": age >= MIN_BLOCK_DAYS,
            "days_remaining": max(0, MIN_BLOCK_DAYS - age)}


def _sessions_for(fingerprints: list[str]) -> dict[str, str]:
    """Tradeable session per fingerprint -- the day a score could first be acted
    on, which is what an outcome is attributed to."""
    from .asx_feed import DB_PATH as ASX_DB, tradeable_session_for
    if not fingerprints:
        return {}
    out: dict[str, str] = {}
    with sqlite3.connect(f"file:{ASX_DB}?mode=ro", uri=True) as ann:
        for i in range(0, len(fingerprints), 900):     # SQLite variable limit
            chunk = fingerprints[i:i + 900]
            marks = ",".join("?" * len(chunk))
            for fp, released in ann.execute(
                    f"SELECT fingerprint, released_at FROM announcements"
                    f" WHERE fingerprint IN ({marks})", chunk):
                s = tradeable_session_for(released or "")
                if s:
                    out[fp] = s
    return out


def evaluate(version: str, horizon: str = "fwd_5d_pct", min_names: int = 5,
             oos_only: bool = True) -> dict[str, Any]:
    """Rank IC and directional accuracy for one version.

    `oos_only` is the whole point and defaults on: rows are kept only if their
    tradeable session is strictly after the version's `frozen_at`. Pass False
    for diagnosis, and do not quote the result as evidence.
    """
    import pandas as pd
    from .signal_outcomes import _connect as so_connect
    from .prompt_ab import _connect as ab_connect

    with _connect() as db:
        row = db.execute("SELECT lane, frozen_at FROM lane_versions WHERE version=?",
                         (version,)).fetchone()
    if not row:
        return {"error": f"{version} not registered -- call register() or mint()"}
    lane, frozen_at = row

    # Two sources, because the production classifier never writes to ab_scores.
    # The frozen lane's real record is signal_outcomes, already one row per
    # ticker-session with the prompt_version that produced it; a tuned lane only
    # ever exists in ab_scores. Preferring ab_scores keeps a retrospective
    # re-score of a production version from being confused with its live record.
    with ab_connect() as db:
        ab = pd.read_sql("SELECT fingerprint, ticker, score FROM ab_scores"
                         " WHERE version=?", db, params=(version,))
    if not ab.empty:
        sess = _sessions_for(ab["fingerprint"].tolist())
        ab["session"] = ab["fingerprint"].map(sess)
        ab = ab.dropna(subset=["session"])
        source = "ab_scores"
    else:
        with so_connect() as c:
            # prompt_version is comma-joined when several versions scored the
            # same ticker-session, so match on substring rather than equality
            ab = pd.read_sql(
                "SELECT ticker, as_of AS session, score FROM signal_outcomes"
                " WHERE score IS NOT NULL AND prompt_version LIKE ?",
                c, params=(f"%{version}%",))
        source = "signal_outcomes"
    if ab.empty:
        return {"version": version, "lane": lane, "frozen_at": frozen_at,
                "n": 0, "note": "no scores stored for this version yet"}
    n_all = len(ab)
    if oos_only:
        ab = ab[ab["session"] > frozen_at]
    if ab.empty:
        return {"version": version, "lane": lane, "frozen_at": frozen_at,
                "n": 0, "n_in_sample_excluded": n_all,
                "note": "no out-of-sample rows yet -- this version has not scored "
                        "anything since it was minted"}

    bench = horizon.replace("fwd_", "sect_")
    with so_connect() as c:
        out = pd.read_sql(
            f"SELECT ticker, as_of, {horizon} AS r, {bench} AS b FROM signal_outcomes"
            f" WHERE {horizon} IS NOT NULL", c)
    if out.empty:
        return {"version": version, "lane": lane, "frozen_at": frozen_at,
                "n": 0, "note": f"no matured {horizon} outcomes yet"}
    out["rel"] = out["r"] - out["b"].fillna(0)

    ab["tilt"] = (ab["score"] - 50).abs()
    picked = ab.sort_values("tilt").groupby(["ticker", "session"]).tail(1)
    j = out.merge(picked[["ticker", "session", "score"]],
                  left_on=["ticker", "as_of"], right_on=["ticker", "session"],
                  how="inner")
    if len(j) < min_names:
        return {"version": version, "lane": lane, "frozen_at": frozen_at,
                "n": int(len(j)), "n_in_sample_excluded": n_all - len(ab),
                "note": "too few matched out-of-sample rows to say anything"}

    ics = [g["score"].corr(g["rel"], method="spearman")
           for _, g in j.groupby("as_of") if len(g) >= min_names]
    ics = [x for x in ics if x == x]
    strong = j[(j.score - 50).abs() >= 15]
    return {
        "version": version, "lane": lane, "frozen_at": frozen_at,
        "source": source, "oos_only": oos_only,
        "n": int(len(j)), "n_dates": len(ics),
        "n_in_sample_excluded": int(n_all - len(ab)),
        "rank_ic": round(float(sum(ics) / len(ics)), 4) if ics else None,
        "mean_score": round(float(j["score"].mean()), 1),
        "strong_calls": int(len(strong)),
        "right_direction_pct": (
            round(float((((strong.score >= 65) & (strong.rel > 0)) |
                         ((strong.score <= 35) & (strong.rel < 0))).mean()) * 100, 1)
            if len(strong) else None),
    }


def evaluate_horizons(version: str, horizons: tuple[str, ...] = HORIZONS) -> dict[str, Any]:
    """The same version across horizons, so "is 5 days enough" can be read off
    the data once a block has matured rather than assumed."""
    return {h: evaluate(version, horizon=h) for h in horizons}


def report(horizon: str = "fwd_5d_pct") -> dict[str, Any]:
    """Every registered version with its out-of-sample record, and the variant
    count that a later 'the tuned one won' claim has to be discounted by."""
    with _connect() as db:
        rows = db.execute("SELECT version, lane, frozen_at, parent, note"
                          " FROM lane_versions ORDER BY lane, frozen_at").fetchall()
    versions = []
    for v, lane, frozen_at, parent, note in rows:
        r = evaluate(v, horizon=horizon)
        r.update({"parent": parent, "note": note})
        versions.append(r)
    tuned = [v for v in versions if v.get("lane") == LANE_TUNED]
    return {
        "horizon": horizon,
        "block": block_status(),
        "n_versions": len(versions),
        "n_tuned_variants": len(tuned),
        "multiple_testing_note": (
            f"{len(tuned)} tuned variant(s) tried. A winner among them is worth "
            f"roughly one variant's worth less than its own p-value suggests; "
            f"compare the best tuned variant's OUT-OF-SAMPLE record against the "
            f"frozen lane, never its in-sample fit."),
        "versions": versions,
    }


def active_version() -> str:
    """The tuned version currently scoring. Only the newest one runs.

    Every registered variant COULD be re-scored daily, but that multiplies the
    quota by the number of variants for no gain: an older variant's
    out-of-sample window ended the moment its successor was minted, and adding
    rows to it after that would silently mix two different prompts under one
    name.
    """
    with _connect() as db:
        row = db.execute(
            "SELECT version FROM lane_versions WHERE lane=?"
            " ORDER BY frozen_at DESC, created_at DESC LIMIT 1",
            (LANE_TUNED,)).fetchone()
    return row[0] if row else ""


def score_session(session: str = "", limit: int = 400) -> dict[str, Any]:
    """Score one session's announcements under the active tuned version.

    A no-op while no tuned version exists, which is the correct state until the
    first real tune: an untuned clone produces the same prompt as production
    and would double the quota to reproduce scores that already exist.
    """
    from datetime import date
    version = active_version()
    if not version:
        return {"scored": 0, "note": "no tuned version registered yet -- nothing "
                                     "to run; the lane costs nothing until the "
                                     "first tune mints one"}
    session = session or date.today().isoformat()
    from .prompt_ab import score_under
    res = score_under(version, session, session, limit=limit)
    res["session"] = session
    return res
