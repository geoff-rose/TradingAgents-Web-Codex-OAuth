"""Score the same announcements under two prompt versions and compare them
against what the stocks actually did (2026-09-01, user request).

**Why the frozen prompts exist.** v5 overwrote v4 in place in `asx_signals.py`,
and the repository has no commit containing either -- so an hour after the
change, v4 existed nowhere but in a reconstruction. `_frozen_prompts.json`
holds all three verbatim, verified by rebuilding v5 and comparing it to the
live prompt. Without that this comparison could not be run at all.

**The design constraint.** v5 is scoring live and its scores accumulate day by
day. v4 will be run retrospectively over the SAME announcements. That is fair
only because everything v5 sees is point-in-time: `context_pack` reads price
history strictly before the tradeable session and prior filings strictly
before the announcement's timestamp. Running v4 later therefore gives it
neither an advantage nor a handicap -- it simply sees less, which is the
difference being measured.

**What NOT to conclude from a single comparison.** The earlier one-sided test
of v5 looked like a clear win -- the 12 worst misses fell a mean 9.8 points --
until the control showed the calls the model got RIGHT fell 13.8 points. v5
was not more discriminating, it was more bearish. So this compares RANKING
skill (IC and directional accuracy), never the average score, because a
version can improve the average by shifting everything down.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

FROZEN_PATH = Path(__file__).with_name("_frozen_prompts.json")
# Versions whose prompt refers to a context block; others are sent without one.
USES_CONTEXT = {"v5-context"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ab_scores (
    fingerprint TEXT NOT NULL,
    version     TEXT NOT NULL,
    ticker      TEXT,
    score       INTEGER,
    reason      TEXT,
    model       TEXT,
    scored_at   TEXT NOT NULL,
    PRIMARY KEY (fingerprint, version)
);
"""


def frozen(version: str) -> str:
    data = json.loads(FROZEN_PATH.read_text())
    if version not in data:
        raise KeyError(f"{version} not frozen; have {sorted(data)}")
    return data[version]


def _connect() -> sqlite3.Connection:
    from .mover_log import DB_PATH
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def score_under(version: str, session_from: str, session_to: str | None = None,
                limit: int = 400, candidate_limit: int | None = None) -> dict[str, Any]:
    """Score every in-universe announcement in the window under `version`.

    Writes only to `ab_scores`. Production `signals`/`ticker_signals` are
    never touched, so a comparison run cannot disturb the live record.

    **`candidate_limit` decouples the candidate pool from the batch size.**
    They used to be one number (`limit * 6`), which coupled two unrelated
    things: how far back the scan reaches, and how much work commits at once.
    Running with limit=100 therefore searched only the newest 600 announcements
    and silently skipped the first two days of an eight-day window -- it scored
    101 of 910 and reported success (2026-09-07). Pass a large
    `candidate_limit` to cover the window and a small `limit` to commit often.
    """
    from . import asx_signals as S
    from .asx_feed import DB_PATH as ASX_DB, tradeable_session_for
    from .context_pack import build as build_context

    system = frozen(version)
    conn = sqlite3.connect(f"file:{ASX_DB}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT a.fingerprint, a.ticker, COALESCE(a.company, u.company) AS company,"
            " a.headline, a.released_at, a.is_halt, a.halt_kind, a.price_sensitive, a.url"
            " FROM announcements a JOIN universe u ON u.ticker = a.ticker"
            " ORDER BY COALESCE(a.released_at, a.seen_at) DESC LIMIT ?",
            (candidate_limit if candidate_limit is not None else limit * 6,))]
    finally:
        conn.close()

    todo = []
    for r in rows:
        sess = tradeable_session_for(r.get("released_at") or "")
        if not sess or sess < session_from or (session_to and sess > session_to):
            continue
        if S._worth_a_call(r):
            todo.append(r)
    # Drop already-scored rows BEFORE applying the limit. The other order
    # spends the batch allowance on rows that are then discarded, so each pass
    # does less work as the done-set grows and the run stalls short of the
    # window -- the same shrinking-window trap as the candidate pool above.
    with _connect() as db:
        done = {x[0] for x in db.execute(
            "SELECT fingerprint FROM ab_scores WHERE version=?", (version,))}
    todo = [r for r in todo if r["fingerprint"] not in done][:limit]
    if not todo:
        return {"version": version, "scored": 0, "reason": "nothing new in window"}

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    scored, failed = 0, 0
    pending: list[tuple] = []

    def flush():
        """Write in short bursts rather than holding one transaction open.

        The connection used to stay open across the whole batch, so the write
        lock was held for as long as the LLM calls took -- about 25 minutes for
        a 400-row batch. swing.db is shared with the scanner, mover_log and the
        paper books, and on 2026-09-07 that starved asx-scanner-refresh during
        ASX trading hours: it returned HTTP 200 and silently wrote nothing.
        Flushing every FLUSH_EVERY rows keeps the lock held for milliseconds.
        """
        if not pending:
            return
        with _connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO ab_scores"
                " (fingerprint, version, ticker, score, reason, model, scored_at)"
                " VALUES (?,?,?,?,?,?,?)", pending)
            db.commit()
        pending.clear()

    FLUSH_EVERY = 20
    for r in todo:
        body = ""
        try:
            from .announcement_body import fetch_body
            body = fetch_body(r["fingerprint"], r["url"]) if r.get("url") else ""
        except Exception:
            pass
        prompt = (f"Ticker: {r['ticker']}\nCompany: {r.get('company') or 'unknown'}\n"
                  f"Headline: {r['headline']}\n"
                  f"Halt: {'yes' if r.get('is_halt') else 'no'}\n"
                  f"Flagged price-sensitive by ASX: "
                  f"{'yes' if r.get('price_sensitive') else 'no'}")
        if body:
            prompt += f"\n\nAnnouncement body:\n{body[:20000]}"
        if version in USES_CONTEXT:
            try:
                ctx = build_context(r["ticker"], r.get("released_at"))
                if ctx:
                    prompt += "\n\n" + ctx
            except Exception:
                pass
        try:
            txt = S._ask_counted(prompt, system=system, kind=f"ab-{version}",
                                 ticker=r["ticker"])
            parsed = S._extract_json(txt) or {}
            sc = int(parsed["score"])
        except Exception:
            failed += 1
            continue
        pending.append((r["fingerprint"], version, r["ticker"], sc,
                        str(parsed.get("reason") or "")[:200],
                        S._MODEL_LABEL, now))
        scored += 1
        if len(pending) >= FLUSH_EVERY:
            flush()
    flush()
    return {"version": version, "scored": scored, "failed": failed,
            "considered": len(todo)}


def compare(version_a: str, version_b: str, horizon: str = "fwd_5d_pct",
            min_names: int = 5) -> dict[str, Any]:
    """Rank IC and directional accuracy for each version, on identical rows.

    `horizon` is a column of `signal_outcomes`; its sector benchmark is
    subtracted so a version is not rewarded for a market or sector move.
    """
    import pandas as pd
    from .signal_outcomes import _connect as so_connect

    bench = horizon.replace("fwd_", "sect_")
    with so_connect() as c:
        out = pd.read_sql(
            f"SELECT ticker, as_of, {horizon} AS r, {bench} AS b FROM signal_outcomes"
            f" WHERE {horizon} IS NOT NULL", c)
    if out.empty:
        return {"error": f"no matured {horizon} rows yet"}
    out["rel"] = out["r"] - out["b"].fillna(0)

    with _connect() as db:
        ab = pd.read_sql("SELECT fingerprint, version, ticker, score FROM ab_scores", db)
    if ab.empty:
        return {"error": "no ab_scores yet -- run score_under() first"}

    # One score per ticker-version: the most material, matching production's
    # netting tie-break rather than a mean.
    ab["tilt"] = (ab["score"] - 50).abs()
    picked = ab.sort_values("tilt").groupby(["ticker", "version"]).tail(1)

    res: dict[str, Any] = {"horizon": horizon, "n_outcomes": int(len(out)), "versions": {}}
    for v in (version_a, version_b):
        m = picked[picked.version == v][["ticker", "score"]]
        j = out.merge(m, on="ticker", how="inner")
        if len(j) < min_names:
            res["versions"][v] = {"n": int(len(j)), "note": "too few matched rows"}
            continue
        ics = []
        for _, g in j.groupby("as_of"):
            if len(g) >= min_names:
                ics.append(g["score"].corr(g["rel"], method="spearman"))
        ics = [x for x in ics if x == x]
        strong = j[(j.score - 50).abs() >= 15]
        res["versions"][v] = {
            "n": int(len(j)), "n_dates": len(ics),
            "rank_ic": round(float(sum(ics) / len(ics)), 4) if ics else None,
            "mean_score": round(float(j["score"].mean()), 1),
            "strong_calls": int(len(strong)),
            "right_direction_pct": (
                round(float((((strong.score >= 65) & (strong.rel > 0)) |
                             ((strong.score <= 35) & (strong.rel < 0))).mean()) * 100, 1)
                if len(strong) else None),
        }
    return res
