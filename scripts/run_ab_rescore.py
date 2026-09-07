#!/usr/bin/env python3
"""Score BOTH prompt versions over the A/B window under ONE model.

Scoring only v4 would have confounded prompt version with model version (v5's
live scores were made under gpt-5.4, withdrawn 2026-09-06), and `compare()`
reads both versions out of `ab_scores` regardless -- it never falls back to
signal_outcomes, so v5 had to be scored into that table anyway.

Paces itself against the real quota headers rather than guessing: ~2,158 calls
is about 108% of a five-hour window, so it parks when the window is nearly
spent and resumes after the reset. score_under() skips fingerprints already
stored for that version, so every pass is resumable and re-running is free.
"""
import os, sys, time, json, httpx

os.environ.setdefault("TRADINGAGENTS_CODEX_MODEL", "gpt-5.6-sol")
sys.path.insert(0, "/opt/tradingagents")

from tradingagents import codex_oauth as C
from tradingagents.prompt_ab import score_under

WINDOW = ("2026-09-01", "2026-09-08")
VERSIONS = ["v4-novelty", "v5-context"]
# score_under fetches limit*6 CANDIDATES before filtering to the window, so a
# small chunk silently truncates the window: at 100 the 600-row pool reached
# only back to 2026-09-02 and the 1st and 2nd were never considered -- it
# scored 101 of 1,079 and reported COMPLETE. 400 gives a 2,400-row pool
# reaching to 2026-08-25. This is the same newest-N shrinking-window trap
# documented in classify_pending().
CHUNK = 80
CANDIDATES = 3000   # pool big enough to cover the whole window, independent of CHUNK
PAUSE_ABOVE = 88          # % of the 5h window at which to park


def quota():
    rt, rf, h = C._token_helpers()
    tok = C.ensure_fresh_token()
    hd = dict(h(tok)); hd["Authorization"] = f"Bearer {tok}"
    hd["Content-Type"] = "application/json"
    with httpx.Client(timeout=60) as cl:
        r = cl.post(C.BASE_URL + "/responses", headers=hd, json={
            "model": "gpt-5.6-sol", "input": [{"role": "user", "content": "hi"}],
            "store": False, "stream": True})
    return (int(r.headers.get("x-codex-primary-used-percent", 0)),
            int(r.headers.get("x-codex-primary-reset-after-seconds", 300)))


def stored_vs_expected(version):
    """(scored so far, eligible in window) -- the check that catches a silently
    truncated candidate pool."""
    import sqlite3
    from tradingagents import asx_signals as S
    from tradingagents.asx_feed import DB_PATH as ASX_DB, tradeable_session_for
    from tradingagents.prompt_ab import _connect
    with _connect() as db:
        have = db.execute("SELECT COUNT(*) FROM ab_scores WHERE version=?",
                          (version,)).fetchone()[0]
    con = sqlite3.connect(f"file:{ASX_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT a.fingerprint, a.headline, a.price_sensitive, a.is_halt"
        " FROM announcements a JOIN universe u ON u.ticker = a.ticker"
        " WHERE a.released_at >= ?", (WINDOW[0],))]
    con.close()
    want = sum(1 for r in rows if S._worth_a_call(r))
    return have, want


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log(f"model={os.environ['TRADINGAGENTS_CODEX_MODEL']} window={WINDOW}")
    # Self-calibrating, because the fixed estimate was wrong by 42%. A/B
    # prompts carry the full announcement body plus the context pack, so they
    # cost ~0.071% of the 5h window each, not the 0.050% measured on shorter
    # prompts. Under-estimating the cost lets a batch sail past the cap and
    # start failing mid-chunk, so this re-measures after every chunk.
    cost_per_call = 0.071
    for version in VERSIONS:
        done_total = 0
        while True:
            used, reset_in = quota()
            if used >= PAUSE_ABOVE:
                log(f"quota {used}% -- parking {reset_in//60}m for the window reset")
                time.sleep(reset_in + 60)
                continue
            # Size the batch to the quota left, so a chunk cannot overshoot the
            # cap mid-flight: the driver only checks BETWEEN chunks, and at
            # ~0.05%/call a blind 400-call chunk from 82% would run past 100%.
            room = max(1, int((PAUSE_ABOVE - used) / cost_per_call))
            res = score_under(version, WINDOW[0], WINDOW[1],
                              limit=min(CHUNK, room), candidate_limit=CANDIDATES)
            n = res.get("scored", 0)
            done_total += n
            used_after, _ = quota()
            if n >= 20 and used_after > used:
                observed = (used_after - used) / n
                cost_per_call = max(0.02, min(0.30, observed))
            log(f"{version}: +{n} (total {done_total}) failed={res.get('failed',0)} "
                f"quota {used}%->{used_after}% cost/call={cost_per_call:.3f}%")
            if n == 0:
                # Never trust scored==0 as "finished" -- that is what truncation
                # looks like too. Verify against the eligible population.
                have, want = stored_vs_expected(version)
                if have < want:
                    log(f"{version}: STALLED at {have}/{want} -- candidate pool "
                        f"is not reaching the whole window")
                    return 1
                log(f"{version}: COMPLETE {have}/{want}")
                break
    log("ALL DONE")


if __name__ == "__main__":
    main()
