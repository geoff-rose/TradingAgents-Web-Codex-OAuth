"""Daily review of the biggest open-to-high moves against what the classifier
said about their news (2026-09-02).

**The question this answers.** Does a LOW score sit in front of a big upward
move? That is the classifier's known weak side: on strong Buys it is
directionally right 50% of the time and on strong Sells 65% -- it spots bad
news and cannot spot good news. Every worst call ranked so far was a high score
that fell. This looks for the opposite error, which nothing currently measures.

**Open-to-high, not open-to-close.** A move that ran and gave it all back still
had news the model should have seen. Using the high asks "did the market react
at all", which is the question about the classifier; using the close asks "was
it a good trade", which is a different question and a harder one. It is NOT
tradeable -- you cannot sell at the high -- so nothing here is a return.

**The base rate is computed every time, and it is the point.** Ranking today's
top 5 and reading their scores is selection on the outcome: most scores sit
near 50, so most big movers will have an unremarkable score no matter how good
the classifier is. `population` holds every scored ticker that session so the
flagged names can be compared against what a typical scored name did. A flag
means "worth a look", never "the classifier failed" -- one session cannot show
that, and the daily rows accumulate so it can eventually be tested properly
through `hypothesis.run()`.

**Announcement attribution uses the tradeable session.** 23.2% of ASX
announcements land after the close and cannot move the price until the next
session; charging them with the move that preceded them is the error
`asx_feed.tradeable_session_for` exists to prevent.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Two different errors, kept apart because they mean different things.
# Measured over 2,907 classified announcements: 50 is the mode at 51% of all
# scores, p25=50, p75=57, and only 8.3% land at or below 45.
#   <=45  the model leaned NEGATIVE and the stock jumped -- a contradiction.
#   46-54 the model made no call at all -- it saw nothing. Far more common,
#         so much weaker evidence, but it is the shape of "cannot spot good
#         news" and pooling it with the above would drown the rare case.
CONTRADICTED_MAX = 45
NO_CALL_MAX = 54
# Open-to-high at or above the 90th percentile of mover_log. NOT a guess:
# calibrate() over 261 finalized rows gives p50=6.06, p75=8.75, p90=11.43.
# mover_log is already screened for movers, so a 5% threshold would have
# flagged more than half of it and meant nothing.
BIG_MOVE_PCT = 11.0
NEWS_WINDOW_DAYS = 3
TOP_N = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS openhigh_review (
    session_date   TEXT NOT NULL,
    ticker         TEXT NOT NULL,
    rank           INTEGER,
    open_high_pct  REAL,
    open           REAL,
    high           REAL,
    close          REAL,
    news_date      TEXT,
    news_age_days  INTEGER,
    headline       TEXT,
    price_sensitive INTEGER,
    n_in_window    INTEGER,
    session_score  INTEGER,
    score          INTEGER,
    signal         TEXT,
    reason         TEXT,
    prompt_version TEXT,
    flagged        INTEGER,
    flag_kind      TEXT,
    reviewed_at    TEXT,
    PRIMARY KEY (session_date, ticker)
);
CREATE TABLE IF NOT EXISTS openhigh_population (
    session_date        TEXT PRIMARY KEY,
    n_movers            INTEGER,
    n_with_news         INTEGER,
    n_price_sensitive   INTEGER,
    n_scored            INTEGER,
    n_low_score         INTEGER,
    n_contradicted      INTEGER,
    n_no_call           INTEGER,
    median_open_high    REAL,
    mean_oh_low_score   REAL,
    mean_oh_high_score  REAL,
    reviewed_at         TEXT
);
"""


def _connect() -> sqlite3.Connection:
    from .mover_log import DB_PATH
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns SCHEMA has gained since the table was first created.

    `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so adding a
    column to SCHEMA leaves the real table one column short and every insert
    fails with "no such column" -- which is exactly how the first run of this
    module broke, after a dry run had already created the table.

    The reference schema is built in memory and diffed, so this stays correct
    without a hand-maintained migration list.
    """
    ref = sqlite3.connect(":memory:")
    ref.executescript(SCHEMA)
    for (table,) in ref.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        have = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
        for _, name, decl, *_rest in ref.execute(f'PRAGMA table_info("{table}")'):
            if name not in have:
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {decl}')
                logger.info("openhigh_review: added column %s.%s", table, name)
    ref.close()


def _latest_session(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT trade_date FROM mover_log WHERE finalized=1 "
        "ORDER BY trade_date DESC LIMIT 1").fetchone()
    return row[0] if row else ""


def _news_for(tickers: list[str], session: str, window: int) -> dict[str, dict]:
    """The announcement most likely to explain the move, per ticker.

    **Not simply the most recent one.** The first run of this picked
    "Securities Trading Policy", "Date of AGM" and an Appendix 3Y as the news
    behind three of the top five movers, and flagged the classifier for scoring
    them 50. Scoring them 50 was RIGHT -- they are administrative filings that
    imply nothing about price, and the stock moved for other reasons. Taking
    the latest announcement manufactures a false alarm every time a routine
    notice happens to land after the real news.

    So: prefer the most recent PRICE-SENSITIVE announcement in the window, and
    fall back to the most recent of any kind only so the row still shows what
    was there. Only price-sensitive news can raise a flag.

    Scored per announcement by fingerprint, with the per-session aggregate from
    `ticker_signals` carried alongside -- a ticker with five announcements has
    one aggregate that hides which one was misread, and one fingerprint score
    that ignores the others. Both are worth seeing.
    """
    from .asx_feed import DB_PATH as ASX_DB, tradeable_session_for
    from .asx_signals import DB_PATH as SIG_DB

    if not tickers:
        return {}
    cutoff = (date.fromisoformat(session) - timedelta(days=window + 4)).isoformat()
    marks = ",".join("?" * len(tickers))
    out: dict[str, dict] = {}

    with sqlite3.connect(f"file:{ASX_DB}?mode=ro", uri=True) as ann:
        rows = ann.execute(
            f"SELECT ticker, fingerprint, headline, released_at, price_sensitive "
            f"FROM announcements WHERE ticker IN ({marks}) AND released_at >= ? "
            f"ORDER BY released_at DESC", (*tickers, cutoff)).fetchall()

    for tkr, fp, headline, released, ps in rows:
        sess = tradeable_session_for(released)
        if not sess or sess > session:
            continue                       # not actionable by this session yet
        age = (date.fromisoformat(session) - date.fromisoformat(sess)).days
        if age > window:
            continue
        cand = {"fingerprint": fp, "headline": headline, "news_date": sess,
                "news_age_days": age, "price_sensitive": int(ps or 0)}
        prev = out.get(tkr)
        if prev is None:
            out[tkr] = dict(cand, n_in_window=1)
        else:
            prev["n_in_window"] += 1
            # rows arrive newest-first, so only a price-sensitive one displaces
            # an incumbent that is not
            if cand["price_sensitive"] and not prev["price_sensitive"]:
                out[tkr] = dict(cand, n_in_window=prev["n_in_window"])

    fps = [v["fingerprint"] for v in out.values() if v["fingerprint"]]
    if fps:
        marks = ",".join("?" * len(fps))
        with sqlite3.connect(f"file:{SIG_DB}?mode=ro", uri=True) as sig:
            scored = {r[0]: r[1:] for r in sig.execute(
                f"SELECT fingerprint, score, signal, reason, prompt_version "
                f"FROM signals WHERE fingerprint IN ({marks})", fps)}
        for v in out.values():
            s = scored.get(v["fingerprint"])
            v.update(zip(("score", "signal", "reason", "prompt_version"),
                         s if s else (None, None, None, None)))

    # Per-session aggregate, for tickers with more than one announcement.
    # Keyed on the ANNOUNCEMENT's tradeable session, not the mover's session:
    # news one day old belongs to yesterday's aggregate, and keying on today
    # returned null for every stale-news row.
    if out:
        keys = {(v["news_date"], t) for t, v in out.items()}
        marks = " OR ".join(["(session_date=? AND ticker=?)"] * len(keys))
        flat = [x for k in keys for x in k]
        with sqlite3.connect(f"file:{SIG_DB}?mode=ro", uri=True) as sig:
            agg = {(r[0], r[1]): r[2] for r in sig.execute(
                f"SELECT session_date, ticker, score FROM ticker_signals"
                f" WHERE {marks}", flat)}
        for tkr, v in out.items():
            v["session_score"] = agg.get((v["news_date"], tkr))
    return out


def review(session: str = "", top_n: int = TOP_N, window: int = NEWS_WINDOW_DAYS,
           store: bool = True) -> dict[str, Any]:
    """Rank the session's movers by open-to-high and report the top N with the
    news that preceded them, alongside the base rate for that session."""
    conn = _connect()
    try:
        session = session or _latest_session(conn)
        if not session:
            return {"error": "no finalized session in mover_log"}

        movers = conn.execute(
            "SELECT ticker, open, high, close FROM mover_log "
            "WHERE trade_date=? AND open IS NOT NULL AND high IS NOT NULL AND open>0",
            (session,)).fetchall()
        if not movers:
            return {"error": f"no finalized rows for {session}", "session": session}

        ranked = sorted(
            ({"ticker": t, "open": o, "high": h, "close": c,
              "open_high_pct": round((h - o) / o * 100, 2)}
             for t, o, h, c in movers),
            key=lambda r: r["open_high_pct"], reverse=True)

        news = _news_for([r["ticker"] for r in ranked], session, window)
        for r in ranked:
            r.update(news.get(r["ticker"], {}))
            sc, big = r.get("score"), r["open_high_pct"] >= BIG_MOVE_PCT
            # An administrative filing implies nothing about price; the model
            # scoring it 50 is correct, not a miss.
            big = big and bool(r.get("price_sensitive"))
            r["flag_kind"] = (
                "contradicted" if big and sc is not None and sc <= CONTRADICTED_MAX
                else "no_call" if big and sc is not None and sc <= NO_CALL_MAX
                else "")
            r["flagged"] = bool(r["flag_kind"])

        # The control. Without this the top-5 list is unreadable: it says
        # nothing that "most scores are near 50" does not already explain.
        scored = [r for r in ranked if r.get("score") is not None]
        lo = [r["open_high_pct"] for r in scored if r["score"] <= CONTRADICTED_MAX]
        hi = [r["open_high_pct"] for r in scored if r["score"] > CONTRADICTED_MAX]
        oh = sorted(r["open_high_pct"] for r in ranked)
        pop = {
            "session_date": session,
            "n_movers": len(ranked),
            "n_with_news": sum(1 for r in ranked if r.get("news_date")),
            "n_price_sensitive": sum(1 for r in ranked if r.get("price_sensitive")),
            "n_scored": len(scored),
            "n_low_score": len(lo),
            "n_contradicted": sum(1 for r in ranked if r["flag_kind"] == "contradicted"),
            "n_no_call": sum(1 for r in ranked if r["flag_kind"] == "no_call"),
            "median_open_high": round(oh[len(oh) // 2], 2) if oh else None,
            "mean_oh_low_score": round(sum(lo) / len(lo), 2) if lo else None,
            "mean_oh_high_score": round(sum(hi) / len(hi), 2) if hi else None,
        }

        top = ranked[:top_n]
        if store:
            now = datetime.now().astimezone().isoformat(timespec="seconds")
            with conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO openhigh_review (session_date, ticker,"
                    " rank, open_high_pct, open, high, close, news_date,"
                    " news_age_days, headline, price_sensitive, n_in_window,"
                    " session_score, score, signal,"
                    " reason, prompt_version, flagged, flag_kind, reviewed_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(session, r["ticker"], i + 1, r["open_high_pct"], r["open"],
                      r["high"], r["close"], r.get("news_date"),
                      r.get("news_age_days"), r.get("headline"),
                      r.get("price_sensitive"), r.get("n_in_window"),
                      r.get("session_score"), r.get("score"), r.get("signal"),
                      r.get("reason"), r.get("prompt_version"), int(r["flagged"]),
                      r["flag_kind"], now)
                     for i, r in enumerate(top)])
                cols = list(pop) + ["reviewed_at"]
                conn.execute(
                    f"INSERT OR REPLACE INTO openhigh_population ({','.join(cols)})"
                    f" VALUES ({','.join('?' * len(cols))})",
                    (*pop.values(), now))
        return {"session": session, "top": top, "population": pop,
                "flagged": [r for r in top if r["flagged"]]}
    finally:
        conn.close()


def calibrate(sessions: int = 60) -> dict[str, Any]:
    """What open-to-high actually looks like across stored sessions, so
    BIG_MOVE_PCT is set from the distribution rather than guessed."""
    conn = _connect()
    try:
        vals = sorted(r[0] for r in conn.execute(
            "SELECT (high-open)/open*100 FROM mover_log WHERE finalized=1"
            " AND open>0 AND high IS NOT NULL AND trade_date >= "
            " (SELECT MIN(trade_date) FROM (SELECT DISTINCT trade_date FROM mover_log"
            "  ORDER BY trade_date DESC LIMIT ?))", (sessions,)))
        if not vals:
            return {"error": "no data"}
        q = lambda p: round(vals[min(len(vals) - 1, int(len(vals) * p))], 2)
        return {"n": len(vals), "p50": q(.50), "p75": q(.75), "p90": q(.90),
                "p95": q(.95), "p99": q(.99), "max": round(vals[-1], 2)}
    finally:
        conn.close()


def recent(limit: int = 20) -> list[dict[str, Any]]:
    """Flagged rows across sessions -- the accumulating record that a real
    test needs. One session proves nothing."""
    conn = _connect()
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT * FROM openhigh_review WHERE flagged=1"
            " ORDER BY session_date DESC, open_high_pct DESC LIMIT ?", (limit,))]
    finally:
        conn.close()


def stored(session: str = "", flag_limit: int = 15) -> dict[str, Any]:
    """Read back what the nightly run wrote, without recomputing.

    The dashboard reads this rather than POSTing a review: rendering a page
    must not depend on IBKR, the announcement feed, or a recompute that would
    silently disagree with the row the timer actually stored.
    """
    conn = _connect()
    try:
        conn.row_factory = sqlite3.Row
        session = session or (conn.execute(
            "SELECT session_date FROM openhigh_population"
            " ORDER BY session_date DESC LIMIT 1").fetchone() or [""])[0]
        if not session:
            return {"session": "", "top": [], "population": None, "flags": [],
                    "history": []}
        top = [dict(r) for r in conn.execute(
            "SELECT * FROM openhigh_review WHERE session_date=? ORDER BY rank",
            (session,))]
        pop = conn.execute("SELECT * FROM openhigh_population WHERE session_date=?",
                           (session,)).fetchone()
        flags = [dict(r) for r in conn.execute(
            "SELECT * FROM openhigh_review WHERE flagged=1"
            " ORDER BY session_date DESC, open_high_pct DESC LIMIT ?", (flag_limit,))]
        # running totals, so the panel can say how thin the record still is
        hist = conn.execute(
            "SELECT COUNT(*) n_sessions, SUM(n_contradicted) contradicted,"
            " SUM(n_no_call) no_call, SUM(n_price_sensitive) price_sensitive"
            " FROM openhigh_population").fetchone()
        return {"session": session, "top": top,
                "population": dict(pop) if pop else None,
                "flags": flags, "history": dict(hist) if hist else {}}
    finally:
        conn.close()
