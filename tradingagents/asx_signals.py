"""AI Buy/Sell signal + bullishness score for ASX announcements.

Score is a single 0-100 bullish/bearish scale (0=Sell, 50=neutral/Hold,
100=Buy) -- the Signal label is derived from the score band, not asked of the
model independently, so "Hold" always means a score near 50, never a
confusingly high or low one. See `_signal_for_score()`.

Every announcement from a top-N-universe company gets classified (~161/day
as of 2026-08-21) -- restricting to ASX's own price-sensitive/halt flag
missed too much: debt issuance, buy-backs, and director's dealings from large
companies aren't flagged price-sensitive but the user wanted them graded too.
See asx_feed.signal_worthy().

Results are cached in our own SQLite db, keyed by asxbrief's `fingerprint`,
so a given announcement is only ever classified once regardless of how many
times the collector's poll loop or a dashboard refresh sees it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import codex_oauth, grok_oauth

# Which OAuth provider backs the classifier. Switched to Codex 2026-08-23 when
# the user's xAI credits ran out; both modules expose the same narrow surface
# (is_configured / ask / is_quota_error / is_auth_error + a quota and an auth
# exception), so this is the only place that has to know which one is active.
# Override with ASX_SIGNALS_PROVIDER=grok to switch back without a code change.
_PROVIDERS = {"codex": codex_oauth, "grok": grok_oauth}
PROVIDER_NAME = os.environ.get("ASX_SIGNALS_PROVIDER", "codex").lower()
_provider = _PROVIDERS.get(PROVIDER_NAME, codex_oauth)

# Model recorded on each cached row. Rows classified under different providers
# coexist -- the column is there so a later accuracy check can separate them
# rather than pooling two different models' scores as if they were one.
_MODEL_LABEL = {"codex": codex_oauth.DEFAULT_MODEL, "grok": "grok-4.3"}.get(PROVIDER_NAME, "unknown")


def provider_status() -> dict[str, Any]:
    """What the classifier is currently using, and whether it can run."""
    return {
        "provider": PROVIDER_NAME,
        "model": _MODEL_LABEL,
        "configured": _provider.is_configured(),
        "login_hint": ("uv run python scripts/codex_login.py" if PROVIDER_NAME == "codex"
                       else "uv run python scripts/grok_login.py"),
    }

logger = logging.getLogger(__name__)

DB_PATH = Path(os.getenv("ASX_SIGNALS_DB_PATH", str(Path.home() / ".tradingagents" / "asx_signals.db")))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    fingerprint  TEXT PRIMARY KEY,
    ticker       TEXT NOT NULL,
    signal       TEXT NOT NULL,   -- 'Buy' | 'Sell' | 'Hold'
    score        INTEGER NOT NULL,
    reason       TEXT,
    model        TEXT,
    prompt_version TEXT,
    classified_at TEXT NOT NULL
);
"""

# Bump whenever _SYSTEM_PROMPT changes in a way that could move scores.
# Without this, rows scored under different prompts pool silently and a later
# accuracy check compares two different graders as if they were one -- the
# same reason `model` is recorded. v2 (2026-08-23) added the
# execution-is-not-news rule after the user flagged TRE's "First Drawdown
# Proceeds Received under Gold Stream" scoring 78 when it should be ~50.
# v5 (2026-09-01) prepends a point-in-time CONTEXT block -- run-in price
# action, 52-week range position, and the company's own earlier announcements
# with the scores given then. Prompted by MVF: the classifier wrote "2H
# momentum and FY27 growth outlook" against the investor presentation, netted
# the ticker Sell 30 anyway, and the stock rose 3.7% that session against an
# index down 0.2% and another 9% the next day. It read the document correctly
# and had nothing to weigh it against. See context_pack.py, and note the
# version bump is what keeps v4 and v5 rows from pooling in the IC panel.
#
# v4 (2026-08-31) added the novelty rules after BC8's FY26 results scored 90
# and moved -1.6% against its own sector. The release's headline figures --
# record revenue, +132% production -- had been public since its 30 July
# quarterly, which the document itself cites as the source. The classifier had
# no way to know that, so it read a restatement as news. Measured on 40
# rescored tickers that day, spearman against sector-relative return went
# +0.181 (v3) -> +0.307 (v4); the advantage survived leave-one-out on every
# name but shrank to +0.034 excluding BC8, so this is promoted on the merits
# of the rules, not on one day of evidence.
PROMPT_VERSION = "v5-context"

# Score is a single bullishness scale, not a separate conviction axis --
# 0 = strong sell, 50 = neutral, 100 = strong buy. Signal is *derived* from
# the score band (see _signal_for_score), not asked of the model separately:
# the earlier design had the LLM emit signal and score independently, which
# produced confusing results like "Hold, 90" (read as "very confident this
# is nothing" rather than "strongly bullish") -- user flagged this 2026-08-21.
# One axis removes that ambiguity by construction.
_SELL_MAX = 35   # score <= this -> Sell
_BUY_MIN = 65    # score >= this -> Buy

_SYSTEM_PROMPT = (
    "You are a terse equity analyst for ASX announcements. Given one "
    "announcement's ticker and headline, judge the short-term (next few days) "
    "bullishness implied purely by this announcement's content. Reply with "
    "ONLY a JSON object, no markdown fences, no commentary: "
    '{"score": <0-100 integer>, "reason": "<=12 words"}. '
    "Score is a single bullish/bearish scale: 100 = strongly bullish (buy), "
    "0 = strongly bearish (sell), 50 = neutral/unclear/routine -- NOT a "
    "separate conviction axis, so a routine announcement with no real signal "
    "should score near 50, not near 0 or 100. Most halts belong near 50: a "
    "halt means news is COMING, not what it says, so only move off 50 if the "
    "headline itself names the reason, e.g. \"pending capital raising\" is "
    "bearish (dilutive, score low), \"pending drill results\"/\"pending trial "
    "results\" is genuinely unclear either way (stay near 50).\n"
    "CRITICAL -- carrying out something already announced is NOT new "
    "information and must score near 50, however positive the underlying "
    "arrangement was. The market priced that arrangement when it was first "
    "disclosed; the follow-through is administrative. This covers first or "
    "subsequent drawdowns, proceeds or funds received, completion, settlement, "
    "commencement, and shares issued under an existing facility, placement or "
    "agreement. Treat wording like \"under\", \"pursuant to\", \"previously "
    "announced\", \"first drawdown\", \"proceeds received\", \"completion of\" "
    "as strong evidence you are looking at execution rather than news. Example: "
    "\"First Drawdown Proceeds Received under Gold Stream\" scores about 50 -- "
    "the stream itself was the news, receiving the money was already expected.\n"
    "You are given ASX's price-sensitive flag only to indicate the "
    "announcement is worth reading. It is a disclosure classification, not a "
    "direction, and plenty of routine mandatory filings carry it. Never move "
    "off 50 because of that flag alone.\n"
    "You are usually given the announcement BODY text as well as the headline. "
    "When present, judge from the body -- headlines are written by the company "
    "and are a lossy, flattering summary. The body is what reveals whether "
    "\"proceeds received\" is $2m or $200m, whether a placement is at a 5% or "
    "40% discount, and whether an event was already disclosed. A body that "
    "refers back to an earlier announcement (\"announced on <date>\", "
    "\"previously announced\", \"conditions precedent satisfied\") is strong "
    "evidence you are reading execution, not news -- score near 50. "
    "If no body is given, judge on the headline alone and, when it does not "
    "itself establish that something new and materially good or bad has "
    "happened, score 50. Missing detail is a reason to stay neutral, never a "
    "licence to guess what the document probably says.\n"
    "WHEN A CONTEXT BLOCK IS SUPPLIED, USE IT. It carries the run-in price "
    "action, where the price sits in its 52-week range, and the company's own "
    "earlier announcements with the scores previously assigned. Judge whether "
    "this announcement is a SURPRISE against that, and in which direction -- "
    "not whether the numbers in it are good or bad in isolation. A weak result "
    "that the market has already marked down, or that a company pre-announced, "
    "is far less bearish than the same result arriving unexpectedly. A "
    "deteriorating full-year number alongside an IMPROVING second half is what "
    "a turnaround looks like, and the direction of travel usually matters more "
    "to the next few days than the level. The context is never itself news: do "
    "not score the price history, score the announcement in light of it.\n"
    "PERIODIC RESULTS ARE MOSTLY CONFIRMATION. Full-year, half-year, "
    "Appendix 4E/4D and annual report releases almost always follow a "
    "quarterly or production report that already disclosed the operational "
    "numbers. Score the SURPRISE against what the market already knew, not "
    "the size of the numbers. If the release cites its own earlier "
    "announcement as the source for production, sales or revenue figures, "
    "those figures are not new and must not drive the score. A results "
    "release containing no genuine surprise belongs near 50 no matter how "
    "strong the absolute numbers are.\n"
    "PERCENTAGE GROWTH OFF A TINY BASE IS NOT GROWTH. A revenue or earnings "
    "increase of several hundred percent against a prior year in which "
    "operations were starting up, suspended or pre-revenue is a ramp-up "
    "artifact. Judge the absolute level and the trajectory, not the "
    "percentage.\n"
    "WHAT IS MISSING MATTERS. In a results release, note the absence of "
    "things a confident company would include: forward guidance (especially "
    "if explicitly DEFERRED to a later date), unit cost figures such as AISC "
    "for a miner, margin detail, or any capital return where cash flow "
    "clearly allows one. Deferred guidance is a mild negative, not a "
    "neutral: it withholds the number the market most wants.\n"
    "BEWARE AGGREGATED HEADLINE FIGURES. Output or revenue that combines the "
    "company's own production with third-party, tolling or joint-venture "
    "volumes overstates what accrues to shareholders. If the release gives "
    "both, judge on the company's own share."
)



def _signal_for_score(score: int) -> str:
    if score <= _SELL_MAX:
        return "Sell"
    if score >= _BUY_MIN:
        return "Buy"
    return "Hold"


def _migrate(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(signals)")}
    if "prompt_version" not in cols:
        conn.execute("ALTER TABLE signals ADD COLUMN prompt_version TEXT")
        conn.commit()


# Every provider call, one row. Exists because the classifier's cost was
# invisible: the only way to know what a day spent was to count rows in
# `signals` and `ticker_signals`, and those undercount badly -- both are
# upserts, so a net re-computed 144 times still looks like one row. That is
# precisely how ~3,000-6,000 redundant netting calls a day stayed hidden
# (2026-08-25). Failures are recorded too: a rejected call generally still
# counts against a provider's quota, and a day full of auth errors should look
# expensive rather than free.
_USAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    called_at    TEXT NOT NULL,      -- UTC ISO8601
    provider     TEXT NOT NULL,
    model        TEXT,
    kind         TEXT NOT NULL,      -- 'classify' | 'net'
    ticker       TEXT,
    ok           INTEGER NOT NULL,
    error        TEXT,
    ms           INTEGER,
    prompt_chars INTEGER
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_at ON llm_calls(called_at);
"""


def _usage_connect() -> sqlite3.Connection:
    conn = _connect()
    conn.executescript(_USAGE_SCHEMA)
    return conn


def _ask_counted(prompt: str, *, system: str, kind: str,
                 ticker: str | None = None) -> str:
    """The single door every provider call goes through, so usage cannot be
    undercounted by a caller that forgets to log. Logging never breaks a
    call: if the counter itself fails the classification still returns."""
    t0 = time.time()
    ok, err, text = 0, None, None
    try:
        text = _provider.ask(prompt, system=system)
        ok = 1
        return text
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"[:200]
        raise
    finally:
        try:
            with _usage_connect() as conn:
                conn.execute(
                    "INSERT INTO llm_calls (called_at, provider, model, kind, ticker, ok,"
                    " error, ms, prompt_chars) VALUES (?,?,?,?,?,?,?,?,?)",
                    (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), PROVIDER_NAME,
                     _MODEL_LABEL, kind, ticker, ok, err,
                     int((time.time() - t0) * 1000), len(prompt) + len(system)),
                )
                conn.commit()
        except Exception:
            logger.debug("usage logging failed", exc_info=True)


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


def get_signals_for(fingerprints: list[str]) -> dict[str, dict[str, Any]]:
    """Bulk-fetch cached signals, keyed by fingerprint. Missing keys just aren't in the dict."""
    if not fingerprints:
        return {}
    with _connect() as conn:
        placeholders = ",".join("?" * len(fingerprints))
        rows = conn.execute(
            f"SELECT fingerprint, signal, score, reason FROM signals "
            f"WHERE fingerprint IN ({placeholders})",
            fingerprints,
        ).fetchall()
    return {r["fingerprint"]: dict(r) for r in rows}


def attach_signals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge cached signal/score/reason onto feed rows in place (by fingerprint)."""
    cached = get_signals_for([r["fingerprint"] for r in rows if r.get("fingerprint")])
    for r in rows:
        hit = cached.get(r.get("fingerprint"))
        r["signal"] = hit["signal"] if hit else None
        r["score"] = hit["score"] if hit else None
        r["signal_reason"] = hit["reason"] if hit else None
    return rows


def _extract_json(text: str) -> dict[str, Any] | None:
    """No structured-output enforcement over the OAuth path -- the model may
    wrap JSON in code fences or add a stray sentence. Pull out the first
    {...} block and parse that."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def classify_one(rec: dict[str, Any], use_body: bool = True) -> dict[str, Any] | None:
    """`use_body` fetches the announcement PDF and includes its text.

    On by default since 2026-08-23: headline-only classification scored TRE's
    "First Drawdown Proceeds Received under Gold Stream" at 78 when the body
    says plainly it is the first drawdown of a facility announced on 13 July --
    administrative, not news. A failed fetch degrades to headline-only rather
    than skipping the announcement; an unreadable document is not a reason to
    leave it unscored.
    """
    body = ""
    if use_body:
        if not (rec.get("url") and rec.get("fingerprint")):
            # Loud, not silent: a caller that forgets to select `url` degrades
            # every classification to headline-only while still looking healthy.
            logger.warning("no url/fingerprint for %s -- classifying on headline only",
                            rec.get("ticker"))
        else:
            try:
                from tradingagents.announcement_body import fetch_body
                body = fetch_body(rec["fingerprint"], rec["url"])
            except Exception as exc:
                logger.warning("body fetch failed for %s: %s", rec.get("ticker"), exc)

    prompt = (
        f"Ticker: {rec['ticker']}\n"
        f"Company: {rec.get('company') or 'unknown'}\n"
        f"Headline: {rec['headline']}\n"
        f"Halt: {'yes (' + rec['halt_kind'] + ')' if rec.get('is_halt') else 'no'}\n"
        f"Flagged price-sensitive by ASX: {'yes' if rec.get('price_sensitive') else 'no'}"
    )
    # Point-in-time context. Best-effort: a failure here degrades to the
    # previous behaviour rather than skipping the announcement, but it is
    # logged, because silently losing the context would leave v5 scoring
    # exactly like v4 while claiming to be different.
    try:
        from .context_pack import build as build_context
        ctx = build_context(rec["ticker"], rec.get("released_at") or rec.get("seen_at"))
        if ctx:
            prompt += "\n\n" + ctx
        else:
            logger.debug("no context available for %s", rec.get("ticker"))
    except Exception as exc:
        logger.warning("context pack failed for %s: %s", rec.get("ticker"), exc)
    if body:
        prompt += f"\n\nAnnouncement body:\n{body}"
    text = _ask_counted(prompt, system=_SYSTEM_PROMPT, kind="classify",
                        ticker=rec.get("ticker"))
    parsed = _extract_json(text)
    if not parsed or "score" not in parsed:
        logger.warning("unparseable signal response for %s: %r", rec["ticker"], text[:200])
        return None
    try:
        score = max(0, min(100, int(parsed["score"])))
    except (TypeError, ValueError):
        logger.warning("non-numeric score for %s: %r", rec["ticker"], parsed.get("score"))
        return None
    return {
        "signal": _signal_for_score(score),
        "score": score,
        "reason": str(parsed.get("reason") or "")[:200],
    }


CANDIDATE_POOL_SIZE = 2000  # see classify_pending's docstring for why this
                            # must be far larger than any per-call `limit`


# Announcement types not worth an LLM call: pure administrative filings that
# cannot move a price. Kept as an explicit, readable list rather than a clever
# regex so it can be audited and edited -- adding a pattern here is a direct
# spending decision.
#
# Deliberately NARROW. An earlier draft also skipped substantial-holding
# notices and director's interest notices, which the user overruled: a
# substantial holder building a stake, or a director buying, is real
# information for a momentum scanner even though it is routine paperwork.
# Only these three are skipped for now (2026-08-25).
#
# Measured share: ~2.6% of in-universe announcements, ~8 calls/day. Small --
# the dominant saving is the netting skip in refresh_ticker_signals, not this.
SKIP_HEADLINE_PATTERNS = [
    r"appendix\s*3g",                                  # notification of ceasing to have a relevant interest
    r"notification\s+regarding\s+unquoted\s+securities",
    r"cleansing\s+notice",
]
_SKIP_RE = re.compile("|".join(SKIP_HEADLINE_PATTERNS), re.I)


def _worth_a_call(rec: dict[str, Any]) -> bool:
    """Whether an announcement earns a classification call.

    **ASX's own price-sensitive flag always wins.** A document whose headline
    looks administrative but which the exchange flagged price-sensitive is
    classified regardless -- the filter is there to skip noise, and must never
    be the reason a flagged announcement goes unread. Same for halts.
    """
    if rec.get("price_sensitive") or rec.get("is_halt"):
        return True
    return not _SKIP_RE.search(rec.get("headline") or "")


def classify_pending(limit: int = 60) -> dict[str, Any]:
    """Classify up to `limit` signal-worthy announcements that aren't cached yet.

    **`limit` bounds how many get CLASSIFIED, not how many candidates are
    considered.** An earlier version passed the same `limit` straight through
    to `signal_worthy(limit=...)`, which fetches the newest N candidates before
    filtering out already-scored ones -- so as the backlog grew, that
    newest-N window filled up with already-scored items and shrank toward
    nothing, while genuinely unscored older announcements sat below the window
    and were never even looked at. Caught 2026-08-24: EVT's actual FY26
    results release (22:35) had fallen out of the newest-60 window by the time
    any batch ran, while two of its OTHER documents got scored (one via
    headline-only fallback, its results slide deck having exceeded the PDF
    size cap) -- so EVT sat at Hold 50 with its real news never read.
    Candidates are now fetched from a pool of `CANDIDATE_POOL_SIZE` (must
    exceed any realistic backlog) and only the classification count is capped
    by `limit`.

    Returns counts; aborts the whole batch immediately on a quota error since
    retrying a billing block wastes calls and every remaining item would hit
    the same wall.
    """
    if not _provider.is_configured():
        return {"classified": 0, "errors": 0, "skipped": 0,
                "error": f"{PROVIDER_NAME} OAuth not configured -- run {provider_status()['login_hint']}"}

    from .asx_feed import signal_worthy

    candidates = [c for c in signal_worthy(limit=CANDIDATE_POOL_SIZE) if _worth_a_call(c)]
    with _connect() as conn:
        have = {r["fingerprint"] for r in conn.execute("SELECT fingerprint FROM signals").fetchall()}
    todo = [c for c in candidates if c["fingerprint"] not in have][:limit]

    classified, errors = 0, 0
    for rec in todo:
        try:
            result = classify_one(rec)
        except (grok_oauth.GrokQuotaExceeded, codex_oauth.CodexQuotaExceeded) as e:
            logger.error("%s quota exceeded, aborting batch: %s", PROVIDER_NAME, e)
            return {"classified": classified, "errors": errors, "skipped": len(todo) - classified - errors,
                     "error": f"{PROVIDER_NAME} quota exceeded"}
        except (grok_oauth.GrokAuthError, codex_oauth.CodexAuthError) as e:
            logger.error("%s auth error, aborting batch: %s", PROVIDER_NAME, e)
            return {"classified": classified, "errors": errors, "skipped": len(todo) - classified - errors,
                     "error": f"auth error -- re-run {provider_status()['login_hint']}"}
        except Exception as e:
            logger.warning("classification failed for %s: %s", rec["ticker"], e)
            errors += 1
            continue

        if result is None:
            errors += 1
            continue

        with _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO signals (fingerprint, ticker, signal, score, reason, model, prompt_version, classified_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (rec["fingerprint"], rec["ticker"], result["signal"], result["score"],
                 result["reason"], _MODEL_LABEL, PROMPT_VERSION, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            )
            conn.commit()
        classified += 1

    return {"classified": classified, "errors": errors, "skipped": len(candidates) - len(todo)}


# ---------------------------------------------------------------------------
# Ticker-level aggregation (added 2026-08-24)
# ---------------------------------------------------------------------------

_TICKER_SCHEMA = """
CREATE TABLE IF NOT EXISTS ticker_signals (
    ticker        TEXT NOT NULL,
    session_date  TEXT NOT NULL,
    signal        TEXT NOT NULL,
    score         INTEGER NOT NULL,
    reason        TEXT,
    n_announcements INTEGER NOT NULL,
    fingerprints  TEXT,
    model         TEXT,
    prompt_version TEXT,
    computed_at   TEXT NOT NULL,
    PRIMARY KEY (ticker, session_date)
);
"""

_COMBINE_PROMPT = (
    "You are a terse equity analyst. Below are ALL of one ASX company's "
    "announcements from a single session, each with a score already assigned "
    "to it individually (0-100, 50 = neutral). Produce ONE net score for the "
    "company for that session.\n"
    "CRITICAL: these are usually NOT independent events. ASX convention splits "
    "a single corporate action across several documents -- a results release, a "
    "media release, statutory accounts, an appendix, a dividend notice, a "
    "trading halt. Treat them as one event described several times and score "
    "the EVENT, not the paperwork. Do not average, and do not let the number of "
    "documents about something make it more important.\n"
    "When genuinely good and bad news arrive together -- strong results "
    "alongside a dilutive capital raising is the classic pairing -- weigh what "
    "the market actually prices over the next few days. A sizable discounted "
    "raise usually dominates good results in the short term.\n"
    "Ignore routine filler (trading halts with no stated reason, appendices, "
    "NTA updates) when something material is present.\n"
    "Reply with ONLY a JSON object, no markdown fences: "
    '{"score": <0-100 integer>, "reason": "<=14 words"}.'
)


def _ticker_connect() -> sqlite3.Connection:
    conn = _connect()
    conn.executescript(_TICKER_SCHEMA)
    return conn


def combine_ticker_day(ticker: str, items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """One net score for a ticker's session from its already-scored announcements.

    **Why not arithmetic on the individual scores.** 34% of scored tickers have
    more than one announcement, and they are usually the same event described
    several times (GNG, 2026-08-23: six documents for one "FY26 results plus
    equity raising" event, scoring 20 to 68). Any mean, max or min over them
    double-counts whichever facet the company published most documents about --
    and `max()` in particular is biased bullish exactly when a company pairs
    good results with a dilutive raise, which is the case that matters most.

    **Why this call is cheap.** The per-announcement scores already read the
    PDFs, so this reasons over their headlines/scores/reasons rather than
    re-reading the bodies. One small call per multi-announcement ticker-session.
    """
    if not items:
        return None
    if len(items) == 1:
        one = items[0]
        return {"signal": one["signal"], "score": one["score"],
                "reason": one.get("reason") or "", "n_announcements": 1}

    lines = []
    for it in items:
        lines.append(
            f"- [{it['score']}] {it['headline']}"
            + (f" -- {it['reason']}" if it.get("reason") else "")
        )
    prompt = f"Company: {ticker}\nAnnouncements this session:\n" + "\n".join(lines)

    text = _ask_counted(prompt, system=_COMBINE_PROMPT, kind="net", ticker=ticker)
    parsed = _extract_json(text)
    if not parsed or "score" not in parsed:
        logger.warning("unparseable combine response for %s: %r", ticker, text[:200])
        return None
    try:
        score = max(0, min(100, int(parsed["score"])))
    except (TypeError, ValueError):
        return None
    return {"signal": _signal_for_score(score), "score": score,
            "reason": str(parsed.get("reason") or "")[:200],
            "n_announcements": len(items)}


def refresh_ticker_signals(session_date: str | None = None) -> dict[str, Any]:
    """Compute a net per-ticker score for every ticker with scored
    announcements in `session_date` (UTC date, defaults to the most recent
    date present).

    Idempotent, and now also CHEAP to re-run: a ticker whose scored
    fingerprint set is unchanged since its stored net -- under the same model
    and prompt version -- is skipped rather than re-netted, because the answer
    cannot have moved. Re-scoring a ticker's announcements under a new
    prompt version re-nets that ticker and only that ticker; switching model
    re-nets everything."""
    from .asx_feed import recent_announcements, session_date_for

    # **Not a plain newest-400.** That window is the same trap documented in
    # `classify_pending`: on a busy session 400 announcements reach back only
    # a few hours, so a ticker whose documents fall below it drops out of
    # netting entirely and its stored net can never be corrected. Caught
    # 2026-08-31 -- BC8's 22:34 UTC results release sat 12 minutes below a
    # window whose oldest row was 22:46, so re-scoring its announcements under
    # v4 left the dashboard showing the v3 score with no error anywhere.
    # Asking for the SESSION when one is named is exact; otherwise fall back
    # to the same generous pool the classifier uses.
    # `session_date` now means the same thing on both sides -- the Sydney
    # session -- so it can be pushed into the query instead of being filtered
    # out afterwards.
    anns = {a["fingerprint"]: a
            for a in recent_announcements(limit=CANDIDATE_POOL_SIZE,
                                          session_date=session_date)}
    with _ticker_connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT fingerprint, ticker, signal, score, reason, prompt_version"
            " FROM signals")]

    by_ticker: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        a = anns.get(r["fingerprint"])
        if not a:
            continue
        # The ASX SESSION date (Sydney), not the UTC date. `stamp[:10]` filed
        # a pre-open announcement under the previous calendar day, so a
        # company that announced at 08:34 and again at 11:00 was netted as
        # TWO sessions -- the net for each computed on half the evidence, and
        # the dashboard showing a morning release under yesterday. Migrated
        # 2026-08-31; 460 of 824 stored rows changed date and 60 pairs merged,
        # each of those a single session that had been split in two.
        day = session_date_for(a.get("released_at") or a.get("seen_at") or "")
        if not day or (session_date and day != session_date):
            continue
        r = {**r, "headline": a["headline"]}
        by_ticker.setdefault((r["ticker"], day), []).append(r)

    if session_date is None and by_ticker:
        latest = max(day for _, day in by_ticker)
        by_ticker = {k: v for k, v in by_ticker.items() if k[1] == latest}

    # What has already been netted, so an unchanged ticker is not re-netted.
    with _ticker_connect() as conn:
        existing = {(r["ticker"], r["session_date"]):
                    (set((r["fingerprints"] or "").split(",")), r["model"], r["prompt_version"])
                    for r in conn.execute(
                        "SELECT ticker, session_date, fingerprints, model, prompt_version"
                        " FROM ticker_signals")}

    computed, failed, unchanged = 0, 0, 0
    for (ticker, day), items in by_ticker.items():
        # **Skip a net that cannot have changed.** This function runs after
        # every classify_pending -- every 10 minutes, all day -- and used to
        # re-net every ticker in the session unconditionally, spending one LLM
        # call each time to re-derive an identical answer. Measured 2026-08-25:
        # a run that classified ONE new announcement still made 15 netting
        # calls. Over 144 runs a day that dwarfed the classification spend it
        # was meant to accompany. A net is a pure function of the scored
        # fingerprints that went into it, so an identical fingerprint set under
        # the same model and prompt version must produce the same answer.
        fps = {i["fingerprint"] for i in items}
        # **Keyed on the INPUTS' prompt versions, not the current global.** A
        # net is a pure function of the announcement scores that feed it, and
        # `_COMBINE_PROMPT` is separate from `_SYSTEM_PROMPT`. Comparing
        # against the global constant meant bumping the CLASSIFY prompt
        # re-netted every ticker in the session -- ~100 calls re-deriving
        # identical answers from unchanged v3 scores, then stamping them v4,
        # mislabelling exactly the boundary PROMPT_VERSION exists to mark.
        # Keying on the inputs re-nets precisely the tickers whose underlying
        # scores actually changed, and records honestly when a session is mixed.
        version_key = ",".join(sorted({(i.get("prompt_version") or "?") for i in items}))
        prior = existing.get((ticker, day))
        if prior and prior[0] == fps and prior[1] == _MODEL_LABEL and prior[2] == version_key:
            unchanged += 1
            continue
        result = combine_ticker_day(ticker, items)
        if not result:
            failed += 1
            continue
        with _ticker_connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ticker_signals (ticker, session_date, signal, score,"
                " reason, n_announcements, fingerprints, model, prompt_version, computed_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ticker, day, result["signal"], result["score"], result["reason"],
                 result["n_announcements"], ",".join(i["fingerprint"] for i in items),
                 _MODEL_LABEL, version_key,
                 time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
            )
            conn.commit()
        computed += 1
    return {"computed": computed, "failed": failed, "unchanged": unchanged,
            "tickers": len(by_ticker)}


def get_ticker_signals(session_date: str | None = None,
                       recent_sessions: int = 2) -> dict[str, dict[str, Any]]:
    """{ticker: net signal}. With no `session_date`, returns the most recent
    `recent_sessions` sessions, newest winning per ticker.

    **Two sessions by default, deliberately**: the scanner joins announcements
    over a 36-hour window because the news that moves a stock is usually
    released after the previous close or before the open. Returning a single
    session here would leave those movers with no net score while their raw
    announcements were still on screen -- a mismatch that silently reintroduces
    the per-announcement view this function exists to replace.
    """
    with _ticker_connect() as conn:
        if session_date:
            rows = conn.execute(
                "SELECT * FROM ticker_signals WHERE session_date=?", (session_date,)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM ticker_signals WHERE session_date IN ("
                "  SELECT DISTINCT session_date FROM ticker_signals"
                "  ORDER BY session_date DESC LIMIT ?"
                ") ORDER BY session_date ASC", (recent_sessions,)).fetchall()
    # ASC ordering means a newer session overwrites an older one per ticker.
    return {r["ticker"]: dict(r) for r in rows}


def group_announcements(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse an announcement feed to one row per company per session.

    **Grouped by ticker AND session date, not ticker alone.** The feed spans
    several days; grouping on ticker would merge Monday's results with
    Thursday's placement into a single meaningless row.

    Each group carries the **net** score from `ticker_signals` where one has
    been computed, falling back to the most material individual score
    (furthest from 50) — never the highest, which is the bullish bias that
    made GNG's six-document "results plus dilutive raise" read as Buy 68.
    Children are kept so the individual announcements stay one click away
    rather than being thrown out.
    """
    from .asx_feed import session_date_for

    nets = get_ticker_signals(recent_sessions=7)

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for it in items:
        stamp = it.get("released_at") or it.get("seen_at") or ""
        # The ASX SESSION date (Sydney), not the UTC date. Grouping on
        # `stamp[:10]` split every session in two at 10:00 Sydney, because
        # anything released before then is still the previous date in UTC --
        # so a company that announced pre-open and again mid-morning showed as
        # two rows on two different days, which is the exact double-counting
        # this grouping exists to remove.
        day = session_date_for(stamp)
        key = (it["ticker"], day)
        g = groups.get(key)
        if g is None:
            g = groups[key] = {
                "ticker": it["ticker"], "company": it.get("company"),
                "session_date": day,
                "latest_at": stamp, "universe_rank": it.get("universe_rank"),
                "is_halt": False, "price_sensitive": False, "keyword_flags": [],
                "children": [],
            }
        g["children"].append(it)
        g["latest_at"] = max(g["latest_at"], stamp)
        g["is_halt"] = g["is_halt"] or bool(it.get("is_halt"))
        g["price_sensitive"] = g["price_sensitive"] or bool(it.get("price_sensitive"))
        for k in it.get("keyword_flags") or []:
            if k not in g["keyword_flags"]:
                g["keyword_flags"].append(k)
        if g["universe_rank"] is None:
            g["universe_rank"] = it.get("universe_rank")
        if not g.get("company"):
            g["company"] = it.get("company")

    out = []
    for (ticker, day), g in groups.items():
        scored = [c for c in g["children"] if c.get("score") is not None]
        net = nets.get(ticker)
        # Only trust the stored net if it was computed for THIS session --
        # otherwise a ticker that announced on two days would show one day's
        # net against the other day's announcements.
        #
        # Both sides key on the Sydney session date since the 2026-08-31
        # migration, so this is a direct comparison again -- the previous
        # version had to match against the children's UTC dates because
        # `ticker_signals` was keyed differently from this grouping.
        use_net = bool(net and net.get("session_date") == day)
        if use_net:
            g.update({"signal": net["signal"], "score": net["score"],
                      "signal_reason": net.get("reason"), "is_net": True})
        elif scored:
            best = max(scored, key=lambda c: abs(c["score"] - 50))
            g.update({"signal": best["signal"], "score": best["score"],
                      "signal_reason": best.get("signal_reason"), "is_net": False})
        else:
            g.update({"signal": None, "score": None, "signal_reason": None, "is_net": False})

        g["n_announcements"] = len(g["children"])
        g["n_scored"] = len(scored)
        g["score_range"] = ([min(c["score"] for c in scored), max(c["score"] for c in scored)]
                             if scored else None)
        g["children"].sort(key=lambda c: (c.get("released_at") or ""), reverse=True)
        # Headline shown on the collapsed row: the most material scored one,
        # else the newest. Picking the newest unconditionally would often show
        # an Appendix filed minutes after the results it accompanies.
        g["headline"] = (max(scored, key=lambda c: abs(c["score"] - 50))["headline"]
                          if scored else g["children"][0]["headline"])
        g["url"] = next((c.get("url") for c in g["children"] if c.get("url")), None)
        out.append(g)

    out.sort(key=lambda g: g["latest_at"], reverse=True)
    return out


FREE_TIER_DAILY = {"gemini": 1000}   # Gemini CLI / Code Assist personal OAuth


def usage_summary(days: int = 14) -> dict[str, Any]:
    """Per-day provider call counts, split by kind.

    Days are **UTC**, matching the `called_at` stamp, and that is what the
    `day` field means -- not the Sydney session. The two differ by 10-11
    hours, so a burst of pre-open Sydney classification lands on the previous
    UTC day. Reported as-is rather than silently re-bucketed, since a
    provider's own daily quota is the thing being tracked here, and that
    resets on the provider's clock rather than the ASX's.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _usage_connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT substr(called_at,1,10) AS day, kind, provider,"
            "       COUNT(*) AS n, SUM(ok) AS n_ok, AVG(ms) AS avg_ms,"
            "       AVG(prompt_chars) AS avg_chars"
            " FROM llm_calls WHERE called_at >= ?"
            " GROUP BY day, kind, provider ORDER BY day DESC", (since,))]

    by_day: dict[str, dict[str, Any]] = {}
    for r in rows:
        d = by_day.setdefault(r["day"], {"day": r["day"], "total": 0, "failed": 0,
                                         "classify": 0, "net": 0, "provider": r["provider"]})
        d["total"] += r["n"]
        d["failed"] += r["n"] - (r["n_ok"] or 0)
        if r["kind"] in ("classify", "net"):
            d[r["kind"]] += r["n"]
    days_out = sorted(by_day.values(), key=lambda x: x["day"], reverse=True)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_row = next((d for d in days_out if d["day"] == today), None)
    complete = [d for d in days_out if d["day"] != today]
    return {
        "days": days_out,
        "today": today_row or {"day": today, "total": 0, "classify": 0, "net": 0, "failed": 0},
        "mean_complete_day": (round(sum(d["total"] for d in complete) / len(complete), 1)
                              if complete else None),
        "n_complete_days": len(complete),
        "provider": PROVIDER_NAME,
        "model": _MODEL_LABEL,
        "free_tier_daily": FREE_TIER_DAILY.get(PROVIDER_NAME),
        # Logging began when this table was added; earlier days read as zero
        # because nothing was recorded, not because nothing was spent.
        "logging_since": (lambda v: v[0] if v else None)(
            [r["m"] for r in _usage_connect().execute(
                "SELECT MIN(called_at) AS m FROM llm_calls")]),
    }
