"""Read-only access to the asxbrief collector's SQLite store.

asxbrief is a separate systemd service (/opt/asxbrief) that independently polls
ASX announcements and writes to its own WAL-mode SQLite db. This module only
reads it -- schema ownership and writes stay in asxbrief.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

DB_PATH = Path(os.getenv("ASXBRIEF_DB_PATH", "/opt/asxbrief/data/asx.db"))

# asxbrief stores `released_at` in UTC. The ASX day does NOT line up with the
# UTC day: Sydney is UTC+10/+11, so a pre-open announcement at 07:30 Sydney is
# 21:30 UTC on the PREVIOUS date. Filtering "today" on the UTC date would
# therefore drop exactly the announcements that matter most -- the ones
# released before the open. Every day filter here is a Sydney-local day
# converted to a UTC range, via zoneinfo rather than a fixed +10 so the
# AEST/AEDT switch is handled instead of silently shifting the window by an
# hour for half the year.
SYDNEY = ZoneInfo("Australia/Sydney")


def sydney_today() -> str:
    """Current ASX calendar date (YYYY-MM-DD), Sydney local."""
    return datetime.now(SYDNEY).date().isoformat()


def session_date_for(stamp: str | None) -> str:
    """The ASX session (Sydney calendar date) a UTC timestamp belongs to.

    The reason this is not `stamp[:10]`: an announcement released at 07:30
    Sydney is 21:30 UTC on the previous date, so slicing the UTC string files
    a company's pre-open announcement under a different day than its 11:00
    one -- splitting a single session in two.
    """
    if not stamp:
        return ""
    try:
        return datetime.fromisoformat(stamp).astimezone(SYDNEY).date().isoformat()
    except ValueError:
        return stamp[:10]


def tradeable_session_for(stamp: str | None) -> str:
    """The ASX session an announcement can first be TRADED in.

    Distinct from `session_date_for`, which answers "what day was this
    published". For evaluation the two differ for roughly a quarter of all
    announcements: measured 2026-08-31 over 6,776 stored announcements, 52.0%
    are pre-open, 24.7% in-session, and **23.2% land after the close** -- those
    cannot move the price until the following session, so scoring them against
    the day they were published charges them with a move that preceded them.

    Rolls over the weekend. Public holidays are NOT handled: that needs an
    exchange calendar, and the failure mode is mild (the session lands on a
    non-trading day and simply matches no row) rather than wrong.
    """
    if not stamp:
        return ""
    try:
        local = datetime.fromisoformat(stamp).astimezone(SYDNEY)
    except ValueError:
        return stamp[:10]
    d = local.date()
    # 16:00 Sydney, derived from the local clock rather than a fixed UTC hour
    # -- the old `stamp[11:13] >= "06"` test silently breaks under daylight
    # saving, when 16:00 Sydney is 05:00 UTC rather than 06:00.
    if local.hour >= 16:
        d += timedelta(days=1)
    while d.weekday() >= 5:          # Sat/Sun -> Monday
        d += timedelta(days=1)
    return d.isoformat()


def sydney_day_bounds(session_date: str) -> tuple[str, str]:
    """[start, end) UTC ISO timestamps spanning one Sydney calendar day, in
    the same format asxbrief writes (`...+00:00`), so they compare correctly
    as strings against `released_at`."""
    start = datetime.fromisoformat(session_date).replace(tzinfo=SYDNEY)
    end = start + timedelta(days=1)
    utc = ZoneInfo("UTC")
    return (start.astimezone(utc).isoformat(), end.astimezone(utc).isoformat())


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def recent_announcements(
    limit: int = 100, kind: str = "all", universe_only: bool = False,
    session_date: str | None = None, ticker: str | None = None,
) -> list[dict[str, Any]]:
    """kind: 'all' | 'halts' | 'sensitive' | 'keywords'

    `ticker` restricts to one company across ALL stored sessions. Searching
    has to reach past the session filter: the stream shows today by default,
    so a client-side filter over the loaded rows can never find the thing a
    search is usually for -- an announcement from an earlier day.

    `session_date` restricts to a single ASX day -- 'today' for the current
    Sydney date, or an explicit 'YYYY-MM-DD'. Without it the query is the
    plain "newest N" it always was, which spans however many days it takes to
    reach `limit` (on a quiet run that reaches back several sessions).

    `universe_only` joins live against asxbrief's `universe` table (current
    top-N by market cap) rather than trusting each row's `in_universe` flag,
    which is only ever set at collection time -- a live join means refreshing
    the universe re-filters all history immediately, not just future rows.

    `keyword_flags` (e.g. ["State Street"]) comes from asxbrief's PDF-keyword
    watch -- only populated for announcement types where a fetch was
    triggered (currently: substantial-holding notices), NULL/empty for
    everything else, not "checked, no match" vs "never checked" distinct at
    this layer. See asxbrief's sources/pdf_text.py for why headline-only
    matching wouldn't work for this.
    """
    if not DB_PATH.exists():
        return []
    # `a.company` is NULL for every row asxbrief has ever written (checked
    # 2026-08-25: 0 of 2510) -- the collector records the ticker but not the
    # name. The `universe` table it also maintains does carry names, and this
    # query already joins it for the rank, so the name comes from there.
    # COALESCE rather than using u.company outright, so that if asxbrief ever
    # starts populating its own column that value wins automatically.
    # Consequence: a ticker outside the top-500 universe has no name available
    # anywhere locally and comes back NULL, which the UI shows as a dash.
    q = (
        "SELECT a.fingerprint, a.ticker, COALESCE(a.company, u.company) AS company, "
        "a.headline, a.released_at, "
        "a.released_local, a.seen_at, a.price_sensitive, a.is_halt, a.halt_kind, "
        "a.url, a.pages, a.keyword_flags, u.rank AS universe_rank FROM announcements a "
        "LEFT JOIN universe u ON u.ticker = a.ticker"
    )
    where, args = [], []
    if ticker:
        where.append("UPPER(a.ticker) = ?")
        args.append(ticker.strip().upper())
    if session_date:
        lo, hi = sydney_day_bounds(sydney_today() if session_date == "today" else session_date)
        where.append("COALESCE(a.released_at, a.seen_at) >= ? AND COALESCE(a.released_at, a.seen_at) < ?")
        args += [lo, hi]
    if universe_only:
        where.append("u.ticker IS NOT NULL")
    if kind == "halts":
        where.append("a.is_halt=1")
    elif kind == "sensitive":
        where.append("a.price_sensitive=1")
    elif kind == "keywords":
        where.append("a.keyword_flags IS NOT NULL AND a.keyword_flags != ''")
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY COALESCE(a.released_at, a.seen_at) DESC LIMIT ?"
    with _connect() as conn:
        rows = conn.execute(q, (*args, limit)).fetchall()
    out = [dict(r) for r in rows]
    for r in out:
        r["keyword_flags"] = r["keyword_flags"].split(",") if r["keyword_flags"] else []
    return out


def signal_worthy(limit: int = 50) -> list[dict[str, Any]]:
    """Every announcement from a top-N-universe company -- the subset worth
    spending an LLM call on. Newest first.

    Originally restricted to price-sensitive/halt only, but plenty of
    material-looking announcements from large companies (debt issuance,
    buy-backs, director's dealings) aren't flagged price-sensitive by ASX
    itself, and the user wants those graded too (2026-08-21) -- in-universe
    membership alone is now the bar, not ASX's own flag.
    """
    if not DB_PATH.exists():
        return []
    q = (
        # `a.url` is required by the classifier's body fetch (added 2026-08-23).
        # Without it `classify_one` silently degrades to headline-only -- which
        # it did, scoring 10 announcements while fetching exactly one body.
        "SELECT a.fingerprint, a.ticker, COALESCE(a.company, u.company) AS company, "
        "a.headline, a.released_at, "
        "a.is_halt, a.halt_kind, a.price_sensitive, a.url, u.rank AS universe_rank "
        "FROM announcements a JOIN universe u ON u.ticker = a.ticker "
        "ORDER BY COALESCE(a.released_at, a.seen_at) DESC LIMIT ?"
    )
    with _connect() as conn:
        rows = conn.execute(q, (limit,)).fetchall()
    return [dict(r) for r in rows]


def health() -> dict[str, Any]:
    if not DB_PATH.exists():
        return {"available": False}
    with _connect() as conn:
        row = conn.execute(
            "SELECT ts, ok, error FROM poll_log ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        count = conn.execute("SELECT COUNT(*) n FROM announcements").fetchone()["n"]
        universe_size = conn.execute("SELECT COUNT(*) n FROM universe").fetchone()["n"]
    return {
        "available": True,
        "count": count,
        "universe_size": universe_size,
        "last_poll_ts": row["ts"] if row else None,
        "last_poll_ok": bool(row["ok"]) if row else None,
        "last_poll_error": row["error"] if row else None,
    }
