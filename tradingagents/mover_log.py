"""Daily OHLC log for scanner movers, and the open-vs-prev-close outcome
report (2026-08-24, user request).

**Why this exists**: "a stock could open 20% up and just drift down all day
and that would be no use" -- the scanner's own thresholds (move >= +8%,
volume >= 3x) are both measured against the PREVIOUS CLOSE, so a stock could
qualify purely on an overnight gap and be actively fading by the time it is
noticed. This logs enough to separate "up on the day" (vs prev close) from
"actually holding the move" (vs open) for every ticker the scanner ever
flagged, and reports the split the user actually asked for.

**One row per (ticker, date)**, upserted every time it appears in a scan.
Every price field is refreshed on each touch, `prev_close` and `open`
included. That is a change from the original design, which wrote them once on
the theory that the opening auction price is fixed for the day. True in
principle, but it made the row unable to heal: a row created before the data
provider published today's bar got yesterday's `prev_close` and `open`, then
later scans corrected `high`/`low`/`close` and left those two stale forever.
The result was a row mixing two sessions -- NXL on 2026-08-25 carried a
prev_close from the 21st against the 25th's high. Now that each row is filed
under its bar's own date, re-reading `open` from that same bar is a no-op in
the normal case and self-healing in the abnormal one. `finalized` distinguishes a row
last touched while the market was open (today's `close` is really "last
traded", could still move) from one confirmed after close.

**Logged for every WATCH-level row, not just FULL setups.** The point is to
measure the general "does a gap hold" question across whatever the scanner
actually surfaces, not just the small subset that also has volume and news --
narrowing the log to FULL rows would answer a different, smaller question.

**Capture does not depend on the page being left open.** `log_movers()` is
called from `scan()` itself, so a browser visit or the 2-minute auto-refresh
populates it during the day, but a `finalize_today()` sweep also runs from a
systemd timer shortly after the ASX close (16:15 Sydney) so the true close is
captured even if nobody was watching -- see the `.service`/`.timer` pair
alongside this module.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "swing.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mover_log (
    ticker      TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    prev_close  REAL NOT NULL,
    open        REAL NOT NULL,
    high        REAL,
    low         REAL,
    close       REAL,
    move_pct_at_log REAL,       -- move vs prev_close when it FIRST qualified as a mover
    volume_ratio_at_log REAL,
    price_band  TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL,
    n_scans     INTEGER NOT NULL DEFAULT 1,
    finalized   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (ticker, trade_date)
);
"""

# Announcement context, snapshotted onto the row at log time rather than
# joined at read time. asxbrief's store keeps only a few days and
# `ticker_signals` is only computed for the last couple of sessions, so a
# read-time join would silently blank out the score column for exactly the
# older rows this table exists to preserve. Recording what the scanner
# actually saw also keeps the log honest as a forward test: a later
# re-classification shouldn't rewrite the score we were looking at on the day.
_ANNOTATION_COLUMNS = {
    "n_announcements": "INTEGER",
    "ai_score": "INTEGER",          # net per-ticker score where one existed
    "ai_is_net": "INTEGER",         # 0 = fell back to a single announcement's score
    "has_price_sensitive": "INTEGER",
    "top_headline": "TEXT",
    "announcement_url": "TEXT",
}


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    have = {r["name"] for r in conn.execute("PRAGMA table_info(mover_log)")}
    for col, typ in _ANNOTATION_COLUMNS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE mover_log ADD COLUMN {col} {typ}")
    return conn


def _annotation_values(r: dict[str, Any]) -> tuple[Any, ...]:
    """The announcement columns pulled off a scanner row, in _ANNOTATION_COLUMNS
    order. `ai_score` is the scanner's own choice of score -- the net
    per-ticker score where asx_signals computed one, else the most material
    single announcement (furthest from 50, never the highest). Deliberately
    not recomputed here: the gap table should show the same number the scanner
    row showed, not a second opinion derived from the same data."""
    n = r.get("n_announcements")
    return (
        int(n) if n is not None else None,
        r.get("ai_score"),
        (1 if r.get("ai_is_net") else 0) if r.get("ai_score") is not None else None,
        (1 if r.get("has_price_sensitive") else 0) if n is not None else None,
        r.get("top_headline"),
        r.get("announcement_url"),
    )


def log_movers(rows: list[dict[str, Any]], as_of: str | None = None) -> int:
    """Upsert one row per (ticker, today) from a scanner result. Returns the
    number of tickers written. `as_of` overrides today's date (UTC), for
    tests and for `finalize_today()` re-stamping a specific day."""
    day = as_of or datetime.now(timezone.utc).date().isoformat()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    n = 0
    with _connect() as conn:
        for r in rows:
            if r.get("last") is None or r.get("prev_close") is None or r.get("open") is None:
                continue
            bar_high = r.get("high") if r.get("high") is not None else r["last"]
            bar_low = r.get("low") if r.get("low") is not None else r["last"]
            existing = conn.execute(
                "SELECT high, low FROM mover_log WHERE ticker=? AND trade_date=?",
                (r["ticker"], day),
            ).fetchone()
            if existing:
                # Take High/Low straight from the bar rather than max/min-ing
                # against what is already stored. The old defensive min/max
                # could only ever widen the range, so a wrong extreme written
                # once was absorbed permanently: NXL's 2026-08-25 row kept a
                # low of 1.635 (the 24th's low) all day after opening at 1.97,
                # because min() preferred the stale smaller number. The daily
                # bar's own High/Low are the running intraday extremes and the
                # row is now filed under that bar's date, so the bar is simply
                # authoritative; a momentarily stale bar self-corrects on the
                # next scan instead of being locked in.
                high, low = bar_high, bar_low
                # **Never clear `finalized` here.** An earlier version set
                # `finalized=0` on every touch, including scans that happen
                # AFTER finalize_today() has already stamped the true close --
                # since a ticker keeps showing as a mover all day (the
                # threshold is against prev_close, which never changes), the
                # 2-minute auto-refresh kept re-logging it and silently undid
                # the finalize timer's work within minutes. Caught 2026-08-24:
                # the timer correctly finalized 33 rows at 16:15 Sydney: by
                # 16:19, only 3 were still finalized. `finalized` is now a
                # one-way stamp for the day, set only by finalize_today().
                # Announcements accumulate through the session and the
                # classifier often scores them well after they land, so these
                # take the LATEST value -- but via COALESCE, so a scan that
                # couldn't reach the signals db degrades to "keep what we had"
                # instead of erasing a score we already recorded.
                conn.execute(
                    "UPDATE mover_log SET prev_close=?, open=?, high=?, low=?, close=?,"
                    " last_seen_at=?, n_scans=n_scans+1,"
                    " n_announcements=COALESCE(?, n_announcements),"
                    " ai_score=COALESCE(?, ai_score), ai_is_net=COALESCE(?, ai_is_net),"
                    " has_price_sensitive=COALESCE(?, has_price_sensitive),"
                    " top_headline=COALESCE(?, top_headline),"
                    " announcement_url=COALESCE(?, announcement_url) "
                    "WHERE ticker=? AND trade_date=?",
                    (r["prev_close"], r["open"], high, low, r["last"], now,
                     *_annotation_values(r), r["ticker"], day),
                )
            else:
                conn.execute(
                    "INSERT INTO mover_log (ticker, trade_date, prev_close, open, high, low, close,"
                    " move_pct_at_log, volume_ratio_at_log, price_band, first_seen_at, last_seen_at,"
                    " n_scans, finalized, n_announcements, ai_score, ai_is_net,"
                    " has_price_sensitive, top_headline, announcement_url)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1,0,?,?,?,?,?,?)",
                    (r["ticker"], day, r["prev_close"], r["open"], bar_high, bar_low, r["last"],
                     r.get("move_pct"), r.get("volume_ratio"), r.get("price_band"), now, now,
                     *_annotation_values(r)),
                )
            n += 1
        conn.commit()
    return n


def finalize_pending(days_back: int = 7, period: str = "1mo") -> dict[str, Any]:
    """Stamp the true close/high/low on every unfinalized row whose session has
    ended, not just today's.

    **Why not just today**: this used to be `finalize_today()`, a one-shot at
    16:15 Sydney. But the scanner keeps running after the close, so any stock
    that first crosses the threshold AFTER the sweep creates a brand-new row
    the sweep has already passed -- and nothing ever finalized it. Measured
    2026-08-25: the sweep ran at 16:15:28 and finalized 43 rows; APX first
    appeared at 16:21 and SVL at 16:31, and both sat permanently "live",
    excluded from the outcome statistics. Three rows from the 24th were stuck
    the same way. Sweeping every unfinalized past session makes this
    self-healing instead of dependent on a single well-timed run.

    **Each row is stamped from the bar matching ITS OWN date**, never
    `iloc[-1]`. The old version took the newest bar, which was fine while it
    only ever ran against today, and would have written today's prices onto
    yesterday's rows the moment it swept anything older -- the same class of
    bug that had the scanner reporting yesterday's move as live.

    A session is only finalized once it has actually ended: a past date
    always, today only after the 16:00 Sydney close.
    """
    import yfinance as yf

    from tradingagents.yf_lock import YF_LOCK

    now_syd = datetime.now(timezone.utc) + timedelta(hours=10)
    today_syd = now_syd.date()
    session_over_today = now_syd.hour >= 16
    since = (today_syd - timedelta(days=days_back)).isoformat()

    with _connect() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT ticker, trade_date FROM mover_log"
            " WHERE finalized=0 AND trade_date>=? ORDER BY trade_date", (since,))]
    # Drop rows for a session still in progress -- their close is not yet real.
    rows = [r for r in rows
            if r["trade_date"] < today_syd.isoformat()
            or (r["trade_date"] == today_syd.isoformat() and session_over_today)]
    if not rows:
        return {"finalized": 0, "pending": 0, "dates": []}

    tickers = sorted({r["ticker"] for r in rows})
    with YF_LOCK:
        data = yf.download([f"{t}.AX" for t in tickers], period=period, interval="1d",
                           group_by="ticker", auto_adjust=False, threads=True, progress=False)

    n, missing = 0, 0
    with _connect() as conn:
        for r in rows:
            t, day = r["ticker"], r["trade_date"]
            try:
                df = data[f"{t}.AX"].dropna(subset=["Close"])
            except (KeyError, TypeError):
                missing += 1
                continue
            idx = [d.date().isoformat() for d in df.index]
            if day not in idx:
                # No bar for that session (holiday, halt, delisting, or simply
                # older than `period`). Left unfinalized rather than stamped
                # from whatever bar happens to be nearest.
                missing += 1
                continue
            bar = df.iloc[idx.index(day)]
            conn.execute(
                "UPDATE mover_log SET high=?, low=?, close=?, finalized=1 "
                "WHERE ticker=? AND trade_date=?",
                (float(bar["High"]), float(bar["Low"]), float(bar["Close"]), t, day),
            )
            n += 1
        conn.commit()
    return {"finalized": n, "pending": missing,
            "dates": sorted({r["trade_date"] for r in rows})}


def finalize_today(period: str = "5d") -> dict[str, Any]:
    """Backward-compatible name for the timer and the /api/scanner/finalize
    endpoint. Now sweeps every unfinalized past session, not only today."""
    return finalize_pending(period=period)


def _row_out(row: sqlite3.Row) -> dict[str, Any]:
    """One row for the API, with the scanner's own setup grade recomputed.

    Not stored: `move_pct_at_log`, `volume_ratio_at_log` and
    `n_announcements` are already on the row, and the grade is just the
    scanner's three thresholds applied to them. Importing those thresholds
    rather than hardcoding 8/3 here means the log can never drift out of step
    with what the movers table calls a FULL setup.
    """
    d = dict(row)
    try:
        from tradingagents.scanner import MIN_MOVE_PCT, MIN_VOLUME_RATIO
    except Exception:
        MIN_MOVE_PCT, MIN_VOLUME_RATIO = 8.0, 3.0
    mv, vr = d.get("move_pct_at_log"), d.get("volume_ratio_at_log")
    d["meets_move"] = bool(mv is not None and mv >= MIN_MOVE_PCT)
    d["meets_volume"] = bool(vr is not None and vr >= MIN_VOLUME_RATIO)
    d["has_news"] = bool(d.get("n_announcements"))
    d["full_setup"] = bool(d["meets_move"] and d["meets_volume"] and d["has_news"])
    d["setup"] = ("full" if d["full_setup"]
                  else ("move+vol" if d["meets_move"] and d["meets_volume"] else "watch"))
    return d


def _classify(row: sqlite3.Row) -> str:
    """The three-way split the user asked for, from ONE row's numbers."""
    if row["close"] is None:
        return "unresolved"
    if row["close"] > row["open"]:
        return "above_open"          # held or built on the gap
    if row["close"] > row["prev_close"]:
        return "above_prev_close_only"   # the user's "gapped 20%, drifted all day" case
    return "at_or_below_prev_close"      # gave the whole move back


def outcomes(days: int = 30, on_date: str | None = None,
             include_provisional: bool = False) -> dict[str, Any]:
    """The report: of the movers the scanner flagged, how many actually held
    above their open by the close versus merely finishing above the previous
    close on drift.

    `on_date` (YYYY-MM-DD) narrows the whole report to a single session,
    which is what the calendar's day cells select: the same three-way split
    and the same rows, just for that day instead of a rolling window.

    **`include_provisional` is for watching TODAY mid-session.** By default
    only `finalized` rows count, because an in-progress row's `close` is
    really "last traded so far" and pooling those into a multi-day statistic
    biases it toward whatever a partial session happens to look like. That
    reasoning holds for the rolling window and NOT for looking at one day as
    it happens -- a provisional outcome is exactly what is useful at 11am. So
    the flag is accepted only alongside `on_date`: a single named session can
    be watched live, the rolling statistic still cannot be contaminated.

    Every row carries `finalized` so provisional ones stay labelled all the
    way to the screen, and the summary reports `n_provisional` separately
    rather than folding them into one undifferentiated count.
    """
    provisional = bool(include_provisional and on_date)
    if on_date:
        where = "trade_date=?" if provisional else "finalized=1 AND trade_date=?"
        params: tuple[Any, ...] = (on_date,)
    else:
        where = "finalized=1 AND trade_date>=?"
        params = ((date.today() - timedelta(days=days)).isoformat(),)
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM mover_log WHERE {where} ORDER BY trade_date DESC, ticker",
            params,
        ).fetchall()

    buckets: dict[str, list[sqlite3.Row]] = {"above_open": [], "above_prev_close_only": [],
                                             "at_or_below_prev_close": []}
    for r in rows:
        k = _classify(r)
        if k in buckets:
            buckets[k].append(r)
    n = len(rows)

    def stats(label: str, group: list[sqlite3.Row]) -> dict[str, Any]:
        if not group:
            return {"label": label, "n": 0, "pct_of_total": 0.0}
        open_to_close = [100 * (r["close"] - r["open"]) / r["open"] for r in group]
        gap_pct = [100 * (r["open"] - r["prev_close"]) / r["prev_close"] for r in group]
        return {
            "label": label, "n": len(group),
            "pct_of_total": round(100 * len(group) / n, 1) if n else 0.0,
            "avg_gap_pct": round(sum(gap_pct) / len(group), 2),
            "avg_open_to_close_pct": round(sum(open_to_close) / len(group), 2),
        }

    return {
        # `n` is every row in the report; `n_finalized` must mean what it says,
        # or a fully provisional day reports itself as fully settled.
        "n_rows": n,
        "n_finalized": sum(1 for r in rows if r["finalized"]),
        "n_provisional": sum(1 for r in rows if not r["finalized"]),
        "provisional_included": provisional,
        "days": days,
        "on_date": on_date,
        "above_open": stats("Closed above open (held/built on the gap)", buckets["above_open"]),
        "above_prev_close_only": stats(
            "Closed above prev close but BELOW open (gapped, then drifted -- the case that "
            "prompted this report)", buckets["above_prev_close_only"]),
        "at_or_below_prev_close": stats(
            "Closed at/below prev close (gave the whole move back)", buckets["at_or_below_prev_close"]),
        "rows": [_row_out(r) for r in rows],
    }


def pending_finalize_count() -> int:
    """How many of today's logged movers are still waiting on the EOD
    finalize sweep -- surfaced on the page so a stale, never-finalized row
    (e.g. the finalize timer failing) is visible rather than silently
    absent from the report."""
    day = datetime.now(timezone.utc).date().isoformat()
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM mover_log WHERE trade_date=? AND finalized=0", (day,)
        ).fetchone()[0]


def daily_summary(start: str | None = None, end: str | None = None) -> dict[str, Any]:
    """Per-session roll-up for the calendar: one entry per trade_date with the
    three-way outcome split, so a month grid can be painted without pulling
    every row for every day.

    Unlike `outcomes()` this counts UNFINALIZED rows too -- but separately,
    in `n_pending`, never mixed into the split. A day whose finalize sweep
    never ran should look visibly incomplete on the calendar rather than
    quietly showing a partial-session split as if it were settled.

    `start`/`end` are inclusive YYYY-MM-DD bounds; omit both for all history
    (the log is one row per mover per day, so a full sweep stays cheap)."""
    where, params = [], []
    if start:
        where.append("trade_date>=?"); params.append(start)
    if end:
        where.append("trade_date<=?"); params.append(end)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM mover_log {clause} ORDER BY trade_date", params).fetchall()

    days: dict[str, dict[str, Any]] = {}
    for r in rows:
        d = days.setdefault(r["trade_date"], {
            "trade_date": r["trade_date"], "n_movers": 0, "n_finalized": 0, "n_pending": 0,
            "above_open": 0, "above_prev_close_only": 0, "at_or_below_prev_close": 0,
        })
        d["n_movers"] += 1
        if not r["finalized"]:
            d["n_pending"] += 1
            continue
        d["n_finalized"] += 1
        k = _classify(r)
        if k in d:
            d[k] += 1

    out = []
    for d in sorted(days.values(), key=lambda x: x["trade_date"]):
        n = d["n_finalized"]
        d["held_pct"] = round(100 * d["above_open"] / n, 1) if n else None
        out.append(d)
    return {
        "days": out,
        "first_date": out[0]["trade_date"] if out else None,
        "last_date": out[-1]["trade_date"] if out else None,
    }


# The ASX closes at 16:00 Sydney = 06:00 UTC, so a session's news window runs
# from the PREVIOUS close to this one. Anchoring on the session rather than on
# "now" (which is what the live scanner's rolling 36h lookback does) is what
# makes the backfill reproducible for a past day.
_SESSION_CLOSE_UTC_HOUR = 6


def backfill_announcements(day: str | None = None, overwrite: bool = False) -> dict[str, Any]:
    """Fill the announcement columns for rows logged before those columns
    existed, by joining asxbrief's store and asx_signals after the fact.

    Only useful for the few days still present in those two databases --
    asxbrief keeps days, `ticker_signals` keeps a couple of sessions. That
    limit is exactly why `log_movers()` snapshots these values going forward
    instead of relying on this. Skips rows that already have a value unless
    `overwrite=True`.
    """
    from .asx_feed import DB_PATH as ASX_DB
    from .asx_feed import session_date_for, tradeable_session_for
    from .asx_signals import get_signals_for, get_ticker_signals

    where = "n_announcements IS NULL" if not overwrite else "1=1"
    params: list[Any] = []
    if day:
        where += " AND trade_date=?"
        params.append(day)
    with _connect() as conn:
        targets = [dict(r) for r in conn.execute(
            f"SELECT ticker, trade_date FROM mover_log WHERE {where}", params)]
    if not targets:
        return {"updated": 0, "rows": 0}
    if not ASX_DB.exists():
        return {"updated": 0, "rows": len(targets), "error": "asxbrief db not found"}

    days = sorted({t["trade_date"] for t in targets})
    # Reach back four calendar days, not one. A session's news window opens at
    # the PREVIOUS TRADING day's close, which for a Monday is the preceding
    # Friday -- three days back, or four across a long weekend. Stepping one
    # calendar day put every Friday-evening announcement outside the fetch, so
    # the weekend roll in `tradeable_session_for` had nothing to roll: four
    # Monday mover rows silently kept zero announcements even after that fix.
    # Over-fetching is free here because `by_key` buckets by tradeable session
    # and only keys matching a mover row are ever read.
    lo = f"{(date.fromisoformat(days[0]) - timedelta(days=4)).isoformat()}T{_SESSION_CLOSE_UTC_HOUR:02d}:00:00"
    hi = f"{days[-1]}T{_SESSION_CLOSE_UTC_HOUR:02d}:00:00"
    conn = sqlite3.connect(f"file:{ASX_DB}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    anns = [dict(r) for r in conn.execute(
        "SELECT fingerprint, ticker, headline, url, price_sensitive, is_halt,"
        " COALESCE(released_at, seen_at) AS at FROM announcements"
        " WHERE COALESCE(released_at, seen_at) >= ? AND COALESCE(released_at, seen_at) < ?",
        (lo, hi))]
    conn.close()

    scores = get_signals_for([a["fingerprint"] for a in anns])
    for a in anns:
        a["ai_score"] = (scores.get(a["fingerprint"]) or {}).get("score")
    # Bucket each announcement into the SESSION it belongs to: anything from a
    # session's close onward is news for the next session, which is the whole
    # point -- a gap is usually explained by an announcement released after the
    # previous close, not during the day it moved.
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for a in anns:
        # `tradeable_session_for` replaces a hand-rolled `+1 day if UTC hour
        # >= 06` test that was wrong twice: it broke under daylight saving
        # (16:00 Sydney is 05:00 UTC in summer, not 06:00) and it rolled a
        # Friday-evening announcement onto Saturday, so it matched no session
        # at all rather than Monday's.
        session = tradeable_session_for(a["at"] or "")
        if session:
            by_key.setdefault((a["ticker"], session), []).append(a)

    nets = get_ticker_signals(recent_sessions=30)

    updated = 0
    with _connect() as db:
        for t in targets:
            group = by_key.get((t["ticker"], t["trade_date"]), [])
            scored = [a for a in group if a["ai_score"] is not None]
            net = nets.get(t["ticker"])
            # A net only counts if it was computed for a session this row's
            # announcements actually fall in -- otherwise a ticker that
            # announced on two days would wear the wrong day's net.
            # `ticker_signals.session_date` is the SYDNEY session date (re-keyed
            # 2026-08-31); comparing it against the announcement's UTC date
            # failed for every pre-open release -- 88 of 180 tickers on the
            # day this was caught -- silently downgrading `ai_is_net` to a
            # single announcement's score instead of the net.
            in_window = {session_date_for(a["at"]) for a in group}
            use_net = bool(net and net.get("session_date") in in_window)
            if use_net:
                ai_score, is_net = net["score"], 1
            elif scored:
                best = max(scored, key=lambda a: abs(a["ai_score"] - 50))
                ai_score, is_net = best["ai_score"], 0
            else:
                ai_score, is_net = None, None
            best_hl = (max(scored, key=lambda a: abs(a["ai_score"] - 50)) if scored
                       else (group[0] if group else None))
            db.execute(
                "UPDATE mover_log SET n_announcements=?, ai_score=?, ai_is_net=?,"
                " has_price_sensitive=?, top_headline=?, announcement_url=?"
                " WHERE ticker=? AND trade_date=?",
                (len(group), ai_score, is_net,
                 1 if any(a["price_sensitive"] for a in group) else 0,
                 best_hl["headline"] if best_hl else None,
                 best_hl["url"] if best_hl else None,
                 t["ticker"], t["trade_date"]),
            )
            updated += 1
        db.commit()
    return {"updated": updated, "rows": len(targets), "announcements_seen": len(anns)}
