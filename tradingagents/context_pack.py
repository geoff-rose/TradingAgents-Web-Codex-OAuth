"""Point-in-time context for one announcement: what a professional would have
on screen when it landed (2026-09-01).

**The problem this addresses.** MVF released FY26 results on 31 August. The
classifier read them correctly -- it wrote "2H momentum and FY27 growth
outlook" against the investor presentation -- and then netted the ticker at
Sell 30, because it had nothing to weigh that against. The stock rose 3.7% that
session against an index down 0.2%, and another 9% the next day. The model did
not lack judgement; it lacked the facts. So this assembles them.

**Everything here is strictly point-in-time, and that is the whole design
constraint.** Two rules, both learned by getting it wrong:

  * Prior announcements are filtered by TIMESTAMP, never by date string. The
    first prototype used `date(released_at) < as_of` and fed the model the very
    announcements it was scoring -- MVF's 08:14 Sydney filings are 30 August in
    UTC. That is the fourth appearance of the UTC/Sydney boundary bug in this
    codebase, and in a context pack it is not a mislabel, it is look-ahead.
  * Price history stops before the TRADEABLE session, via
    `tradeable_session_for`. For a pre-open announcement that is the previous
    close; for one released after 16:00 it is that day's close, because the
    market had a full session to react before the news could be traded.

**Cost.** This adds tokens to calls that already happen, not new calls -- the
provider's limit is calls. Price frames are cached per (ticker, session) so a
company filing seven documents costs one download, not seven.
"""

from __future__ import annotations

import functools
import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 120          # of prior announcements to show
MAX_PRIOR = 12               # most recent N, so a serial filer cannot flood it


@functools.lru_cache(maxsize=512)
def _price_context(ticker: str, before_session: str) -> dict[str, Any] | None:
    """Price facts using only bars STRICTLY BEFORE `before_session`."""
    import yfinance as yf

    from .symbols import resolve_market, yf_symbol
    from .yf_lock import YF_LOCK
    try:
        sym = yf_symbol(ticker, resolve_market(ticker))
        with YF_LOCK:
            df = yf.download(sym, period="1y", interval="1d", auto_adjust=False,
                             progress=False)
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.droplevel(1)
        df = df.dropna(subset=["Close"])
        if df.empty:
            return None
        dates = [d.date().isoformat() for d in df.index]
        cut = next((i for i, d in enumerate(dates) if d >= before_session), len(dates))
        prior = df.iloc[:cut]
        if len(prior) < 25:
            return None
        close = prior["Close"]
        px = float(close.iloc[-1])

        def mv(n: int) -> float | None:
            if len(close) <= n:
                return None
            base = float(close.iloc[-1 - n])
            return round((px / base - 1) * 100, 1) if base else None

        hi = float(prior["High"].max())
        lo = float(prior["Low"].min())
        vol20 = float(prior["Volume"].tail(20).mean())
        return {
            "last_close": round(px, 4),
            "d1": mv(1), "d3": mv(3), "m1": mv(21), "m3": mv(63),
            "m12": mv(min(251, len(close) - 1)),
            "range_pos": round((px - lo) / (hi - lo) * 100) if hi > lo else None,
            "hi52": round(hi, 3), "lo52": round(lo, 3),
            "avg_vol_20d": int(vol20) if vol20 == vol20 else None,
        }
    except Exception as exc:
        logger.debug("price context failed for %s: %s", ticker, exc)
        return None


def _prior_announcements(ticker: str, released_at: str,
                         session: str) -> list[dict[str, Any]]:
    """Earlier announcements for this ticker, with the score we gave them.

    Fetched on the raw TIMESTAMP -- a date comparison leaks same-session
    filings straight into the prompt (see the module docstring) -- and then
    narrowed to STRICTLY EARLIER SESSIONS.

    Dropping same-session documents is deliberate. Including them made a
    document's score depend on how many of its siblings had been processed
    first, so re-running a batch in a different order produced different
    scores and a net stopped being a pure function of its inputs. Combining
    the documents of one event is what `combine_ticker_day` already does;
    doing it here as well would double-count the same filings and destroy
    reproducibility for nothing.
    """
    from pathlib import Path

    from .asx_feed import DB_PATH as ASX_DB
    if not ASX_DB.exists() or not released_at:
        return []
    from .asx_feed import tradeable_session_for
    conn = sqlite3.connect(f"file:{ASX_DB}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        raw = [dict(r) for r in conn.execute(
            "SELECT fingerprint, headline, released_local, price_sensitive,"
            " COALESCE(released_at, seen_at) AS at FROM announcements"
            " WHERE ticker=? AND COALESCE(released_at, seen_at) < ?"
            " ORDER BY COALESCE(released_at, seen_at) DESC LIMIT ?",
            (ticker, released_at, MAX_PRIOR * 4))]
    finally:
        conn.close()
    rows = [r for r in raw if tradeable_session_for(r["at"]) != session][:MAX_PRIOR]
    if not rows:
        return []
    sig_db = Path.home() / ".tradingagents" / "asx_signals.db"
    scores: dict[str, int] = {}
    if sig_db.exists():
        c2 = sqlite3.connect(f"file:{sig_db}?mode=ro", uri=True, timeout=5.0)
        try:
            q = ",".join("?" * len(rows))
            scores = {r[0]: r[1] for r in c2.execute(
                f"SELECT fingerprint, score FROM signals WHERE fingerprint IN ({q})",
                [r["fingerprint"] for r in rows])}
        except Exception:
            pass
        finally:
            c2.close()
    for r in rows:
        r["score"] = scores.get(r["fingerprint"])
    return rows


def build(ticker: str, released_at: str | None) -> str:
    """The context block, as plain text for the prompt. Empty when nothing
    useful is known -- an empty block is better than a block of dashes, which
    reads to the model as meaningful absence."""
    from .asx_feed import tradeable_session_for
    from .sectors import benchmark_index, get_map

    ticker = (ticker or "").strip().upper()
    if not ticker:
        return ""
    session = tradeable_session_for(released_at) if released_at else ""
    lines: list[str] = []

    p = _price_context(ticker, session) if session else None
    if p:
        moves = "  ".join(f"{k} {v:+.1f}%" for k, v in
                          (("1d", p["d1"]), ("3d", p["d3"]), ("1m", p["m1"]),
                           ("3m", p["m3"]), ("12m", p["m12"])) if v is not None)
        lines.append(f"Last close before this announcement: {p['last_close']}")
        if moves:
            lines.append(f"Price change into it: {moves}")
        if p["range_pos"] is not None:
            lines.append(f"52-week range {p['lo52']}-{p['hi52']}; currently "
                         f"{p['range_pos']}% of the way up that range")

    prior = _prior_announcements(ticker, released_at, session) if released_at else []
    if prior:
        lines.append("Earlier announcements from this company, from PREVIOUS "
                     "sessions (most recent first, with the score assigned then):")
        for r in prior:
            sc = f"score {r['score']}" if r["score"] is not None else "unscored"
            ps = " [price-sensitive]" if r["price_sensitive"] else ""
            lines.append(f"  - {r['released_local']}{ps} ({sc}): {r['headline'][:90]}")

    m = get_map([ticker]).get(ticker) or {}
    if m.get("sector_name"):
        fit = "" if benchmark_index(m) != "^AXJO" else " (weak fit; benchmarked to the ASX 200)"
        lines.append(f"Sector: {m['sector_name']}{fit}")

    if not lines:
        return ""
    return (
        "CONTEXT (all of it dated strictly BEFORE this announcement; use it to "
        "judge whether the news is a surprise, and in which direction, rather "
        "than judging the numbers in isolation):\n" + "\n".join(lines))
