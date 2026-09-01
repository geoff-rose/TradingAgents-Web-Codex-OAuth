"""Watchlist: tickers pinned from the scanner or the announcement feed, with a
live-ish quote line and a one-click hand-off into the paper-trade log
(2026-08-24, user request).

**Why a separate table and not a paper trade with zero size**: a watchlist
entry is an *intention to look*, not a position. Conflating the two would
pollute the momentum log's P&L statistics with rows that were never traded --
the same reason `momentum_trades` is kept apart from `swing_trades`.

**Quotes are delayed.** The provider seam is `get_quotes()`: today it reads
yfinance daily bars (same source as the scanner, so the watchlist and the
movers table can never disagree), and the user intends to point it at IBKR
for live quotes later. Everything above this module talks in terms of the
returned dict, so that switch is one function.

**The ask is validated before it is used, not assumed good.** Measured
2026-08-25 at 09:41 Sydney, BHP came back bid 73.49 / ask 64.00 against a 67.12
last -- a crossed book, and the same inversion held for every ticker tried.
That is not bad data: 07:00-10:00 Sydney is the ASX **pre-open auction**, where
orders queue without matching, so bid above ask is the correct state of the
book and is what a broker screen shows too. It does mean a pre-open ask is an
indicative auction level, not something anyone will sell you at, so prefilling
a trade entry from it would put a price in the log that never existed.
Confirmed the same day at 10:35 Sydney, once the 20-minute delay had cleared
the auction: BHP 67.58/67.59, CBA 159.03/159.07, DTL 11.14/11.21 -- tight,
uncrossed, last inside the spread, all three accepted by the guard. So the
provider's ASX book is sound; only the auction phase is unusable.

`get_quote_detail()` therefore validates the ask and reports which side the
caller actually got; during continuous trading a sane ask passes through, and
so will IBKR's when it is wired in.

**The ask is ~20 minutes old.** That delay is why the guard still matters after
10:00 -- for the first ~20 minutes of the session the quote on offer is still
the auction's. It also means a prefilled entry is a 20-minute-old offer, not a
live one, which the dialog says out loud so it is checked before logging.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "swing.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    ticker        TEXT PRIMARY KEY,
    company       TEXT,
    added_at      TEXT NOT NULL,
    source        TEXT,            -- scanner | announcements | manual
    ai_score      INTEGER,         -- score at the moment it was added
    headline      TEXT,            -- what prompted it, if anything
    url           TEXT,
    note          TEXT
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def add(ticker: str, company: str | None = None, source: str | None = None,
        ai_score: int | None = None, headline: str | None = None,
        url: str | None = None, note: str | None = None) -> dict[str, Any]:
    """Pin a ticker. Re-adding one already present refreshes the context
    (score/headline) and keeps the ORIGINAL `added_at` -- "when did this first
    catch my eye" is the useful field, and clobbering it on every re-add from
    a still-running scan would reset it to now every ten minutes.

    `source` defaults to None rather than 'manual' so that re-adding without
    one cannot overwrite how the ticker was ORIGINALLY found; 'manual' is
    applied only when inserting a genuinely new row."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise ValueError("ticker is required")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not company:
        # Resolve once, here, rather than on every render. Comes from the same
        # asxbrief `universe` table the announcement feed uses, so the two
        # pages cannot disagree about what a company is called. A ticker
        # outside the top 500 simply stays unnamed and shows a dash.
        try:
            from .symbols import company_name
            company = company_name(ticker, "AU")
        except Exception:
            company = None
    with _connect() as conn:
        existing = conn.execute("SELECT ticker FROM watchlist WHERE ticker=?", (ticker,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE watchlist SET company=COALESCE(?, company), source=COALESCE(?, source),"
                " ai_score=COALESCE(?, ai_score), headline=COALESCE(?, headline),"
                " url=COALESCE(?, url), note=COALESCE(?, note) WHERE ticker=?",
                (company, source, ai_score, headline, url, note, ticker))
        else:
            conn.execute(
                "INSERT INTO watchlist (ticker, company, added_at, source, ai_score, headline, url, note)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (ticker, company, now, source or "manual", ai_score, headline, url, note))
        conn.commit()
    return {"ticker": ticker, "added": not existing}


def remove(ticker: str) -> dict[str, Any]:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM watchlist WHERE ticker=?", ((ticker or "").strip().upper(),))
        conn.commit()
    return {"removed": cur.rowcount}


def list_items() -> list[dict[str, Any]]:
    with _connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM watchlist ORDER BY added_at DESC")]


# ---------------------------------------------------------------------------
# Quote provider seam. Swap the body of `get_quotes` for IBKR later; the
# returned shape is the contract everything above depends on.
# ---------------------------------------------------------------------------

QUOTE_SOURCE = "yfinance-delayed"


def get_quotes(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """{ticker: {prev_close, open, high, low, last, volume, ...}}.

    One batched `yf.download` rather than a per-ticker `fast_info` call: the
    scanner already reads the same daily bars this way, and going per-ticker
    would turn a 30-name watchlist into 30 round trips on every refresh.

    `last` is the current daily bar's Close, which during the session is the
    last traded price -- delayed, and explicitly labelled as such by
    QUOTE_SOURCE rather than presented as a live quote.
    """
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        return {}
    import yfinance as yf

    from .yf_lock import YF_LOCK

    out: dict[str, dict[str, Any]] = {}
    # Shares the scanner's lock: a browser with both pages open would otherwise
    # have the watchlist refresh and a scan download at the same moment.
    # `.AX` was hard-coded here, so a US name on the watchlist silently showed
    # "no quote" -- the symbol simply did not exist. `resolve_market` is the
    # same lookup the paper-trade side already uses (universe table, then a
    # probe), so both agree on what a ticker is.
    from .symbols import resolve_market, yf_symbol
    sym = {t: yf_symbol(t, resolve_market(t)) for t in tickers}
    with YF_LOCK:
        data = yf.download(sorted(set(sym.values())), period="5d", interval="1d",
                           group_by="ticker", auto_adjust=False, threads=True, progress=False)
    for t in tickers:
        try:
            df = data[sym[t]].dropna(subset=["Close"])
        except (KeyError, TypeError):
            continue
        if df.empty:
            continue
        today = df.iloc[-1]
        prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else None
        last = float(today["Close"])
        out[t] = {
            "prev_close": round(prev_close, 4) if prev_close else None,
            "open": round(float(today["Open"]), 4),
            "high": round(float(today["High"]), 4),
            "low": round(float(today["Low"]), 4),
            "last": round(last, 4),
            "volume": int(today["Volume"]) if today["Volume"] == today["Volume"] else None,
            "move_pct": round((last - prev_close) / prev_close * 100, 2) if prev_close else None,
            "as_of": str(df.index[-1].date()),
            "source": QUOTE_SOURCE,
        }
    return out


def _ask_is_usable(bid: Any, ask: Any, last: Any) -> bool:
    """Whether a quoted ask can be trusted enough to prefill a trade entry.

    Rejects three cases: a missing or non-positive ask; a CROSSED book (bid
    above ask), which is the normal state during the 07:00-10:00 Sydney
    pre-open auction where orders queue without matching -- an indicative
    auction level is not a price anyone will fill at; and an ask more than 20%
    away from the last trade, which means a figure stale from another session.
    """
    try:
        ask = float(ask)
    except (TypeError, ValueError):
        return False
    if ask <= 0:
        return False
    try:
        if bid is not None and float(bid) > 0 and float(bid) > ask:
            return False
    except (TypeError, ValueError):
        pass
    try:
        last = float(last)
        if last > 0 and abs(ask - last) / last > 0.20:
            return False
    except (TypeError, ValueError):
        pass
    return True


def get_quote_detail(ticker: str) -> dict[str, Any]:
    """The single-ticker quote behind the Trade button's price prefill.

    Returns `entry_price` plus `entry_basis` naming where it came from -- 'ask'
    when a usable ask exists, otherwise 'last'. The caller shows that label, so
    a fallback is visible rather than silently passing a last trade off as an
    offer.
    """
    ticker = (ticker or "").strip().upper()
    import yfinance as yf

    quotes = get_quotes([ticker])
    q = quotes.get(ticker, {})
    bid = ask = None
    try:
        from .yf_lock import YF_LOCK
        with YF_LOCK:
            from .symbols import resolve_market, yf_symbol
            info = yf.Ticker(yf_symbol(ticker, resolve_market(ticker))).info
        bid, ask = info.get("bid"), info.get("ask")
    except Exception:
        pass

    last = q.get("last")
    usable = _ask_is_usable(bid, ask, last)
    return {
        "ticker": ticker,
        **q,
        "bid": bid, "ask": ask,
        "ask_usable": usable,
        "entry_price": round(float(ask), 4) if usable else last,
        "entry_basis": "ask" if usable else "last",
        "quote_source": QUOTE_SOURCE,
    }


def items_with_quotes() -> dict[str, Any]:
    items = list_items()
    quotes = get_quotes([i["ticker"] for i in items])
    for i in items:
        i["quote"] = quotes.get(i["ticker"])
    return {"items": items, "quote_source": QUOTE_SOURCE,
            "n_quoted": sum(1 for i in items if i["quote"])}


def backfill_companies(force: bool = False) -> dict[str, Any]:
    """Fill company names for rows added before they were resolved at add
    time. Skips rows that already have one unless `force`."""
    from .symbols import company_name

    rows = list_items()
    updated = []
    for r in rows:
        if r.get("company") and not force:
            continue
        name = company_name(r["ticker"], "AU")
        if not name:
            continue
        with _connect() as conn:
            conn.execute("UPDATE watchlist SET company=? WHERE ticker=?", (name, r["ticker"]))
            conn.commit()
        updated.append({"ticker": r["ticker"], "company": name})
    return {"updated": len(updated), "rows": len(rows), "detail": updated}
