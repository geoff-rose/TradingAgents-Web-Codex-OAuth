"""Which exchange a ticker belongs to, and what yfinance calls it.

**Why this exists**: every price path in this project assumed ASX and appended
`.AX` unconditionally. That silently broke the moment a US ticker was logged:
five of eight open paper trades (BE, CCJ, OKLO, INTC, UUUU) were being marked
to market as `INTC.AX` and friends, which return nothing, so those trades sat
with no last price and no P&L indefinitely -- inert rather than wrong-looking,
which is worse. Found 2026-08-25.

**Resolved once and stored, not re-derived per price fetch.** A trade or
watchlist row records its market at creation. Re-detecting on every
mark-to-market would mean a network probe per row per refresh, and would let a
row silently change exchange if a probe failed once.

**Detection order** (`resolve_market`): an explicit caller hint wins; then
asxbrief's `universe` table, which is a local lookup and definitive for the
ASX top 500; then a probe of `.AX` before the bare symbol. AU is tried first
because this is an ASX-focused tool and several ASX codes collide with US
ones. The probe is the only network cost and happens once per new ticker.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

ASX_DB = Path(os.getenv("ASXBRIEF_DB_PATH", "/opt/asxbrief/data/asx.db"))

AU, US = "AU", "US"


def yf_symbol(ticker: str, market: str = AU) -> str:
    """The symbol yfinance expects. ASX codes carry `.AX`; US ones are bare."""
    t = (ticker or "").strip().upper()
    return f"{t}.AX" if (market or AU).upper() == AU else t


def _in_asx_universe(ticker: str) -> bool:
    if not ASX_DB.exists():
        return False
    try:
        conn = sqlite3.connect(f"file:{ASX_DB}?mode=ro", uri=True, timeout=5.0)
        try:
            row = conn.execute(
                "SELECT 1 FROM universe WHERE ticker=?", (ticker,)).fetchone()
            return row is not None
        finally:
            conn.close()
    except Exception:
        return False


def _has_data(symbol: str) -> bool:
    try:
        import yfinance as yf

        from .yf_lock import YF_LOCK
        with YF_LOCK:
            df = yf.Ticker(symbol).history(period="5d")
        return not df.empty
    except Exception:
        return False


def resolve_market(ticker: str, hint: str | None = None) -> str:
    """'AU' or 'US' for a ticker. Falls back to AU when nothing resolves, so a
    transient network failure cannot silently reclassify an ASX holding."""
    t = (ticker or "").strip().upper()
    if hint and hint.upper() in (AU, US):
        return hint.upper()
    if not t:
        return AU
    if _in_asx_universe(t):
        return AU
    if _has_data(f"{t}.AX"):
        return AU
    if _has_data(t):
        return US
    logger.info("could not resolve a market for %s; defaulting to AU", t)
    return AU


def company_name(ticker: str, market: str = AU) -> str | None:
    """A display name for a ticker, or None if nothing resolves.

    ASX names come from asxbrief's `universe` table -- a local lookup, and the
    same source the announcement feed uses, so the two pages cannot disagree
    about what a company is called. It only covers the top 500; anything
    outside falls through to the provider.

    US names come from yfinance's `info`, which is a network call. Resolved
    once and stored by the caller rather than looked up per render.
    """
    t = (ticker or "").strip().upper()
    if not t:
        return None
    if (market or AU).upper() == AU and ASX_DB.exists():
        try:
            conn = sqlite3.connect(f"file:{ASX_DB}?mode=ro", uri=True, timeout=5.0)
            try:
                row = conn.execute(
                    "SELECT company FROM universe WHERE ticker=?", (t,)).fetchone()
                if row and row[0]:
                    return str(row[0])
            finally:
                conn.close()
        except Exception:
            pass
    try:
        import yfinance as yf

        from .yf_lock import YF_LOCK
        with YF_LOCK:
            info = yf.Ticker(yf_symbol(t, market)).info
        name = info.get("longName") or info.get("shortName")
        return str(name) if name else None
    except Exception:
        return None
