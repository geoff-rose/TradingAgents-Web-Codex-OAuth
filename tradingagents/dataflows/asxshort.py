"""ASX short interest data from asxshort.app (public API, no auth required).

Data has a ~4 trading-day lag per ASIC reporting requirements.
Only meaningful for ASX-listed tickers (*.AX suffix).
"""

from __future__ import annotations

import logging
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

_BASE = "https://api.asxshort.app/api/v1"
_TIMEOUT = 10


def _bare(ticker: str) -> str:
    """Strip .AX suffix for the API (it expects bare codes like 'BHP')."""
    return ticker.upper().replace(".AX", "")


def fetch_short_position(ticker: str) -> dict | None:
    """Return the current short position snapshot for *ticker*, or None on error."""
    code = _bare(ticker)
    try:
        r = requests.get(f"{_BASE}/stocks/{code}", timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("asxshort: failed to fetch %s: %s", ticker, exc)
        return None


def fetch_short_history(ticker: str, days: int = 60) -> list[dict]:
    """Return daily short interest history for the last *days* trading days."""
    code = _bare(ticker)
    try:
        r = requests.get(f"{_BASE}/stocks/{code}/history", params={"days": days}, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return data.get("history", [])
    except Exception as exc:
        logger.warning("asxshort: failed to fetch history for %s: %s", ticker, exc)
        return []


def build_short_interest_block(ticker: str) -> str:
    """Fetch and format short interest data as a structured text block for LLM prompts.

    Returns a plain-text block with current position, trend, and lending data.
    Gracefully degrades — always returns a string even on API failure.
    """
    if not ticker.upper().endswith(".AX"):
        return f"<short_interest>\nNot applicable — {ticker} is not an ASX-listed security.\n</short_interest>"

    snapshot = fetch_short_position(ticker)
    history = fetch_short_history(ticker, days=60)

    if snapshot is None:
        return f"<short_interest>\nShort interest data unavailable for {ticker}.\n</short_interest>"

    sp = snapshot.get("short_position") or {}
    current_pct = sp.get("percentage")
    shares_short = sp.get("shares")
    liability_aud = sp.get("liability_aud")
    report_date = sp.get("report_date", "unknown")
    rank = snapshot.get("rank")
    rank_change = snapshot.get("rank_change")  # positive = more shorted vs last period

    lending = snapshot.get("lending_summary") or {}
    total_borrowed = lending.get("total_borrowed")
    top_borrowers = lending.get("top_borrowers", [])

    lines = [
        f"=== Short Interest: {ticker} ===",
        f"Report date: {report_date}  (note: ASIC data has ~4 trading-day lag)",
        "",
        "--- Current Position ---",
    ]

    if current_pct is not None:
        lines.append(f"Short interest:  {current_pct:.2f}% of issued shares")
    if shares_short is not None:
        lines.append(f"Shares short:    {shares_short:,.0f}")
    if liability_aud is not None:
        lines.append(f"Short liability: A${liability_aud / 1e6:.1f}M")
    if rank is not None:
        direction = ""
        if rank_change is not None:
            direction = f"  (rank changed {rank_change:+d} vs prior period)"
        lines.append(f"ASX short rank:  #{rank} (1 = most shorted){direction}")

    if history:
        lines += ["", "--- 60-Day Trend ---"]
        # Summarise the trend: first vs last entry
        first = history[0]
        last = history[-1]
        first_pct = first.get("short_percentage", 0)
        last_pct = last.get("short_percentage", 0)
        delta = last_pct - first_pct
        trend_word = "INCREASING" if delta > 0.1 else "DECREASING" if delta < -0.1 else "STABLE"
        lines.append(f"Trend ({first['date']} → {last['date']}): {trend_word}  ({first_pct:.2f}% → {last_pct:.2f}%)")

        # Recent 10 data points as a table
        lines.append("")
        lines.append("Recent daily short %:")
        for entry in history[-10:]:
            lines.append(f"  {entry['date']}  {entry.get('short_percentage', 0):.2f}%")

    if top_borrowers:
        lines += ["", "--- Securities Lending (top institutional borrowers) ---"]
        for b in top_borrowers[:5]:
            entity = b.get("entity", "Unknown")
            bshares = b.get("shares", 0)
            lines.append(f"  {entity}: {bshares:,.0f} shares")
        if total_borrowed:
            lines.append(f"  Total borrowed: {total_borrowed:,.0f} shares")

    return "<short_interest>\n" + "\n".join(lines) + "\n</short_interest>"
