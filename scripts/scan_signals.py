#!/usr/bin/env python3
"""Scan all saved reports and list their final trading signals.

Usage:
    uv run python scripts/scan_signals.py [--buy] [--sell] [--hold] [--all]

Defaults to showing BUY signals only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LOGS_DIR = Path.home() / ".tradingagents" / "logs"

# Check order matters: longer/more-specific first to avoid "buy" matching inside "strong buy"
_SIGNALS_ORDERED = [
    "strong buy",
    "strong sell",
    "overweight",
    "underweight",
    "hold",
    "buy",
    "sell",
]

_BULLISH = {"overweight", "strong buy", "buy"}
_BEARISH = {"underweight", "strong sell", "sell"}


def _canonicalize(raw: str) -> tuple[str, str]:
    r = raw.strip().lower()
    if r in _BULLISH:
        return "BUY", raw.title()
    if r in _BEARISH:
        return "SELL", raw.title()
    return "HOLD", raw.title()


def parse_signal(text: str) -> tuple[str, str]:
    """Extract trading signal from the header of a report (first 1 000 chars).

    Strategy:
    1. Extract all **...** bold spans from the header.
    2. Check each span for a signal word.
    3. Skip spans where the signal word is clearly negated ("not a strong buy").

    This avoids false positives from body text like "This is not a strong buy setup."
    because those phrases are not wrapped in bold markdown.
    """
    header = text[:1000]

    # Also try explicit "Rating: SIGNAL" pattern (handles "**Rating: Underweight**")
    rating_match = re.search(
        r"Rating:\s*(Strong\s+Buy|Strong\s+Sell|Overweight|Underweight|Hold|Buy|Sell)",
        header,
        re.I,
    )
    if rating_match:
        return _canonicalize(rating_match.group(1))

    # Extract all bold spans: **...** (non-greedy, no newlines)
    bold_spans = re.findall(r"\*\*([^*\n]{1,120})\*\*", header)
    for span in bold_spans:
        span_lower = span.lower()
        for signal in _SIGNALS_ORDERED:
            if signal not in span_lower:
                continue
            # Avoid spans that negate the signal ("not a strong buy")
            idx = span_lower.find(signal)
            prefix = span_lower[:idx]
            if re.search(r"\bnot\b", prefix):
                continue
            return _canonicalize(signal.title())

    return "UNKNOWN", "Unknown"


def iter_reports():
    """Yield (ticker, date, decision_text) for every saved report, newest per ticker."""
    if not LOGS_DIR.exists():
        return

    for ticker_dir in sorted(LOGS_DIR.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        found = False

        # New format: logs/{ticker}/{date}/reports/final_trade_decision.md
        for date_dir in sorted(ticker_dir.iterdir(), reverse=True):
            if not date_dir.is_dir() or date_dir.name == "TradingAgentsStrategy_logs":
                continue
            md = date_dir / "reports" / "final_trade_decision.md"
            if md.exists():
                yield ticker, date_dir.name, md.read_text()
                found = True
                break

        if found:
            continue

        # Legacy format: logs/{ticker}/TradingAgentsStrategy_logs/full_states_log_{date}.json
        legacy = ticker_dir / "TradingAgentsStrategy_logs"
        if legacy.exists():
            for f in sorted(legacy.glob("full_states_log_*.json"), reverse=True):
                try:
                    raw = json.loads(f.read_text())
                    dec = raw.get("final_trade_decision") or ""
                    if dec:
                        date = f.stem.replace("full_states_log_", "")
                        yield ticker, date, dec
                except Exception:
                    pass
                break


def main():
    want_buy     = "--buy"  in sys.argv or "--all" in sys.argv
    want_sell    = "--sell" in sys.argv or "--all" in sys.argv
    want_hold    = "--hold" in sys.argv or "--all" in sys.argv
    want_unknown = "--all"  in sys.argv

    # Default: BUY only
    if not any(a in sys.argv for a in ("--buy", "--sell", "--hold", "--all")):
        want_buy = True

    results: list[tuple[str, str, str, str]] = []
    for ticker, date, text in iter_reports():
        category, label = parse_signal(text)
        results.append((ticker, date, category, label))

    buy_count  = sum(1 for _, _, c, _ in results if c == "BUY")
    sell_count = sum(1 for _, _, c, _ in results if c == "SELL")
    hold_count = sum(1 for _, _, c, _ in results if c == "HOLD")
    unk_count  = sum(1 for _, _, c, _ in results if c == "UNKNOWN")

    print(f"\nScanned {len(results)} reports: "
          f"{buy_count} BUY  {sell_count} SELL  {hold_count} HOLD  {unk_count} UNKNOWN\n")

    filtered = [r for r in results if (
        (r[2] == "BUY"     and want_buy)  or
        (r[2] == "SELL"    and want_sell) or
        (r[2] == "HOLD"    and want_hold) or
        (r[2] == "UNKNOWN" and want_unknown)
    )]

    if not filtered:
        print("No matching reports.")
        return

    for ticker, date, category, label in filtered:
        print(f"  {ticker:<12}  {date}  [{label}]")

    print()


if __name__ == "__main__":
    main()
