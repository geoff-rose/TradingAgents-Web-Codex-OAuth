"""Deterministic quality checks and compact action summaries for saved reports.

The language models remain responsible for the analysis, but these checks make
missing evidence, date leakage, and execution levels visible to the operator.
They are deliberately non-blocking: a questionable report is saved with a
warning rather than silently presented as complete.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Mapping


REPORT_KEYS = (
    "market_report",
    "fundamentals_report",
    "news_report",
    "sentiment_report",
    "short_interest_report",
    "final_trade_decision",
)
_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_SCORE_RE = re.compile(r"(?:score\s*:\s*|total\s*\|\s*\*{0,2})(\d{1,3})\s*/\s*100", re.I)
_RATING_RE = re.compile(r"^\s*##\s*Rating:\s*\*{0,2}([^(*\n]+)", re.I | re.M)
_ANALYST_SIGNAL_RE = re.compile(r"\*{0,2}Signal\*{0,2}\s*:\s*\[?([^\]\n]+)", re.I)
_ASOF_CONTEXT_RE = re.compile(
    r"(?:as\s+of|as-at|report\s+date|analysis\s+date|current\s+date|"
    r"data\s+as|retrieved(?:\s+snapshot)?\s+through|snapshot\s+through)",
    re.I,
)


def _label_number(text: str, label: str) -> float | None:
    match = re.search(
        rf"(?:^|\n)\s*\*{{0,2}}(?:Preliminary\s+)?{re.escape(label)}\*{{0,2}}\s*:\s*\$?([0-9]+(?:\.[0-9]+)?)",
        text or "",
        re.I,
    )
    return float(match.group(1)) if match else None


def build_action_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the fields useful on a dashboard without asking an LLM."""
    decision = str(state.get("final_trade_decision") or "")
    trader = str(state.get("trader_investment_plan") or "")
    rating_match = _RATING_RE.search(decision)
    score_match = _SCORE_RE.search(decision)
    target = _label_number(decision, "Price Target")
    return {
        "rating": rating_match.group(1).strip().title() if rating_match else None,
        "score": int(score_match.group(1)) if score_match else None,
        "entry_price": _label_number(trader, "Entry Price"),
        "stop_loss": _label_number(trader, "Stop Loss"),
        "price_target": target,
        "analyst_signal": (
            _ANALYST_SIGNAL_RE.search(decision).group(1).strip().title()
            if _ANALYST_SIGNAL_RE.search(decision) else None
        ),
    }


def validate_report_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Check completeness and obvious as-of-date violations in a final state."""
    trade_date = str(state.get("trade_date") or "")
    missing = [key for key in REPORT_KEYS if not str(state.get(key) or "").strip()]
    all_text = "\n".join(str(state.get(key) or "") for key in REPORT_KEYS)
    future_dates: list[str] = []
    if trade_date:
        try:
            cutoff = date.fromisoformat(trade_date)
            candidates = []
            for match in _DATE_RE.finditer(all_text):
                value = match.group(1)
                if date.fromisoformat(value) <= cutoff:
                    continue
                # Future event dates (e.g. a dividend payment date) are valid;
                # only flag future dates used to describe the evidence cutoff.
                prefix = all_text[max(0, match.start() - 160):match.start()]
                # Keep the nearest sentence/line only so an earlier "report
                # date" does not classify an unrelated future event date.
                context = re.split(r"[.\n]", prefix)[-1]
                if _ASOF_CONTEXT_RE.search(context):
                    candidates.append(value)
            future_dates = sorted(set(candidates))
        except ValueError:
            future_dates = []

    warnings: list[str] = []
    if missing:
        warnings.append("Missing report sections: " + ", ".join(missing))
    if future_dates:
        warnings.append(
            "Dates after the analysis date detected: " + ", ".join(future_dates)
        )

    summary = build_action_summary(state)
    if summary["score"] is not None and summary["score"] >= 70 and missing:
        warnings.append("High score is based on incomplete analyst evidence")
    status = "error" if missing else "warning" if warnings else "ok"
    return {
        "status": status,
        "missing_reports": missing,
        "future_dates": future_dates,
        "warnings": warnings,
        "action_summary": summary,
    }
