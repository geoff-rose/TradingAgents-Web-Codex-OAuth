"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager — scored decision
# ---------------------------------------------------------------------------

_SCORE_THRESHOLDS = [
    (70, PortfolioRating.BUY),
    (55, PortfolioRating.OVERWEIGHT),
    (38, PortfolioRating.HOLD),
    (22, PortfolioRating.UNDERWEIGHT),
]


def _rating_from_score(total: int) -> PortfolioRating:
    for threshold, rating in _SCORE_THRESHOLDS:
        if total >= threshold:
            return rating
    return PortfolioRating.SELL


class PortfolioDecision(BaseModel):
    """Scored decision produced by the Portfolio Manager.

    Score each dimension using the rubric in the prompt, then provide
    brief reasoning for each score and an overall thesis. The final
    rating is derived automatically from the total score — do not invent
    a separate rating field.
    """

    # ── Scores (integers, must stay within the stated max) ──────────────
    technical_score: int = Field(
        description=(
            "Technical trend score 0–25. "
            "20-25: clear confirmed uptrend, price well above rising 50d & 200d MA, positive MACD, RSI 50–70. "
            "14-19: constructive, above 200d MA with improving momentum. "
            "8-13: mixed, near MAs, no clear direction. "
            "3-7: weakening, below 50d MA or MACD turning negative. "
            "0-2: clear downtrend, below both MAs, falling 200d."
        ),
    )
    technical_reasoning: str = Field(
        description="One or two sentences explaining the technical score.",
    )

    fundamentals_score: int = Field(
        description=(
            "Fundamentals quality score 0–25. "
            "20-25: excellent — high margins (>20% net), strong ROE (>20%), clean balance sheet, growing revenue. "
            "14-19: good — solid profitability, manageable debt, stable earnings. "
            "8-13: average — adequate margins, some balance sheet concerns, mixed growth. "
            "3-7: weak — thin margins, deteriorating earnings or heavy debt. "
            "0-2: poor — losses, solvency concerns, fundamental deterioration."
        ),
    )
    fundamentals_reasoning: str = Field(
        description="One or two sentences explaining the fundamentals score.",
    )

    valuation_score: int = Field(
        description=(
            "Valuation attractiveness score 0–20. "
            "17-20: attractive — meaningful discount to peers/history, PEG <1, or yield well above sector peers. Quantifiably cheap. "
            "12-16: fair value — at or modest premium to sector peers; growth/quality justifies the multiple. "
            "6-11: expensive — notable stretch in multiples relative to growth, limited margin of safety. "
            "2-5: very expensive — extreme multiples, strong growth already fully priced in. "
            "0-1: bubble territory — no rational valuation support."
        ),
    )
    valuation_reasoning: str = Field(
        description="One or two sentences explaining the valuation score.",
    )

    sentiment_score: int = Field(
        description=(
            "Sentiment and news score 0–15. "
            "12-15: strongly positive — clear bullish company-specific news flow, positive analyst upgrades or revisions, concrete catalysts. "
            "8-11: mildly positive — recent results beat or minor positive announcements; constructive tone with company-specific evidence. "
            "5-7: neutral — no material company-specific news in the past 14 days, or mixed signals. A positive macro/sector backdrop (e.g. gold price strength, sector tailwinds) WITHOUT a company-specific catalyst scores in this band, not higher. "
            "1-4: negative — concerning headlines, earnings misses, analyst downgrades, or cautious guidance. "
            "0: severely negative — material adverse news or a significant sentiment breakdown."
        ),
    )
    sentiment_reasoning: str = Field(
        description="One or two sentences explaining the sentiment score.",
    )

    risk_score: int = Field(
        description=(
            "Risk profile score 0–15 (higher = LOWER risk). "
            "Balance-sheet leverage, solvency, and cash generation are already scored under fundamentals_score — do NOT re-score them here. Score only volatility, concentration, and execution/regulatory/event risk. "
            "12-15: low risk — ATR <2% daily, diversified revenue base across markets/products/customers, no pending binary catalysts, no material execution or regulatory overhang. "
            "8-11: moderate risk — ATR 2–4% daily, some concentration (single major product, customer, or region) or execution uncertainty, no existential threat. "
            "4-7: elevated risk — ATR 4–8% daily OR meaningful concentration/execution/regulatory exposure. Single-asset or single-country operations are capped at 7 regardless of business quality. "
            "0-3: high risk — ATR >8% daily, binary outcome risk (e.g. single regulatory approval, trial readout, contract renewal), or an event that could reprice the stock sharply independent of its underlying fundamentals."
        ),
    )
    risk_reasoning: str = Field(
        description="One or two sentences explaining the risk score.",
    )

    # ── Narrative ────────────────────────────────────────────────────────
    investment_thesis: str = Field(
        description=(
            "Overall investment thesis: what the scores mean together, the key "
            "bull and bear points from the debate, and the recommended portfolio action."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional price target in the instrument's quote currency.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional holding period, e.g. '6–12 months'.",
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision scorecard to markdown."""
    total = (
        decision.technical_score
        + decision.fundamentals_score
        + decision.valuation_score
        + decision.sentiment_score
        + decision.risk_score
    )
    # Clamp individual scores to their maxima before summing
    total = min(total, 100)
    rating = _rating_from_score(total)

    lines = [
        f"## Rating: **{rating.value}** (Score: {total}/100)",
        "",
        "### Scorecard",
        "",
        "| Dimension | Score | Max | Reasoning |",
        "|-----------|------:|----:|-----------|",
        f"| Technical Trend     | {decision.technical_score:>2} | 25 | {decision.technical_reasoning} |",
        f"| Fundamentals        | {decision.fundamentals_score:>2} | 25 | {decision.fundamentals_reasoning} |",
        f"| Valuation           | {decision.valuation_score:>2} | 20 | {decision.valuation_reasoning} |",
        f"| Sentiment / News    | {decision.sentiment_score:>2} | 15 | {decision.sentiment_reasoning} |",
        f"| Risk (lower = worse)| {decision.risk_score:>2} | 15 | {decision.risk_reasoning} |",
        f"| **Total**           | **{total}** | **100** | |",
        "",
        "**Thresholds**: Buy ≥70 · Overweight ≥55 · Hold ≥38 · Underweight ≥22 · Sell <22",
        "",
        "### Investment Thesis",
        "",
        decision.investment_thesis,
    ]
    if decision.price_target is not None:
        lines.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        lines.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(lines)
