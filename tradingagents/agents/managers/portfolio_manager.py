"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, score this stock across five dimensions and deliver the final trading decision.

{instrument_context}

---

## Scoring Rubric

Score each dimension independently using the evidence from the analysts. Be accurate — do not cluster scores in the middle. Use the full range.

### 1. Technical Trend (0–25)
- **20–25**: Clear confirmed uptrend — price well above rising 50d & 200d MAs, positive MACD, RSI 50–70
- **14–19**: Constructive — above 200d MA with improving momentum, or recovering from a minor dip
- **8–13**: Mixed/neutral — price near MAs, momentum flat, no clear direction
- **3–7**: Weakening — below 50d MA or MACD turning negative, trend under pressure
- **0–2**: Clear downtrend — below both MAs, falling 200d, negative MACD

### 2. Fundamentals Quality (0–25)
- **20–25**: Excellent — net margin >20%, ROE >20%, clean balance sheet, growing revenue
- **14–19**: Good — solid profitability, manageable debt, stable or growing earnings
- **8–13**: Average — adequate margins, some balance sheet concerns, mixed growth
- **3–7**: Weak — thin margins, deteriorating earnings or heavy debt load
- **0–2**: Poor — losses, solvency concerns, fundamental deterioration

### 3. Valuation Attractiveness (0–20)
- **16–20**: Attractive — meaningful discount to peers/history, PEG <1, or strong yield
- **11–15**: Fair value — in line with sector peers on most metrics
- **6–10**: Slightly expensive — premium to peers, some quality justification
- **2–5**: Expensive — stretched multiples, limited margin of safety
- **0–1**: Very expensive — extreme multiples

### 4. Sentiment & News (0–15)
- **12–15**: Strong positive — bullish company-specific news flow, positive analyst revisions, clear catalysts
- **8–11**: Mildly positive — recent results beat, minor positive announcements, constructive tone
- **5–7**: Neutral — no material company-specific news in the past 14 days, or mixed signals; a supportive macro/sector backdrop alone (e.g. gold price, sector tailwind) does NOT justify above this band without a company-specific catalyst
- **1–4**: Negative — concerning headlines, earnings misses, analyst downgrades, cautious guidance
- **0**: Severely negative — material adverse news or significant sentiment breakdown

### 5. Risk Profile (0–15) — higher score = LOWER risk
Balance-sheet leverage, solvency, and cash generation are already scored under Fundamentals — do NOT re-score them here. Score only volatility, concentration, and execution/regulatory/event risk.
- **12–15**: Low risk — ATR <2% daily, diversified revenue base across markets/products/customers, no pending binary catalysts, no material execution or regulatory overhang
- **8–11**: Moderate risk — ATR 2–4% daily, some concentration (single major product, customer, or region) or execution uncertainty, no existential threat
- **4–7**: Elevated risk — ATR 4–8% daily OR meaningful concentration/execution/regulatory exposure. Single-asset or single-country operations cap this band at 7 regardless of business quality.
- **0–3**: High risk — ATR >8% daily, binary outcome risk (e.g., single regulatory approval, trial readout, contract renewal), or an event that could reprice the stock sharply independent of its underlying fundamentals

---

## Rating Thresholds (derived automatically from total score)
- **70–100 → Buy**
- **55–69 → Overweight**
- **38–54 → Hold**
- **22–37 → Underweight**
- **0–21 → Sell**

---

## Context
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
## Risk Analysts Debate History
{history}

---

Score each dimension honestly based on the evidence. Then write a concise investment thesis summarising what the scores mean together and the recommended portfolio action.{get_language_instruction()}"""

        _FORMAT_FALLBACK = (
            "IMPORTANT — format your entire response exactly as follows "
            "(use these exact headers; do not add extra sections):\n\n"
            "## Rating: **<Buy|Overweight|Hold|Underweight|Sell>** (Score: <N>/100)\n\n"
            "### Scorecard\n\n"
            "| Dimension | Score | Max | Reasoning |\n"
            "|-----------|------:|----:|-----------|\n"
            "| Technical Trend      | <0–25> | 25 | <one sentence> |\n"
            "| Fundamentals         | <0–25> | 25 | <one sentence> |\n"
            "| Valuation            | <0–20> | 20 | <one sentence> |\n"
            "| Sentiment / News     | <0–15> | 15 | <one sentence> |\n"
            "| Risk (lower = worse) | <0–15> | 15 | <one sentence> |\n"
            "| **Total**            | **<N>** | **100** | |\n\n"
            "**Thresholds**: Buy ≥70 · Overweight ≥55 · Hold ≥38 · Underweight ≥22 · Sell <22\n\n"
            "### Investment Thesis\n\n"
            "<2–4 sentences summarising key bull/bear points and recommended action>"
        )

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
            fallback_format=_FORMAT_FALLBACK,
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
