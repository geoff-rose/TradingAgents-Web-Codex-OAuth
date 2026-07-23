"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: The bull arguments outweighed the bear case. Recommend entering or adding at current prices. This is the normal positive rating — not reserved for perfect situations. Every stock debated will have bear arguments; Buy means the bulls won, not that bears had nothing to say.
- **Overweight**: The bull side had an edge but one material, stock-specific reservation (not just "risks exist") justifies a more cautious sizing.
- **Hold**: The debate was genuinely balanced — neither side clearly prevailed.
- **Underweight**: The bear side had an edge. Reduce exposure.
- **Sell**: The bear arguments clearly outweighed the bull case.

**Calibration guidance**: Think like a sell-side analyst. Across a broad universe, expect roughly 30–40% Buy, 40% Hold, 20% Underweight/Sell. If you're clustering in the middle, you're not making a call — you're avoiding one. Be decisive: if the bulls won the debate, say Buy. "The bull case was credible but there are risks" describes every single Buy-rated stock in existence — that phrasing alone is not a reason to drop to Overweight.

---

**Debate History:**
{history}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
