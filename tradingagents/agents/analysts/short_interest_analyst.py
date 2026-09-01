"""Short interest analyst — ASX short selling data from asxshort.app.

Pre-fetches current short position, 60-day trend, and institutional lending
data before invoking the LLM. No tool-calling; all data is in the prompt.

Only meaningful for ASX tickers (.AX). For non-ASX instruments the block
will note the data is not applicable and the LLM will acknowledge this.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.dataflows.asxshort import build_short_interest_block


def create_short_interest_analyst(llm):
    """Create a short interest analyst node for the trading graph."""

    def short_interest_analyst_node(state):
        ticker = state["company_of_interest"]
        trade_date = state["trade_date"]
        instrument_context = build_instrument_context(ticker)

        short_block = build_short_interest_block(ticker)

        system_message = _build_system_message(
            ticker=ticker,
            trade_date=trade_date,
            short_block=short_block,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    "\n{system_message}\n"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=trade_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm
        result = chain.invoke(state["messages"])

        return {
            "messages": [result],
            "short_interest_report": result.content,
        }

    return short_interest_analyst_node


def _build_system_message(*, ticker: str, trade_date: str, short_block: str) -> str:
    return f"""You are a short interest analyst specialising in ASX securities. Your task is to analyse short selling data for {ticker} as of {trade_date} and produce a concise, actionable report for the trading team.

## Short interest data (pre-fetched)

{short_block}

## How to interpret this data

**Short interest % benchmarks (ASX context):**
- < 1%: Minimal shorting — stock is not a meaningful short target
- 1–3%: Low-to-moderate shorting — normal range for large-caps
- 3–6%: Elevated — meaningful bearish conviction from institutional participants
- 6–10%: High — significant institutional concern or known short thesis
- > 10%: Very high — strong consensus short, elevated short-squeeze risk if thesis breaks

**Trend direction matters as much as the level:**
- Rising short interest = growing bearish conviction; can precede price weakness OR indicate a crowded trade vulnerable to a squeeze
- Falling short interest = shorts covering; often a bullish signal as bears reduce exposure
- Stable high short = entrenched short thesis; watch for catalysts that could force a squeeze

**Securities lending (top borrowers):**
Large names (Goldman Sachs, J.P. Morgan, Citadel) borrowing heavily signals institutional short positioning — this is "smart money" betting against the stock. Multiple large borrowers = broad-based consensus short.

**Data lag caveat:**
ASIC requires short position disclosure with a ~4 trading-day delay. Very recent price moves may not yet be reflected in the short data.

**Short squeeze conditions:**
A squeeze can occur when short interest is high AND:
1. The stock starts rising sharply (shorts face mounting losses)
2. A positive catalyst surprises (earnings beat, contract win, regulatory approval)
3. Borrow availability tightens (lenders recall stock)

## Output

Produce a short interest report covering:

1. **Current short positioning** — level, how it compares to ASX norms, and what it implies about institutional sentiment
2. **Trend analysis** — is short interest building, unwinding, or stable? What does this signal?
3. **Institutional conviction** — key borrowers and what their presence implies
4. **Short squeeze assessment** — conditions present or absent; risk level for bears
5. **Signal for the trading team** — Bullish / Bearish / Neutral signal from the short data, with confidence level and key caveats (especially data lag)
6. **Markdown table** summarising: current short %, trend direction, ASX rank, key borrowers, squeeze risk level

{get_language_instruction()}

Finally, conclude your report with a section using **exactly** this format (do not skip it):

## Analyst Signal
**Signal:** [one of: Buy | Overweight | Hold | Underweight | Sell]
**Confidence:** [0-100]/100
**Outlook:** [0-100]/100
**Rationale:** [One sentence summarising what the short interest data implies for the stock]

Outlook is a directional score: 100 = short data is strongly bullish (low/falling shorts, shorts covering), 0 = strongly bearish (high/rising shorts, heavy institutional borrowing), 50 = neutral/mixed."""
