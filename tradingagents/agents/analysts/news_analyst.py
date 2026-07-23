from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_language_instruction,
    get_news,
)
from tradingagents.dataflows.afr import fetch_afr_headlines


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)

        tools = [
            get_news,
            get_global_news,
        ]

        # Pre-fetch AFR headlines for ASX stocks and inject into the system message.
        # AFR is Australia's leading financial newspaper; its public news sitemap
        # is listed in robots.txt for machine consumption.
        afr_block = ""
        if ticker.upper().endswith(".AX"):
            start_dt = (
                datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=14)
            ).strftime("%Y-%m-%d")
            afr_data = fetch_afr_headlines(
                ticker=ticker,
                start_date=start_dt,
                end_date=current_date,
            )
            afr_block = (
                "\n\n### AFR (Australian Financial Review) headlines — public sitemap, past 14 days\n"
                "These are headline titles only from AFR's publicly listed news sitemap. "
                "No article content has been fetched.\n\n"
                f"<start_of_afr>\n{afr_data}\n<end_of_afr>"
            )

        system_message = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for company-specific or targeted news searches, and get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + afr_block
            + "\n\n**IMPORTANT — news vacuum rule**: If there has been no material company-specific news (earnings, guidance, major operational update, M&A, regulatory decision) in the past 14 days, explicitly state this and treat it as a NEUTRAL signal for the company — not positive. A positive macro backdrop (e.g. gold price strength, sector tailwinds) is a macro factor, not a company-specific catalyst. Do NOT use macro tailwinds alone to justify an Overweight or Buy signal for the individual company. Distinguish clearly between (a) company-specific catalysts and (b) sector/macro context, and weight your signal accordingly. Silence is neutral, not bullish."
            + "\n\nFinally, conclude your report with a section using **exactly** this format (do not skip it):\n\n## Analyst Signal\n**Signal:** [one of: Buy | Overweight | Hold | Underweight | Sell]\n**Confidence:** [0-100]/100\n**Outlook:** [0-100]/100\n**Rationale:** [One sentence summarising your key reasoning]\n\nOutlook is a directional score: 100 = maximally bullish signals, 0 = maximally bearish, 50 = neutral/mixed. It reflects the overall quality of the data for the company right now, independent of your confidence in the signal."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
