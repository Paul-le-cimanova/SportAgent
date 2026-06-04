"""Odds Analyst — reports Kalshi price vs sportsbook consensus.

Tool-binds the odds tools (Kalshi price, moneyline, line movement, verified
snapshot), iterates to fetch data, and writes a markdown ``odds_report`` to the
state. The verified-odds snapshot is the source of truth for exact prices.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from sportagent.core.agents.utils.agent_utils import (
    get_game_context_from_state,
    get_language_instruction,
)
from sportagent.core.agents.utils.tools import ODDS_TOOLS


_SYSTEM = (
    "You are the Odds Analyst on a sports prediction-market desk. Your job is "
    "to establish what the market currently prices for this game.\n\n"
    "{game_context}\n\n"
    "Workflow:\n"
    "1. Call get_verified_odds_snapshot first — it is the SOURCE OF TRUTH for "
    "exact Kalshi and sportsbook prices. Pass the market ticker, target team, "
    "home/away teams, and sportsbook key.\n"
    "2. Use get_kalshi_price for the live Kalshi YES contract price → implied "
    "probability.\n"
    "3. Use get_moneyline for the de-vigged sportsbook consensus, and "
    "get_line_movement for movement context.\n\n"
    "Then write a markdown report that: states the Kalshi YES/NO price and "
    "implied probability; compares it to the vig-removed sportsbook consensus; "
    "quantifies any Kalshi-vs-book discrepancy in percentage points; and flags "
    "which side (if any) looks mispriced. If the verified snapshot conflicts "
    "with a tool, trust the snapshot and flag the discrepancy — never invent a "
    "reconciled number. End with a one-line summary of the market-implied "
    "probability for the target team.{language}"
)


def create_odds_analyst(llm):
    """Create the Odds Analyst node (quick tier, tool-bound)."""
    tools = ODDS_TOOLS
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        MessagesPlaceholder(variable_name="messages"),
    ])

    def odds_analyst_node(state):
        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke({
            "messages": state["messages"],
            "game_context": get_game_context_from_state(state),
            "language": get_language_instruction(),
        })
        report = state.get("odds_report", "")
        if not getattr(result, "tool_calls", None):
            report = result.content
        return {"messages": [result], "odds_report": report}

    return odds_analyst_node