"""News/Injury Analyst — surfaces availability and lineup news.

Tool-binds the news tools (injury news, lineup news), iterates to fetch
headlines for both teams, and writes a markdown ``news_report``. Quantifies the
likely impact of injuries/availability on the favored side.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from sportagent.core.agents.utils.agent_utils import (
    get_game_context_from_state,
    get_language_instruction,
)
from sportagent.core.agents.utils.tools import NEWS_TOOLS


_SYSTEM = (
    "You are the News/Injury Analyst on a sports prediction-market desk. Your "
    "job is to surface late-breaking availability and lineup information that "
    "moves the line.\n\n"
    "{game_context}\n\n"
    "For BOTH teams, use get_injury_news (out / questionable / probable, rest "
    "decisions) and get_lineup_news (starting lineup and rotation changes, "
    "beat-writer reports). Search each team separately.\n\n"
    "Then write a markdown report that: lists key injuries/availability by team "
    "with status; notes starting-lineup or rest decisions; flags late-breaking "
    "news that could move the price; and quantifies the likely impact on the "
    "favored side (e.g. 'a missing starter typically shifts win probability "
    "several points'). If a tool returned a placeholder, say data was "
    "unavailable rather than inventing injuries. End with a net read of which "
    "team availability favors.{language}"
)


def create_news_injury_analyst(llm):
    """Create the News/Injury Analyst node (quick tier, tool-bound)."""
    tools = NEWS_TOOLS
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        MessagesPlaceholder(variable_name="messages"),
    ])

    def news_injury_analyst_node(state):
        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke({
            "messages": state["messages"],
            "game_context": get_game_context_from_state(state),
            "language": get_language_instruction(),
        })
        report = state.get("news_report", "")
        if not getattr(result, "tool_calls", None):
            report = result.content
        return {"messages": [result], "news_report": report}

    return news_injury_analyst_node