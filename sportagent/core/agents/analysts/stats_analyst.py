"""Stats Analyst — evidence-based on-court edge factors.

Tool-binds the stats tools (team stats, recent form, h2h, standings, rest
status), iterates to fetch data, and writes a markdown ``stats_report``. The
sport adapter supplies the key-factor hints so this analyst stays sport-aware.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from sportagent.core.agents.utils.agent_utils import (
    get_game_context_from_state,
    get_language_instruction,
)
from sportagent.core.agents.utils.tools import STATS_TOOLS


_SYSTEM = (
    "You are the Stats Analyst on a sports prediction-market desk. Your job is "
    "to produce evidence-based, quantitative edge factors for this game.\n\n"
    "{game_context}\n\n"
    "{key_factors}\n\n"
    "Use your tools to gather: each team's season record/standing "
    "(get_team_stats), recent form over the last ~10 games (get_recent_form), "
    "head-to-head this season (get_h2h), league standings (get_standings), and "
    "rest / back-to-back status heading into the game (get_rest_status). "
    "Compare net efficiency, pace, recent form, head-to-head, rest, and "
    "home/away context.\n\n"
    "Then write a markdown report with concrete edge factors and a summary "
    "table. Be specific and cite the numbers your tools returned; if a tool "
    "returned a placeholder, say so rather than inventing data. End with which "
    "team the on-court evidence favors and how strongly.{language}"
)


def create_stats_analyst(llm, key_factors_prompt: str = ""):
    """Create the Stats Analyst node (quick tier, tool-bound).

    ``key_factors_prompt`` comes from the sport adapter (NBA/NFL/...) so the
    analyst weighs what matters in that sport.
    """
    tools = STATS_TOOLS
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        MessagesPlaceholder(variable_name="messages"),
    ])

    def stats_analyst_node(state):
        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke({
            "messages": state["messages"],
            "game_context": get_game_context_from_state(state),
            "key_factors": key_factors_prompt,
            "language": get_language_instruction(),
        })
        report = state.get("stats_report", "")
        if not getattr(result, "tool_calls", None):
            report = result.content
        return {"messages": [result], "stats_report": report}

    return stats_analyst_node