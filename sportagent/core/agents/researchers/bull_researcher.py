"""Bull Researcher — argues the YES case for the target team.

Reads all four analyst reports + the running debate from state, argues why the
true probability of the target YES contract exceeds the market price, counters
the bear, and appends its turn to ``investment_debate_state``.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from sportagent.core.agents.utils.agent_utils import (
    get_game_context_from_state,
    get_language_instruction,
)


_SYSTEM = (
    "You are the Bull Researcher on a sports prediction-market desk. You argue "
    "the YES case for the target team: why its true probability of winning "
    "exceeds the current market price.\n\n"
    "{game_context}\n\n"
    "Ground every claim in the analyst reports and the verified-odds snapshot "
    "below. Engage conversationally and directly rebut the bear's latest "
    "points; do not just repeat your prior arguments. Cite specific evidence "
    "(stats, injuries, sentiment, line value).{language}\n\n"
    "=== Verified odds snapshot ===\n{verified_odds}\n\n"
    "=== Odds report ===\n{odds_report}\n\n"
    "=== Stats report ===\n{stats_report}\n\n"
    "=== News/Injury report ===\n{news_report}\n\n"
    "=== Sentiment report ===\n{sentiment_report}\n\n"
    "=== Debate so far ===\n{history}\n\n"
    "=== Bear's last argument ===\n{bear_response}\n"
)


def create_bull_researcher(llm):
    """Create the Bull Researcher node (quick tier)."""

    def bull_node(state):
        debate = state.get("investment_debate_state", {})
        system = _SYSTEM.format(
            game_context=get_game_context_from_state(state),
            language=get_language_instruction(),
            verified_odds=state.get("verified_odds", ""),
            odds_report=state.get("odds_report", ""),
            stats_report=state.get("stats_report", ""),
            news_report=state.get("news_report", ""),
            sentiment_report=state.get("sentiment_report", ""),
            history=debate.get("history", ""),
            bear_response=debate.get("current_response", ""),
        )
        messages = [
            SystemMessage(content=system),
            HumanMessage(content="Make your next bull argument."),
        ]
        result = llm.invoke(messages)
        argument = f"Bull: {getattr(result, 'content', str(result))}"

        new_debate = dict(debate)
        new_debate["bull_history"] = (debate.get("bull_history", "") + "\n" + argument).strip()
        new_debate["history"] = (debate.get("history", "") + "\n" + argument).strip()
        new_debate["current_response"] = argument
        new_debate["count"] = debate.get("count", 0) + 1
        return {"investment_debate_state": new_debate}

    return bull_node