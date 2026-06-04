"""Bear Researcher — argues the NO case for the target team.

Reads all four analyst reports + the running debate from state, argues why the
market over-rates the target team (variance, matchup weaknesses, injury risk,
overreaction to public narrative), counters the bull, and appends its turn to
``investment_debate_state``.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from sportagent.core.agents.utils.agent_utils import (
    get_game_context_from_state,
    get_language_instruction,
)


_SYSTEM = (
    "You are the Bear Researcher on a sports prediction-market desk. You argue "
    "the NO case for the target team: why the market OVER-rates it and its true "
    "probability is below the current price.\n\n"
    "{game_context}\n\n"
    "Ground every claim in the analyst reports and the verified-odds snapshot "
    "below. Emphasize single-game variance, matchup weaknesses, injury/rest "
    "risk, and overreaction to public narrative. Engage conversationally and "
    "directly rebut the bull's latest points; do not just repeat your prior "
    "arguments. Cite specific evidence.{language}\n\n"
    "=== Verified odds snapshot ===\n{verified_odds}\n\n"
    "=== Odds report ===\n{odds_report}\n\n"
    "=== Stats report ===\n{stats_report}\n\n"
    "=== News/Injury report ===\n{news_report}\n\n"
    "=== Sentiment report ===\n{sentiment_report}\n\n"
    "=== Debate so far ===\n{history}\n\n"
    "=== Bull's last argument ===\n{bull_response}\n"
)


def create_bear_researcher(llm):
    """Create the Bear Researcher node (quick tier)."""

    def bear_node(state):
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
            bull_response=debate.get("current_response", ""),
        )
        messages = [
            SystemMessage(content=system),
            HumanMessage(content="Make your next bear argument."),
        ]
        result = llm.invoke(messages)
        argument = f"Bear: {getattr(result, 'content', str(result))}"

        new_debate = dict(debate)
        new_debate["bear_history"] = (debate.get("bear_history", "") + "\n" + argument).strip()
        new_debate["history"] = (debate.get("history", "") + "\n" + argument).strip()
        new_debate["current_response"] = argument
        new_debate["count"] = debate.get("count", 0) + 1
        return {"investment_debate_state": new_debate}

    return bear_node