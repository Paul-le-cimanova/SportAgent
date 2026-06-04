"""Neutral Risk Debator — balances aggressive and conservative stances.

Quick-tier agent. Reads the trader's position + analyst reports + the running
risk debate, and argues for a realistic stake given edge confidence and
variance, flagging where either extreme overreaches. Appends its turn to
``risk_debate_state``.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from sportagent.core.agents.utils.agent_utils import (
    get_game_context_from_state,
    get_language_instruction,
)


_SYSTEM = (
    "You are the Neutral Risk Debator on a sports prediction-market desk. You "
    "balance the aggressive and conservative views: argue for a realistic "
    "stake that reflects both the edge confidence and the single-game "
    "variance, and flag where either extreme overreaches.\n\n"
    "{game_context}\n\n"
    "Ground your case in the trader's position, the verified-odds snapshot, and "
    "the analyst reports. Engage directly with the aggressive and conservative "
    "arguments; do not just repeat yourself.{language}\n\n"
    "=== Verified odds snapshot ===\n{verified_odds}\n\n"
    "=== Trader position proposal ===\n{trader_position_plan}\n\n"
    "=== Analyst reports ===\n{reports}\n\n"
    "=== Risk debate so far ===\n{history}\n"
)


def create_neutral_debator(llm):
    """Create the Neutral Risk Debator node (quick tier)."""

    def neutral_node(state):
        risk = state.get("risk_debate_state", {})
        reports = "\n\n".join([
            state.get("odds_report", ""),
            state.get("stats_report", ""),
            state.get("news_report", ""),
            state.get("sentiment_report", ""),
        ])
        system = _SYSTEM.format(
            game_context=get_game_context_from_state(state),
            language=get_language_instruction(),
            verified_odds=state.get("verified_odds", ""),
            trader_position_plan=state.get("trader_position_plan", ""),
            reports=reports,
            history=risk.get("history", ""),
        )
        messages = [
            SystemMessage(content=system),
            HumanMessage(content="Make your next neutral risk argument."),
        ]
        result = llm.invoke(messages)
        argument = f"Neutral: {getattr(result, 'content', str(result))}"

        new_risk = dict(risk)
        new_risk["neutral_history"] = (
            risk.get("neutral_history", "") + "\n" + argument
        ).strip()
        new_risk["history"] = (risk.get("history", "") + "\n" + argument).strip()
        new_risk["current_neutral_response"] = argument
        new_risk["latest_speaker"] = "Neutral"
        new_risk["count"] = risk.get("count", 0) + 1
        return {"risk_debate_state": new_risk}

    return neutral_node