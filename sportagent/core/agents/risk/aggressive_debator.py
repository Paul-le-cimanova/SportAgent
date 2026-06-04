"""Aggressive Risk Debator — champions pressing a real edge.

Quick-tier agent. Reads the trader's position + analyst reports + the running
risk debate, and argues for a full (capped) Kelly stake when edge and confidence
are high, challenging over-caution. Appends its turn to ``risk_debate_state``.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from sportagent.core.agents.utils.agent_utils import (
    get_game_context_from_state,
    get_language_instruction,
)


_SYSTEM = (
    "You are the Aggressive Risk Debator on a sports prediction-market desk. "
    "You champion pressing a genuine edge: when the edge and confidence are "
    "high, argue for the full (capped) Kelly stake and challenge needless "
    "caution that leaves value on the table.\n\n"
    "{game_context}\n\n"
    "Ground your case in the trader's position, the verified-odds snapshot, and "
    "the analyst reports. Engage directly with the conservative and neutral "
    "arguments; do not just repeat yourself.{language}\n\n"
    "=== Verified odds snapshot ===\n{verified_odds}\n\n"
    "=== Trader position proposal ===\n{trader_position_plan}\n\n"
    "=== Analyst reports ===\n{reports}\n\n"
    "=== Risk debate so far ===\n{history}\n"
)


def create_aggressive_debator(llm):
    """Create the Aggressive Risk Debator node (quick tier)."""

    def aggressive_node(state):
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
            HumanMessage(content="Make your next aggressive risk argument."),
        ]
        result = llm.invoke(messages)
        argument = f"Aggressive: {getattr(result, 'content', str(result))}"

        new_risk = dict(risk)
        new_risk["aggressive_history"] = (
            risk.get("aggressive_history", "") + "\n" + argument
        ).strip()
        new_risk["history"] = (risk.get("history", "") + "\n" + argument).strip()
        new_risk["current_aggressive_response"] = argument
        new_risk["latest_speaker"] = "Aggressive"
        new_risk["count"] = risk.get("count", 0) + 1
        return {"risk_debate_state": new_risk}

    return aggressive_node