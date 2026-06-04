"""Conservative Risk Debator — protects bankroll against variance.

Quick-tier agent. Reads the trader's position + analyst reports + the running
risk debate, and argues for fractional staking or HOLD when single-game
variance, injury/late-news risk, or a thin edge threaten the bankroll. Appends
its turn to ``risk_debate_state``.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from sportagent.core.agents.utils.agent_utils import (
    get_game_context_from_state,
    get_language_instruction,
)


_SYSTEM = (
    "You are the Conservative Risk Debator on a sports prediction-market desk. "
    "You protect the bankroll: emphasize single-game variance, injury and "
    "late-news risk, and thin-edge traps; argue for fractional staking or HOLD "
    "when the edge is not clearly worth the risk.\n\n"
    "{game_context}\n\n"
    "Ground your case in the trader's position, the verified-odds snapshot, and "
    "the analyst reports. Engage directly with the aggressive and neutral "
    "arguments; do not just repeat yourself.{language}\n\n"
    "=== Verified odds snapshot ===\n{verified_odds}\n\n"
    "=== Trader position proposal ===\n{trader_position_plan}\n\n"
    "=== Analyst reports ===\n{reports}\n\n"
    "=== Risk debate so far ===\n{history}\n"
)


def create_conservative_debator(llm):
    """Create the Conservative Risk Debator node (quick tier)."""

    def conservative_node(state):
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
            HumanMessage(content="Make your next conservative risk argument."),
        ]
        result = llm.invoke(messages)
        argument = f"Conservative: {getattr(result, 'content', str(result))}"

        new_risk = dict(risk)
        new_risk["conservative_history"] = (
            risk.get("conservative_history", "") + "\n" + argument
        ).strip()
        new_risk["history"] = (risk.get("history", "") + "\n" + argument).strip()
        new_risk["current_conservative_response"] = argument
        new_risk["latest_speaker"] = "Conservative"
        new_risk["count"] = risk.get("count", 0) + 1
        return {"risk_debate_state": new_risk}

    return conservative_node