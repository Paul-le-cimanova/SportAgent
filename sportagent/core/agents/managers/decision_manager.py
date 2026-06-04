"""Decision Manager — synthesizes the risk debate, emits the FinalRecommendation.

Deep-tier agent and the terminal node. Reads the risk debate, the investment
plan, the trader's position, the verified-odds snapshot, and any prior settled
lessons (``past_context``). Emits a structured ``FinalRecommendation`` (free-text
fallback) rendered to ``final_recommendation`` and the risk debate's judge
decision. All numbers are grounded in the verified snapshot; the deterministic
``probability.py`` helpers supply implied probability / edge so the LLM never
invents the quantitative figures.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from sportagent.core.agents.schemas import FinalRecommendation, render_final_recommendation
from sportagent.core.agents.utils import probability as prob
from sportagent.core.agents.utils.agent_utils import (
    get_game_context_from_state,
    get_language_instruction,
)
from sportagent.core.agents.utils.structured import invoke_structured_or_freetext


_SYSTEM = (
    "You are the Decision Manager on a sports prediction desk and the final "
    "authority. Your PRIMARY job is to predict the GAME WINNER: name which team "
    "wins and your probability that they win (>0.5), plus a confidence level and "
    "plain-language reasoning grounded in the analyst reports.\n\n"
    "{game_context}\n\n"
    "Use the EXACT team names from the game context for `predicted_winner` — "
    "never invent or abbreviate. Base the winner call on the stats, news/injury, "
    "odds, and sentiment reports plus the risk debate.\n\n"
    "SECONDARY (betting view): also produce the Kalshi call — BUY YES, BUY NO, "
    "or HOLD. Ground EVERY number in the verified-odds snapshot. The "
    "market-implied probability for the target is {implied_pct} (from the "
    "snapshot). HOLD if the edge is within the no-trade band of "
    "{no_trade_band_pct}. Incorporate the prior settled-outcome lessons if any "
    "are present below.{language}\n\n"
    "=== Verified odds snapshot ===\n{verified_odds}\n\n"
    "=== Investment plan (edge thesis) ===\n{investment_plan}\n\n"
    "=== Trader position proposal ===\n{trader_position_plan}\n\n"
    "=== Risk debate ===\n{risk_history}\n\n"
    "=== Prior settled-outcome lessons ===\n{past_context}\n"
)


def _extract_implied_prob(verified_odds: str) -> float:
    """Best-effort parse of the Kalshi implied probability from the snapshot.

    The snapshot renders ``→ implied 58.0%``. Falls back to 0.5 when absent so
    the prompt always has a numeric anchor; the deterministic value is still the
    one the renderer reports.
    """
    match = re.search(r"implied\s+([0-9]+(?:\.[0-9]+)?)\s*%", verified_odds)
    if match:
        return prob.clamp_prob(float(match.group(1)) / 100.0)
    return 0.5


def create_decision_manager(llm):
    """Create the Decision Manager node (deep tier, structured output)."""

    def decision_manager_node(state):
        from sportagent.core.dataflows.config import get_config

        config = get_config()
        no_trade_band = config.get("no_trade_band", 0.03)
        verified_odds = state.get("verified_odds", "")
        implied = _extract_implied_prob(verified_odds)

        risk_debate = state.get("risk_debate_state", {})
        system = _SYSTEM.format(
            game_context=get_game_context_from_state(state),
            language=get_language_instruction(),
            implied_pct=f"{implied * 100:.1f}%",
            no_trade_band_pct=f"{no_trade_band * 100:.1f}pp",
            verified_odds=verified_odds,
            investment_plan=state.get("investment_plan", ""),
            trader_position_plan=state.get("trader_position_plan", ""),
            risk_history=risk_debate.get("history", ""),
            past_context=state.get("past_context", "") or "(none)",
        )
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=(
                "Deliver the final recommendation. PRIMARY: predicted_winner "
                "(exact team name), win_probability (>0.5), confidence, and "
                "reasoning explaining why that team wins. SECONDARY (betting "
                "view): action, estimated and implied probabilities, edge, and "
                "recommended stake."
            )),
        ]
        markdown, _parsed = invoke_structured_or_freetext(
            llm, FinalRecommendation, messages, render_final_recommendation
        )

        new_risk = dict(risk_debate)
        new_risk["judge_decision"] = markdown
        new_risk["latest_speaker"] = "Judge"
        return {
            "final_recommendation": markdown,
            "risk_debate_state": new_risk,
        }

    return decision_manager_node