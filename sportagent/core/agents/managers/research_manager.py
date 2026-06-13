"""Research Manager — judges the bull/bear debate, emits an EdgeThesis.

Deep-tier agent. Reads the full research debate + all analyst reports + the
verified-odds snapshot, commits to an estimated true probability for the target
outcome, and decides whether it diverges enough from the market price to justify
a position. Emits a structured ``EdgeThesis`` (free-text fallback) rendered to
``investment_plan`` and recorded as the debate's judge decision.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from sportagent.core.agents.schemas import (
    EdgeThesis,
    ThreeWayEdgeThesis,
    render_edge_thesis,
    render_three_way_edge_thesis,
)
from sportagent.core.agents.utils.agent_utils import (
    get_game_context_from_state,
    get_language_instruction,
)
from sportagent.core.agents.utils.structured import invoke_structured_or_freetext


_SYSTEM = (
    "You are the Research Manager (judge) on a sports prediction-market desk. "
    "You weigh the bull/bear debate and the analyst evidence, then commit to an "
    "estimated TRUE probability that the target YES contract resolves.\n\n"
    "{game_context}\n\n"
    "Decide a lean: YES if your estimate materially exceeds the market price, "
    "NO if it is materially below, or NO-EDGE if there is no meaningful "
    "divergence. Ground your estimate in the verified-odds snapshot (the source "
    "of truth for the market price) and the analyst reports. Be decisive — "
    "reserve NO-EDGE for genuinely balanced cases.{language}\n\n"
    "=== Verified odds snapshot ===\n{verified_odds}\n\n"
    "=== Odds report ===\n{odds_report}\n\n"
    "=== Stats report ===\n{stats_report}\n\n"
    "=== News/Injury report ===\n{news_report}\n\n"
    "=== Sentiment report ===\n{sentiment_report}\n\n"
    "=== Research debate ===\n{history}\n"
)


def create_research_manager(llm):
    """Create the Research Manager node (deep tier, structured output)."""

    def research_manager_node(state):
        debate = state.get("investment_debate_state", {})
        three_way = state.get("outcome_structure", "two_way") == "three_way"
        system = _SYSTEM.format(
            game_context=get_game_context_from_state(state),
            language=get_language_instruction(),
            verified_odds=state.get("verified_odds", ""),
            odds_report=state.get("odds_report", ""),
            stats_report=state.get("stats_report", ""),
            news_report=state.get("news_report", ""),
            sentiment_report=state.get("sentiment_report", ""),
            history=debate.get("history", ""),
        )
        if three_way:
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=(
                    "This is a THREE-WAY soccer market. Commit to a probability "
                    "vector: prob_home, prob_draw, prob_away (the code normalizes "
                    "them to sum to 1) — never treat the draw as a leftover. Add "
                    "rationale and key factors."
                )),
            ]
            markdown, _parsed = invoke_structured_or_freetext(
                llm, ThreeWayEdgeThesis, messages, render_three_way_edge_thesis
            )
        else:
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=(
                    "Commit to your edge thesis: lean, estimated probability, "
                    "rationale, and key factors."
                )),
            ]
            markdown, _parsed = invoke_structured_or_freetext(
                llm, EdgeThesis, messages, render_edge_thesis
            )

        new_debate = dict(debate)
        new_debate["judge_decision"] = markdown
        return {
            "investment_plan": markdown,
            "investment_debate_state": new_debate,
        }

    return research_manager_node