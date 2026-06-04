"""Sentiment Analyst — structured read of public/community lean.

Unlike the other analysts, this one does not tool-bind. The graph pre-fetches
Reddit posts (and public betting %) via ``get_reddit_sentiment`` and injects the
text into the prompt. The analyst emits a structured ``SentimentReport`` (with a
free-text fallback) and writes the rendered markdown to ``sentiment_report``.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from sportagent.core.agents.schemas import SentimentReport, render_sentiment_report
from sportagent.core.agents.utils.agent_utils import (
    get_game_context_from_state,
    get_language_instruction,
)
from sportagent.core.agents.utils.structured import invoke_structured_or_freetext
from sportagent.core.dataflows.interface import route_to_vendor


_SYSTEM = (
    "You are the Sentiment Analyst on a sports prediction-market desk. Your job "
    "is to gauge public and community lean toward the target team and separate "
    "signal from noise.\n\n"
    "{game_context}\n\n"
    "Read the pre-fetched Reddit / community data below. Assess: the public "
    "betting lean and any CONTRARIAN signal (heavy public money on one side can "
    "flag value on the other), the r/nba mood, and notable narratives. Be "
    "explicit about data confidence — if sources are thin or returned "
    "placeholders, say so and lower your confidence.{language}\n\n"
    "Community data:\n{sentiment_data}"
)


def create_sentiment_analyst(llm):
    """Create the Sentiment Analyst node (quick tier, structured output)."""

    def sentiment_analyst_node(state):
        target = state.get("target_team", "") or state.get("market_ticker", "")
        sentiment_data = route_to_vendor("get_reddit_sentiment", target)

        system = _SYSTEM.format(
            game_context=get_game_context_from_state(state),
            language=get_language_instruction(),
            sentiment_data=sentiment_data,
        )
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=(
                "Produce your structured sentiment report for the target team."
            )),
        ]
        markdown, _parsed = invoke_structured_or_freetext(
            llm, SentimentReport, messages, render_sentiment_report
        )
        return {"sentiment_report": markdown}

    return sentiment_analyst_node