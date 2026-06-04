"""Shared agent helpers: game-identity context, msg-clear, language instruction.

These small utilities are used across every agent factory:

- ``get_game_context_from_state`` injects the deterministic matchup identity
  (anti-hallucination) into each agent prompt.
- ``create_msg_delete`` returns the Msg-Clear node that wipes accumulated
  messages between analysts and inserts a context-anchored placeholder.
- ``get_language_instruction`` localizes output when a non-English language is
  configured (empty string for English, so no extra tokens are spent).
"""

from __future__ import annotations

from typing import Any, Mapping

from langchain_core.messages import HumanMessage, RemoveMessage


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Empty string for English (default), so no extra tokens are used. Applied to
    every agent whose output reaches the saved report.
    """
    from sportagent.core.dataflows.config import get_config

    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_game_context(
    market_ticker: str,
    target_team: str,
    home_team: str,
    away_team: str,
    game_date: str,
) -> str:
    """Render the deterministic game-identity context string.

    Used at run start (when the matchup is resolved) and as a fallback when a
    state lacks a precomputed ``game_context``. Mirrors the doc-06 example so
    every agent anchors to the exact teams and never substitutes.
    """
    matchup = ""
    if away_team and home_team:
        matchup = f"{away_team} @ {home_team}"
    elif target_team:
        matchup = target_team
    date_part = f", {game_date}" if game_date else ""
    target_part = (
        f" YES resolves on {target_team}." if target_team else ""
    )
    return (
        f"Market `{market_ticker}`: {matchup}{date_part}.{target_part} "
        "Use these exact teams; do not substitute or invent a different matchup."
    )


def get_game_context_from_state(state: Mapping[str, Any]) -> str:
    """Return the game-identity context for the current run.

    Prefers the precomputed ``game_context`` stored at run start; falls back to
    building it from the identity fields on the state (no network lookup), so a
    consumer is never forced to resolve identity mid-graph.
    """
    context = state.get("game_context")
    if isinstance(context, str) and context.strip():
        return context
    return build_game_context(
        str(state.get("market_ticker", "")),
        str(state.get("target_team", "")),
        str(state.get("home_team", "")),
        str(state.get("away_team", "")),
        str(state.get("game_date", "")),
    )


def create_msg_delete():
    """Return a Msg-Clear node that wipes messages + adds a context placeholder.

    The placeholder must not be a bare ``"Continue"``: some OpenAI-compatible
    providers interpret that literally. Anchoring it to the resolved game
    context and date keeps the next analyst on-task even if the provider treats
    the placeholder as a standalone request.
    """

    def delete_messages(state):
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        game_context = get_game_context_from_state(state)
        game_date = state.get("game_date", "the scheduled date")
        placeholder = HumanMessage(
            content=(
                f"Proceed with your assigned analysis for this workflow. "
                f"{game_context} The game is on {game_date}."
            )
        )
        return {"messages": removal_operations + [placeholder]}

    return delete_messages