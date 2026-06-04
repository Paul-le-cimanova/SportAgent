"""Typed state for the SportAgent LangGraph pipeline.

``GameState`` extends LangGraph's ``MessagesState`` and carries the structured
documents that pass between phases: analyst reports, the verified-odds snapshot,
the research and risk debate transcripts, the trader's position, and the final
recommendation. Natural-language dialogue is confined to the two debate
sub-states (``DebateState``, ``RiskDebateState``); everything else is a typed
field so context does not degrade across turns (see design doc 06).
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import MessagesState


class DebateState(TypedDict):
    """Bull/Bear research debate transcript (Phase II)."""

    bull_history: str          # YES-case cumulative arguments
    bear_history: str          # NO-case cumulative arguments
    history: str               # Full interleaved transcript
    current_response: str      # Latest turn (prefixed "Bull:" / "Bear:")
    judge_decision: str        # Research Manager's EdgeThesis (rendered)
    count: int                 # Debate turns so far


class RiskDebateState(TypedDict):
    """Aggressive/Conservative/Neutral risk debate transcript (Phase IV)."""

    aggressive_history: str
    conservative_history: str
    neutral_history: str
    history: str
    latest_speaker: str        # "Aggressive" / "Conservative" / "Neutral" / "Judge"
    current_aggressive_response: str
    current_conservative_response: str
    current_neutral_response: str
    judge_decision: str        # Decision Manager's final call (rendered)
    count: int


class GameState(MessagesState):
    """Top-level shared state for a single market analysis run."""

    # --- Identity -----------------------------------------------------------
    market_ticker: str         # Kalshi market (e.g., KXNBAGAME-...)
    game_context: str          # Deterministic matchup identity (anti-hallucination)
    target_team: str           # Team the YES contract resolves on (2-way)
    game_date: str             # YYYY-MM-DD
    sport: str                 # "nba" | "nfl" | "mlb" | "soccer" (drives adapter)
    outcome_structure: str     # "two_way" | "three_way"
    market_refs: list          # Kalshi contract(s): 1 for two_way, 3 for three_way

    # --- Phase I: Analyst reports ------------------------------------------
    odds_report: str
    stats_report: str
    news_report: str
    sentiment_report: str

    # --- Verified deterministic snapshot (source of truth for prices) ------
    verified_odds: str

    # --- Phase II: Research debate -----------------------------------------
    investment_debate_state: DebateState
    investment_plan: str       # EdgeThesis from Research Manager (rendered)

    # --- Phase III: Trader -------------------------------------------------
    trader_position_plan: str  # PositionProposal from Trader (rendered)

    # --- Phase IV-V: Risk debate + final -----------------------------------
    risk_debate_state: RiskDebateState
    final_recommendation: str  # FinalRecommendation from Decision Manager (rendered)

    # --- Memory ------------------------------------------------------------
    past_context: str          # Prior settled-outcome lessons injected at run start


def empty_debate_state() -> DebateState:
    """Return a zeroed ``DebateState`` for run initialization."""
    return DebateState(
        bull_history="",
        bear_history="",
        history="",
        current_response="",
        judge_decision="",
        count=0,
    )


def empty_risk_debate_state() -> RiskDebateState:
    """Return a zeroed ``RiskDebateState`` for run initialization."""
    return RiskDebateState(
        aggressive_history="",
        conservative_history="",
        neutral_history="",
        history="",
        latest_speaker="",
        current_aggressive_response="",
        current_conservative_response="",
        current_neutral_response="",
        judge_decision="",
        count=0,
    )