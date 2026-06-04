"""Initial-state construction for a SportAgent run.

``create_initial_state`` builds the ``GameState`` dict the LangGraph pipeline
starts from. Identity fields (market ticker, resolved game context, target team,
date, sport, outcome structure, contracts) are seeded from the resolved
``MarketRef`` / game context; all report fields start empty; the two debate
sub-states are zeroed via the ``empty_*`` helpers; and any prior settled-outcome
lessons are injected as ``past_context``.

The first message seeds the analyst message channel with the market ticker so
the first tool-bound analyst has a concrete anchor (see design doc 06 §3).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from sportagent.core.agents.utils.game_state import (
    empty_debate_state,
    empty_risk_debate_state,
)


def create_initial_state(
    market_ref: Any,
    game_context: str,
    past_context: str = "",
    *,
    verified_odds: str = "",
) -> Dict[str, Any]:
    """Build the initial ``GameState`` dict for a run.

    Args:
        market_ref: The resolved ``MarketRef`` (sport/ticker/teams/date/outcome
            structure/contracts) from the sport adapter. A mapping with the same
            keys is also accepted for flexibility.
        game_context: The deterministic matchup-identity string injected into
            every agent prompt (anti-hallucination anchor).
        past_context: Prior settled-outcome lessons to inject for the Decision
            Manager (empty string when there is no memory).
        verified_odds: Optional pre-computed verified-odds snapshot (source of
            truth for exact prices). May be filled later by the graph.

    Returns:
        A ``GameState``-shaped dict ready to pass to the compiled graph.
    """
    ref = _as_attrs(market_ref)

    market_ticker = ref.get("market_ticker", "")
    target_team = ref.get("target_team", "")
    home_team = ref.get("home_team", "")
    away_team = ref.get("away_team", "")
    game_date = ref.get("game_date", "")
    sport = ref.get("sport", "nba")
    outcome_structure = ref.get("outcome_structure", "two_way")
    contracts = ref.get("contracts", {}) or {}
    market_refs: List[str] = list(contracts.values()) if contracts else (
        [market_ticker] if market_ticker else []
    )

    return {
        # Message channel — seed with the ticker as the first human anchor.
        "messages": [("human", market_ticker or game_context)],
        # Identity
        "market_ticker": market_ticker,
        "game_context": game_context,
        "target_team": target_team,
        "home_team": home_team,
        "away_team": away_team,
        "game_date": game_date,
        "sport": sport,
        "outcome_structure": outcome_structure,
        "market_refs": market_refs,
        # Phase I analyst reports
        "odds_report": "",
        "stats_report": "",
        "news_report": "",
        "sentiment_report": "",
        # Verified deterministic snapshot (source of truth for prices)
        "verified_odds": verified_odds,
        # Phase II research debate
        "investment_debate_state": empty_debate_state(),
        "investment_plan": "",
        # Phase III trader
        "trader_position_plan": "",
        # Phase IV-V risk debate + final
        "risk_debate_state": empty_risk_debate_state(),
        "final_recommendation": "",
        # Memory
        "past_context": past_context,
    }


def _as_attrs(market_ref: Any) -> Dict[str, Any]:
    """Return a uniform dict view of a ``MarketRef`` dataclass or mapping."""
    if market_ref is None:
        return {}
    if isinstance(market_ref, dict):
        return market_ref
    keys = (
        "sport",
        "market_ticker",
        "target_team",
        "home_team",
        "away_team",
        "game_date",
        "outcome_structure",
        "contracts",
    )
    return {k: getattr(market_ref, k, None) for k in keys}