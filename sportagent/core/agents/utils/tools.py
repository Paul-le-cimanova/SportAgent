"""LangChain ``@tool`` wrappers over the data-routing interface.

Each tool delegates to ``interface.route_to_vendor(method, ...)`` so the active
vendor is resolved from config at call time. Tools are grouped to match the
analysts that bind them (odds / stats / news). The verified-odds snapshot tool
calls ``odds_validator.build_verified_odds_snapshot`` directly — it is a
deterministic anti-hallucination artifact, not a routed vendor method.

Every underlying fetcher fails open (returns a placeholder string), so these
wrappers never raise; the LLM simply sees the placeholder text.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from sportagent.core.dataflows import odds_validator
from sportagent.core.dataflows.interface import route_to_vendor


# --- Odds Analyst tools ------------------------------------------------------


@tool
def get_kalshi_price(market_ticker: str) -> str:
    """Get the Kalshi YES/NO contract price (cents) and implied probability for a market ticker."""
    return str(route_to_vendor("get_kalshi_price", market_ticker))


@tool
def get_moneyline(sport_key: str, home_team: str, away_team: str) -> str:
    """Get the de-vigged sportsbook-consensus moneyline probabilities for a matchup.

    sport_key is The Odds API key (e.g. 'basketball_nba').
    """
    return str(route_to_vendor("get_moneyline", sport_key, home_team, away_team))


@tool
def get_line_movement(sport_key: str, home_team: str, away_team: str) -> str:
    """Get sportsbook line-movement context for a matchup (current consensus on the free tier)."""
    return str(route_to_vendor("get_line_movement", sport_key, home_team, away_team))


@tool
def get_verified_odds_snapshot(
    market_ticker: str,
    target_team: str,
    home_team: str,
    away_team: str,
    sport_key: str,
) -> str:
    """Get the deterministic verified-odds snapshot (Kalshi vs sportsbook consensus).

    This is the SOURCE OF TRUTH for exact prices and implied probabilities. If
    another tool conflicts with it, flag the discrepancy rather than inventing a
    reconciled number.
    """
    return odds_validator.build_verified_odds_snapshot(
        market_ticker=market_ticker,
        target_team=target_team,
        home_team=home_team,
        away_team=away_team,
        sport_key=sport_key,
    )


# --- Stats Analyst tools -----------------------------------------------------


@tool
def get_team_stats(team: str, season: Optional[int] = None) -> str:
    """Get a team's season record and standing (net record, conference rank)."""
    return str(route_to_vendor("get_team_stats", team, season))


@tool
def get_recent_form(team: str, n_games: int = 10, season: Optional[int] = None) -> str:
    """Get a team's last-N game results (form heading into the matchup)."""
    return str(route_to_vendor("get_recent_form", team, n_games, season))


@tool
def get_h2h(team_a: str, team_b: str, season: Optional[int] = None) -> str:
    """Get head-to-head results between two teams this season."""
    return str(route_to_vendor("get_h2h", team_a, team_b, season))


@tool
def get_standings(season: Optional[int] = None) -> str:
    """Get the league standings summary."""
    return str(route_to_vendor("get_standings", season))


@tool
def get_rest_status(team: str, game_date: str) -> str:
    """Get a team's rest / back-to-back status heading into game_date (YYYY-MM-DD)."""
    return str(route_to_vendor("get_rest_status", team, game_date))


# --- News/Injury Analyst tools -----------------------------------------------


@tool
def get_injury_news(team: str, date: Optional[str] = None) -> str:
    """Get recent injury / availability news for a team (out/questionable/probable)."""
    return str(route_to_vendor("get_injury_news", team, date))


@tool
def get_lineup_news(team: str, date: Optional[str] = None) -> str:
    """Get recent starting-lineup / rotation news for a team."""
    return str(route_to_vendor("get_lineup_news", team, date))


# --- Tool groups (bound per analyst) -----------------------------------------

ODDS_TOOLS = [get_kalshi_price, get_moneyline, get_line_movement, get_verified_odds_snapshot]
STATS_TOOLS = [get_team_stats, get_recent_form, get_h2h, get_standings, get_rest_status]
NEWS_TOOLS = [get_injury_news, get_lineup_news]

# Lookup by tool method name, for adapter-driven stats-tool selection.
STATS_TOOL_BY_NAME = {
    "get_team_stats": get_team_stats,
    "get_recent_form": get_recent_form,
    "get_h2h": get_h2h,
    "get_standings": get_standings,
    "get_rest_status": get_rest_status,
}