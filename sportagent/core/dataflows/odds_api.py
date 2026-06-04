"""The Odds API client — sportsbook moneylines for cross-checking Kalshi.

Free tier (~500 req/mo) at the-odds-api.com. Parameterized by sport key so the
same client serves NBA/NFL/MLB/soccer:

    basketball_nba · americanfootball_nfl · baseball_mlb · soccer_* (e.g.
    soccer_fifa_world_cup)

We fetch h2h (moneyline) odds, convert to implied probabilities, and remove vig
(via probability.devig / devig_multi) to produce a sportsbook-consensus
probability the Odds Analyst compares against Kalshi.

Fails open: returns a placeholder string on any error. Registers its tools with
the routing interface at import.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from sportagent.core.agents.utils import probability as prob
from sportagent.core.dataflows.interface import register_vendor_method

logger = logging.getLogger(__name__)

_BASE = "https://api.the-odds-api.com/v4"
_TIMEOUT = 10.0


def _api_key() -> Optional[str]:
    return os.environ.get("THE_ODDS_API_KEY")


def _fetch_h2h(sport_key: str) -> Any:
    """Fetch raw h2h odds for a sport key, or an error string."""
    key = _api_key()
    if not key:
        return "<odds api key missing: set THE_ODDS_API_KEY>"
    url = f"{_BASE}/sports/{sport_key}/odds"
    params = {"apiKey": key, "regions": "us", "markets": "h2h", "oddsFormat": "american"}
    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 — fail open
        logger.warning("The Odds API fetch failed for %s: %s", sport_key, exc)
        return f"<sportsbook odds unavailable: {type(exc).__name__}>"


def _match_event(events: List[dict], home_team: str, away_team: str) -> Optional[dict]:
    """Find the event matching the given teams (case-insensitive substring)."""
    h, a = home_team.lower(), away_team.lower()
    for ev in events:
        home = (ev.get("home_team") or "").lower()
        away = (ev.get("away_team") or "").lower()
        names = f"{home} {away}"
        if (h in names or any(t in home or t in away for t in h.split())) and (
            a in names or any(t in home or t in away for t in a.split())
        ):
            return ev
    return None


def _consensus_h2h_probs(event: dict) -> Dict[str, float]:
    """Average de-vigged h2h probabilities across books for an event.

    Returns ``{team_name: prob}``. Two-way and three-way (with 'Draw') both
    supported. Empty dict if no usable odds.
    """
    # Collect per-book outcome→american-odds, convert to raw probs, de-vig,
    # then average across books.
    per_outcome: Dict[str, List[float]] = {}
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = market.get("outcomes", [])
            raw = []
            names = []
            for o in outcomes:
                price = o.get("price")
                name = o.get("name")
                if price is None or name is None:
                    continue
                names.append(name)
                raw.append(prob.american_to_prob(price))
            if len(raw) < 2:
                continue
            devigged = prob.devig_multi(raw)
            for name, p in zip(names, devigged):
                per_outcome.setdefault(name, []).append(p)

    return {name: sum(ps) / len(ps) for name, ps in per_outcome.items() if ps}


# --- Registered tool implementations -----------------------------------------


def get_moneyline(
    sport_key: str,
    home_team: str,
    away_team: str,
) -> str:
    """Sportsbook-consensus moneyline probabilities for a matchup.

    Returns a formatted string with de-vigged consensus implied probabilities
    per outcome (and the draw, for soccer), or a placeholder on failure.
    """
    events = _fetch_h2h(sport_key)
    if isinstance(events, str):  # error placeholder
        return events
    if not isinstance(events, list) or not events:
        return f"<no sportsbook odds returned for {sport_key}>"

    event = _match_event(events, home_team, away_team)
    if event is None:
        return f"<no sportsbook match found for {away_team} @ {home_team} in {sport_key}>"

    probs = _consensus_h2h_probs(event)
    if not probs:
        return f"<no usable h2h odds for {away_team} @ {home_team}>"

    n_books = len(event.get("bookmakers", []))
    lines = [
        f"Sportsbook consensus (h2h, vig-removed, {n_books} books) — "
        f"{away_team} @ {home_team}:"
    ]
    for name, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name}: {p * 100:.1f}%")
    return "\n".join(lines)


def get_line_movement(
    sport_key: str,
    home_team: str,
    away_team: str,
) -> str:
    """Line-movement summary.

    The free tier does not expose historical snapshots, so v1 returns the
    current consensus with a note. (A paid tier or periodic snapshotting would
    populate true movement; documented as a known limitation.)
    """
    current = get_moneyline(sport_key, home_team, away_team)
    if current.startswith("<"):
        return current
    return (
        "Line movement: historical snapshots unavailable on the free tier; "
        "showing current consensus only.\n" + current
    )


# Register with the routing interface (default vendor key: "odds_api").
register_vendor_method("get_moneyline", "odds_api", get_moneyline)
register_vendor_method("get_line_movement", "odds_api", get_line_movement)