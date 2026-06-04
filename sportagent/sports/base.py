"""SportAdapter protocol + registry.

The core pipeline is sport-agnostic. Each sport supplies a ``SportAdapter``
that handles the two things that genuinely differ by sport:

1. **Market structure** — how a matchup maps to Kalshi contract(s), and whether
   the outcome is 2-way (team A vs B) or 3-way (soccer win/draw/loss).
2. **Stats + key factors** — which stats tools to use and what matters in this
   sport (net rating / starting pitcher / QB+weather / xG).

``SportAgentGraph`` looks up the adapter for a market's sport via ``get_adapter``
and runs the identical core pipeline. Adding a sport = implement one adapter and
register it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

OutcomeStructure = Literal["two_way", "three_way"]


@dataclass
class MarketRef:
    """A resolved Kalshi market reference for a matchup.

    For 2-way sports, ``contracts`` has one entry keyed by the target team.
    For 3-way soccer, it has three entries: home / draw / away.
    """

    sport: str
    market_ticker: str                 # primary contract (2-way target)
    target_team: str                   # team the primary YES resolves on
    home_team: str
    away_team: str
    game_date: str                     # YYYY-MM-DD
    outcome_structure: OutcomeStructure = "two_way"
    # outcome label -> kalshi contract ticker (e.g. {"home": "...", "draw": "...", "away": "..."})
    contracts: Dict[str, str] = field(default_factory=dict)
    sportsbook_key: str = ""           # The Odds API sport key


@runtime_checkable
class SportAdapter(Protocol):
    """Per-sport behavior behind the sport-agnostic core."""

    sport: str

    def resolve_market(self, query: str, config: Optional[dict] = None) -> MarketRef:
        """Map a matchup/market query to a ``MarketRef`` (Kalshi contract(s))."""

    def outcome_structure(self) -> OutcomeStructure:
        """``\"two_way\"`` (team A vs B) or ``\"three_way\"`` (win/draw/loss)."""

    def stats_tools(self) -> List[str]:
        """Names of the data-tool methods the Stats Analyst should use."""

    def key_factors_prompt(self) -> str:
        """Prompt hints: what matters in this sport (for the Stats Analyst)."""

    def sportsbook_key(self) -> str:
        """The Odds API sport key, e.g. ``\"basketball_nba\"``."""

    def settle(self, market_ref: MarketRef, result: Optional[str], scores: Any) -> Optional[str]:
        """Map a settled result to an outcome label for the reflection loop.

        Returns ``\"yes\"``/``\"no\"`` for 2-way, or ``\"home\"``/``\"draw\"``/``\"away\"``
        for 3-way; ``None`` if unsettled.
        """


# --- Registry ----------------------------------------------------------------

_ADAPTERS: Dict[str, SportAdapter] = {}


def register_adapter(adapter: SportAdapter) -> None:
    """Register a sport adapter under its ``sport`` key."""
    _ADAPTERS[adapter.sport] = adapter


def get_adapter(sport: str) -> Optional[SportAdapter]:
    """Return the registered adapter for ``sport`` (e.g. ``\"nba\"``) or None."""
    return _ADAPTERS.get(sport)


def available_sports() -> List[str]:
    """List registered sport keys."""
    return sorted(_ADAPTERS.keys())