"""NBA SportAdapter — 2-way game-winner markets.

Resolves a matchup query to a Kalshi NBA game-winner contract, supplies the
Stats Analyst's NBA tools + key-factor hints, and maps settlement to yes/no.

Importing this module also imports ``sports.nba.stats`` so its data tools
self-register with the routing interface, and registers the adapter itself.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

from sportagent.core.dataflows import kalshi
from sportagent.sports.base import MarketRef, OutcomeStructure, register_adapter

# Import stats module for its side-effect of registering its data tools.
from sportagent.sports.nba import stats as _nba_stats  # noqa: F401

logger = logging.getLogger(__name__)

_SPORTSBOOK_KEY = "basketball_nba"
# Kalshi NBA game-winner series ticker (discovered/confirmed via /events). Kalshi
# game-winner markets live under the ``KXNBAGAME`` series; ``KXNBA`` is the
# season-long Finals series (single-team), so it is NOT used for matchup scans.
_NBA_SERIES_HINTS = ("KXNBAGAME",)

# Kalshi market titles use city/full names ("San Antonio", "New York"); user
# queries often use nicknames ("Spurs", "Knicks"). Map nickname -> the tokens
# that may appear in a Kalshi title so the matchup scan can match either form.
_NBA_TEAM_ALIASES = {
    "hawks": ["atlanta", "hawks"],
    "celtics": ["boston", "celtics"],
    "nets": ["brooklyn", "nets"],
    "hornets": ["charlotte", "hornets"],
    "bulls": ["chicago", "bulls"],
    "cavaliers": ["cleveland", "cavaliers", "cavs"],
    "mavericks": ["dallas", "mavericks", "mavs"],
    "nuggets": ["denver", "nuggets"],
    "pistons": ["detroit", "pistons"],
    "warriors": ["golden state", "warriors"],
    "rockets": ["houston", "rockets"],
    "pacers": ["indiana", "pacers"],
    "clippers": ["la clippers", "los angeles clippers", "clippers"],
    "lakers": ["la lakers", "los angeles lakers", "lakers"],
    "grizzlies": ["memphis", "grizzlies"],
    "heat": ["miami", "heat"],
    "bucks": ["milwaukee", "bucks"],
    "timberwolves": ["minnesota", "timberwolves", "wolves"],
    "pelicans": ["new orleans", "pelicans"],
    "knicks": ["new york", "knicks"],
    "thunder": ["oklahoma city", "thunder", "okc"],
    "magic": ["orlando", "magic"],
    "76ers": ["philadelphia", "76ers", "sixers"],
    "suns": ["phoenix", "suns"],
    "trail blazers": ["portland", "trail blazers", "blazers"],
    "kings": ["sacramento", "kings"],
    "spurs": ["san antonio", "spurs"],
    "raptors": ["toronto", "raptors"],
    "jazz": ["utah", "jazz"],
    "wizards": ["washington", "wizards"],
}


def _team_aliases(name: str) -> List[str]:
    """Return the lowercased title tokens that may identify ``name``.

    Matches on nickname, city, or any alias; falls back to the raw lowercased
    name so unknown inputs still attempt a literal match.
    """
    n = name.strip().lower()
    if not n:
        return []
    # Direct nickname hit.
    if n in _NBA_TEAM_ALIASES:
        return _NBA_TEAM_ALIASES[n]
    # Reverse: the query is a city/alias that appears in some entry.
    for aliases in _NBA_TEAM_ALIASES.values():
        if any(n == a or n in a or a in n for a in aliases):
            return aliases
    return [n]


class NBAAdapter:
    """SportAdapter implementation for NBA game-winner markets."""

    sport = "nba"

    def outcome_structure(self) -> OutcomeStructure:
        return "two_way"

    def sportsbook_key(self) -> str:
        return _SPORTSBOOK_KEY

    def stats_tools(self) -> List[str]:
        return [
            "get_team_stats",
            "get_recent_form",
            "get_h2h",
            "get_rest_status",
            "get_four_factors",
            "get_elo_winprob",
        ]

    def key_factors_prompt(self) -> str:
        return (
            "For NBA, weigh: net rating (offensive/defensive efficiency), pace, "
            "recent form (last 10), head-to-head this season, rest vs "
            "back-to-back (B2B teams underperform), home/away splits, and key "
            "injuries/availability. A B2B or a missing star is often the single "
            "biggest edge the market underprices.\n\n"
            "ALWAYS call get_elo_winprob(home_team, away_team) first to get a "
            "REAL computed Elo win probability — anchor your estimate to it "
            "rather than inventing a model. Then call get_four_factors for BOTH "
            "teams: compare eFG%, turnover rate, offensive-rebound rate, and "
            "free-throw rate. Turnover rate and free-throw rate decide close "
            "games — a team that protects the ball and gets to the line wins "
            "tight playoff games even when overall talent is even. Cite the Elo "
            "number and the Four-Factors gaps explicitly; do NOT claim to have "
            "run any model you did not call as a tool. Finally, remember that a "
            "single NBA game is high-variance: even a clear Four-Factors edge "
            "only nudges a near-coin-flip, so keep single-game probabilities "
            "honest (rarely above ~65% without a major injury)."
        )

    def resolve_market(
        self,
        query: str,
        config: Optional[dict] = None,
        game_date: Optional[str] = None,
    ) -> MarketRef:
        """Resolve a matchup query to an NBA game-winner ``MarketRef``.

        ``query`` may be a Kalshi market ticker directly (preferred, exact) or a
        free-text matchup like ``\"Spurs @ Knicks\"``. When given a ticker, we
        fetch the market to confirm and pull team/date metadata where available.

        ``game_date`` (YYYY-MM-DD, from the wizard's schedule selection) is used
        to (a) prefer the Kalshi market for that exact game date and (b) stamp
        the resolved ``MarketRef.game_date`` with the date the user actually
        picked rather than the market's settlement ``close_time``.
        """
        # Direct ticker path: looks like an uppercase Kalshi ticker.
        if re.match(r"^[A-Z0-9][A-Z0-9\-]+$", query) and any(
            h in query.upper() for h in _NBA_SERIES_HINTS
        ):
            return self._from_ticker(query, config, game_date)

        # Free-text matchup path: try to discover the market via /markets.
        return self._from_matchup(query, config, game_date)

    # --- helpers ---

    def _from_ticker(
        self, ticker: str, config: Optional[dict], game_date: Optional[str] = None
    ) -> MarketRef:
        market = kalshi.get_market(ticker, config)
        home = away = target = ""
        resolved_date = ""
        if "error" not in market:
            m = market.get("market", market)
            # Kalshi market dicts vary; pull what we can, leave blanks otherwise.
            target = m.get("yes_sub_title") or m.get("title") or ""
            resolved_date = (m.get("close_time") or m.get("expiration_time") or "")[:10]
        # Prefer the caller-supplied (wizard-selected) game date when present.
        return MarketRef(
            sport=self.sport,
            market_ticker=ticker,
            target_team=target,
            home_team=home,
            away_team=away,
            game_date=game_date or resolved_date,
            outcome_structure="two_way",
            contracts={"yes": ticker},
            sportsbook_key=_SPORTSBOOK_KEY,
        )

    def _from_matchup(
        self, query: str, config: Optional[dict], game_date: Optional[str] = None
    ) -> MarketRef:
        # Parse "AWAY @ HOME" or "A vs B".
        away, home = _parse_matchup(query)
        home_aliases = _team_aliases(home)
        away_aliases = _team_aliases(away)

        # Best-effort discovery: scan open NBA game-winner markets for one
        # mentioning both teams (nickname or city). When multiple games match
        # (e.g. a series Game 1/2/3), prefer the one whose game date matches the
        # wizard-selected ``game_date``; otherwise pick the one closing soonest.
        # The target (YES side) is set to the HOME team's contract.
        ticker = ""
        target = home or query
        resolved_date = ""
        best_close = None
        date_matched = False
        for series in _NBA_SERIES_HINTS:
            resp = kalshi.get_markets(
                series_ticker=series, status="open", limit=200, config=config
            )
            if "error" in resp:
                continue
            for m in resp.get("markets", []):
                title = f"{m.get('title', '')} {m.get('yes_sub_title', '')}".lower()
                home_hit = not home_aliases or any(a in title for a in home_aliases)
                away_hit = not away_aliases or any(a in title for a in away_aliases)
                if not (home and away and home_hit and away_hit):
                    continue
                # Prefer the contract whose YES side is the home team.
                sub = (m.get("yes_sub_title") or "").lower()
                is_home_contract = any(a in sub for a in home_aliases) if home_aliases else True
                if not is_home_contract:
                    continue
                close = m.get("close_time") or m.get("expiration_time") or ""
                # A market matches the selected date if that date appears in the
                # ticker (Kalshi encodes it, e.g. ...-26JUN05...) or its rules.
                this_ticker = m.get("ticker", "")
                matches_date = bool(game_date) and _ticker_matches_date(
                    this_ticker, m, game_date
                )
                if game_date and date_matched and not matches_date:
                    # Already locked onto a date-matched market; ignore others.
                    continue
                take = False
                if matches_date and not date_matched:
                    take = True  # first date-exact match wins over soonest-close
                elif matches_date and date_matched:
                    take = best_close is None or (close and close < best_close)
                elif not date_matched:
                    take = best_close is None or (close and close < best_close)
                if take:
                    if matches_date:
                        date_matched = True
                    best_close = close
                    ticker = this_ticker
                    target = m.get("yes_sub_title") or home
                    resolved_date = close[:10]
        return MarketRef(
            sport=self.sport,
            market_ticker=ticker or f"<unresolved:{query}>",
            target_team=target,
            home_team=home,
            away_team=away,
            # Stamp the wizard-selected date when given (it's the game the user
            # actually picked); fall back to the market close date otherwise.
            game_date=game_date or resolved_date,
            outcome_structure="two_way",
            contracts={"yes": ticker} if ticker else {},
            sportsbook_key=_SPORTSBOOK_KEY,
        )

    def settle(
        self, market_ref: MarketRef, result: Optional[str], scores: Any
    ) -> Optional[str]:
        """Map Kalshi ``result`` (yes/no) to the settled outcome label."""
        if result in ("yes", "no"):
            return result
        return None


def _ticker_matches_date(ticker: str, market: dict, game_date: str) -> bool:
    """True if a Kalshi market corresponds to ``game_date`` (YYYY-MM-DD).

    Kalshi NBA game tickers encode the originally-scheduled date as ``YYMMMDD``
    (e.g. ``KXNBAGAME-26JUN05NYKSAS-SAS`` → 2026-06-05). We also check the
    market's ``rules_primary`` text, which names the scheduled date.
    """
    try:
        import datetime as _dt

        dt = _dt.datetime.strptime(game_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return False
    code = dt.strftime("%y%b%d").upper()  # e.g. 26JUN05
    if code in (ticker or "").upper():
        return True
    rules = (market.get("rules_primary") or "")
    # Rules read e.g. "... originally scheduled for Jun 5, 2026 ...".
    human = dt.strftime("%b %-d, %Y") if hasattr(dt, "strftime") else ""
    return bool(human) and human in rules


def _parse_matchup(query: str) -> tuple[str, str]:
    """Parse '\"Away @ Home\"' or '\"A vs B\"' into (away, home).

    For '@', the team after '@' is home. For 'vs', order is (first, second)
    treated as (away, home) by convention.
    """
    q = query.strip()
    if "@" in q:
        away, home = q.split("@", 1)
        return away.strip(), home.strip()
    for sep in (" vs ", " v ", " - ", "vs"):
        if sep in q:
            a, b = q.split(sep, 1)
            return a.strip(), b.strip()
    return "", q


# Instantiate + register the adapter at import.
register_adapter(NBAAdapter())