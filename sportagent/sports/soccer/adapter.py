"""Soccer SportAdapter — 3-way (win/draw/loss) match markets.

Resolves a soccer matchup query to a set of Kalshi match-winner contracts
(home / draw / away), supplies the Stats Analyst's soccer tools + key-factor
hints, and maps settlement to home/draw/away. World Cup advancement/futures
markets are single YES/NO (the 2-way path) with a longer settlement horizon.

Importing this module also imports ``sports.soccer.stats`` so its data tools
self-register with the routing interface, and registers the adapter itself.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Iterator, List, Optional

from sportagent.core.dataflows import kalshi
from sportagent.sports.base import MarketRef, OutcomeStructure, register_adapter

# Import stats module for its side-effect of registering its data tools.
from sportagent.sports.soccer import stats as _soccer_stats  # noqa: F401

logger = logging.getLogger(__name__)

# Default sportsbook key (overridden per competition via resolve_market's
# config["competition"]). The Odds API uses soccer_* keys.
_DEFAULT_SPORTSBOOK_KEY = "soccer_epl"

# Competition name/key → The Odds API sportsbook key.
_COMPETITION_SPORTSBOOK = {
    "epl": "soccer_epl",
    "premier league": "soccer_epl",
    "champions league": "soccer_uefa_champs_league",
    "world cup": "soccer_fifa_world_cup",
    "la liga": "soccer_spain_la_liga",
    "serie a": "soccer_italy_serie_a",
    "bundesliga": "soccer_germany_bundesliga",
    "ligue 1": "soccer_france_ligue_one",
}

# Kalshi soccer match-winner series hints (the win/draw/loss contracts live
# under per-competition series). Verified live against the Kalshi API:
#   - ``KXWCGAME``  : FIFA World Cup match-winner (3-way: TeamA / TeamB / Tie),
#                     events like ``KXWCGAME-26JUN27COLPOR`` with contracts
#                     ``-COL`` / ``-POR`` / ``-TIE``. This is the primary v0.4.0
#                     target. The draw contract's ``yes_sub_title`` is "Tie".
# (``KXUCL``/``KXLALIGA`` exist but are season-winner FUTURES, not 3-way match
# markets — those route through the 2-way ``advancement``/``futures`` path.)
_SOCCER_SERIES_HINTS = ("KXWCGAME",)

# Series that are season-winner FUTURES (single YES/NO per team), not 3-way
# match markets — e.g. ``KXUCL-27`` ("… win the Champions League?"),
# ``KXLALIGA-27`` ("… La Liga Champion").
_SOCCER_FUTURES_SERIES = ("KXUCL", "KXLALIGA", "KXEPL", "KXSERIEA", "KXBUNDESLIGA")

# Market types: match (3-way), advancement / futures (2-way YES/NO).
_MARKET_TYPES = ("match", "advancement", "futures")


def _sportsbook_key_for(competition: Optional[str]) -> str:
    """Resolve a competition string to a The Odds API soccer key."""
    if not competition:
        return _DEFAULT_SPORTSBOOK_KEY
    key = competition.strip().lower()
    if key.startswith("soccer_"):
        return key
    for name, sb_key in _COMPETITION_SPORTSBOOK.items():
        if name in key:
            return sb_key
    return _DEFAULT_SPORTSBOOK_KEY


class SoccerAdapter:
    """SportAdapter implementation for soccer 3-way match markets."""

    sport = "soccer"

    def outcome_structure(self) -> OutcomeStructure:
        # Match markets are 3-way. Advancement/futures markets are 2-way; the
        # market_type is read from the resolved MarketRef when present, but the
        # adapter's default structure is three_way (the primary v-soccer target).
        return "three_way"

    def sportsbook_key(self) -> str:
        return _DEFAULT_SPORTSBOOK_KEY

    def stats_tools(self) -> List[str]:
        return [
            "get_team_xg",
            "get_league_table",
            "get_soccer_recent_form",
            "get_soccer_h2h",
        ]

    def key_factors_prompt(self) -> str:
        return (
            "For SOCCER, this is a THREE-WAY market: home win, DRAW, and away "
            "win. You must produce three probabilities that sum to 1 — never "
            "forget the draw, which the market frequently misprices.\n\n"
            "ALWAYS call get_team_xg(team) FIRST for BOTH sides — it returns a "
            "REAL aggregated xG-for / xG-against per game (the single best "
            "predictor of underlying quality, more stable than raw results). "
            "Anchor your three probabilities to the net-xG gap rather than "
            "inventing an xG model. Then weigh: recent form (last 5, W/D/L — "
            "call get_soccer_recent_form for both sides), the group standings "
            "(get_league_table), "
            "home/away splits (home advantage is real but smaller than in the "
            "US sports), head-to-head history (get_soccer_h2h), fixture "
            "congestion / rest (a midweek European tie before a league game "
            "saps legs), and key absences (suspensions from accumulated cards "
            "or a red card last match, plus injuries to creative or defensive "
            "leaders). Tight, evenly-matched fixtures and low-scoring sides "
            "raise the draw probability — a cagey 0-0 or 1-1 is a genuine, "
            "well-priced outcome, not a residual. Cite the form and table "
            "numbers explicitly; do NOT invent an xG model you did not call. "
            "Keep single-match probabilities honest: soccer is low-scoring and "
            "high-variance, so even a clear favorite rarely exceeds ~65% to win "
            "outright, and the draw is commonly 22-30%."
        )

    def resolve_market(
        self,
        query: str,
        config: Optional[dict] = None,
        game_date: Optional[str] = None,
    ) -> MarketRef:
        """Resolve a matchup query to a soccer ``MarketRef``.

        ``query`` may be a Kalshi market/event ticker directly (preferred,
        exact) or a free-text matchup like ``"Arsenal vs Chelsea"``. The
        competition (for the sportsbook key) and ``market_type`` (match /
        advancement / futures) are read from ``config`` when present.
        """
        cfg = config or {}
        competition = cfg.get("competition") or cfg.get("soccer_competition")
        market_type = (cfg.get("market_type") or "match").strip().lower()
        if market_type not in _MARKET_TYPES:
            market_type = "match"
        sportsbook_key = _sportsbook_key_for(competition)

        # Advancement/futures markets are single YES/NO (2-way path).
        if market_type in ("advancement", "futures"):
            return self._resolve_two_way(
                query, cfg, game_date, sportsbook_key, market_type
            )

        # Direct ticker/event path.
        if re.match(r"^[A-Z0-9][A-Z0-9\-]+$", query) and any(
            h in query.upper() for h in _SOCCER_SERIES_HINTS
        ):
            return self._from_event(query, cfg, game_date, sportsbook_key)

        return self._from_matchup(query, cfg, game_date, sportsbook_key)

    # --- helpers ---

    def _from_event(
        self,
        event_ticker: str,
        config: dict,
        game_date: Optional[str],
        sportsbook_key: str,
    ) -> MarketRef:
        """Resolve the three match contracts from a Kalshi event ticker.

        Kalshi World Cup match events (``KXWCGAME-…``) carry three contracts —
        e.g. ``-COL`` / ``-POR`` / ``-TIE`` for "Colombia vs Portugal Winner?".
        The draw contract's ``yes_sub_title`` is "Tie"; each team contract's
        ``yes_sub_title`` names its team. We first derive the two team names
        from the event title so ``_classify_contract`` can label home/away, then
        fall back to title order for any unlabeled team contract.
        """
        resp = kalshi.get_markets(event_ticker=event_ticker, config=config)
        contracts: dict[str, str] = {}
        home = away = ""
        resolved_date = ""
        markets = resp.get("markets", []) if "error" not in resp else []

        # First pass: derive the two team names from a market title ("A vs B").
        for m in markets:
            home, away = self._teams_from_market(m, home, away)
            if home and away:
                break
        home_l, away_l = home.lower(), away.lower()

        # Second pass: classify each contract using the resolved team names.
        unlabeled: list[str] = []
        for m in markets:
            ticker = m.get("ticker", "")
            label = self._classify_contract(m, home_l, away_l)
            if label and label not in contracts:
                contracts[label] = ticker
            elif label is None and ticker:
                # A team contract we couldn't name-match (e.g. abbreviated
                # sub-title) — hold for positional fallback below.
                unlabeled.append(ticker)
            resolved_date = (
                m.get("close_time") or m.get("expiration_time") or resolved_date
            )[:10]

        # Positional fallback: assign any unlabeled team contracts to the open
        # home/away slots in the title's order (first-named team is "home").
        for ticker in unlabeled:
            if "home" not in contracts:
                contracts["home"] = ticker
            elif "away" not in contracts:
                contracts["away"] = ticker

        primary = contracts.get("home") or (
            next(iter(contracts.values())) if contracts else event_ticker
        )
        return MarketRef(
            sport=self.sport,
            market_ticker=primary,
            target_team=home,
            home_team=home,
            away_team=away,
            game_date=game_date or resolved_date,
            outcome_structure="three_way",
            contracts=contracts,
            sportsbook_key=sportsbook_key,
        )

    def _from_matchup(
        self,
        query: str,
        config: dict,
        game_date: Optional[str],
        sportsbook_key: str,
    ) -> MarketRef:
        """Discover the three match contracts by scanning soccer markets."""
        away, home = _parse_matchup(query)
        home_l = home.lower()
        away_l = away.lower()

        # Collect the matching fixture's contracts, grouped by event ticker, so
        # we never mix two different editions of the same pairing. Scan ALL
        # statuses (not just open) so a finalized/closed game — e.g. a past
        # World Cup match used for settlement/backtests — still resolves. Both
        # team names must appear in the title to belong to this fixture; names
        # are matched NORMALIZED so "Bosnia-Herzegovina" matches Kalshi's
        # "Bosnia and Herzegovina".
        events: dict[str, list[dict]] = {}
        if home and away:
            for series in _SOCCER_SERIES_HINTS:
                for m in _iter_series_markets(series, config):
                    title = f"{m.get('title', '')} {m.get('yes_sub_title', '')}"
                    if not (_name_matches(home, title) and _name_matches(away, title)):
                        continue
                    evt = m.get("event_ticker") or _event_ticker_of(m.get("ticker", ""))
                    events.setdefault(evt, []).append(m)

        chosen = _pick_event(events, game_date)
        contracts: dict[str, str] = {}
        resolved_date = ""
        for m in chosen:
            label = self._classify_contract(m, home_l, away_l)
            if label and label not in contracts:
                contracts[label] = m.get("ticker", "")
                resolved_date = (
                    m.get("close_time") or m.get("expiration_time") or resolved_date
                )[:10]

        primary = contracts.get("home") or (
            next(iter(contracts.values())) if contracts else f"<unresolved:{query}>"
        )
        return MarketRef(
            sport=self.sport,
            market_ticker=primary,
            target_team=home,
            home_team=home,
            away_team=away,
            game_date=game_date or resolved_date,
            outcome_structure="three_way",
            contracts=contracts,
            sportsbook_key=sportsbook_key,
        )

    def _resolve_two_way(
        self,
        query: str,
        config: dict,
        game_date: Optional[str],
        sportsbook_key: str,
        market_type: str,
    ) -> MarketRef:
        """Resolve an advancement/futures market as a single YES/NO contract."""
        ticker = query
        target = query
        resolved_date = ""
        if re.match(r"^[A-Z0-9][A-Z0-9\-]+$", query):
            market = kalshi.get_market(query, config)
            if "error" not in market:
                m = market.get("market", market)
                target = m.get("yes_sub_title") or m.get("title") or query
                resolved_date = (
                    m.get("close_time") or m.get("expiration_time") or ""
                )[:10]
        return MarketRef(
            sport=self.sport,
            market_ticker=ticker,
            target_team=target,
            home_team="",
            away_team="",
            game_date=game_date or resolved_date,
            outcome_structure="two_way",
            contracts={"yes": ticker},
            sportsbook_key=sportsbook_key,
        )

    @staticmethod
    def _classify_contract(
        market: dict, home_l: str = "", away_l: str = ""
    ) -> Optional[str]:
        """Classify a Kalshi contract as home / draw / away.

        A soccer match event exposes three contracts; the draw is identified by
        a 'draw'/'tie' token, otherwise the contract's YES side names a team.
        """
        sub = market.get("yes_sub_title") or market.get("title") or ""
        sub_l = sub.lower()
        if "draw" in sub_l or "tie" in sub_l:
            return "draw"
        if home_l and _name_matches(home_l, sub):
            return "home"
        if away_l and _name_matches(away_l, sub):
            return "away"
        # Without a team-name match, fall back to positional assignment upstream.
        return None

    @staticmethod
    def _teams_from_market(market: dict, home: str, away: str) -> tuple[str, str]:
        """Best-effort extraction of home/away team names from a market dict.

        Kalshi World Cup match titles read "Colombia vs Portugal Winner?" — the
        trailing "Winner?"/"Winner" suffix (and any "?" punctuation) is stripped
        so the team name isn't polluted (e.g. "Portugal Winner?" → "Portugal").
        """
        if home and away:
            return home, away
        title = market.get("title", "") or ""
        # Strip the Kalshi match-winner question suffix, e.g. "… Winner?".
        for suffix in (" Winner?", " Winner", " winner?", " winner"):
            if title.endswith(suffix):
                title = title[: -len(suffix)]
                break
        title = title.rstrip(" ?").strip()
        # Common Kalshi soccer titles read "Home vs Away" / "Away @ Home".
        for sep in (" vs ", " v ", " @ ", " - "):
            if sep in title:
                a, b = title.split(sep, 1)
                if sep == " @ ":
                    return b.strip(" ?").strip(), a.strip(" ?").strip()  # "Away @ Home"
                return a.strip(" ?").strip(), b.strip(" ?").strip()
        return home, away

    def settle(
        self, market_ref: MarketRef, result: Optional[str], scores: Any
    ) -> Optional[str]:
        """Map a settled result to home/draw/away (3-way) or yes/no (2-way).

        For a 3-way match, the realized outcome is derived from ``scores`` when
        present (``{"home": int, "away": int}``); otherwise the Kalshi YES/NO
        result on the home contract is mapped (yes→home, no→away, with the draw
        only inferable from scores).
        """
        if getattr(market_ref, "outcome_structure", "two_way") != "three_way":
            return result if result in ("yes", "no") else None

        # Prefer the actual scoreline when available (only way to see a draw).
        if isinstance(scores, dict):
            hg = scores.get("home")
            ag = scores.get("away")
            if isinstance(hg, int) and isinstance(ag, int):
                if hg > ag:
                    return "home"
                if hg < ag:
                    return "away"
                return "draw"

        # Fallback: map the home contract's settled result.
        if result == "yes":
            return "home"
        if result == "no":
            return "away"
        return None


def _normalize_team(name: str) -> str:
    """Normalize a team/nation name for tolerant matching.

    Lowercases, strips accents, and collapses separators (``-``, ``&``, ``.``,
    ``/``, ``,``) plus the filler word ``and`` to spaces, so variants like
    ``"Bosnia-Herzegovina"`` and ``"Bosnia and Herzegovina"`` compare equal.
    """
    if not name:
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    lowered = ascii_name.lower()
    for sep in ("-", "&", ".", "/", ",", "(", ")"):
        lowered = lowered.replace(sep, " ")
    tokens = [t for t in lowered.split() if t and t != "and"]
    return " ".join(tokens)


def _name_matches(query_name: str, candidate_text: str) -> bool:
    """True if ``query_name`` matches ``candidate_text`` after normalization.

    A match is either a normalized substring (in either direction) or all of the
    query's normalized tokens being present in the candidate — tolerant of
    hyphen/``and`` variants while staying specific enough not to cross-match
    different nations.
    """
    q = _normalize_team(query_name)
    c = _normalize_team(candidate_text)
    if not q or not c:
        return False
    if q in c or c in q:
        return True
    q_tokens = q.split()
    c_tokens = set(c.split())
    return bool(q_tokens) and all(tok in c_tokens for tok in q_tokens)


def _iter_series_markets(
    series: str,
    config: dict,
    status: Optional[str] = None,
    max_pages: int = 20,
) -> Iterator[dict]:
    """Yield markets for a Kalshi series, paging via ``cursor``. Fail-open.

    ``status`` filters by Kalshi market status (``open``/``closed``/…); pass
    ``None`` (the default) to include EVERY status — a finalized/closed past
    fixture must still resolve for settlement/backtests. The single-page
    ``limit=200`` scan missed fixtures beyond the first page of the full World
    Cup bracket; this walks every page until the cursor is empty.
    """
    cursor: Optional[str] = None
    pages = 0
    while pages < max_pages:
        resp = kalshi.get_markets(
            series_ticker=series,
            status=status,
            limit=200,
            cursor=cursor,
            config=config,
        )
        if "error" in resp:
            return
        markets = resp.get("markets", [])
        for m in markets:
            yield m
        cursor = resp.get("cursor")
        pages += 1
        if not cursor or not markets:
            return


def _event_ticker_of(market_ticker: str) -> str:
    """Derive the event ticker from a contract ticker.

    Kalshi World Cup contracts look like ``KXWCGAME-26JUN12CANBIH-CAN``; the
    event ticker is everything before the final ``-<leg>`` segment, e.g.
    ``KXWCGAME-26JUN12CANBIH``. Returns the input unchanged if it has no leg.
    """
    if market_ticker.count("-") >= 2:
        return market_ticker.rsplit("-", 1)[0]
    return market_ticker


def _pick_event(events: dict, game_date: Optional[str]) -> list:
    """Pick the best matching event's markets from ``{event_ticker: [markets]}``.

    Prefers an event whose close/expiration date matches ``game_date`` (so the
    right edition of a repeated pairing is chosen), otherwise the event with the
    most contracts (closest to a full 3-way set). Returns ``[]`` when empty.
    """
    if not events:
        return []
    groups = list(events.values())
    if game_date:
        for markets in groups:
            for m in markets:
                d = (m.get("close_time") or m.get("expiration_time") or "")[:10]
                if d == game_date:
                    return markets
    groups.sort(key=len, reverse=True)
    return groups[0]


def _parse_matchup(query: str) -> tuple[str, str]:
    """Parse '"Away @ Home"' or '"Home vs Away"' into (away, home).

    For '@', the team after '@' is home (mirrors the NBA adapter). For soccer's
    'vs' form the convention is "Home vs Away" (the wizard emits this order), so
    the FIRST team is home and the second is away.
    """
    q = query.strip()
    if "@" in q:
        away, home = q.split("@", 1)
        return away.strip(), home.strip()
    for sep in (" vs ", " v ", " - ", "vs"):
        if sep in q:
            home, away = q.split(sep, 1)
            return away.strip(), home.strip()
    return "", q


# Instantiate + register the adapter at import.
register_adapter(SoccerAdapter())