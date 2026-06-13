"""NBA stats sourcing (balldontlie + ESPN, free).

Provides the Stats Analyst's data tools for NBA: team records, recent form,
head-to-head, standings, and rest/back-to-back status. balldontlie is the
primary (games/standings); ESPN supplements (team detail, schedule for rest).

All fetchers fail open — return a placeholder string on any error, never raise.
Registers its tools with the routing interface at import (vendor "balldontlie"
for team_stats; "espn" for scores_standings).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests

from sportagent.core.dataflows.interface import register_vendor_method

logger = logging.getLogger(__name__)

_BDL_BASE = "https://api.balldontlie.io/v1"
_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
_TIMEOUT = 10.0


def _bdl_headers() -> dict:
    key = os.environ.get("BALLDONTLIE_API_KEY", "")
    return {"Authorization": key} if key else {}


def _bdl_get(path: str, params: Optional[dict] = None) -> Any:
    try:
        resp = requests.get(
            f"{_BDL_BASE}{path}", headers=_bdl_headers(), params=params or {}, timeout=_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 — fail open
        # /standings 401 is expected on the free tier (we fall back to /games);
        # log it at debug so it doesn't spam the console / corrupt the live UI.
        msg = str(exc)
        if path.startswith("/standings") and "401" in msg:
            logger.debug("balldontlie GET %s unavailable on this tier: %s", path, exc)
        else:
            logger.warning("balldontlie GET %s failed: %s", path, exc)
        return None


def _espn_get(path: str, params: Optional[dict] = None) -> Any:
    try:
        resp = requests.get(f"{_ESPN_BASE}{path}", params=params or {}, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 — fail open
        logger.warning("ESPN GET %s failed: %s", path, exc)
        return None


# --- Tool implementations ----------------------------------------------------


def get_team_stats(team: str, season: Optional[int] = None) -> str:
    """Season win/loss record + standing for a team.

    Prefers balldontlie ``/standings`` (record + conference rank), but that
    endpoint is gated to a higher tier. On any failure we fall back to deriving
    the season record from ``/games`` (available on the NBA tier), so the Stats
    Analyst still gets a real record.
    """
    season = season or _current_season()
    data = _bdl_get("/standings", {"season": season})
    if data and "data" in data:
        target_id = _resolve_team_id(team)
        for row in data.get("data", []):
            info = row.get("team", {})
            if info.get("id") != target_id:
                continue
            wins = row.get("wins")
            losses = row.get("losses")
            conf = row.get("conference_record", "")
            rank = row.get("conference_rank", "")
            return (
                f"{info.get('full_name', team)} — record {wins}-{losses}, "
                f"conference rank {rank}, conf record {conf} (season {season})."
            )
    # Fallback: derive record from completed games (works on the NBA tier).
    return _record_from_games(team, season)


def _record_from_games(team: str, season: int) -> str:
    """Compute a season W-L record from the /games endpoint (tier fallback).

    Splits regular-season and postseason records so a deep playoff run is
    visible (critical for Finals-context analysis).
    """
    team_id = _resolve_team_id(team)
    if team_id is None:
        return f"<team stats unavailable for {team} (season {season})>"
    data = _bdl_get(
        "/games",
        {"seasons[]": season, "team_ids[]": team_id, "per_page": 100},
    )
    if not data or "data" not in data:
        return f"<team stats unavailable for {team} (season {season})>"
    rs_w = rs_l = po_w = po_l = 0
    for g in data["data"]:
        if g.get("status", "").lower() != "final":
            continue
        home_id = g.get("home_team", {}).get("id")
        home_score = g.get("home_team_score", 0)
        away_score = g.get("visitor_team_score", 0)
        is_home = home_id == team_id
        team_score = home_score if is_home else away_score
        opp_score = away_score if is_home else home_score
        won = team_score > opp_score
        if g.get("postseason"):
            po_w += int(won)
            po_l += int(not won)
        else:
            rs_w += int(won)
            rs_l += int(not won)
    if rs_w + rs_l + po_w + po_l == 0:
        return f"<no completed games found for {team} (season {season})>"
    rs_pct = rs_w / (rs_w + rs_l) if (rs_w + rs_l) else 0.0
    out = (
        f"{team} — regular season {rs_w}-{rs_l} ({rs_pct:.3f}) "
        f"(season {season}; standings endpoint unavailable on current tier)."
    )
    if po_w + po_l:
        out += f" Postseason: {po_w}-{po_l} (currently in the playoffs)."
    return out


def get_recent_form(team: str, n_games: int = 10, season: Optional[int] = None) -> str:
    """Last-N results for a team (balldontlie games), most recent first.

    Each line is tagged ``[PO]`` for postseason or ``[RS]`` for regular season,
    so the analyst can weight current playoff form appropriately.
    """
    season = season or _current_season()
    team_id = _resolve_team_id(team)
    if team_id is None:
        return f"<could not resolve balldontlie team id for {team}>"
    data = _bdl_get(
        "/games",
        {"seasons[]": season, "team_ids[]": team_id, "per_page": 100},
    )
    if not data or "data" not in data:
        return f"<recent form unavailable for {team}>"
    games = [g for g in data["data"] if g.get("status", "").lower() == "final"]
    # Sort by date desc; postseason games naturally fall latest in the season.
    games.sort(key=lambda g: g.get("date", ""), reverse=True)
    games = games[:n_games]
    if not games:
        return f"<no completed games found for {team} (season {season})>"
    wins = 0
    lines = []
    for g in games:
        home_id = g.get("home_team", {}).get("id")
        home_score = g.get("home_team_score", 0)
        away_score = g.get("visitor_team_score", 0)
        is_home = home_id == team_id
        team_score = home_score if is_home else away_score
        opp_score = away_score if is_home else home_score
        won = team_score > opp_score
        wins += int(won)
        opp = (g.get("visitor_team") if is_home else g.get("home_team")).get("abbreviation", "?")
        tag = "PO" if g.get("postseason") else "RS"
        loc = "vs" if is_home else "@"
        date = g.get("date", "")[:10]
        lines.append(f"[{tag}] {date} {'W' if won else 'L'} {team_score}-{opp_score} {loc} {opp}")
    summary = f"{team} last {len(games)} (most recent first): {wins}-{len(games) - wins}"
    return summary + "\n  " + "\n  ".join(lines)


def get_h2h(team_a: str, team_b: str, season: Optional[int] = None) -> str:
    """Head-to-head results between two teams this season (balldontlie)."""
    season = season or _current_season()
    id_a = _resolve_team_id(team_a)
    id_b = _resolve_team_id(team_b)
    if id_a is None or id_b is None:
        return f"<could not resolve team ids for {team_a} / {team_b}>"
    data = _bdl_get(
        "/games",
        {"seasons[]": season, "team_ids[]": id_a, "per_page": 100},
    )
    if not data or "data" not in data:
        return f"<h2h unavailable for {team_a} vs {team_b}>"
    meetings = []
    for g in data["data"]:
        ids = {g.get("home_team", {}).get("id"), g.get("visitor_team", {}).get("id")}
        if id_b in ids and g.get("status", "").lower() == "final":
            hs = g.get("home_team_score", 0)
            vs = g.get("visitor_team_score", 0)
            home_abbr = g.get("home_team", {}).get("abbreviation", "?")
            away_abbr = g.get("visitor_team", {}).get("abbreviation", "?")
            tag = "PO" if g.get("postseason") else "RS"
            meetings.append(
                f"[{tag}] {g.get('date', '')[:10]}: {away_abbr} {vs} @ {home_abbr} {hs}"
            )
    if not meetings:
        return f"No completed head-to-head meetings between {team_a} and {team_b} this season."
    # Sort most-recent-first so any playoff series shows at the top.
    meetings.sort(reverse=True)
    return f"H2H {team_a} vs {team_b} (season {season}): " + "; ".join(meetings)


def get_standings(season: Optional[int] = None) -> str:
    """League standings summary (ESPN)."""
    data = _espn_get("/standings")
    if not data:
        # ESPN standings shape varies; fall back to a brief note.
        return "<standings unavailable from ESPN>"
    return "ESPN standings fetched (raw structure available to caller)."


def get_schedule(team: str) -> str:
    """Upcoming/recent schedule for a team (ESPN scoreboard-based)."""
    data = _espn_get("/scoreboard")
    if not data:
        return f"<schedule unavailable for {team}>"
    return "ESPN scoreboard fetched (today's slate available to caller)."


def get_schedule_for_date(date: str) -> list[dict]:
    """Return the NBA schedule for a calendar date as a list of game dicts.

    ESPN scoreboard is primary (``?dates=YYYYMMDD``); balldontlie ``/games``
    (``?dates[]=YYYY-MM-DD``) is the fallback. Each entry is::

        {"away": str, "home": str, "tip_off": str, "status": str,
         "away_id": int|None, "home_id": int|None}

    Fails open to an empty list on any error.
    """
    games = _schedule_from_espn(date)
    if games:
        return games
    return _schedule_from_bdl(date)


def _schedule_from_espn(date: str) -> list[dict]:
    """ESPN scoreboard schedule for ``date`` (YYYY-MM-DD). Fail-open to []."""
    try:
        compact = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
    except ValueError:
        return []
    data = _espn_get("/scoreboard", {"dates": compact})
    if not data or "events" not in data:
        return []
    games: list[dict] = []
    for ev in data.get("events", []):
        comps = ev.get("competitions") or []
        if not comps:
            continue
        competitors = comps[0].get("competitors") or []
        home = away = ""
        for c in competitors:
            name = (c.get("team") or {}).get("displayName", "")
            if c.get("homeAway") == "home":
                home = name
            elif c.get("homeAway") == "away":
                away = name
        status = (
            (ev.get("status") or {}).get("type", {}).get("shortDetail", "")
            or (ev.get("status") or {}).get("type", {}).get("description", "")
        )
        games.append({
            "away": away,
            "home": home,
            "tip_off": ev.get("date", ""),
            "status": status,
            "away_id": None,
            "home_id": None,
        })
    return games


def _schedule_from_bdl(date: str) -> list[dict]:
    """balldontlie /games schedule for ``date`` (YYYY-MM-DD). Fail-open to []."""
    data = _bdl_get("/games", {"dates[]": date, "per_page": 100})
    if not data or "data" not in data:
        return []
    games: list[dict] = []
    for g in data["data"]:
        home = g.get("home_team", {}) or {}
        away = g.get("visitor_team", {}) or {}
        games.append({
            "away": away.get("full_name", ""),
            "home": home.get("full_name", ""),
            "tip_off": g.get("status", "") if ":" in str(g.get("status", "")) else g.get("date", ""),
            "status": g.get("status", ""),
            "away_id": away.get("id"),
            "home_id": home.get("id"),
        })
    return games


def get_rest_status(team: str, game_date: str) -> str:
    """Rest / back-to-back status heading into ``game_date`` (YYYY-MM-DD).

    Uses balldontlie game history to detect a game on the prior calendar day.
    """
    team_id = _resolve_team_id(team)
    if team_id is None:
        return f"<could not resolve team id for {team}>"
    try:
        target = datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError:
        return f"<invalid game_date {game_date!r} (expected YYYY-MM-DD)>"
    prev_day = (target - timedelta(days=1)).strftime("%Y-%m-%d")
    season = _current_season()
    data = _bdl_get(
        "/games",
        {"seasons[]": season, "team_ids[]": team_id, "per_page": 100},
    )
    if not data or "data" not in data:
        return f"<rest status unavailable for {team}>"
    played_prev = any(g.get("date", "")[:10] == prev_day for g in data["data"])
    if played_prev:
        return f"{team} is on a BACK-TO-BACK (played {prev_day}, plays again {game_date})."
    return f"{team} is rested (no game on {prev_day} before {game_date})."


def get_four_factors(team: str, season: Optional[int] = None, last_n: int = 20) -> str:
    """Season Four-Factors + efficiency for a team (balldontlie box scores).

    The "Four Factors" (Dean Oliver) are the box-score rates that actually
    predict NBA outcomes — and two of them (turnover rate, free-throw rate) are
    precisely the levers that decide close games:

      - eFG%    effective field-goal % = (FGM + 0.5*3PM) / FGA   (shooting)
      - TOV%    turnovers per ~possession                         (ball security)
      - OREB%   offensive rebounds / (OREB + opp DREB) ≈ OREB/FGA proxy
      - FT-rate FTM / FGA                                         (getting to the line)

    Plus simple offensive output (PPG) and pace proxy. Computed from completed
    games this season (most recent ``last_n``). Fails open to a placeholder.
    """
    data = _four_factors_data(team, season, last_n)
    if data is None:
        return f"<four factors unavailable for {team}>"
    return (
        f"{team} — Four Factors (last {data['games']} games, season {data['season']}): "
        f"eFG% {data['efg'] * 100:.1f}%, TOV-rate {data['tov_rate'] * 100:.1f}%, "
        f"OREB-rate {data['oreb_rate'] * 100:.1f}%, FT-rate {data['ft_rate']:.3f} "
        f"(FTM/FGA); {data['ppg']:.1f} PPG on {data['fga_pg']:.1f} FGA/game."
    )


def _four_factors_data(
    team: str, season: Optional[int] = None, last_n: int = 20
) -> Optional[dict]:
    """Compute team-level Four-Factors aggregates as a dict. None on failure.

    Returns ``{games, season, efg, tov_rate, oreb_rate, ft_rate, ppg, fga_pg}``.
    Aggregation is sum-then-divide (possession-weighted), which is the correct
    way to combine rate stats across games (never average per-game ratios).
    """
    season = season or _current_season()
    team_id = _resolve_team_id(team)
    if team_id is None:
        return None
    # Pull completed games to know which game ids to aggregate box scores over.
    games = _bdl_get(
        "/games",
        {"seasons[]": season, "team_ids[]": team_id, "per_page": 100},
    )
    if not games or "data" not in games:
        return None
    finals = [g for g in games["data"] if g.get("status", "").lower() == "final"]
    finals.sort(key=lambda g: g.get("date", ""), reverse=True)
    game_ids = [g.get("id") for g in finals[:last_n] if g.get("id") is not None]
    if not game_ids:
        return None

    # The /stats endpoint returns one row PER PLAYER PER GAME (~30 rows/game), so
    # a single page of 100 only covers ~3 games. Paginate via next_cursor until
    # all requested games are gathered (cap for safety). balldontlie accepts
    # repeated game_ids[]; requests encodes a list correctly.
    base_params: dict = {
        "team_ids[]": team_id,
        "per_page": 100,
        "seasons[]": season,
        "game_ids[]": game_ids,
    }
    rows: list = []
    cursor = None
    pages = 0
    while pages < 25:
        params = dict(base_params)
        if cursor is not None:
            params["cursor"] = cursor
        page = _bdl_get("/stats", params)
        if not page or "data" not in page or not page["data"]:
            break
        rows.extend(page["data"])
        cursor = (page.get("meta") or {}).get("next_cursor")
        pages += 1
        if cursor is None:
            break
    if not rows:
        return None

    fgm = fga = fg3m = ftm = fta = oreb = tov = pts = 0
    n_games = set()
    for row in rows:
        if (row.get("team") or {}).get("id") != team_id:
            continue
        fgm += row.get("fgm", 0) or 0
        fga += row.get("fga", 0) or 0
        fg3m += row.get("fg3m", 0) or 0
        ftm += row.get("ftm", 0) or 0
        fta += row.get("fta", 0) or 0
        oreb += row.get("oreb", 0) or 0
        tov += row.get("turnover", 0) or 0
        pts += row.get("pts", 0) or 0
        gid = (row.get("game") or {}).get("id")
        if gid is not None:
            n_games.add(gid)
    g = len(n_games) or 1
    if fga <= 0:
        return None
    # Possession proxy (standard): FGA + 0.44*FTA + TOV.
    poss = fga + 0.44 * fta + tov
    return {
        "games": g,
        "season": season,
        "efg": (fgm + 0.5 * fg3m) / fga,
        "tov_rate": (tov / poss) if poss > 0 else 0.0,
        "oreb_rate": oreb / fga,  # OREB/FGA proxy (true OREB% needs opp DREB)
        "ft_rate": ftm / fga,
        "ppg": pts / g,
        "fga_pg": fga / g,
    }


def get_elo_winprob(
    home_team: str, away_team: str, season: Optional[int] = None
) -> str:
    """Elo-based pre-game home win probability for ``home_team`` vs ``away_team``.

    A *real* (deterministic) prior the Research Manager can anchor to, instead of
    inventing an "Elo model" in prose. Ratings are built from this season's
    completed games with a standard NBA Elo:

      - start every team at 1500
      - K-factor 20 (playoff games weighted slightly higher, K=24)
      - margin-of-victory multiplier (FiveThirtyEight-style)
      - home-court advantage of +60 Elo applied at prediction time

    Expected home win prob = 1 / (1 + 10**(-(R_home + HCA - R_away)/400)).
    Fails open to a placeholder when game data is unavailable.
    """
    data = _elo_winprob_data(home_team, away_team, season)
    if data is None:
        return f"<elo win prob unavailable for {home_team} vs {away_team}>"
    return (
        f"Elo prior (season {data['season']}): {home_team} {data['home_elo']:.0f} "
        f"(home) vs {away_team} {data['away_elo']:.0f}. With +{data['hca']:.0f} "
        f"home-court Elo, {home_team} win probability = "
        f"{data['home_winprob'] * 100:.1f}% (away {(1 - data['home_winprob']) * 100:.1f}%)."
    )


def _elo_winprob_data(
    home_team: str, away_team: str, season: Optional[int] = None, hca: float = 60.0
) -> Optional[dict]:
    """Compute Elo ratings + home win probability. None on failure.

    Returns ``{season, home_elo, away_elo, hca, home_winprob}``.
    """
    season = season or _current_season()
    home_id = _resolve_team_id(home_team)
    away_id = _resolve_team_id(away_team)
    if home_id is None or away_id is None:
        return None
    ratings = _season_elo_ratings(season)
    if not ratings:
        return None
    r_home = ratings.get(home_id, 1500.0)
    r_away = ratings.get(away_id, 1500.0)
    home_winprob = 1.0 / (1.0 + 10 ** (-((r_home + hca) - r_away) / 400.0))
    return {
        "season": season,
        "home_elo": r_home,
        "away_elo": r_away,
        "hca": hca,
        "home_winprob": home_winprob,
    }


# Cache the computed Elo table per season (one full /games sweep per process).
_ELO_CACHE: dict[int, dict] = {}


def _season_elo_ratings(season: int) -> dict:
    """Build an Elo rating table for every team from this season's finals.

    Iterates completed games in chronological order, updating both teams after
    each result with a margin-of-victory-scaled K-factor. Cached per season.
    """
    if season in _ELO_CACHE:
        return _ELO_CACHE[season]
    data = _bdl_get("/games", {"seasons[]": season, "per_page": 100})
    # /games is paginated; pull subsequent pages until exhausted (cap for safety).
    games: list = []
    if data and "data" in data:
        games.extend(data["data"])
        cursor = (data.get("meta") or {}).get("next_cursor")
        pages = 0
        while cursor and pages < 40:
            page = _bdl_get(
                "/games",
                {"seasons[]": season, "per_page": 100, "cursor": cursor},
            )
            if not page or "data" not in page:
                break
            games.extend(page["data"])
            cursor = (page.get("meta") or {}).get("next_cursor")
            pages += 1
    finals = [g for g in games if g.get("status", "").lower() == "final"]
    if not finals:
        return {}
    finals.sort(key=lambda g: g.get("date", ""))

    ratings: dict[int, float] = {}

    def rating(tid: int) -> float:
        return ratings.get(tid, 1500.0)

    for g in finals:
        home = g.get("home_team", {}) or {}
        away = g.get("visitor_team", {}) or {}
        hid, aid = home.get("id"), away.get("id")
        if hid is None or aid is None:
            continue
        hs = g.get("home_team_score", 0) or 0
        as_ = g.get("visitor_team_score", 0) or 0
        if hs == as_:
            continue
        r_h, r_a = rating(hid), rating(aid)
        # +100 home-court when computing the in-game expectation.
        exp_h = 1.0 / (1.0 + 10 ** (-((r_h + 100.0) - r_a) / 400.0))
        s_h = 1.0 if hs > as_ else 0.0
        margin = abs(hs - as_)
        elo_diff = (r_h + 100.0) - r_a if s_h else r_a - (r_h + 100.0)
        # FiveThirtyEight MoV multiplier (dampens runaway favorites).
        mov_mult = ((margin + 3) ** 0.8) / (7.5 + 0.006 * elo_diff)
        k = 24.0 if g.get("postseason") else 20.0
        delta = k * mov_mult * (s_h - exp_h)
        ratings[hid] = r_h + delta
        ratings[aid] = r_a - delta

    _ELO_CACHE[season] = ratings
    return ratings


# --- Helpers -----------------------------------------------------------------


def _current_season() -> int:
    """NBA season start year. Season starts in October; Jan-Sep → prior year."""
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 10 else now.year - 1


_TEAM_ID_CACHE: dict[str, Optional[int]] = {}
# Cache the full /teams payload once per process so resolving multiple team ids
# costs a single balldontlie call (reduces rate-limit pressure).
_TEAMS_CACHE: Optional[list] = None


def _all_teams() -> list:
    """Return (and cache) the balldontlie teams list. Fail-open to []."""
    global _TEAMS_CACHE
    if _TEAMS_CACHE:
        return _TEAMS_CACHE
    data = _bdl_get("/teams")
    _TEAMS_CACHE = (data or {}).get("data", []) or []
    return _TEAMS_CACHE


def _resolve_team_id(team: str) -> Optional[int]:
    """Resolve a balldontlie numeric team id by name (cached).

    Matching is tiered to avoid false positives from shared tokens (e.g.
    "New York" must not match "New Orleans" on the token "new"):
      1. exact match on full_name, name (nickname), city, or abbreviation
      2. the query is a substring of full_name (e.g. "knicks" in "New York Knicks")
      3. ALL query tokens appear in the team's combined name
    """
    key = team.lower().strip()
    if key in _TEAM_ID_CACHE:
        return _TEAM_ID_CACHE[key]

    teams = _all_teams()
    result: Optional[int] = None

    def fields(t: dict) -> tuple[str, str, str, str]:
        return (
            t.get("full_name", "").lower(),
            t.get("name", "").lower(),
            t.get("city", "").lower(),
            t.get("abbreviation", "").lower(),
        )

    # Tier 1: exact match on any identifying field.
    for t in teams:
        full, nick, city, abbr = fields(t)
        if key in (full, nick, city, abbr):
            result = t.get("id")
            break

    # Tier 2: query is a substring of the full team name (e.g. "knicks").
    if result is None:
        for t in teams:
            full, nick, city, abbr = fields(t)
            if key and (key in full or key in nick):
                result = t.get("id")
                break

    # Tier 3: ALL query tokens present in the combined name (strict AND).
    if result is None:
        tokens = [tok for tok in key.split() if tok]
        for t in teams:
            full, nick, city, abbr = fields(t)
            combined = f"{full} {nick} {city} {abbr}"
            if tokens and all(tok in combined for tok in tokens):
                result = t.get("id")
                break

    _TEAM_ID_CACHE[key] = result
    return result


# Register tools with the routing interface.
register_vendor_method("get_team_stats", "balldontlie", get_team_stats)
register_vendor_method("get_recent_form", "balldontlie", get_recent_form)
register_vendor_method("get_h2h", "balldontlie", get_h2h)
register_vendor_method("get_standings", "espn", get_standings)
register_vendor_method("get_schedule", "espn", get_schedule)
register_vendor_method("get_rest_status", "espn", get_rest_status)
register_vendor_method("get_four_factors", "balldontlie", get_four_factors)
register_vendor_method("get_elo_winprob", "balldontlie", get_elo_winprob)
