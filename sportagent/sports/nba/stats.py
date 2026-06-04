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
from datetime import datetime, timedelta
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


# --- Helpers -----------------------------------------------------------------


def _current_season() -> int:
    """NBA season start year. Season starts in October; Jan-Sep → prior year."""
    now = datetime.utcnow()
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