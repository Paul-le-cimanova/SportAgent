"""Soccer / World Cup stats sourcing (balldontlie FIFA World Cup API).

Provides the Stats Analyst's data tools for soccer: group standings, team
recent form (W/D/L with scorelines), head-to-head, an xG-grounded team form
read (the single best soccer signal), and a scheduled-fixtures fetcher for the
wizard. Reuses the existing ``BALLDONTLIE_API_KEY`` (same key/header as NBA) —
no separate vendor key.

API: https://api.balldontlie.io/fifa/worldcup/v1 (header ``Authorization: <key>``).
Covers the 2018/2022/2026 editions; ``seasons[]`` defaults to 2026 when omitted.

All fetchers **fail open** — return a placeholder string on any error, never
raise. Registers its tools with the routing interface at import (vendor
``balldontlie``).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Optional

import requests

from sportagent.core.dataflows.interface import register_vendor_method

logger = logging.getLogger(__name__)

_BDL_FIFA_BASE = "https://api.balldontlie.io/fifa/worldcup/v1"
_TIMEOUT = 10.0


def _bdl_headers() -> dict:
    key = os.environ.get("BALLDONTLIE_API_KEY", "")
    return {"Authorization": key} if key else {}


def _bdl_get(path: str, params: Optional[dict] = None) -> Any:
    """GET a balldontlie FIFA path. Fail-open to None on any error."""
    try:
        resp = requests.get(
            f"{_BDL_FIFA_BASE}{path}",
            headers=_bdl_headers(),
            params=params or {},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 — fail open
        logger.warning("balldontlie FIFA GET %s failed: %s", path, exc)
        return None


# --- Tool implementations ----------------------------------------------------


def get_group_standings(competition: str = "", season: Optional[int] = None) -> str:
    """Group-stage standings for the requested World Cup edition.

    ``competition`` is accepted for interface symmetry with other sports but is
    unused (the FIFA API is World-Cup-scoped). Fails open to a placeholder.
    """
    params: dict = {}
    if season:
        params["seasons[]"] = season
    data = _bdl_get("/group_standings", params)
    rows = (data or {}).get("data") if isinstance(data, dict) else None
    if not rows:
        return "<group standings unavailable>"
    # Group rows by group name, ordered by position.
    groups: dict[str, list] = {}
    for row in rows:
        gname = (row.get("group") or {}).get("name", "?")
        groups.setdefault(gname, []).append(row)
    lines = ["World Cup group standings:"]
    for gname in sorted(groups):
        # The API's group name may already include the "Group " prefix.
        label = gname if gname.lower().startswith("group") else f"Group {gname}"
        lines.append(f"  {label}:")
        for row in sorted(groups[gname], key=lambda r: r.get("position", 99)):
            team = (row.get("team") or {}).get("name", "?")
            lines.append(
                f"    {row.get('position', '?')}. {team} — {row.get('points', 0)} pts "
                f"({row.get('won', 0)}W-{row.get('drawn', 0)}D-{row.get('lost', 0)}L, "
                f"GF {row.get('goals_for', 0)}/GA {row.get('goals_against', 0)}, "
                f"GD {row.get('goal_difference', 0):+d})"
            )
    return "\n".join(lines)


def get_recent_form(team: str, competition: str = "", n_matches: int = 5) -> str:
    """Last-N completed results for ``team`` (most recent first, W/D/L).

    Reads the FIFA ``/matches`` endpoint for the team, tagged with the scoreline
    so the analyst can read current form (incl. draws, which matter for the
    3-way market). Fails open.
    """
    team_id = _resolve_team_id(team)
    if team_id is None:
        return f"<could not resolve balldontlie FIFA team id for {team}>"
    matches = _team_matches(team_id)
    finals = [m for m in matches if _is_completed(m)]
    finals.sort(key=lambda m: m.get("datetime", ""), reverse=True)
    finals = finals[:n_matches]
    if not finals:
        return f"<no completed matches found for {team}>"
    wins = draws = losses = 0
    lines = []
    for m in finals:
        home = (m.get("home_team") or {}).get("name", "?")
        away = (m.get("away_team") or {}).get("name", "?")
        hs, as_ = m.get("home_score"), m.get("away_score")
        is_home = (m.get("home_team") or {}).get("id") == team_id
        if hs is None or as_ is None:
            continue
        team_goals = hs if is_home else as_
        opp_goals = as_ if is_home else hs
        if team_goals > opp_goals:
            res = "W"
            wins += 1
        elif team_goals == opp_goals:
            res = "D"
            draws += 1
        else:
            res = "L"
            losses += 1
        date = (m.get("datetime", "") or "")[:10]
        lines.append(f"  {date} {res} {home} {hs}-{as_} {away}")
    summary = f"{team} last {len(lines)} (most recent first): {wins}W-{draws}D-{losses}L"
    return summary + "\n" + "\n".join(lines)


def get_h2h(team_a: str, team_b: str, competition: str = "", limit: int = 5) -> str:
    """Recent head-to-head results between two nations (FIFA matches)."""
    id_a = _resolve_team_id(team_a)
    id_b = _resolve_team_id(team_b)
    if id_a is None or id_b is None:
        return f"<could not resolve team ids for {team_a} / {team_b}>"
    matches = _team_matches(id_a, all_seasons=True)
    meetings = []
    for m in matches:
        ids = {
            (m.get("home_team") or {}).get("id"),
            (m.get("away_team") or {}).get("id"),
        }
        if id_b not in ids or not _is_completed(m):
            continue
        home = (m.get("home_team") or {}).get("name", "?")
        away = (m.get("away_team") or {}).get("name", "?")
        hs, as_ = m.get("home_score"), m.get("away_score")
        if hs is None or as_ is None:
            continue
        date = (m.get("datetime", "") or "")[:10]
        meetings.append(f"  {date}: {home} {hs}-{as_} {away}")
    if not meetings:
        return f"No completed head-to-head meetings found between {team_a} and {team_b}."
    meetings.sort(reverse=True)
    return (
        f"H2H {team_a} vs {team_b} (most recent first):\n"
        + "\n".join(meetings[:limit])
    )


def get_team_xg(team: str, competition: str = "", last_n: int = 5) -> str:
    """xG-grounded form read for ``team`` — the single best soccer signal.

    Aggregates ``expected_goals`` (xG for) and the opponent's xG (xG against)
    across the team's most recent completed matches via ``/team_match_stats``,
    plus possession and shots-on-target. A *real* number the Research Manager
    can anchor to instead of inventing an xG model. Fails open.
    """
    data = _team_xg_data(team, last_n)
    if data is None:
        return f"<xG form unavailable for {team}>"
    return (
        f"{team} — xG form (last {data['games']} completed matches): "
        f"{data['xg_for_pg']:.2f} xG for/game vs {data['xg_against_pg']:.2f} xG "
        f"against/game (net {data['xg_for_pg'] - data['xg_against_pg']:+.2f}); "
        f"{data['poss_pg']:.0f}% possession, {data['sot_pg']:.1f} shots on "
        f"target/game. A positive net-xG side controls chances and is "
        f"underrated by results-based form."
    )


def _team_xg_data(team: str, last_n: int = 5) -> Optional[dict]:
    """Aggregate team-level xG for/against across recent matches. None on fail."""
    team_id = _resolve_team_id(team)
    if team_id is None:
        return None
    matches = _team_matches(team_id)
    finals = [m for m in matches if _is_completed(m)]
    finals.sort(key=lambda m: m.get("datetime", ""), reverse=True)
    match_ids = [m.get("id") for m in finals[:last_n] if m.get("id") is not None]
    if not match_ids:
        return None
    # One call for all the recent matches (rate-limit friendly): the
    # /team_match_stats endpoint accepts repeated match_ids[] params.
    stats = _bdl_get("/team_match_stats", {"match_ids[]": match_ids, "per_page": 100})
    rows = (stats or {}).get("data") if isinstance(stats, dict) else None
    if not rows:
        return None
    # Group the returned rows by match so we can pick our row + the opponent's.
    by_match: dict[Any, list] = {}
    for r in rows:
        by_match.setdefault(r.get("match_id"), []).append(r)
    xg_for = xg_against = poss = sot = 0.0
    counted = 0
    for mid in match_ids:
        pair = by_match.get(mid)
        if not pair:
            continue
        ours = next((r for r in pair if r.get("team_id") == team_id), None)
        theirs = next((r for r in pair if r.get("team_id") != team_id), None)
        if ours is None:
            continue
        xg_for += float(ours.get("expected_goals") or 0.0)
        if theirs is not None:
            xg_against += float(theirs.get("expected_goals") or 0.0)
        poss += float(ours.get("possession_pct") or 0.0)
        sot += float(ours.get("shots_on_target") or 0.0)
        counted += 1
    if counted == 0:
        return None
    return {
        "games": counted,
        "xg_for_pg": xg_for / counted,
        "xg_against_pg": xg_against / counted,
        "poss_pg": poss / counted,
        "sot_pg": sot / counted,
    }


def get_schedule_for_date(date: str, competition: str = "") -> list[dict]:
    """Return the World Cup fixtures for a calendar date as a list of dicts.

    Each entry is::

        {"away": str, "home": str, "kick_off": str, "status": str,
         "competition": str}

    The FIFA ``/matches`` endpoint isn't date-filterable, so we page through the
    edition's matches and keep those whose ``datetime`` falls on ``date``. Fails
    open to an empty list.
    """
    try:
        target = datetime.strptime(date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return []
    games: list[dict] = []
    cursor = None
    pages = 0
    while pages < 20:
        params: dict = {"per_page": 100}
        if cursor is not None:
            params["cursor"] = cursor
        data = _bdl_get("/matches", params)
        rows = (data or {}).get("data") if isinstance(data, dict) else None
        if not rows:
            break
        for m in rows:
            dt = (m.get("datetime", "") or "")[:10]
            if dt != target.isoformat():
                continue
            stage = (m.get("stage") or {}).get("name", "")
            group = (m.get("group") or {}).get("name", "")
            label = stage + (f" Group {group}" if group else "")
            games.append({
                "away": (m.get("away_team") or {}).get("name", "TBD"),
                "home": (m.get("home_team") or {}).get("name", "TBD"),
                "kick_off": m.get("datetime", ""),
                "status": m.get("status", ""),
                "competition": label or "World Cup",
            })
        cursor = (data.get("meta") or {}).get("next_cursor") if isinstance(data, dict) else None
        pages += 1
        if cursor is None:
            break
    return games


# --- Helpers -----------------------------------------------------------------

# Cache resolved team ids + the teams list per process (rate-limit friendly).
_TEAM_ID_CACHE: dict[str, Optional[int]] = {}
_TEAMS_CACHE: Optional[list] = None


def _all_teams() -> list:
    """Return (and cache) the FIFA teams list. Fail-open to []."""
    global _TEAMS_CACHE
    if _TEAMS_CACHE is not None:
        return _TEAMS_CACHE
    data = _bdl_get("/teams")
    _TEAMS_CACHE = (data or {}).get("data", []) if isinstance(data, dict) else []
    return _TEAMS_CACHE


def _resolve_team_id(team: str) -> Optional[int]:
    """Resolve a FIFA numeric team id by nation name/abbreviation (cached).

    Matching is tiered: exact name/abbreviation, then a substring of the full
    name, then all query tokens present in the combined name.
    """
    key = team.lower().strip()
    if key in _TEAM_ID_CACHE:
        return _TEAM_ID_CACHE[key]
    teams = _all_teams()
    result: Optional[int] = None

    def fields(t: dict) -> tuple[str, str]:
        return (
            (t.get("name") or "").lower(),
            (t.get("abbreviation") or "").lower(),
        )

    for t in teams:
        name, abbr = fields(t)
        if key in (name, abbr):
            result = t.get("id")
            break
    if result is None:
        for t in teams:
            name, _abbr = fields(t)
            if key and key in name:
                result = t.get("id")
                break
    if result is None:
        tokens = [tok for tok in key.split() if tok]
        for t in teams:
            name, abbr = fields(t)
            combined = f"{name} {abbr}"
            if tokens and all(tok in combined for tok in tokens):
                result = t.get("id")
                break

    _TEAM_ID_CACHE[key] = result
    return result


def _team_matches(team_id: int, all_seasons: bool = False) -> list:
    """Return all matches for a team (paginated). Fail-open to []."""
    out: list = []
    cursor = None
    pages = 0
    while pages < 20:
        params: dict = {"team_ids[]": team_id, "per_page": 100}
        if all_seasons:
            params["seasons[]"] = [2018, 2022, 2026]
        if cursor is not None:
            params["cursor"] = cursor
        data = _bdl_get("/matches", params)
        rows = (data or {}).get("data") if isinstance(data, dict) else None
        if not rows:
            break
        out.extend(rows)
        cursor = (data.get("meta") or {}).get("next_cursor") if isinstance(data, dict) else None
        pages += 1
        if cursor is None:
            break
    return out


def _is_completed(match: dict) -> bool:
    """True if a match has finished (status completed + both scores present)."""
    status = (match.get("status") or "").lower()
    if status not in ("completed", "finished", "final"):
        return False
    return match.get("home_score") is not None and match.get("away_score") is not None


# Register tools with the routing interface (vendor "balldontlie").
register_vendor_method("get_league_table", "balldontlie", get_group_standings)
register_vendor_method("get_soccer_recent_form", "balldontlie", get_recent_form)
register_vendor_method("get_soccer_h2h", "balldontlie", get_h2h)
register_vendor_method("get_team_xg", "balldontlie", get_team_xg)