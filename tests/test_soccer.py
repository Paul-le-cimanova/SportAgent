"""Tests for the Soccer / World Cup (3-way) adapter pack + 3-way plumbing.

Covers the v0.4.0 milestone:

1. 3-way de-vig + best-leg Kelly selection (incl. the draw) in probability.py.
2. The SoccerAdapter's 3-contract resolution (mocked Kalshi) + settlement.
3. The 3-way verified-odds snapshot rendering all three legs.
4. The Trader's deterministic best-leg extraction from the 3-way snapshot/thesis.
5. 3-way settlement scored via brier_multi in reflection.py.

All network access is monkeypatched, so these run offline + deterministically.
"""

import pytest

from sportagent.core.agents.utils import probability as p
from sportagent.sports.base import MarketRef
from sportagent.sports.soccer.adapter import SoccerAdapter


# --- 3-way de-vig + best-leg Kelly ------------------------------------------


def test_devig_3way_normalizes():
    h, d, a = p.devig_3way(0.55, 0.30, 0.25)
    assert h + d + a == pytest.approx(1.0)
    assert h > d > a


def test_best_leg_selects_draw_when_most_underpriced():
    # Draw est 0.34 vs implied 0.25 is the biggest edge (+0.09).
    leg = p.best_leg([0.40, 0.34, 0.26], [40, 25, 30])
    assert leg is not None
    assert leg.label == "draw"
    assert leg.stake_pct > 0.0


def test_best_leg_holds_when_fairly_priced():
    assert p.best_leg([0.40, 0.30, 0.30], [40, 30, 30]) is None


def test_kelly_3way_negative_edge_no_stake():
    legs = p.kelly_3way([0.30, 0.30, 0.40], [40, 30, 30])
    # Home is overpriced (est 0.30 vs implied 0.40) → no stake.
    home = next(l for l in legs if l.label == "home")
    assert home.stake_pct == 0.0


# --- SoccerAdapter contract resolution (mocked Kalshi) ----------------------


def _fake_markets_response():
    """Three Kalshi contracts for a single soccer match (home/draw/away)."""
    return {
        "markets": [
            {"ticker": "KXSOCCER-ARSCHE-ARS", "yes_sub_title": "Arsenal",
             "title": "Arsenal vs Chelsea", "close_time": "2026-06-12T19:00:00Z"},
            {"ticker": "KXSOCCER-ARSCHE-DRAW", "yes_sub_title": "Draw",
             "title": "Arsenal vs Chelsea", "close_time": "2026-06-12T19:00:00Z"},
            {"ticker": "KXSOCCER-ARSCHE-CHE", "yes_sub_title": "Chelsea",
             "title": "Arsenal vs Chelsea", "close_time": "2026-06-12T19:00:00Z"},
        ]
    }


def test_adapter_resolves_three_contracts(monkeypatch):
    from sportagent.sports.soccer import adapter as soccer_adapter

    monkeypatch.setattr(
        soccer_adapter.kalshi,
        "get_markets",
        lambda *a, **k: _fake_markets_response(),
    )
    ad = SoccerAdapter()
    ref = ad.resolve_market("Arsenal vs Chelsea", {})
    assert ref.outcome_structure == "three_way"
    assert ref.contracts.get("home") == "KXSOCCER-ARSCHE-ARS"
    assert ref.contracts.get("draw") == "KXSOCCER-ARSCHE-DRAW"
    assert ref.contracts.get("away") == "KXSOCCER-ARSCHE-CHE"
    assert ref.home_team == "Arsenal"
    assert ref.away_team == "Chelsea"


def test_adapter_outcome_structure_is_three_way():
    assert SoccerAdapter().outcome_structure() == "three_way"


def test_adapter_sportsbook_key_per_competition(monkeypatch):
    from sportagent.sports.soccer import adapter as soccer_adapter

    monkeypatch.setattr(
        soccer_adapter.kalshi, "get_markets", lambda *a, **k: {"markets": []}
    )
    ad = SoccerAdapter()
    ref = ad.resolve_market(
        "France vs Brazil", {"competition": "World Cup", "market_type": "match"}
    )
    assert ref.sportsbook_key == "soccer_fifa_world_cup"


def test_adapter_advancement_is_two_way(monkeypatch):
    from sportagent.sports.soccer import adapter as soccer_adapter

    monkeypatch.setattr(
        soccer_adapter.kalshi, "get_market", lambda *a, **k: {"error": "x"}
    )
    ad = SoccerAdapter()
    ref = ad.resolve_market(
        "KXWCADV-FRA", {"market_type": "advancement"}
    )
    assert ref.outcome_structure == "two_way"
    assert ref.contracts == {"yes": "KXWCADV-FRA"}


# --- Settlement → home/draw/away --------------------------------------------


def _three_way_ref():
    return MarketRef(
        sport="soccer",
        market_ticker="KXSOCCER-ARSCHE-ARS",
        target_team="Arsenal",
        home_team="Arsenal",
        away_team="Chelsea",
        game_date="2026-06-12",
        outcome_structure="three_way",
        contracts={
            "home": "KXSOCCER-ARSCHE-ARS",
            "draw": "KXSOCCER-ARSCHE-DRAW",
            "away": "KXSOCCER-ARSCHE-CHE",
        },
        sportsbook_key="soccer_epl",
    )


def test_settle_from_scores_home_win():
    ad = SoccerAdapter()
    assert ad.settle(_three_way_ref(), None, {"home": 2, "away": 1}) == "home"


def test_settle_from_scores_draw():
    ad = SoccerAdapter()
    assert ad.settle(_three_way_ref(), None, {"home": 1, "away": 1}) == "draw"


def test_settle_from_scores_away_win():
    ad = SoccerAdapter()
    assert ad.settle(_three_way_ref(), None, {"home": 0, "away": 2}) == "away"


def test_settle_fallback_from_result():
    ad = SoccerAdapter()
    assert ad.settle(_three_way_ref(), "yes", None) == "home"
    assert ad.settle(_three_way_ref(), "no", None) == "away"


# --- 3-way verified-odds snapshot -------------------------------------------


def test_three_way_snapshot_renders_all_legs(monkeypatch):
    from sportagent.core.dataflows import odds_validator

    def fake_get_market(ticker, config=None):
        prices = {
            "KXSOCCER-ARSCHE-ARS": "0.45",
            "KXSOCCER-ARSCHE-DRAW": "0.28",
            "KXSOCCER-ARSCHE-CHE": "0.27",
        }
        return {"market": {"last_price_dollars": prices.get(ticker, "0.50")}}

    monkeypatch.setattr(odds_validator.kalshi, "get_market", fake_get_market)
    # No sportsbook key → fetch returns an error string (fail open).
    snap = odds_validator.build_verified_odds_snapshot(
        market_ticker="KXSOCCER-ARSCHE-ARS",
        target_team="Arsenal",
        home_team="Arsenal",
        away_team="Chelsea",
        sport_key="",
        outcome_structure="three_way",
        contracts={
            "home": "KXSOCCER-ARSCHE-ARS",
            "draw": "KXSOCCER-ARSCHE-DRAW",
            "away": "KXSOCCER-ARSCHE-CHE",
        },
    )
    assert "Kalshi home" in snap
    assert "Kalshi draw" in snap
    assert "Kalshi away" in snap
    assert "45.0%" in snap and "28.0%" in snap and "27.0%" in snap


# --- Trader best-leg extraction ---------------------------------------------


def test_trader_three_way_extractors():
    from sportagent.core.agents.trader.trader import (
        _extract_three_way_estimates,
        _extract_three_way_leg_prices,
    )

    plan = (
        "**3-Way Probability Vector (normalized to sum to 1):**\n"
        "- Home win: 0.500 (50.0%)\n"
        "- Draw: 0.300 (30.0%)\n"
        "- Away win: 0.200 (20.0%)\n"
    )
    est = _extract_three_way_estimates(plan)
    assert est == pytest.approx([0.5, 0.3, 0.2])

    snap = (
        "Kalshi home (T): YES 40c → implied 40.0%\n"
        "Kalshi draw (T): YES 28c → implied 28.0%\n"
        "Kalshi away (T): YES 22c → implied 22.0%\n"
    )
    prices = _extract_three_way_leg_prices(snap)
    assert prices == {"home": 40.0, "draw": 28.0, "away": 22.0}


# --- 3-way reflection scoring (brier_multi) ---------------------------------


def test_three_way_brier_scores_vector():
    from sportagent.core.graph.reflection import _three_way_brier

    raw = (
        "Home win: 0.500\nDraw: 0.300\nAway win: 0.200\n"
    )
    # Draw was the realized outcome (index 1).
    score = _three_way_brier(raw, "draw")
    assert score == pytest.approx(p.brier_multi([0.5, 0.3, 0.2], 1))


def test_three_way_brier_none_when_unparseable():
    from sportagent.core.graph.reflection import _three_way_brier

    assert _three_way_brier("no vector here", "home") is None


def test_won_three_way_leg_in_action():
    from sportagent.core.graph.reflection import _won

    assert _won("BUY YES on the DRAW leg", "draw") is True
    assert _won("BUY YES on the HOME leg", "away") is False
    assert _won("HOLD", "draw") is None


# --- balldontlie FIFA stats (mocked HTTP) -----------------------------------


def _patch_fifa(monkeypatch, responses):
    """Patch soccer stats._bdl_get to serve canned responses keyed by path prefix.

    ``responses`` maps a path prefix (e.g. "/teams", "/matches",
    "/team_match_stats") to the dict the real API would return. Also resets the
    module's process-level caches so each test is isolated.
    """
    from sportagent.sports.soccer import stats as s

    s._TEAMS_CACHE = None
    s._TEAM_ID_CACHE.clear()

    def fake_get(path, params=None):
        for prefix, payload in responses.items():
            if path.startswith(prefix):
                return payload
        return None

    monkeypatch.setattr(s, "_bdl_get", fake_get)
    return s


_FIFA_TEAMS = {
    "data": [
        {"id": 1, "name": "Brazil", "abbreviation": "BRA"},
        {"id": 2, "name": "Argentina", "abbreviation": "ARG"},
    ]
}


def test_fifa_resolve_team_id_by_name(monkeypatch):
    s = _patch_fifa(monkeypatch, {"/teams": _FIFA_TEAMS})
    assert s._resolve_team_id("Brazil") == 1
    assert s._resolve_team_id("ARG") == 2
    assert s._resolve_team_id("Nowhere") is None


def test_fifa_recent_form_counts_wdl(monkeypatch):
    matches = {
        "data": [
            {"id": 10, "status": "completed", "datetime": "2026-06-10T18:00:00Z",
             "home_team": {"id": 1, "name": "Brazil"}, "away_team": {"id": 2, "name": "Argentina"},
             "home_score": 2, "away_score": 1},
            {"id": 11, "status": "completed", "datetime": "2026-06-05T18:00:00Z",
             "home_team": {"id": 3, "name": "Spain"}, "away_team": {"id": 1, "name": "Brazil"},
             "home_score": 1, "away_score": 1},
            {"id": 12, "status": "scheduled", "datetime": "2026-06-20T18:00:00Z",
             "home_team": {"id": 1, "name": "Brazil"}, "away_team": {"id": 4, "name": "France"},
             "home_score": None, "away_score": None},
        ]
    }
    s = _patch_fifa(monkeypatch, {"/teams": _FIFA_TEAMS, "/matches": matches})
    out = s.get_recent_form("Brazil")
    # 1 win (2-1) + 1 draw (1-1); the scheduled match is excluded.
    assert "1W-1D-0L" in out
    assert "2-1" in out and "1-1" in out


def test_fifa_team_xg_aggregates(monkeypatch):
    matches = {
        "data": [
            {"id": 10, "status": "completed", "datetime": "2026-06-10T18:00:00Z",
             "home_team": {"id": 1}, "away_team": {"id": 2},
             "home_score": 2, "away_score": 1},
            {"id": 11, "status": "completed", "datetime": "2026-06-05T18:00:00Z",
             "home_team": {"id": 1}, "away_team": {"id": 3},
             "home_score": 0, "away_score": 0},
        ]
    }
    team_stats = {
        "data": [
            {"match_id": 10, "team_id": 1, "expected_goals": 2.0, "possession_pct": 60, "shots_on_target": 6},
            {"match_id": 10, "team_id": 2, "expected_goals": 0.8, "possession_pct": 40, "shots_on_target": 2},
            {"match_id": 11, "team_id": 1, "expected_goals": 2.0, "possession_pct": 60, "shots_on_target": 6},
            {"match_id": 11, "team_id": 3, "expected_goals": 0.8, "possession_pct": 40, "shots_on_target": 2},
        ]
    }
    s = _patch_fifa(monkeypatch, {
        "/teams": _FIFA_TEAMS,
        "/matches": matches,
        "/team_match_stats": team_stats,
    })
    data = s._team_xg_data("Brazil", last_n=5)
    assert data is not None
    assert data["games"] == 2
    assert data["xg_for_pg"] == pytest.approx(2.0)
    assert data["xg_against_pg"] == pytest.approx(0.8)
    out = s.get_team_xg("Brazil")
    assert "2.00 xG for/game" in out and "0.80 xG against/game" in out


def test_fifa_group_standings_renders(monkeypatch):
    standings = {
        "data": [
            {"group": {"name": "A"}, "position": 1, "team": {"name": "Brazil"},
             "points": 7, "won": 2, "drawn": 1, "lost": 0,
             "goals_for": 5, "goals_against": 1, "goal_difference": 4},
            {"group": {"name": "A"}, "position": 2, "team": {"name": "Spain"},
             "points": 4, "won": 1, "drawn": 1, "lost": 1,
             "goals_for": 3, "goals_against": 3, "goal_difference": 0},
        ]
    }
    s = _patch_fifa(monkeypatch, {"/group_standings": standings})
    out = s.get_group_standings()
    assert "Group A" in out
    assert "Brazil" in out and "7 pts" in out
    assert "GD +4" in out


def test_fifa_schedule_for_date_filters(monkeypatch):
    matches = {
        "data": [
            {"id": 10, "status": "scheduled", "datetime": "2026-06-12T18:00:00Z",
             "home_team": {"name": "Brazil"}, "away_team": {"name": "Argentina"},
             "stage": {"name": "Group Stage"}, "group": {"name": "A"}},
            {"id": 11, "status": "scheduled", "datetime": "2026-06-13T18:00:00Z",
             "home_team": {"name": "Spain"}, "away_team": {"name": "France"},
             "stage": {"name": "Group Stage"}, "group": {"name": "B"}},
        ]
    }
    s = _patch_fifa(monkeypatch, {"/matches": matches})
    games = s.get_schedule_for_date("2026-06-12")
    assert len(games) == 1
    assert games[0]["home"] == "Brazil" and games[0]["away"] == "Argentina"


def test_fifa_stats_fail_open(monkeypatch):
    # All API calls return None → every tool returns a placeholder, never raises.
    from sportagent.sports.soccer import stats as s

    s._TEAMS_CACHE = None
    s._TEAM_ID_CACHE.clear()
    monkeypatch.setattr(s, "_bdl_get", lambda *a, **k: None)
    assert s.get_group_standings().startswith("<")
    assert s.get_recent_form("Brazil").startswith("<")
    assert s.get_team_xg("Brazil").startswith("<")
    assert s.get_schedule_for_date("2026-06-12") == []
