"""Tests for the new quantitative stats layer + the trader probability-parse fix.

Three things are covered:

1. ``trader._extract_prob`` — the regression that silently fell back to 0.5 when
   the edge thesis rendered the estimate as ``0.610 (61.0%)`` (decimal followed
   by a parenthesized percent). This is the bug that made the Trader read 50%
   while the Research Manager emitted 61%.
2. ``stats._elo_winprob_data`` — deterministic Elo math (home-court, symmetry).
3. ``stats._four_factors_data`` — Four-Factors aggregation over paginated,
   per-player box-score rows (the small-sample pagination fix).

All network access is monkeypatched, so these run offline and deterministically.
"""

import pytest

from sportagent.core.agents.trader.trader import _extract_prob
from sportagent.sports.nba import stats


# --- Trader probability parsing (the plumbing bug) --------------------------


def test_extract_prob_decimal_then_percent():
    """The exact edge-thesis render shape that triggered the 0.5 fallback bug."""
    text = "**Estimated Probability (target YES):** 0.610 (61.0%)"
    assert _extract_prob(text, "Estimated Probability") == pytest.approx(0.61)


def test_extract_prob_percent_only():
    assert _extract_prob(
        "Estimated Probability: 61.0%", "Estimated Probability"
    ) == pytest.approx(0.61)


def test_extract_prob_decimal_only():
    assert _extract_prob(
        "Estimated Probability 0.58", "Estimated Probability"
    ) == pytest.approx(0.58)


def test_extract_prob_fifty_fifty():
    assert _extract_prob(
        "Estimated Probability: 0.500 (50.0%)", "Estimated Probability"
    ) == pytest.approx(0.50)


def test_extract_prob_missing_label_defaults_half():
    assert _extract_prob("no probability stated here", "Estimated Probability") == 0.5


def test_extract_prob_full_certainty():
    assert _extract_prob(
        "Estimated Probability: 1.000 (100.0%)", "Estimated Probability"
    ) == pytest.approx(1.0)


# --- Elo win probability ----------------------------------------------------


def test_elo_winprob_equal_ratings_is_home_edge(monkeypatch):
    """Equal Elo → home team favored purely by the home-court adjustment."""
    monkeypatch.setattr(stats, "_resolve_team_id", lambda t: 1 if "home" in t else 2)
    monkeypatch.setattr(stats, "_season_elo_ratings", lambda s: {1: 1500.0, 2: 1500.0})
    data = stats._elo_winprob_data("home", "away", season=2025, hca=60.0)
    assert data is not None
    # +60 Elo home edge → ~58.6% home.
    assert data["home_winprob"] == pytest.approx(0.5862, abs=1e-3)
    assert data["home_winprob"] > 0.5


def test_elo_winprob_symmetry(monkeypatch):
    """Stronger away team can overcome home-court when the gap is large enough."""
    monkeypatch.setattr(stats, "_resolve_team_id", lambda t: 1 if "home" in t else 2)
    monkeypatch.setattr(stats, "_season_elo_ratings", lambda s: {1: 1400.0, 2: 1600.0})
    data = stats._elo_winprob_data("home", "away", season=2025, hca=60.0)
    assert data is not None
    # Away is 200 Elo better, home gets +60 → net -140 for home → home underdog.
    assert data["home_winprob"] < 0.5


def test_elo_winprob_unresolved_team_is_none(monkeypatch):
    monkeypatch.setattr(stats, "_resolve_team_id", lambda t: None)
    assert stats._elo_winprob_data("x", "y") is None


# --- Four Factors aggregation (pagination + rate math) ----------------------


def _fake_stats_pages(team_id, n_games, players_per_game=3):
    """Build paginated /stats responses: one row per player per game.

    Each player contributes identical lines so the team totals are easy to
    assert. Pages hold 100 rows each (mirrors balldontlie's per_page cap).
    """
    rows = []
    for gid in range(n_games):
        for _ in range(players_per_game):
            rows.append({
                "team": {"id": team_id},
                "game": {"id": gid},
                "fgm": 10, "fga": 20, "fg3m": 4,
                "ftm": 5, "fta": 6, "oreb": 3, "turnover": 2, "pts": 29,
            })
    # Chunk into pages of 100 with cursors.
    pages = []
    for i in range(0, len(rows), 100):
        chunk = rows[i:i + 100]
        nxt = i + 100 if i + 100 < len(rows) else None
        pages.append({"data": chunk, "meta": {"next_cursor": nxt}})
    return pages


def test_four_factors_paginates_and_aggregates(monkeypatch):
    """20 games × 3 players = 60 rows over one page → all 20 games counted."""
    team_id = 7
    monkeypatch.setattr(stats, "_resolve_team_id", lambda t: team_id)

    finals = {"data": [
        {"id": i, "status": "Final", "date": f"2026-01-{i + 1:02d}"}
        for i in range(20)
    ]}
    pages = _fake_stats_pages(team_id, n_games=20)
    page_iter = iter(pages)

    def fake_get(path, params=None):
        if path == "/games":
            return finals
        if path == "/stats":
            return next(page_iter)
        return None

    monkeypatch.setattr(stats, "_bdl_get", fake_get)

    data = stats._four_factors_data("anyteam", season=2025, last_n=20)
    assert data is not None
    assert data["games"] == 20
    # Per-game team line ×3 players: fga=60, fgm=30, fg3m=12, ftm=15, fta=18, tov=6.
    # Totals over 20 games scale linearly; rates are scale-invariant:
    # eFG% = (30 + 0.5*12) / 60 = 0.60
    assert data["efg"] == pytest.approx(0.60)
    # FT-rate = ftm/fga = 15/60 = 0.25
    assert data["ft_rate"] == pytest.approx(0.25)
    # TOV-rate = tov / (fga + 0.44*fta + tov) = 6 / (60 + 7.92 + 6) = 0.08116
    assert data["tov_rate"] == pytest.approx(6 / (60 + 0.44 * 18 + 6), abs=1e-4)


def test_four_factors_multipage(monkeypatch):
    """>100 rows must paginate; ensure cursor following gathers every game."""
    team_id = 9
    monkeypatch.setattr(stats, "_resolve_team_id", lambda t: team_id)
    finals = {"data": [
        {"id": i, "status": "Final", "date": f"2026-02-{i + 1:02d}"}
        for i in range(20)
    ]}
    # 20 games × 12 players = 240 rows → 3 pages of 100/100/40.
    pages = _fake_stats_pages(team_id, n_games=20, players_per_game=12)
    assert len(pages) == 3  # sanity: it really did span multiple pages
    page_iter = iter(pages)

    def fake_get(path, params=None):
        if path == "/games":
            return finals
        if path == "/stats":
            return next(page_iter)
        return None

    monkeypatch.setattr(stats, "_bdl_get", fake_get)
    data = stats._four_factors_data("anyteam", season=2025, last_n=20)
    assert data is not None
    assert data["games"] == 20  # all games counted despite 3-page span


def test_four_factors_no_games_is_none(monkeypatch):
    monkeypatch.setattr(stats, "_resolve_team_id", lambda t: 1)
    monkeypatch.setattr(stats, "_bdl_get", lambda path, params=None: {"data": []})
    assert stats._four_factors_data("anyteam", season=2025) is None