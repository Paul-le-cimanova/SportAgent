"""Tests for the deterministic probability & sizing layer."""

import pytest

from sportagent.core.agents.utils import probability as p


def test_implied_prob():
    assert p.implied_prob(58) == pytest.approx(0.58)
    assert p.implied_prob(1) == pytest.approx(0.01)
    assert p.implied_prob(99) == pytest.approx(0.99)


def test_implied_prob_clamps():
    assert p.implied_prob(0) == 0.0
    assert p.implied_prob(150) == 1.0


def test_american_to_prob_favorite():
    assert p.american_to_prob(-150) == pytest.approx(0.6)


def test_american_to_prob_underdog():
    assert p.american_to_prob(150) == pytest.approx(0.4)


def test_devig_sums_to_one():
    a, b = p.devig(0.6, 0.45)
    assert a + b == pytest.approx(1.0)
    assert a > b  # 0.6 side stays the bigger probability


def test_devig_zero_total():
    assert p.devig(0.0, 0.0) == (0.5, 0.5)


def test_devig_multi_sums_to_one():
    out = p.devig_multi([0.5, 0.3, 0.3])
    assert sum(out) == pytest.approx(1.0)
    assert len(out) == 3


def test_devig_multi_zero_total():
    out = p.devig_multi([0.0, 0.0, 0.0])
    assert out == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_devig_3way_sums_to_one():
    h, d, a = p.devig_3way(0.5, 0.3, 0.3)
    assert h + d + a == pytest.approx(1.0)
    assert h > a  # the 0.5 side stays the biggest probability


def test_devig_3way_zero_total():
    h, d, a = p.devig_3way(0.0, 0.0, 0.0)
    assert (h, d, a) == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_kelly_3way_labels_and_edges():
    legs = p.kelly_3way([0.50, 0.30, 0.20], [40, 28, 22])
    assert [leg.label for leg in legs] == ["home", "draw", "away"]
    assert legs[0].edge == pytest.approx(0.10)
    assert legs[1].edge == pytest.approx(0.02)
    assert legs[2].edge == pytest.approx(-0.02)
    # Positive-edge legs get a stake; negative-edge legs do not.
    assert legs[0].stake_pct > 0.0
    assert legs[2].stake_pct == 0.0


def test_best_leg_picks_largest_positive_edge():
    leg = p.best_leg([0.50, 0.30, 0.20], [40, 28, 22])
    assert leg is not None
    assert leg.label == "home"
    assert leg.edge == pytest.approx(0.10)


def test_best_leg_can_pick_the_draw():
    # Draw is the most underpriced leg here (0.34 est vs 0.25 implied).
    leg = p.best_leg([0.40, 0.34, 0.26], [40, 25, 30])
    assert leg is not None
    assert leg.label == "draw"


def test_best_leg_holds_when_no_edge_clears_band():
    assert p.best_leg([0.40, 0.30, 0.30], [40, 30, 30]) is None


def test_best_leg_respects_no_trade_band():
    # A 2pp edge does NOT clear the default 3pp band → HOLD.
    assert p.best_leg([0.42, 0.30, 0.28], [40, 30, 30]) is None
    # But a wider band-free call surfaces it.
    leg = p.best_leg([0.42, 0.30, 0.28], [40, 30, 30], no_trade_band=0.0)
    assert leg is not None
    assert leg.label == "home"


def test_edge():
    assert p.edge(0.64, 0.58) == pytest.approx(0.06)
    assert p.edge(0.50, 0.58) == pytest.approx(-0.08)


def test_kelly_full():
    # full Kelly at estimate 0.64, price 58c
    assert p.kelly_fraction(0.64, 58, cap=1.0) == pytest.approx(0.142857, abs=1e-4)


def test_kelly_quarter():
    assert p.kelly_fraction(0.64, 58, cap=0.25) == pytest.approx(0.035714, abs=1e-4)


def test_kelly_no_edge_is_zero():
    assert p.kelly_fraction(0.50, 58) == 0.0
    assert p.kelly_fraction(0.40, 58) == 0.0


def test_kelly_degenerate_price():
    assert p.kelly_fraction(0.9, 100) == 0.0
    assert p.kelly_fraction(0.9, 0) == 0.0


def test_recommended_stake_capped():
    # large edge would exceed max_stake_pct → capped
    assert p.recommended_stake(0.70, 50, kelly_cap=0.5, max_stake_pct=0.05) == pytest.approx(0.05)


def test_recommended_stake_below_cap():
    s = p.recommended_stake(0.60, 58, kelly_cap=0.25, max_stake_pct=0.05)
    assert 0.0 <= s <= 0.05


def test_brier():
    assert p.brier(0.64, 1) == pytest.approx(0.1296)
    assert p.brier(0.64, 0) == pytest.approx(0.4096)


def test_brier_multi():
    assert p.brier_multi([0.5, 0.3, 0.2], 0) == pytest.approx(0.38)
    # perfect prediction → 0
    assert p.brier_multi([1.0, 0.0, 0.0], 0) == pytest.approx(0.0)


def test_brier_multi_draw_outcome():
    # Settled result is the draw (index 1).
    assert p.brier_multi([0.5, 0.3, 0.2], 1) == pytest.approx(0.78)


def test_clamp_prob():
    assert p.clamp_prob(-0.5) == 0.0
    assert p.clamp_prob(1.5) == 1.0
    assert p.clamp_prob(0.42) == pytest.approx(0.42)