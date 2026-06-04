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


def test_clamp_prob():
    assert p.clamp_prob(-0.5) == 0.0
    assert p.clamp_prob(1.5) == 1.0
    assert p.clamp_prob(0.42) == pytest.approx(0.42)