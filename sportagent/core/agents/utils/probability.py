"""Deterministic probability & sizing math for SportAgent.

All quantitative logic lives here, never in an LLM. Functions convert market
prices to implied probabilities, remove sportsbook vig, compute edge, size
positions via fractional/capped Kelly, and score calibration via Brier.

Conventions:
- Kalshi contract prices are in **cents** (1-99); a YES price of 58c implies a
  58% probability.
- Probabilities are floats in [0.0, 1.0].
- Edge is ``estimate - implied`` (positive = model thinks the side is
  underpriced by the market).
"""

from __future__ import annotations

from typing import Sequence, Tuple


def implied_prob(price_cents: float) -> float:
    """Convert a Kalshi contract price (cents, 1-99) to implied probability.

    >>> implied_prob(58)
    0.58
    """
    return max(0.0, min(1.0, float(price_cents) / 100.0))


def american_to_prob(moneyline: float) -> float:
    """Convert an American moneyline to a raw (vig-inclusive) probability.

    Positive lines (underdogs): 100 / (ml + 100).
    Negative lines (favorites): -ml / (-ml + 100).

    >>> round(american_to_prob(-150), 4)
    0.6
    >>> round(american_to_prob(150), 4)
    0.4
    """
    ml = float(moneyline)
    if ml < 0:
        return (-ml) / ((-ml) + 100.0)
    return 100.0 / (ml + 100.0)


def devig(prob_a: float, prob_b: float) -> Tuple[float, float]:
    """Remove vig from a two-sided book by normalizing probs to sum to 1.

    >>> a, b = devig(0.6, 0.45)
    >>> round(a + b, 6)
    1.0
    """
    total = prob_a + prob_b
    if total <= 0:
        return 0.5, 0.5
    return prob_a / total, prob_b / total


def devig_multi(probs: Sequence[float]) -> list[float]:
    """Remove vig across N outcomes (e.g. soccer win/draw/loss) → sum to 1.

    >>> out = devig_multi([0.5, 0.3, 0.3])
    >>> round(sum(out), 6)
    1.0
    """
    total = sum(probs)
    if total <= 0:
        n = len(probs) or 1
        return [1.0 / n] * len(probs)
    return [p / total for p in probs]


def edge(estimate: float, implied: float) -> float:
    """Model probability minus market-implied probability.

    Positive means the model thinks the outcome is underpriced.

    >>> round(edge(0.64, 0.58), 4)
    0.06
    """
    return float(estimate) - float(implied)


def kelly_fraction(estimate: float, price_cents: float, cap: float = 0.25) -> float:
    """Fractional, capped Kelly stake for a YES contract at ``price_cents``.

    For a binary contract priced at ``p`` (probability units) that pays 1 on
    win, full-Kelly fraction f* = (estimate - p) / (1 - p). We then apply a
    fractional cap (default quarter-Kelly) and floor negative values at 0.

    >>> round(kelly_fraction(0.64, 58, cap=1.0), 4)  # full Kelly
    0.1429
    >>> round(kelly_fraction(0.64, 58, cap=0.25), 4)  # quarter Kelly
    0.0357
    >>> kelly_fraction(0.50, 58)  # no edge → no stake
    0.0
    """
    p = implied_prob(price_cents)
    if p >= 1.0 or p <= 0.0:
        return 0.0
    full = (float(estimate) - p) / (1.0 - p)
    if full <= 0:
        return 0.0
    return full * cap


def recommended_stake(
    estimate: float,
    price_cents: float,
    kelly_cap: float = 0.25,
    max_stake_pct: float = 0.05,
) -> float:
    """Recommended stake as a fraction of bankroll, capped by ``max_stake_pct``.

    >>> round(recommended_stake(0.70, 50, kelly_cap=0.5, max_stake_pct=0.05), 4)
    0.05
    """
    f = kelly_fraction(estimate, price_cents, cap=kelly_cap)
    return min(f, max_stake_pct)


def brier(estimate: float, outcome: int) -> float:
    """Brier score for a single binary prediction (lower = better calibrated).

    ``outcome`` is 1 if the predicted side won, else 0.

    >>> brier(0.64, 1)
    0.1296
    >>> brier(0.64, 0)
    0.4096
    """
    return (float(estimate) - float(outcome)) ** 2


def brier_multi(estimates: Sequence[float], outcome_index: int) -> float:
    """Multi-class Brier score (e.g. soccer home/draw/away).

    ``estimates`` is the probability vector; ``outcome_index`` is the index of
    the realized outcome.

    >>> round(brier_multi([0.5, 0.3, 0.2], 0), 4)
    0.38
    """
    total = 0.0
    for i, p in enumerate(estimates):
        target = 1.0 if i == outcome_index else 0.0
        total += (p - target) ** 2
    return total


def clamp_prob(p: float) -> float:
    """Clamp a probability into [0.0, 1.0]."""
    return max(0.0, min(1.0, float(p)))