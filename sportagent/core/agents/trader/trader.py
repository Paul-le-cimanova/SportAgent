"""Trader — converts the edge thesis into a concrete PositionProposal.

Quick-tier agent. Reads the investment plan (edge thesis), the verified-odds
snapshot, and the game context. The deterministic ``probability.py`` helpers
compute implied probability, edge, and the fractional/capped Kelly stake — the
LLM is given these numbers and instructed to use them rather than guessing. The
trader picks BUY YES / BUY NO / HOLD (HOLD inside the no-trade band) and emits a
structured ``PositionProposal`` (free-text fallback) → ``trader_position_plan``.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from sportagent.core.agents.schemas import PositionProposal, render_position_proposal
from sportagent.core.agents.utils import probability as prob
from sportagent.core.agents.utils.agent_utils import (
    get_game_context_from_state,
    get_language_instruction,
)
from sportagent.core.agents.utils.structured import invoke_structured_or_freetext


_SYSTEM = (
    "You are the Trader on a sports prediction-market desk. Convert the "
    "Research Manager's edge thesis into a concrete position: BUY YES, BUY NO, "
    "or HOLD.\n\n"
    "{game_context}\n\n"
    "Use the DETERMINISTIC numbers below for sizing — do not invent your own. "
    "The Kalshi YES price implies {implied_pct}; the edge thesis estimates the "
    "true probability at {estimate_pct}; the computed edge is {edge_pp} and the "
    "fractional/capped Kelly stake is {stake_pct} of bankroll. HOLD if the edge "
    "is within the no-trade band of {no_trade_band_pct}. If the estimate is "
    "above the price, the value is on YES; if below, on NO.{language}\n\n"
    "=== Verified odds snapshot ===\n{verified_odds}\n\n"
    "=== Investment plan (edge thesis) ===\n{investment_plan}\n"
)


def _extract_prob(text: str, label: str) -> float:
    """Parse the probability following ``label`` from an edge-thesis block.

    The Research Manager renders the estimate as::

        **Estimated Probability (target YES):** 0.610 (61.0%)

    so the decimal (``0.610``) is the canonical value and the parenthesized
    percentage (``61.0%``) is a human-readable echo. The previous regex tried to
    read the first number *then* a trailing ``%`` — which silently failed here
    because ``0.610`` is not followed by ``%`` (it is followed by the decimal's
    own paren), so it fell through to the 0.5 default and corrupted every
    downstream edge/Kelly calc. We now parse, in priority order:

    1. a decimal in [0, 1] immediately after the label (``0.610`` → 0.610), then
    2. a percentage anywhere after the label (``61.0%`` → 0.61),

    and only fall back to 0.5 if neither is present.
    """
    after_label = re.search(re.escape(label) + r"(.*)", text)
    segment = after_label.group(1) if after_label else text

    # 1) Percentage echo first when present (``61.0%`` → 0.61). Checked before
    #    the bare decimal so a value like ``61.0%`` is never mis-read as the
    #    leading ``6``/``1`` digits of a decimal token.
    pct = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", segment)
    if pct:
        return prob.clamp_prob(float(pct.group(1)) / 100.0)

    # 2) Canonical decimal form: a 0.xxx (or 1.0) token. Require the decimal
    #    point so a bare integer is not mistaken for a probability.
    dec = re.search(r"(0\.[0-9]+|1\.0+|0|1)\b", segment)
    if dec:
        val = float(dec.group(1))
        if 0.0 <= val <= 1.0:
            return prob.clamp_prob(val)

    return 0.5


def _extract_kalshi_price_cents(verified_odds: str) -> float:
    """Parse the Kalshi YES price in cents from the snapshot (fallback 50)."""
    match = re.search(r"YES[^0-9]*([0-9]{1,2})c", verified_odds)
    if match:
        return float(match.group(1))
    return 50.0


def create_trader(llm):
    """Create the Trader node (quick tier, structured output)."""

    def trader_node(state):
        from sportagent.core.dataflows.config import get_config

        config = get_config()
        no_trade_band = config.get("no_trade_band", 0.03)
        kelly_cap = config.get("kelly_cap", 0.25)
        max_stake_pct = config.get("max_stake_pct", 0.05)

        verified_odds = state.get("verified_odds", "")
        investment_plan = state.get("investment_plan", "")

        # Deterministic quantitative inputs (never left to the LLM).
        price_cents = _extract_kalshi_price_cents(verified_odds)
        implied = prob.implied_prob(price_cents)
        estimate = _extract_prob(investment_plan, "Estimated Probability")
        edge_val = prob.edge(estimate, implied)
        # Size on whichever side the edge favors; for a NO lean we'd size on the
        # NO contract symmetrically, so use the magnitude of the directional edge.
        if edge_val >= 0:
            stake = prob.recommended_stake(
                estimate, price_cents, kelly_cap=kelly_cap, max_stake_pct=max_stake_pct
            )
        else:
            no_price_cents = max(1.0, 100.0 - price_cents)
            stake = prob.recommended_stake(
                1.0 - estimate, no_price_cents,
                kelly_cap=kelly_cap, max_stake_pct=max_stake_pct,
            )
        if abs(edge_val) < no_trade_band:
            stake = 0.0

        system = _SYSTEM.format(
            game_context=get_game_context_from_state(state),
            language=get_language_instruction(),
            implied_pct=f"{implied * 100:.1f}%",
            estimate_pct=f"{estimate * 100:.1f}%",
            edge_pp=f"{edge_val * 100:+.1f}pp",
            stake_pct=f"{stake * 100:.2f}%",
            no_trade_band_pct=f"{no_trade_band * 100:.1f}pp",
            verified_odds=verified_odds,
            investment_plan=investment_plan,
        )
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=(
                "Propose the position: action, reasoning, estimated "
                "probability, edge, and suggested stake (use the deterministic "
                "numbers given)."
            )),
        ]
        markdown, _parsed = invoke_structured_or_freetext(
            llm, PositionProposal, messages, render_position_proposal
        )
        return {"trader_position_plan": markdown}

    return trader_node