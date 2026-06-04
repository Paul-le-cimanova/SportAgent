"""Deterministic verified-odds snapshot (anti-hallucination source of truth).

The Odds Analyst is an LLM that can confabulate exact prices. This module
computes a ground-truth snapshot — Kalshi YES/NO contract price (→ implied
probability) cross-checked against the de-vigged sportsbook consensus — that
the analyst is instructed to treat as authoritative for any exact price claim.
Deterministic, no LLM involved.

Fails open: returns a snapshot string that clearly marks any unavailable
source rather than raising.
"""

from __future__ import annotations

import logging
from typing import Optional

from sportagent.core.agents.utils import probability as prob
from sportagent.core.dataflows import kalshi

logger = logging.getLogger(__name__)


def build_verified_odds_snapshot(
    *,
    market_ticker: str,
    target_team: str,
    home_team: str,
    away_team: str,
    sport_key: str,
    config: Optional[dict] = None,
) -> str:
    """Render a ground-truth odds snapshot for a 2-way market.

    Pulls the Kalshi YES price for ``market_ticker`` (implied prob for
    ``target_team``) and the sportsbook consensus for the matchup, converts
    both to implied probability, and reports any discrepancy. The Odds Analyst
    treats this block as the source of truth for exact prices.
    """
    lines = [f"## Verified odds snapshot — {away_team} @ {home_team}"]
    lines.append(f"Target (YES resolves on): {target_team}")

    # --- Kalshi side ---
    market = kalshi.get_market(market_ticker, config)
    kalshi_prob: Optional[float] = None
    if "error" in market:
        lines.append(f"Kalshi ({market_ticker}): <unavailable: {market['error']}>")
    else:
        price = kalshi.extract_price_cents(market)
        if price is None:
            lines.append(f"Kalshi ({market_ticker}): <no usable price>")
        else:
            kalshi_prob = prob.implied_prob(price)
            lines.append(
                f"Kalshi ({market_ticker}): YES ({target_team}) {price}c "
                f"→ implied {kalshi_prob * 100:.1f}%"
            )

    # --- Sportsbook side ---
    # Lazy import to avoid a circular import when the dataflows package eagerly
    # imports vendor modules at init time.
    from sportagent.core.dataflows.odds_api import (
        _consensus_h2h_probs,
        _fetch_h2h,
        _match_event,
    )

    book_prob: Optional[float] = None
    events = _fetch_h2h(sport_key)
    if isinstance(events, str):
        lines.append(f"Sportsbook: {events}")
    else:
        event = _match_event(events, home_team, away_team) if isinstance(events, list) else None
        if not event:
            lines.append(
                f"Sportsbook: <no match found for {away_team} @ {home_team} in {sport_key}>"
            )
        else:
            probs = _consensus_h2h_probs(event)
            if probs:
                # Map the target team to its consensus probability (substring).
                t = target_team.lower()
                for name, p in probs.items():
                    if t in name.lower() or any(tok in name.lower() for tok in t.split()):
                        book_prob = p
                        break
                consensus = " / ".join(
                    f"{n} {p * 100:.1f}%" for n, p in sorted(probs.items(), key=lambda kv: -kv[1])
                )
                lines.append(f"Sportsbook consensus (vig-removed): {consensus}")
            else:
                lines.append("Sportsbook: <no usable h2h odds>")

    # --- Discrepancy / price source of truth ---
    if kalshi_prob is not None and book_prob is not None:
        diff_pp = (kalshi_prob - book_prob) * 100
        direction = "above" if diff_pp >= 0 else "below"
        lines.append(
            f"Discrepancy: Kalshi {abs(diff_pp):.1f}pp {direction} book consensus on {target_team}."
        )
        lines.append(
            f"PRICE SOURCE OF TRUTH: Kalshi YES implied {kalshi_prob * 100:.1f}% "
            f"on {target_team} (sportsbook consensus {book_prob * 100:.1f}% as cross-check)."
        )
    elif kalshi_prob is None and book_prob is not None:
        # Kalshi had no usable price — fall back to the de-vigged sportsbook
        # consensus as the implied-probability source of truth, clearly labelled.
        lines.append(
            f"PRICE SOURCE OF TRUTH (FALLBACK): Kalshi has no usable price, so use "
            f"the de-vigged sportsbook consensus — implied {book_prob * 100:.1f}% "
            f"on {target_team} — as the market-implied probability."
        )
    elif kalshi_prob is not None and book_prob is None:
        lines.append(
            f"PRICE SOURCE OF TRUTH: Kalshi YES implied {kalshi_prob * 100:.1f}% "
            f"on {target_team} (no sportsbook cross-check available)."
        )
    else:
        lines.append(
            "PRICE SOURCE OF TRUTH: <unavailable — neither Kalshi nor sportsbook "
            "consensus returned a usable price; treat market-implied probability "
            "as unknown and lean on the model estimate>."
        )

    lines.append(
        "Use this snapshot as the source of truth for exact prices and implied "
        "probabilities. If another tool conflicts with it, flag the discrepancy "
        "rather than inventing a reconciled number."
    )
    return "\n".join(lines)
