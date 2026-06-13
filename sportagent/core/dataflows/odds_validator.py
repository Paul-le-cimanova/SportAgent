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
    outcome_structure: str = "two_way",
    contracts: Optional[dict] = None,
) -> str:
    """Render a ground-truth odds snapshot for a 2-way or 3-way market.

    Pulls the Kalshi YES price for ``market_ticker`` (implied prob for
    ``target_team``) and the sportsbook consensus for the matchup, converts
    both to implied probability, and reports any discrepancy. The Odds Analyst
    treats this block as the source of truth for exact prices.

    For ``outcome_structure == "three_way"`` (soccer), ``contracts`` maps
    ``home``/``draw``/``away`` to Kalshi tickers and all three legs are rendered
    against the de-vigged 3-way sportsbook consensus.
    """
    if outcome_structure == "three_way":
        return _build_three_way_snapshot(
            home_team=home_team,
            away_team=away_team,
            sport_key=sport_key,
            contracts=contracts or {},
            config=config,
        )
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


def _build_three_way_snapshot(
    *,
    home_team: str,
    away_team: str,
    sport_key: str,
    contracts: dict,
    config: Optional[dict] = None,
) -> str:
    """Render a 3-way (home/draw/away) verified-odds snapshot for soccer.

    Each leg shows the Kalshi YES implied probability for its contract beside
    the de-vigged 3-way sportsbook consensus. The Odds Analyst treats this as
    the source of truth for exact prices and the market-implied probability
    vector (including the often-mispriced draw).
    """
    lines = [f"## Verified odds snapshot (3-way) — {home_team} vs {away_team}"]
    lines.append("Outcomes: HOME win / DRAW / AWAY win (probabilities sum to 1).")

    # --- Kalshi side: one contract per leg. ---
    kalshi_legs: dict[str, Optional[float]] = {"home": None, "draw": None, "away": None}
    for label in ("home", "draw", "away"):
        ticker = contracts.get(label, "")
        if not ticker:
            lines.append(f"Kalshi {label}: <no contract resolved>")
            continue
        market = kalshi.get_market(ticker, config)
        if "error" in market:
            lines.append(f"Kalshi {label} ({ticker}): <unavailable: {market['error']}>")
            continue
        price = kalshi.extract_price_cents(market)
        if price is None:
            lines.append(f"Kalshi {label} ({ticker}): <no usable price>")
            continue
        p = prob.implied_prob(price)
        kalshi_legs[label] = p
        lines.append(
            f"Kalshi {label} ({ticker}): YES {price}c → implied {p * 100:.1f}%"
        )

    # --- Sportsbook side: de-vigged 3-way consensus. ---
    from sportagent.core.dataflows.odds_api import (
        _consensus_h2h_probs,
        _fetch_h2h,
        _match_event,
    )

    book_legs: dict[str, Optional[float]] = {"home": None, "draw": None, "away": None}
    events = _fetch_h2h(sport_key)
    if isinstance(events, str):
        lines.append(f"Sportsbook: {events}")
    else:
        event = _match_event(events, home_team, away_team) if isinstance(events, list) else None
        if not event:
            lines.append(
                f"Sportsbook: <no match found for {home_team} vs {away_team} in {sport_key}>"
            )
        else:
            probs = _consensus_h2h_probs(event)
            if probs:
                h_l, a_l = home_team.lower(), away_team.lower()
                for name, p in probs.items():
                    n = name.lower()
                    if "draw" in n or "tie" in n:
                        book_legs["draw"] = p
                    elif h_l in n or any(tok in n for tok in h_l.split()):
                        book_legs["home"] = p
                    elif a_l in n or any(tok in n for tok in a_l.split()):
                        book_legs["away"] = p
                consensus = " / ".join(
                    f"{n} {p * 100:.1f}%" for n, p in sorted(probs.items(), key=lambda kv: -kv[1])
                )
                lines.append(f"Sportsbook consensus (3-way, vig-removed): {consensus}")
            else:
                lines.append("Sportsbook: <no usable 3-way odds>")

    # --- Per-leg source of truth. ---
    lines.append("PRICE SOURCE OF TRUTH (per leg, prefer Kalshi, else sportsbook):")
    for label in ("home", "draw", "away"):
        k = kalshi_legs[label]
        b = book_legs[label]
        if k is not None and b is not None:
            lines.append(
                f"  {label}: Kalshi implied {k * 100:.1f}% "
                f"(sportsbook {b * 100:.1f}% cross-check)"
            )
        elif k is not None:
            lines.append(f"  {label}: Kalshi implied {k * 100:.1f}% (no sportsbook cross-check)")
        elif b is not None:
            lines.append(
                f"  {label}: FALLBACK sportsbook de-vigged {b * 100:.1f}% "
                f"(Kalshi price unavailable)"
            )
        else:
            lines.append(f"  {label}: <unavailable — treat market-implied as unknown>")

    lines.append(
        "Estimate THREE true probabilities (home/draw/away) that sum to 1, then "
        "the Trader compares each against its leg price above and bets the "
        "best-edge leg (incl. the draw) or HOLDs. Use these prices as the source "
        "of truth; flag conflicts rather than inventing reconciled numbers."
    )
    return "\n".join(lines)
