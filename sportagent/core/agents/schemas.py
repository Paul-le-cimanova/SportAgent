"""Pydantic schemas for SportAgent's structured-output agents.

Four agents emit structured output: the Sentiment Analyst (``SentimentReport``),
the Research Manager (``EdgeThesis``), the Trader (``PositionProposal``), and the
Decision Manager (``FinalRecommendation``). Each schema's field descriptions
double as the model's output instructions; each has a ``render_*()`` helper that
turns the parsed instance back into the markdown shape the rest of the pipeline
(state fields, saved reports, memory log) already consumes.

All probability fields are in [0, 1]; edge is ``estimate - implied``; stake
percentages are bankroll fractions in [0, 1].
"""

from __future__ import annotations

from enum import Enum
from typing import List, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared vocabularies
# ---------------------------------------------------------------------------


class Action(str, Enum):
    """The terminal action vocabulary (see design doc 06 §7)."""

    BUY_YES = "BUY YES"   # model prob materially exceeds price → back target
    BUY_NO = "BUY NO"     # model prob materially below price → fade target
    HOLD = "HOLD"         # edge within the no-trade band; fairly priced


class Lean(str, Enum):
    """Research Manager's directional lean for the target outcome."""

    YES = "YES"
    NO = "NO"
    NO_EDGE = "NO-EDGE"


class SentimentBand(str, Enum):
    """Public/community lean toward the target team."""

    STRONG_FOR = "Strong For"
    LEAN_FOR = "Lean For"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    LEAN_AGAINST = "Lean Against"
    STRONG_AGAINST = "Strong Against"


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentReport(BaseModel):
    """Structured public-sentiment read for the target team."""

    band: SentimentBand = Field(
        description=(
            "Overall public/community lean toward the target team. One of: "
            "Strong For / Lean For / Neutral / Mixed / Lean Against / Strong "
            "Against. Use Mixed when sources clearly diverge; Neutral only when "
            "sources are genuinely silent or non-committal."
        ),
    )
    lean_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Sentiment intensity 0–10. 0 = maximally against the target team, "
            "5 = neutral, 10 = maximally for. Be mindful of the contrarian "
            "signal: very heavy public money on one side can flag value on the "
            "other. Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence given data quality/volume. 'low' if a source returned a "
            "placeholder or fewer than ~5 posts; 'medium' if sparse; 'high' if "
            "all sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment write-up: (1) source-by-source breakdown citing "
            "post counts/notable threads; (2) public betting lean and any "
            "contrarian read; (3) dominant narratives; (4) signal vs noise and "
            "data-confidence caveats."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a ``SentimentReport`` to markdown."""
    return "\n".join([
        f"**Public Lean:** **{report.band.value}** "
        f"(Score: {report.lean_score:.1f}/10 toward target)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class EdgeThesis(BaseModel):
    """Research Manager's judged edge thesis (hand-off to the Trader)."""

    lean: Lean = Field(
        description=(
            "Directional lean for the target outcome: YES (true probability "
            "exceeds the market price), NO (market over-rates the target), or "
            "NO-EDGE (no material divergence). Reserve NO-EDGE for genuinely "
            "balanced cases."
        ),
    )
    estimated_probability: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your committed estimate of the true probability that the target "
            "YES contract resolves, as a decimal in [0, 1] (e.g. 0.62)."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational synthesis of the bull/bear debate, ending with which "
            "arguments carried the call. Ground claims in the analyst reports and "
            "the verified-odds snapshot."
        ),
    )
    key_factors: List[str] = Field(
        description=(
            "The 3–6 highest-weight factors behind the estimate (e.g. 'Spurs on "
            "a back-to-back', 'Knicks missing starting C')."
        ),
    )


def render_edge_thesis(thesis: EdgeThesis) -> str:
    """Render an ``EdgeThesis`` to markdown."""
    factors = "\n".join(f"- {f}" for f in thesis.key_factors) or "- (none cited)"
    return "\n".join([
        f"**Lean:** {thesis.lean.value}",
        f"**Estimated Probability (target YES):** {thesis.estimated_probability:.3f} "
        f"({thesis.estimated_probability * 100:.1f}%)",
        "",
        f"**Rationale:** {thesis.rationale}",
        "",
        "**Key Factors:**",
        factors,
    ])


class ThreeWayEdgeThesis(BaseModel):
    """Research Manager's 3-way (soccer) edge thesis — a probability vector.

    Instead of one target-YES probability, soccer requires three estimates —
    home win / draw / away win — that the deterministic ``probability.py`` layer
    normalizes to sum to 1 and the Trader uses to pick the best-edge leg. The
    LLM only estimates the three numbers; it never does the vector arithmetic.
    """

    prob_home: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your committed estimate of the probability the HOME team wins, as a "
            "decimal in [0, 1]. Together with prob_draw and prob_away this should "
            "be a coherent vector (the code normalizes it to sum to 1)."
        ),
    )
    prob_draw: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your committed estimate of the probability the match is a DRAW, as a "
            "decimal in [0, 1]. Do NOT default this to a residual — the draw is a "
            "genuine outcome (commonly 0.22-0.30 in tight matches)."
        ),
    )
    prob_away: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your committed estimate of the probability the AWAY team wins, as a "
            "decimal in [0, 1]."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational synthesis of the bull/bear debate, ending with which "
            "arguments carried the call. Ground claims in the analyst reports and "
            "the verified-odds snapshot (which lists all three leg prices)."
        ),
    )
    key_factors: List[str] = Field(
        description=(
            "The 3-6 highest-weight factors behind the vector (e.g. 'Home side "
            "unbeaten in 6', 'both defenses strong → draw likely', 'away top "
            "scorer suspended')."
        ),
    )


def render_three_way_edge_thesis(thesis: ThreeWayEdgeThesis) -> str:
    """Render a ``ThreeWayEdgeThesis`` to markdown (normalized vector)."""
    from sportagent.core.agents.utils import probability as prob

    h, d, a = prob.devig_3way(thesis.prob_home, thesis.prob_draw, thesis.prob_away)
    factors = "\n".join(f"- {f}" for f in thesis.key_factors) or "- (none cited)"
    return "\n".join([
        "**3-Way Probability Vector (normalized to sum to 1):**",
        f"- Home win: {h:.3f} ({h * 100:.1f}%)",
        f"- Draw: {d:.3f} ({d * 100:.1f}%)",
        f"- Away win: {a:.3f} ({a * 100:.1f}%)",
        "",
        f"**Rationale:** {thesis.rationale}",
        "",
        "**Key Factors:**",
        factors,
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class PositionProposal(BaseModel):
    """Trader's concrete position proposal (sizing comes from probability.py)."""

    action: Action = Field(
        description=(
            "The position to take. Exactly one of BUY YES / BUY NO / HOLD. "
            "HOLD if the edge is within the no-trade band."
        ),
    )
    reasoning: str = Field(
        description=(
            "Two to four sentences converting the edge thesis into a position, "
            "anchored in the verified-odds snapshot and the research plan."
        ),
    )
    estimated_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Your working estimate of the true target-YES probability, [0, 1].",
    )
    edge: float = Field(
        description=(
            "Estimated probability minus the market-implied probability "
            "(estimate - implied). Positive favors BUY YES, negative BUY NO."
        ),
    )
    suggested_stake_pct: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Suggested stake as a fraction of bankroll [0, 1], taken from the "
            "deterministic Kelly helper in probability.py. 0.0 for HOLD."
        ),
    )


def render_position_proposal(proposal: PositionProposal) -> str:
    """Render a ``PositionProposal`` to markdown."""
    return "\n".join([
        f"**Action:** {proposal.action.value}",
        "",
        f"**Reasoning:** {proposal.reasoning}",
        "",
        f"**Estimated Probability:** {proposal.estimated_probability:.3f} "
        f"({proposal.estimated_probability * 100:.1f}%)",
        f"**Edge:** {proposal.edge:+.3f} ({proposal.edge * 100:+.1f}pp)",
        f"**Suggested Stake:** {proposal.suggested_stake_pct * 100:.2f}% of bankroll",
        "",
        f"FINAL POSITION PROPOSAL: **{proposal.action.value}**",
    ])


# ---------------------------------------------------------------------------
# Decision Manager
# ---------------------------------------------------------------------------


class FinalRecommendation(BaseModel):
    """Decision Manager's final recommendation — winner-first (MVP headline).

    The PRIMARY output is a game-winner prediction: which team wins and with
    what probability/confidence. The betting view (Kalshi action / edge / Kelly
    stake) is retained as a secondary, optional section — still computed, just
    no longer the headline.
    """

    # --- Primary: winner prediction (the MVP headline) -----------------------
    predicted_winner: str = Field(
        description=(
            "The exact team name you predict will WIN the game. Use one of the "
            "two teams from the resolved game context — do not invent or "
            "abbreviate. This is the headline of the recommendation."
        ),
    )
    win_probability: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your estimated probability that ``predicted_winner`` wins, as a "
            "decimal in [0, 1] (e.g. 0.64 for 64%). Must be > 0.5 since this is "
            "the team you favor."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the winner call given probability margin and "
            "variance. 'high' for a clear favorite, 'low' for a near coin-flip."
        ),
    )
    reasoning: str = Field(
        description=(
            "Decisive plain-language synthesis of the analyst reports and risk "
            "debate explaining WHY this team is favored. Ground claims in the "
            "stats/news/odds reports. Incorporate prior settled lessons if "
            "present in context."
        ),
    )

    # --- Secondary: betting view (optional, retained) ------------------------
    action: Action = Field(
        description=(
            "Betting view only (secondary). The Kalshi call: BUY YES / BUY NO / "
            "HOLD. HOLD if the edge vs. the market price is within the no-trade "
            "band."
        ),
    )
    estimated_probability: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Estimate of the true target-YES probability for the Kalshi "
            "contract, [0, 1]. (The YES target may differ from predicted_winner; "
            "this is the contract-resolution probability, not the winner prob.)"
        ),
    )
    implied_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Market-implied probability from the verified-odds snapshot, [0, 1].",
    )
    edge: float = Field(
        description="estimated_probability - implied_probability.",
    )
    recommended_stake_pct: float = Field(
        ge=0.0,
        le=1.0,
        description="Recommended bankroll fraction [0, 1]; 0.0 for HOLD.",
    )


def render_final_recommendation(rec: FinalRecommendation, sport: str = "nba") -> str:
    """Render a ``FinalRecommendation`` to markdown (winner-first headline).

    ``sport`` selects the headline emoji (⚽ for soccer, 🏀 for NBA, etc.).
    """
    from sportagent.sports.base import sport_icon

    win_pct = rec.win_probability * 100
    loser_pct = 100 - win_pct
    icon = sport_icon(sport)
    return "\n".join([
        f"{icon} **PREDICTION: {rec.predicted_winner} win — {win_pct:.0f}%** "
        f"(opponent {loser_pct:.0f}%)",
        f"**Confidence:** {rec.confidence.capitalize()}",
        "",
        f"**Why:** {rec.reasoning}",
        "",
        "---",
        "**Betting view (secondary):**",
        f"- Action: {rec.action.value}",
        f"- Estimated target-YES probability: {rec.estimated_probability:.3f} "
        f"({rec.estimated_probability * 100:.1f}%)",
        f"- Implied probability: {rec.implied_probability:.3f} "
        f"({rec.implied_probability * 100:.1f}%)",
        f"- Edge: {rec.edge:+.3f} ({rec.edge * 100:+.1f}pp)",
        f"- Suggested stake: {rec.recommended_stake_pct * 100:.2f}% of bankroll",
        "",
        f"FINAL PREDICTION: **{rec.predicted_winner}** to win "
        f"({win_pct:.0f}%) · FINAL RECOMMENDATION: **{rec.action.value}**",
    ])
