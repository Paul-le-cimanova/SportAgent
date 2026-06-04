"""Deterministic action extraction from the Decision Manager's output.

``parse_action(text)`` maps a markdown recommendation to one of the three
terminal actions — ``BUY YES`` / ``BUY NO`` / ``HOLD`` — using a labelled-line
regex first (``FINAL RECOMMENDATION:`` / ``Action:``), then a keyword fallback.
No LLM is involved; this is used by the orchestrator and the memory log so the
recorded action is stable and reproducible (see design docs 06 §7 and 07 §6).
"""

from __future__ import annotations

import re

# Canonical action strings. These mirror ``schemas.Action`` values but are
# defined inline so this lightweight, deterministic module never triggers the
# agents package import chain (which pulls in langchain).
BUY_YES = "BUY YES"
BUY_NO = "BUY NO"
HOLD = "HOLD"

# Labels whose trailing value carries the action, most authoritative first.
_LABEL_PATTERNS = (
    re.compile(r"FINAL\s+RECOMMENDATION:\s*\**\s*([A-Za-z ]+)", re.IGNORECASE),
    re.compile(r"FINAL\s+POSITION\s+PROPOSAL:\s*\**\s*([A-Za-z ]+)", re.IGNORECASE),
    re.compile(r"\*\*Action:\*\*\s*([A-Za-z ]+)", re.IGNORECASE),
    re.compile(r"\bAction:\s*\**\s*([A-Za-z ]+)", re.IGNORECASE),
)


def _normalize(fragment: str) -> str | None:
    """Map a captured label fragment to a canonical action, or None."""
    f = fragment.strip().upper()
    if not f:
        return None
    if "BUY YES" in f or re.match(r"^YES\b", f):
        return BUY_YES
    if "BUY NO" in f or re.match(r"^NO\b", f):
        return BUY_NO
    if "HOLD" in f:
        return HOLD
    return None


def parse_action(text: str) -> str:
    """Extract the terminal action from a recommendation string.

    Tries the labelled-line patterns in priority order, then a whole-text
    keyword fallback. Defaults to ``HOLD`` (the safe no-trade action) when no
    action can be determined.
    """
    if not text:
        return HOLD

    for pattern in _LABEL_PATTERNS:
        match = pattern.search(text)
        if match:
            action = _normalize(match.group(1))
            if action:
                return action

    # Keyword fallback over the whole text (order matters: YES/NO before HOLD).
    upper = text.upper()
    if "BUY YES" in upper:
        return BUY_YES
    if "BUY NO" in upper:
        return BUY_NO
    if "HOLD" in upper:
        return HOLD
    return HOLD


def process_signal(final_recommendation: str) -> str:
    """Public alias used by the orchestrator (see doc 07 §6)."""
    return parse_action(final_recommendation)


# Winner-prediction extraction (the MVP headline). The Decision Manager renders
# ``🏀 **PREDICTION: <Team> win — 64%** (opponent 36%)`` and a trailing
# ``FINAL PREDICTION: **<Team>** to win (64%)`` line.
_WINNER_PATTERNS = (
    re.compile(r"FINAL\s+PREDICTION:\s*\*{0,2}\s*(.+?)\s+to\s+win", re.IGNORECASE),
    re.compile(r"PREDICTION:\s*\*{0,2}\s*(.+?)\s+win[^A-Za-z]", re.IGNORECASE),
)
_WIN_PROB_PATTERN = re.compile(r"win[^0-9%]*([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE)


def parse_winner(text: str) -> tuple[str, float]:
    """Extract ``(predicted_winner, win_probability)`` from a rendered rec.

    Returns ``("", 0.0)`` when no winner headline is present (e.g. a free-text
    fallback). ``win_probability`` is a decimal in [0, 1].
    """
    if not text:
        return "", 0.0
    winner = ""
    for pattern in _WINNER_PATTERNS:
        match = pattern.search(text)
        if match:
            winner = match.group(1).strip().strip("*").strip()
            if winner:
                break
    prob = 0.0
    pm = _WIN_PROB_PATTERN.search(text)
    if pm:
        prob = min(1.0, max(0.0, float(pm.group(1)) / 100.0))
    return winner, prob
