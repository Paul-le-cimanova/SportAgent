"""Settled-outcome reflection (Phase B of the decision log).

After a game settles, ``resolve_pending_entries`` walks the still-pending log
entries, fetches each market's result (Kalshi ``result`` field, with no fallback
in v1 beyond the placeholder), scores the call (won/lost, realized edge, Brier),
asks the quick LLM for a constrained 2–4 sentence reflection, and rewrites the
entry atomically with the resolved tag line + reflection (see design doc 07).

Calibration over win/loss: the Brier score ``(estimate - outcome)²`` is the key
metric — a correct 64% edge still loses ~36% of the time, so the reflection
distinguishes *good process / bad luck* from *bad process*.

Fails open throughout: any fetch/parse/LLM error skips that entry rather than
raising, so resolution never crashes the next run.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from sportagent.core.agents.utils import memory
from sportagent.core.agents.utils import probability as prob
from sportagent.core.dataflows import kalshi

logger = logging.getLogger(__name__)


_REFLECTION_SYSTEM = (
    "You are the reflection writer on a sports prediction-market desk. Given a "
    "settled game and the recommendation made before it, write EXACTLY 2-4 "
    "sentences of plain prose. Cover: (1) was the directional call correct? "
    "(cite the result and the edge); (2) which part of the thesis held or "
    "failed; (3) one concrete lesson for the next similar game. Crucially, "
    "distinguish good process / bad luck (a correct edge that lost the coin "
    "flip) from a genuine misread — a single game is high variance, so judge "
    "the decision quality against the market price, not the raw outcome."
)


def _parse_tag_fields(fields: List[str]) -> Dict[str, Any]:
    """Parse the pipe-separated tag fields into a metadata dict."""
    out: Dict[str, Any] = {
        "date": fields[0] if len(fields) > 0 else "",
        "market_ticker": fields[1] if len(fields) > 1 else "",
        "action": fields[2] if len(fields) > 2 else "",
    }
    for f in fields[3:]:
        m = re.match(r"est\s+([0-9.]+)", f)
        if m:
            out["estimated_probability"] = float(m.group(1))
            continue
        m = re.match(r"impl\s+([0-9.]+)", f)
        if m:
            out["implied_probability"] = float(m.group(1))
            continue
    return out


# 3-way settlement: map the realized soccer outcome to the index used by
# probability.brier_multi (home/draw/away).
_THREE_WAY_INDEX = {"home": 0, "draw": 1, "away": 2}


def _won(action: str, result: str) -> Optional[bool]:
    """Did the recommended side hit? None for HOLD (calibration-only).

    Handles both 2-way (yes/no) and 3-way (home/draw/away) settlement: for a
    3-way market the recommendation names the leg (e.g. "BUY YES on the DRAW
    leg"), so a hit is when the settled outcome label appears in the action.
    """
    a = action.strip().upper()
    r = result.strip().lower()
    if r in _THREE_WAY_INDEX:
        # 3-way: the chosen leg is encoded in the action text.
        if "HOLD" in a:
            return None
        return r in a.lower()
    if a == "BUY YES":
        return result == "yes"
    if a == "BUY NO":
        return result == "no"
    return None  # HOLD


def _three_way_brier(raw_entry: str, outcome: str) -> Optional[float]:
    """Multi-class Brier for a 3-way settlement, or None if unparseable.

    Reads the home/draw/away probability vector the Trader/Research Manager
    rendered into the entry and scores it against the realized outcome index.
    """
    idx = _THREE_WAY_INDEX.get(outcome.strip().lower())
    if idx is None:
        return None
    vec = []
    for label in ("Home win", "Draw", "Away win"):
        m = re.search(rf"{label}:\s*([01]\.[0-9]+)", raw_entry)
        if not m:
            return None
        vec.append(float(m.group(1)))
    total = sum(vec) or 1.0
    vec = [v / total for v in vec]
    return prob.brier_multi(vec, idx)


def _build_resolved_tag(meta: Dict[str, Any], result: str, brier: float) -> str:
    """Construct the resolved tag line replacing the pending one."""
    est = meta.get("estimated_probability", 0.5)
    impl = meta.get("implied_probability", 0.5)
    won = _won(meta.get("action", ""), result)
    status = "HOLD" if won is None else ("WON" if won else "LOST")
    return (
        f"[{meta.get('date', '')} | {meta.get('market_ticker', '')} | "
        f"{meta.get('action', '')} | est {est:.2f} | impl {impl:.2f} | "
        f"{status} | brier {brier:.2f}]"
    )


def resolve_pending_entries(
    quick_llm: Any = None,
    config: Optional[dict] = None,
) -> int:
    """Resolve any pending entries whose markets have settled.

    Returns the number of entries successfully resolved. ``quick_llm`` is the
    quick-tier model used for the reflection prose; if None, a deterministic
    placeholder reflection is written instead (still fails open).
    """
    pending = memory.find_pending_entries(config)
    resolved_count = 0

    for item in pending:
        ticker = item.get("market_ticker", "")
        if not ticker:
            continue

        market = kalshi.get_market(ticker, config)
        result = kalshi.extract_result(market)
        raw = item.get("raw", "")
        # 3-way soccer entries carry a home/draw/away vector; detect + score
        # them with the multi-class Brier when the market exposes a 3-way result.
        three_way_result = _extract_three_way_result(market, raw)
        if result is None and three_way_result is None:
            # Unsettled or unavailable — leave pending for a later pass.
            continue

        meta = _parse_tag_fields(item.get("fields", []))
        if three_way_result is not None:
            result = three_way_result
            brier_score = _three_way_brier(raw, three_way_result)
            if brier_score is None:
                # Fall back to the binary scoring path on a parse miss.
                outcome = 1.0 if result in ("yes", "home") else 0.0
                brier_score = prob.brier(meta.get("estimated_probability", 0.5), outcome)
        else:
            outcome = 1.0 if result == "yes" else 0.0
            est = meta.get("estimated_probability", 0.5)
            brier_score = prob.brier(est, outcome)

        reflection = _generate_reflection(
            quick_llm, item.get("raw", ""), meta, result, brier_score
        )

        new_tag = _build_resolved_tag(meta, result, brier_score)
        new_entry = _rewrite_entry(item.get("raw", ""), new_tag, reflection)
        if memory.replace_entry(item.get("raw", ""), new_entry, config):
            resolved_count += 1

    return resolved_count


def _extract_three_way_result(market: Dict[str, Any], raw_entry: str) -> Optional[str]:
    """Return a 3-way settled outcome (home/draw/away) or None.

    Only applies to entries that carry a 3-way probability vector (so 2-way
    markets keep their existing yes/no path untouched). The realized outcome is
    read from the Kalshi market's ``result_sub_title``/``result`` where it names
    home/draw/away; falls back to None when the market is unsettled.
    """
    if "Home win:" not in raw_entry or "Draw:" not in raw_entry:
        return None
    if not market or "error" in market:
        return None
    m = market.get("market", market)
    text = " ".join(
        str(m.get(k, "")) for k in ("result_sub_title", "result", "title")
    ).lower()
    if "draw" in text or "tie" in text:
        return "draw"
    # When the home contract settled YES it's a home win; NO → away win.
    res = m.get("result")
    if res == "yes":
        return "home"
    if res == "no":
        return "away"
    return None


def _generate_reflection(
    quick_llm: Any,
    raw_entry: str,
    meta: Dict[str, Any],
    result: str,
    brier_score: float,
) -> str:
    """Produce the 2–4 sentence reflection (LLM, with deterministic fallback)."""
    won = _won(meta.get("action", ""), result)
    outcome_label = (
        "HOLD (no position)" if won is None
        else ("the call WON" if won else "the call LOST")
    )
    if quick_llm is None:
        return (
            f"{outcome_label}; market resolved {result.upper()}. "
            f"Brier {brier_score:.2f} vs the {meta.get('estimated_probability', 0.5):.2f} "
            "estimate. (Auto-generated: no LLM available for prose reflection.)"
        )

    from langchain_core.messages import HumanMessage, SystemMessage

    human = (
        f"Settled result: {result.upper()}. Outcome for our position: "
        f"{outcome_label}. Brier score: {brier_score:.3f}.\n\n"
        f"Our prior recommendation entry:\n{raw_entry}"
    )
    try:
        out = quick_llm.invoke([
            SystemMessage(content=_REFLECTION_SYSTEM),
            HumanMessage(content=human),
        ])
        text = getattr(out, "content", str(out)).strip()
        return text or outcome_label
    except Exception as exc:  # noqa: BLE001 — fail open
        logger.warning("Reflection LLM call failed for %s: %s", meta.get("market_ticker"), exc)
        return (
            f"{outcome_label}; market resolved {result.upper()}. "
            f"Brier {brier_score:.2f}. (Reflection LLM unavailable.)"
        )


def _rewrite_entry(raw_entry: str, new_tag: str, reflection: str) -> str:
    """Rewrite an entry: swap the tag line and append a REFLECTION section."""
    lines = raw_entry.strip().splitlines()
    # Replace the first tag line (starts with '[') with the resolved tag.
    rewritten: List[str] = []
    swapped = False
    for line in lines:
        if not swapped and line.strip().startswith("[") and line.strip().endswith("]"):
            rewritten.append(new_tag)
            swapped = True
        else:
            rewritten.append(line)
    if not swapped:
        rewritten.insert(0, new_tag)

    body = "\n".join(rewritten).strip()
    # Drop any pre-existing reflection, then append the fresh one.
    if "REFLECTION:" in body:
        body = body.split("REFLECTION:", 1)[0].strip()
    return f"{body}\n\nREFLECTION:\n{reflection.strip()}"