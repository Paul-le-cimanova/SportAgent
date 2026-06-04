"""Report saving for a completed SportAgent run.

Mirrors TradingAgents' ``save_report_to_disk`` (doc 10 §4) but writes to
``~/.sportagent/results/<matchup>/<date>/``:

  - ``complete_report.md``   : the full assembled report (winner headline +
                               every section)
  - per-section markdown     : odds_report.md / stats_report.md / news_report.md
                               / sentiment_report.md / investment_plan.md /
                               trader_position_plan.md / final_recommendation.md
  - ``full_state.json``      : the raw final ``GameState`` (JSON-serialisable
                               fields only)

Everything fails open: a write error is reported but never raises.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# (state key, filename stem, human section title) in report order.
_SECTIONS: List[Tuple[str, str, str]] = [
    ("odds_report", "odds_report", "Odds Analysis"),
    ("stats_report", "stats_report", "Stats Analysis"),
    ("news_report", "news_report", "News & Injury Analysis"),
    ("sentiment_report", "sentiment_report", "Sentiment Analysis"),
    ("verified_odds", "verified_odds", "Verified Odds Snapshot"),
    ("investment_plan", "investment_plan", "Research Team Decision (Edge Thesis)"),
    ("trader_position_plan", "trader_position_plan", "Trader Position Proposal"),
    ("final_recommendation", "final_recommendation", "Final Recommendation"),
]


def results_root() -> Path:
    """Base directory for saved results (``~/.sportagent/results``)."""
    return Path.home() / ".sportagent" / "results"


def _slug(text: str) -> str:
    """Filesystem-safe slug for a matchup/dir component."""
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "game"


def build_matchup_label(state: Dict[str, object]) -> str:
    """Human matchup label like ``away-at-home`` for the directory name."""
    away = str(state.get("away_team", "")).strip()
    home = str(state.get("home_team", "")).strip()
    if away and home:
        return f"{_slug(away)}-at-{_slug(home)}"
    ticker = str(state.get("market_ticker", "")).strip()
    return _slug(ticker) if ticker else "game"


def assemble_complete_report(state: Dict[str, object]) -> str:
    """Assemble the combined markdown report from the final state."""
    away = str(state.get("away_team", "")).strip()
    home = str(state.get("home_team", "")).strip()
    date = str(state.get("game_date", "")).strip()
    title = f"# SportAgent report — {away} @ {home}".rstrip()
    if date:
        title += f"  ({date})"

    parts: List[str] = [title, ""]
    # Winner headline first (parsed from the final recommendation).
    final_rec = str(state.get("final_recommendation", "")).strip()
    try:
        from sportagent.core.graph.signal_processing import parse_winner

        winner, win_prob = parse_winner(final_rec)
        if winner:
            parts.append(
                f"## 🏀 Prediction: {winner} to win — {win_prob * 100:.0f}%"
            )
            parts.append("")
    except Exception:  # noqa: BLE001 — headline is best-effort
        pass

    for key, _stem, heading in _SECTIONS:
        content = str(state.get(key, "")).strip()
        if not content:
            continue
        parts.append(f"## {heading}")
        parts.append("")
        parts.append(content)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def save_report(
    state: Dict[str, object], *, base_dir: Optional[Path] = None
) -> Tuple[bool, str]:
    """Write the complete report + per-section files + full_state.json.

    Returns ``(ok, path_or_error)``. Never raises.
    """
    try:
        base = base_dir or results_root()
        matchup = build_matchup_label(state)
        date = str(state.get("game_date", "")).strip() or datetime.now().strftime(
            "%Y-%m-%d"
        )
        out_dir = base / matchup / date
        out_dir.mkdir(parents=True, exist_ok=True)

        # Combined report.
        (out_dir / "complete_report.md").write_text(
            assemble_complete_report(state), encoding="utf-8"
        )

        # Per-section files (only non-empty).
        for key, stem, _heading in _SECTIONS:
            content = str(state.get(key, "")).strip()
            if content:
                (out_dir / f"{stem}.md").write_text(content, encoding="utf-8")

        # Raw state (JSON-serialisable fields only).
        serialisable: Dict[str, object] = {}
        for k, v in state.items():
            if k == "messages":
                continue  # LangChain message objects aren't JSON-friendly.
            try:
                json.dumps(v)
                serialisable[k] = v
            except (TypeError, ValueError):
                serialisable[k] = str(v)
        (out_dir / "full_state.json").write_text(
            json.dumps(serialisable, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return True, str(out_dir)
    except Exception as exc:  # noqa: BLE001 — fail open
        return False, f"{type(exc).__name__}: {exc}"