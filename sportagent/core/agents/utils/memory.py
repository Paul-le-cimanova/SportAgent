"""Settled-outcome decision log (cross-game learning).

Append-only markdown log at ``~/.sportagent/memory/sport_memory.md`` (path from
config ``memory_log_path``). Three lifecycle phases (see design doc 07):

- **Phase A — store**: append a *pending* entry at the end of each run.
- **Phase B — resolve**: once a game settles, score the call (won/lost, realized
  edge, Brier) and rewrite the entry with a reflection.
- **Phase C — inject**: gather recent same-matchup + cross-game lessons into the
  ``past_context`` string fed to the Decision Manager at run start.

Every function fails open: I/O or parse errors degrade to empty/no-op behavior
rather than raising, so a missing or malformed log never crashes a run.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ENTRY_END = "<!-- ENTRY_END -->"
# Tag line: [date | market | action | est p | impl q | status | brier b]
_TAG_RE = re.compile(r"^\[(?P<body>.+?)\]\s*$", re.MULTILINE)


def _memory_path(config: Optional[dict] = None) -> str:
    """Resolve the memory-log path from config (with a stable default)."""
    from sportagent.core.dataflows.config import get_config

    cfg = config or get_config()
    return cfg.get(
        "memory_log_path",
        os.path.join(os.path.expanduser("~"), ".sportagent", "memory", "sport_memory.md"),
    )


def _read_log(path: str) -> str:
    """Return the full log text, or empty string if absent/unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""
    except Exception as exc:  # noqa: BLE001 — fail open
        logger.warning("Memory log read failed (%s): %s", path, exc)
        return ""


def _atomic_write(path: str, content: str) -> bool:
    """Atomically write ``content`` to ``path`` (temp file + replace)."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        directory = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, path)
        return True
    except Exception as exc:  # noqa: BLE001 — fail open
        logger.warning("Memory log write failed (%s): %s", path, exc)
        return False


def _split_entries(log: str) -> List[str]:
    """Split the log into entry blocks (delimiter retained-free)."""
    if not log.strip():
        return []
    parts = [p.strip() for p in log.split(_ENTRY_END)]
    return [p for p in parts if p]


def _entry_market(entry: str) -> str:
    """Extract the market ticker from an entry's tag line (best-effort)."""
    match = _TAG_RE.search(entry)
    if not match:
        return ""
    fields = [f.strip() for f in match.group("body").split("|")]
    return fields[1] if len(fields) > 1 else ""


def _entry_is_pending(entry: str) -> bool:
    """True if the entry's tag line is still marked ``pending``."""
    match = _TAG_RE.search(entry)
    if not match:
        return False
    return "pending" in match.group("body").lower()


def _entry_has_reflection(entry: str) -> bool:
    return "REFLECTION:" in entry


# --- Phase A: store ----------------------------------------------------------


def append_pending_entry(
    *,
    date: str,
    market_ticker: str,
    action: str,
    estimated_probability: float,
    implied_probability: float,
    recommendation_markdown: str,
    config: Optional[dict] = None,
) -> bool:
    """Append a *pending* decision entry (idempotent per market+date).

    Returns True on success. If an entry for the same market+date already
    exists, this is a no-op (idempotency guard) and returns True.
    """
    path = _memory_path(config)
    log = _read_log(path)

    # Idempotency: skip if a same-market+date tag line already present.
    guard = f"{date} | {market_ticker} |"
    if guard in log:
        logger.info("Memory: pending entry already exists for %s on %s", market_ticker, date)
        return True

    edge = estimated_probability - implied_probability
    tag = (
        f"[{date} | {market_ticker} | {action} | "
        f"est {estimated_probability:.2f} | impl {implied_probability:.2f} | "
        f"edge {edge:+.2f} | pending]"
    )
    entry = "\n".join([
        "",
        tag,
        "",
        "RECOMMENDATION:",
        recommendation_markdown.strip(),
        "",
        _ENTRY_END,
        "",
    ])
    return _atomic_write(path, log + entry)


# --- Phase C: inject ---------------------------------------------------------


def get_past_context(
    *,
    market_ticker: str = "",
    teams: Optional[List[str]] = None,
    max_matchup: int = 5,
    max_cross_game: int = 3,
    config: Optional[dict] = None,
) -> str:
    """Build the ``past_context`` block for the Decision Manager.

    Gathers up to ``max_matchup`` recent *resolved* entries matching the same
    market/teams (full text incl. reflection) plus up to ``max_cross_game``
    recent cross-game reflections (reflection only). Returns "" when there is
    no usable memory.
    """
    path = _memory_path(config)
    entries = _split_entries(_read_log(path))
    if not entries:
        return ""

    resolved = [e for e in entries if _entry_has_reflection(e) and not _entry_is_pending(e)]
    if not resolved:
        return ""

    teams = teams or []
    teams_lc = [t.lower() for t in teams if t]

    def _matches_matchup(entry: str) -> bool:
        if market_ticker and market_ticker in entry:
            return True
        entry_lc = entry.lower()
        return any(t in entry_lc for t in teams_lc)

    # Most recent last in the log → iterate reversed for recency.
    matchup_entries: List[str] = []
    for entry in reversed(resolved):
        if _matches_matchup(entry):
            matchup_entries.append(entry)
        if len(matchup_entries) >= max_matchup:
            break

    cross_game: List[str] = []
    for entry in reversed(resolved):
        if entry in matchup_entries:
            continue
        reflection = _extract_reflection(entry)
        if reflection:
            cross_game.append(reflection)
        if len(cross_game) >= max_cross_game:
            break

    sections: List[str] = []
    if matchup_entries:
        sections.append("### Prior same-matchup decisions")
        sections.extend(matchup_entries)
    if cross_game:
        sections.append("### Recent cross-game lessons")
        sections.extend(f"- {r}" for r in cross_game)
    return "\n\n".join(sections).strip()


def _extract_reflection(entry: str) -> str:
    """Return just the reflection prose from an entry (or empty string)."""
    if "REFLECTION:" not in entry:
        return ""
    after = entry.split("REFLECTION:", 1)[1].strip()
    return after.strip()


# --- Phase B helpers (resolution wiring lives in reflection.py) --------------


def find_pending_entries(config: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Return parsed metadata for each still-pending entry in the log."""
    path = _memory_path(config)
    entries = _split_entries(_read_log(path))
    pending: List[Dict[str, Any]] = []
    for entry in entries:
        if not _entry_is_pending(entry):
            continue
        match = _TAG_RE.search(entry)
        if not match:
            continue
        fields = [f.strip() for f in match.group("body").split("|")]
        pending.append({
            "raw": entry,
            "market_ticker": _entry_market(entry),
            "fields": fields,
        })
    return pending


def replace_entry(old_entry: str, new_entry: str, config: Optional[dict] = None) -> bool:
    """Replace a single entry block in the log atomically.

    ``old_entry``/``new_entry`` are the block bodies *without* the trailing
    ``<!-- ENTRY_END -->`` delimiter.
    """
    path = _memory_path(config)
    log = _read_log(path)
    if not log or old_entry.strip() not in log:
        return False
    updated = log.replace(old_entry.strip(), new_entry.strip(), 1)
    return _atomic_write(path, updated)