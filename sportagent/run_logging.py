"""Per-run file logging for SportAgent.

Captures the full run log (DEBUG+, with tracebacks) to a timestamped file under
``~/.sportagent/logs/`` so a failed or noisy run is fully diagnosable even when
the live UI mutes/redirects the console.

Usage:
    from sportagent.run_logging import setup_run_logging
    log_path = setup_run_logging(matchup="spurs-at-knicks", verbose=False)
    ...run...
    # tell the user where the log is, especially on error.

``verbose=True`` (e.g. ``analyze --debug``) also lets logs through to the
console at INFO so you can watch them live.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

_LOG_DIR = Path.home() / ".sportagent" / "logs"
_FILE_HANDLER: Optional[logging.Handler] = None
_LAST_PATH: Optional[Path] = None


def _slug(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "run"


def log_dir() -> Path:
    """Return (and create) the log directory."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR


def last_log_path() -> Optional[Path]:
    """Path of the most recently configured run log (or None)."""
    return _LAST_PATH


def setup_run_logging(
    matchup: str = "run", *, verbose: bool = False
) -> Optional[Path]:
    """Attach a per-run file handler to the root ``sportagent`` logger.

    Returns the log file path, or ``None`` if the file couldn't be created
    (fails open — never blocks a run).

    The file always captures DEBUG+ with full tracebacks. When ``verbose`` is
    set, a console handler at INFO is added too. Safe to call repeatedly; the
    previous run's file handler is detached first.
    """
    global _FILE_HANDLER, _LAST_PATH

    root = logging.getLogger("sportagent")
    root.setLevel(logging.DEBUG)
    root.propagate = False

    # Detach a previous run's file handler so logs don't bleed across runs.
    if _FILE_HANDLER is not None:
        try:
            root.removeHandler(_FILE_HANDLER)
            _FILE_HANDLER.close()
        except Exception:  # noqa: BLE001
            pass
        _FILE_HANDLER = None

    try:
        d = log_dir()
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = d / f"run-{_slug(matchup)}-{ts}.log"
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(fh)
        _FILE_HANDLER = fh
        _LAST_PATH = path
    except Exception:  # noqa: BLE001 — fail open, no run-log file
        _LAST_PATH = None

    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(ch)

    return _LAST_PATH