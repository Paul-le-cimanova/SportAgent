"""Onboarding + environment preflight for SportAgent.

Single source of truth for the API keys / secrets the app reads from the
environment (loaded from a gitignored ``.env``). Provides:

- ``ENV_KEYS``            : declarative spec of every key (group, required,
                           signup URL, secret flag, optional validator).
- ``read_env_file`` /
  ``write_env_file``      : comment-preserving ``.env`` read/merge/write.
- ``check_environment``   : structured preflight report (presence + optional
                           live "ping" checks), every check **fails open**.
- ``run_setup_wizard``    : interactive ``rich``/Typer wizard that writes
                           ``.env``.

Design notes:
- Mirrors the rest of the data layer: every validator/ping **fails open** and
  never raises — a dead source yields ``INVALID`` with a fix hint, not a crash.
- The Kalshi ``.pem`` is never copied or printed; only its path is stored and
  its readability/format validated via the existing loader.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# --- Status constants --------------------------------------------------------

STATUS_OK = "OK"
STATUS_MISSING = "MISSING"
STATUS_INVALID = "INVALID"
STATUS_SKIPPED = "SKIPPED"


# --- Env-key specification ---------------------------------------------------


@dataclass(frozen=True)
class EnvKey:
    """Declarative spec for a single environment variable."""

    name: str
    group: str
    required: bool
    description: str
    signup_url: str = ""
    secret: bool = True
    default: str = ""
    # Optional offline validator: (value, env) -> (ok, hint). Never raises.
    validator: Optional[Callable[[str, Dict[str, str]], Tuple[bool, str]]] = field(
        default=None, repr=False
    )


def _validate_pem(value: str, _env: Dict[str, str]) -> Tuple[bool, str]:
    """Validate that ``value`` points to a readable RSA private-key PEM."""
    if not value:
        return False, "Path is empty."
    path = Path(value).expanduser()
    if not path.exists():
        return False, f"File not found: {path}"
    try:
        from sportagent.core.dataflows.kalshi_auth import _load_private_key

        _load_private_key(str(path))
        return True, "Valid RSA private key."
    except Exception as exc:  # noqa: BLE001 — fail open
        return False, f"Not a valid RSA private key: {type(exc).__name__}: {exc}"


def _validate_kalshi_env(value: str, _env: Dict[str, str]) -> Tuple[bool, str]:
    if value.strip().lower() in ("demo", "prod"):
        return True, ""
    return False, "Must be 'demo' or 'prod'."


# Ordered list — drives both the wizard prompts and the doctor report.
ENV_KEYS: List[EnvKey] = [
    # --- LLM provider ---
    EnvKey(
        name="ANTHROPIC_API_KEY",
        group="LLM (Anthropic)",
        required=True,
        description="Anthropic API key (default deep+quick provider).",
        signup_url="https://console.anthropic.com/settings/keys",
    ),
    EnvKey(
        name="OPENAI_API_KEY",
        group="LLM (OpenAI)",
        required=False,
        description="OpenAI API key (only if llm_provider=openai).",
        signup_url="https://platform.openai.com/api-keys",
    ),
    # --- Kalshi ---
    EnvKey(
        name="KALSHI_ACCESS_KEY_ID",
        group="Kalshi",
        required=True,
        description="Kalshi Access Key ID (demo first).",
        signup_url="https://kalshi.com/account/profile",
    ),
    EnvKey(
        name="KALSHI_PRIVATE_KEY_PATH",
        group="Kalshi",
        required=True,
        description="Path to your Kalshi RSA private-key .pem file.",
        signup_url="https://kalshi.com/account/profile",
        secret=False,
        default="./secrets/kalshi_private_key.pem",
        validator=_validate_pem,
    ),
    EnvKey(
        name="KALSHI_ENV",
        group="Kalshi",
        required=False,
        description="Kalshi environment: 'demo' or 'prod'.",
        secret=False,
        default="demo",
        validator=_validate_kalshi_env,
    ),
    # --- Sports data ---
    EnvKey(
        name="THE_ODDS_API_KEY",
        group="Sports data",
        required=True,
        description="The Odds API key (sportsbook odds; free ~500 req/mo).",
        signup_url="https://the-odds-api.com/",
    ),
    EnvKey(
        name="BALLDONTLIE_API_KEY",
        group="Sports data",
        required=True,
        description="balldontlie API key (NBA team/player stats).",
        signup_url="https://app.balldontlie.io/",
    ),
    # --- News ---
    EnvKey(
        name="OPENWEB_NINJA_API_KEY",
        group="News",
        required=False,
        description="OpenWeb Ninja direct API key (X-API-Key; injury/news).",
        signup_url="https://www.openwebninja.com/",
    ),
    EnvKey(
        name="OPENWEB_NINJA_NEWS_HOST",
        group="News",
        required=False,
        description="OpenWeb Ninja base URL override (default api.openwebninja.com).",
        secret=False,
        default="https://api.openwebninja.com",
        signup_url="https://www.openwebninja.com/",
    ),
]


def env_keys_by_group() -> Dict[str, List[EnvKey]]:
    """Group ``ENV_KEYS`` preserving declaration order within each group."""
    grouped: Dict[str, List[EnvKey]] = {}
    for spec in ENV_KEYS:
        grouped.setdefault(spec.group, []).append(spec)
    return grouped


# --- .env file read / write (comment-preserving) -----------------------------


def default_env_path() -> Path:
    """Return the project ``.env`` path (cwd-based; matches dotenv loading)."""
    return Path(os.getcwd()) / ".env"


def read_env_file(path: Optional[Path] = None) -> Dict[str, str]:
    """Parse a ``.env`` into a dict (best-effort; ignores comments/blanks)."""
    path = path or default_env_path()
    result: Dict[str, str] = {}
    try:
        if not path.exists():
            return result
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    except Exception:  # noqa: BLE001 — fail open to an empty mapping
        return result
    return result


def write_env_file(
    values: Dict[str, str], path: Optional[Path] = None
) -> Tuple[bool, str]:
    """Merge ``values`` into ``.env`` atomically, preserving unknown lines.

    Existing keys are updated in place; new keys are appended under a managed
    block. Returns ``(ok, message)`` and never raises.
    """
    path = path or default_env_path()
    try:
        existing_lines: List[str] = []
        if path.exists():
            existing_lines = path.read_text(encoding="utf-8").splitlines()

        seen: set = set()
        out: List[str] = []
        for raw in existing_lines:
            stripped = raw.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.partition("=")[0].strip()
                if key in values:
                    out.append(f"{key}={values[key]}")
                    seen.add(key)
                    continue
            out.append(raw)

        appended = [k for k in values if k not in seen]
        if appended:
            if out and out[-1].strip():
                out.append("")
            out.append("# --- Added by `sportagent setup` ---")
            for key in appended:
                out.append(f"{key}={values[key]}")

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return True, f"Wrote {len(values)} value(s) to {path}"
    except Exception as exc:  # noqa: BLE001 — fail open
        return False, f"Failed to write {path}: {type(exc).__name__}: {exc}"


# --- Live "ping" checks (each fails open) ------------------------------------


def _ping_kalshi(env: Dict[str, str]) -> Tuple[bool, str]:
    """Signed ``GET /events?limit=1`` against the configured Kalshi env."""
    try:
        from sportagent.core.dataflows import kalshi
        from sportagent.core.dataflows.config import get_config

        resp = kalshi.get_events(limit=1, config=get_config())
        if isinstance(resp, dict) and "error" in resp:
            return False, str(resp["error"])
        return True, "Kalshi reachable + signature accepted."
    except Exception as exc:  # noqa: BLE001 — fail open
        return False, f"{type(exc).__name__}: {exc}"


def _ping_odds_api(env: Dict[str, str]) -> Tuple[bool, str]:
    """Probe The Odds API sports endpoint (cheap; doesn't spend event quota)."""
    try:
        import requests

        key = env.get("THE_ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY", "")
        if not key:
            return False, "No key."
        resp = requests.get(
            "https://api.the-odds-api.com/v4/sports",
            params={"apiKey": key},
            timeout=10.0,
        )
        if resp.status_code == 200:
            return True, "The Odds API key accepted."
        return False, f"HTTP {resp.status_code}: {resp.text[:120]}"
    except Exception as exc:  # noqa: BLE001 — fail open
        return False, f"{type(exc).__name__}: {exc}"


def _ping_balldontlie(env: Dict[str, str]) -> Tuple[bool, str]:
    """Probe balldontlie teams endpoint with the API key header."""
    try:
        import requests

        key = env.get("BALLDONTLIE_API_KEY") or os.environ.get(
            "BALLDONTLIE_API_KEY", ""
        )
        if not key:
            return False, "No key."
        resp = requests.get(
            "https://api.balldontlie.io/v1/teams",
            headers={"Authorization": key},
            params={"per_page": 1},
            timeout=10.0,
        )
        if resp.status_code == 200:
            return True, "balldontlie key accepted."
        return False, f"HTTP {resp.status_code}: {resp.text[:120]}"
    except Exception as exc:  # noqa: BLE001 — fail open
        return False, f"{type(exc).__name__}: {exc}"


def _ping_openweb_news(env: Dict[str, str]) -> Tuple[bool, str]:
    """Probe the OpenWeb Ninja direct news API with the X-API-Key header."""
    try:
        import requests

        key = env.get("OPENWEB_NINJA_API_KEY") or os.environ.get(
            "OPENWEB_NINJA_API_KEY", ""
        )
        if not key:
            return False, "No key (optional)."
        base = (
            env.get("OPENWEB_NINJA_NEWS_HOST")
            or os.environ.get("OPENWEB_NINJA_NEWS_HOST", "")
        ).strip() or "https://api.openwebninja.com"
        if not base.startswith(("http://", "https://")):
            base = f"https://{base}"
        resp = requests.get(
            f"{base.rstrip('/')}/realtime-news-data/search",
            headers={"X-API-Key": key},
            params={"query": "NBA", "limit": 1},
            timeout=10.0,
        )
        if resp.status_code == 200:
            return True, "OpenWeb Ninja key accepted."
        return False, f"HTTP {resp.status_code}: {resp.text[:120]}"
    except Exception as exc:  # noqa: BLE001 — fail open
        return False, f"{type(exc).__name__}: {exc}"


# Group -> live ping. LLM keys are presence-only (no spend on probe).
_PINGS: Dict[str, Callable[[Dict[str, str]], Tuple[bool, str]]] = {
    "Kalshi": _ping_kalshi,
    "Sports data (Odds)": _ping_odds_api,
    "Sports data (balldontlie)": _ping_balldontlie,
    "News (OpenWeb Ninja)": _ping_openweb_news,
}


@dataclass
class CheckResult:
    """One row of the preflight report."""

    name: str
    group: str
    required: bool
    status: str
    hint: str = ""


def check_environment(
    *, live: bool = False, env: Optional[Dict[str, str]] = None
) -> List[CheckResult]:
    """Build a structured preflight report for every ``ENV_KEYS`` entry.

    Args:
        live: If True, run the live ping checks (small quota use); each fails
            open. If False, presence + offline validators only.
        env: Optional override mapping (defaults to ``os.environ``).

    Returns:
        A list of ``CheckResult`` in ``ENV_KEYS`` order, with per-group live
        ping rows appended when ``live=True``.
    """
    source: Dict[str, str] = dict(os.environ if env is None else env)
    results: List[CheckResult] = []

    for spec in ENV_KEYS:
        value = source.get(spec.name, "").strip()
        if not value:
            status = STATUS_MISSING if spec.required else STATUS_SKIPPED
            hint = "" if not spec.required else f"Set {spec.name} ({spec.signup_url})"
            results.append(CheckResult(spec.name, spec.group, spec.required, status, hint))
            continue
        if spec.validator is not None:
            ok, hint = spec.validator(value, source)
            results.append(
                CheckResult(
                    spec.name,
                    spec.group,
                    spec.required,
                    STATUS_OK if ok else STATUS_INVALID,
                    "" if ok else hint,
                )
            )
        else:
            results.append(CheckResult(spec.name, spec.group, spec.required, STATUS_OK))

    if live:
        for label, ping in _PINGS.items():
            ok, hint = ping(source)
            results.append(
                CheckResult(
                    f"ping: {label}",
                    label,
                    False,
                    STATUS_OK if ok else STATUS_INVALID,
                    "" if ok else hint,
                )
            )

    return results


def required_missing(results: List[CheckResult]) -> List[CheckResult]:
    """Return the required checks that are MISSING or INVALID."""
    return [
        r
        for r in results
        if r.required and r.status in (STATUS_MISSING, STATUS_INVALID)
    ]


def is_ready(results: Optional[List[CheckResult]] = None) -> bool:
    """True when no required key is missing/invalid."""
    results = results if results is not None else check_environment(live=False)
    return not required_missing(results)


# --- Interactive wizard ------------------------------------------------------


def run_setup_wizard(
    *, path: Optional[Path] = None, console=None
) -> Tuple[bool, str]:
    """Interactive ``.env`` setup wizard. Returns ``(ok, message)``.

    Uses ``rich``/``typer`` for prompts when available; falls back to plain
    ``input()`` so it works even without those installed.
    """
    path = path or default_env_path()
    existing = read_env_file(path)

    try:
        from rich.console import Console
        from rich.prompt import Prompt

        console = console or Console()
        _print = console.print
        _have_rich = True
    except Exception:  # noqa: BLE001 — degrade to plain prompts
        _have_rich = False

        def _print(*args, **kwargs):  # type: ignore[no-redef]
            print(*args)

    def _prompt(label: str, *, default: str, secret: bool) -> str:
        if _have_rich:
            return Prompt.ask(label, default=default or None, password=secret) or ""
        suffix = f" [{default}]" if default else ""
        raw = input(f"{label}{suffix}: ").strip()
        return raw or default

    _print("\n[bold]SportAgent setup[/bold] — configure API keys (written to .env)\n"
           if _have_rich else
           "\nSportAgent setup — configure API keys (written to .env)\n")

    collected: Dict[str, str] = {}
    for group, specs in env_keys_by_group().items():
        header = f"— {group} —"
        _print(f"\n[bold cyan]{header}[/bold cyan]" if _have_rich else f"\n{header}")
        for spec in specs:
            req = "required" if spec.required else "optional"
            if spec.signup_url:
                _print(
                    f"  {spec.description} ({req})  → {spec.signup_url}"
                    if not _have_rich
                    else f"  [dim]{spec.description} ({req}) → {spec.signup_url}[/dim]"
                )
            else:
                _print(f"  {spec.description} ({req})")
            current = existing.get(spec.name, "") or spec.default
            value = _prompt(f"  {spec.name}", default=current, secret=spec.secret)
            if value:
                collected[spec.name] = value
                if spec.validator is not None:
                    ok, hint = spec.validator(value, collected)
                    if not ok:
                        _print(f"    ! {hint}" if not _have_rich else f"    [yellow]! {hint}[/yellow]")

    ok, message = write_env_file(collected, path)
    _print(("\n[green]" if _have_rich and ok else "\n") + message + ("[/green]" if _have_rich and ok else ""))
    return ok, message