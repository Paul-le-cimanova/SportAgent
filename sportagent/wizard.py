"""Interactive game-picker wizard (questionary) for SportAgent.

Mirrors the TradingAgents wizard (doc 10 §2) but is **schedule-driven**: the
user never types team names. The flow is:

  1. select sport            (currently NBA)
  2. date picker             (Today / Tomorrow / Upcoming 7d / specific FUTURE
                              date — past dates are rejected)
  3. pick a game             (arrow-select from the fetched schedule list)
  4. research depth          (Shallow=1 / Medium=3 / Deep=5 → debate rounds)
  5. provider / models       (Anthropic / OpenAI; quick + deep model IDs)

Returns a ``WizardResult`` the CLI turns into a query + config overrides.

questionary is the primary UI; every prompt degrades to plain ``input()`` so
the wizard still runs in a non-interactive / minimal environment. All schedule
fetches fail open to an empty list (handled by the caller).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --- Static option tables ----------------------------------------------------

_SPORTS = [
    ("NBA (basketball)", "nba"),
    ("Soccer / World Cup (3-way)", "soccer"),
]

# Soccer competitions (label, The-Odds-API sportsbook key / football-data code).
_SOCCER_COMPETITIONS = [
    ("Premier League (EPL)", "soccer_epl"),
    ("UEFA Champions League", "soccer_uefa_champs_league"),
    ("FIFA World Cup", "soccer_fifa_world_cup"),
    ("La Liga (Spain)", "soccer_spain_la_liga"),
    ("Serie A (Italy)", "soccer_italy_serie_a"),
    ("Bundesliga (Germany)", "soccer_germany_bundesliga"),
    ("Ligue 1 (France)", "soccer_france_ligue_one"),
]

# Soccer market types (label, value). Match = 3-way; advancement/futures = 2-way.
_SOCCER_MARKET_TYPES = [
    ("Match winner (3-way: home/draw/away)", "match"),
    ("Advancement (to advance/reach stage — YES/NO)", "advancement"),
    ("Tournament winner / futures (YES/NO)", "futures"),
]

_DEPTHS = [
    ("Shallow — 1 debate round (fast)", 1),
    ("Medium — 3 debate rounds (balanced)", 3),
    ("Deep — 5 debate rounds (thorough)", 5),
]

_PROVIDERS = [
    ("Anthropic (Claude)", "anthropic"),
    ("OpenAI (GPT)", "openai"),
]

# Quick + deep model menus per provider (label, model_id). \"Custom\" is appended.
_MODELS: Dict[str, Dict[str, List[tuple]]] = {
    "anthropic": {
        "deep": [
            ("Claude Opus 4.8 (deep, default)", "claude-opus-4-8"),
            ("Claude Sonnet 4.5", "claude-sonnet-4-5-20250929"),
        ],
        "quick": [
            ("Claude Haiku 4.5 (quick, default)", "claude-haiku-4-5-20251001"),
            ("Claude Sonnet 4.5", "claude-sonnet-4-5-20250929"),
        ],
    },
    "openai": {
        "deep": [
            ("GPT-5.5 (deep)", "gpt-5.5"),
            ("GPT-5.4 (deep)", "gpt-5.4"),
        ],
        "quick": [
            ("GPT-5.4-mini (quick)", "gpt-5.4-mini"),
            ("GPT-5.5 (quick)", "gpt-5.5"),
        ],
    },
}


@dataclass
class WizardResult:
    """Everything the CLI needs to launch an analysis."""

    sport: str
    game_date: str            # YYYY-MM-DD
    away: str
    home: str
    query: str                # "Away @ Home" matchup string for resolve_market
    research_depth: int
    provider: str
    deep_llm: str
    quick_llm: str
    config_overrides: Dict[str, object] = field(default_factory=dict)
    # Soccer-only: competition sportsbook key + market type (match/advancement/futures).
    competition: str = ""
    market_type: str = "match"


# --- questionary / plain-input helpers ---------------------------------------


def _q():
    """Return the questionary module or None when unavailable."""
    try:
        import questionary

        return questionary
    except Exception:  # noqa: BLE001 — degrade to plain input()
        return None


def _select(prompt: str, choices: List[tuple], default_index: int = 0):
    """Arrow-select one option; ``choices`` is a list of (label, value).

    Falls back to a numbered plain-input menu when questionary is missing.
    Returns the chosen value, or None if the user aborts.
    """
    q = _q()
    if q is not None:
        labels = [c[0] for c in choices]
        answer = q.select(
            prompt,
            choices=labels,
            default=labels[default_index] if 0 <= default_index < len(labels) else None,
        ).ask()
        if answer is None:
            return None
        for label, value in choices:
            if label == answer:
                return value
        return None
    # Plain fallback.
    print(f"\n{prompt}")
    for i, (label, _) in enumerate(choices, 1):
        print(f"  {i}. {label}")
    raw = input(f"  Choose [1-{len(choices)}] (default {default_index + 1}): ").strip()
    if not raw:
        return choices[default_index][1]
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(choices):
            return choices[idx][1]
    except ValueError:
        pass
    return choices[default_index][1]


def _text(prompt: str, default: str = "") -> str:
    """Free-text prompt with a default."""
    q = _q()
    if q is not None:
        answer = q.text(prompt, default=default).ask()
        return (answer or default).strip()
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw or default


# --- Date picker -------------------------------------------------------------


def _pick_date(today: Optional[_dt.date] = None) -> Optional[str]:
    """Date picker → an ISO date string (YYYY-MM-DD), or None if aborted.

    Only current/future dates are allowed (MVP: no backtesting). A specific
    past date is rejected with a re-prompt.
    """
    today = today or _dt.date.today()
    choice = _select(
        "When is the game?",
        [
            ("Today", "today"),
            ("Tomorrow", "tomorrow"),
            ("Upcoming 7 days (pick a day)", "upcoming"),
            ("Specific future date (YYYY-MM-DD)", "specific"),
        ],
    )
    if choice is None:
        return None
    if choice == "today":
        return today.isoformat()
    if choice == "tomorrow":
        return (today + _dt.timedelta(days=1)).isoformat()
    if choice == "upcoming":
        days = [
            (
                (today + _dt.timedelta(days=d)).strftime("%a %Y-%m-%d")
                + (" (today)" if d == 0 else ""),
                (today + _dt.timedelta(days=d)).isoformat(),
            )
            for d in range(0, 7)
        ]
        return _select("Pick a day:", days)
    # specific
    while True:
        raw = _text("Enter a future date (YYYY-MM-DD)", default=today.isoformat())
        try:
            parsed = _dt.datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            print("  ! Invalid format — expected YYYY-MM-DD.")
            continue
        if parsed < today:
            print("  ! Past dates are not supported (current/future games only).")
            continue
        return parsed.isoformat()


# --- Game picker -------------------------------------------------------------


def _fetch_schedule(sport: str, date: str, competition: str = "") -> List[dict]:
    """Fetch the schedule for ``sport`` on ``date`` (fail-open to [])."""
    if sport == "nba":
        try:
            from sportagent.sports.nba.stats import get_schedule_for_date

            return get_schedule_for_date(date) or []
        except Exception:  # noqa: BLE001 — fail open
            return []
    if sport == "soccer":
        try:
            from sportagent.sports.soccer.stats import get_schedule_for_date

            return get_schedule_for_date(date, competition or "soccer_epl") or []
        except Exception:  # noqa: BLE001 — fail open
            return []
    return []


def _pick_game(sport: str, date: str, competition: str = "") -> Optional[dict]:
    """Pick a game from the fetched schedule, or None when none/aborted."""
    games = _fetch_schedule(sport, date, competition)
    if not games:
        print(f"\n  No {sport.upper()} games found on {date}.")
        return None
    choices = []
    for g in games:
        away, home = g.get("away", "?"), g.get("home", "?")
        status = g.get("status", "")
        label = f"{away} @ {home}" + (f"  ({status})" if status else "")
        choices.append((label, g))
    return _select(f"Pick a game on {date}:", choices)


# --- Top-level wizard --------------------------------------------------------


def run_game_wizard() -> Optional[WizardResult]:
    """Run the full schedule-driven wizard. Returns None if aborted/no game."""
    sport = _select("Select a sport:", _SPORTS)
    if sport is None:
        return None

    # Soccer adds a competition picker + market-type selector before the date.
    competition = ""
    market_type = "match"
    if sport == "soccer":
        competition = _select("Select a competition:", _SOCCER_COMPETITIONS)
        if competition is None:
            return None
        market_type = _select("Market type:", _SOCCER_MARKET_TYPES, default_index=0)
        if market_type is None:
            market_type = "match"

    date = _pick_date()
    if date is None:
        return None

    game = _pick_game(sport, date, competition)
    if game is None:
        return None
    away, home = game.get("away", ""), game.get("home", "")

    depth = _select("Research depth:", _DEPTHS, default_index=0)
    if depth is None:
        depth = 1

    provider = _select("LLM provider:", _PROVIDERS, default_index=0)
    if provider is None:
        provider = "anthropic"

    deep_choices = list(_MODELS[provider]["deep"]) + [("Custom model ID…", "__custom__")]
    deep_llm = _select("Deep-think model (managers):", deep_choices, default_index=0)
    if deep_llm == "__custom__":
        deep_llm = _text("Custom deep model ID", default=_MODELS[provider]["deep"][0][1])

    quick_choices = list(_MODELS[provider]["quick"]) + [("Custom model ID…", "__custom__")]
    quick_llm = _select("Quick-think model (analysts):", quick_choices, default_index=0)
    if quick_llm == "__custom__":
        quick_llm = _text("Custom quick model ID", default=_MODELS[provider]["quick"][0][1])

    # Soccer titles read "Home vs Away"; use a vs-form query so the soccer
    # adapter parses the sides correctly (NBA keeps the "Away @ Home" form).
    query = f"{home} vs {away}" if sport == "soccer" else f"{away} @ {home}"
    overrides: Dict[str, object] = {
        "llm_provider": provider,
        "deep_think_llm": deep_llm,
        "quick_think_llm": quick_llm,
        "max_debate_rounds": depth,
        "max_risk_rounds": depth,
    }
    if sport == "soccer":
        overrides["competition"] = competition
        overrides["market_type"] = market_type
    return WizardResult(
        sport=sport,
        game_date=date,
        away=away,
        home=home,
        query=query,
        research_depth=depth,
        provider=provider,
        deep_llm=deep_llm,
        quick_llm=quick_llm,
        config_overrides=overrides,
        competition=competition,
        market_type=market_type,
    )
