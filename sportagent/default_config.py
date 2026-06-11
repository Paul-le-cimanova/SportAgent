"""Default configuration for SportAgent + ``SPORTAGENT_*`` env-var overrides.

Single source of truth for config keys. To expose a key for environment
override, add a row to ``_ENV_OVERRIDES`` — coercion follows the type of the
existing default, so users can keep writing plain strings in their ``.env``.
"""

import os

_SPORTAGENT_HOME = os.path.join(os.path.expanduser("~"), ".sportagent")

# env var -> config key. Coercion driven by the type of the existing default.
_ENV_OVERRIDES = {
    "SPORTAGENT_LLM_PROVIDER":      "llm_provider",
    "SPORTAGENT_LLM_AUTH_METHOD":   "llm_auth_method",
    "SPORTAGENT_DEEP_THINK_LLM":    "deep_think_llm",
    "SPORTAGENT_QUICK_THINK_LLM":   "quick_think_llm",
    "SPORTAGENT_OUTPUT_LANGUAGE":   "output_language",
    "SPORTAGENT_MAX_DEBATE_ROUNDS": "max_debate_rounds",
    "SPORTAGENT_MAX_RISK_ROUNDS":   "max_risk_rounds",
    "SPORTAGENT_CHECKPOINT_ENABLED": "checkpoint_enabled",
    "SPORTAGENT_KALSHI_ENV":        "kalshi_env",
    "SPORTAGENT_TEMPERATURE":       "temperature",
    "SPORTAGENT_NO_TRADE_BAND":     "no_trade_band",
    "SPORTAGENT_KELLY_CAP":         "kelly_cap",
    "SPORTAGENT_MEMORY_LOG_PATH":   "memory_log_path",
}


def _coerce(value: str, reference):
    """Coerce an env-var string to the type of the existing default value."""
    if isinstance(reference, bool):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value


def _apply_env_overrides(config: dict) -> dict:
    """Apply ``SPORTAGENT_*`` env vars to the config dict in-place."""
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        config[key] = _coerce(raw, config.get(key))
    return config


DEFAULT_CONFIG = _apply_env_overrides({
    # Paths
    "results_dir": os.getenv(
        "SPORTAGENT_RESULTS_DIR", os.path.join(_SPORTAGENT_HOME, "logs")
    ),
    "data_cache_dir": os.getenv(
        "SPORTAGENT_CACHE_DIR", os.path.join(_SPORTAGENT_HOME, "cache")
    ),
    "memory_log_path": os.getenv(
        "SPORTAGENT_MEMORY_LOG_PATH",
        os.path.join(_SPORTAGENT_HOME, "memory", "sport_memory.md"),
    ),
    # Optional cap on resolved memory-log entries. None disables rotation.
    "memory_log_max_entries": None,

    # LLM settings
    "llm_provider": "anthropic",          # "anthropic" or "openai"
    "llm_auth_method": "api_key",         # "api_key" or "cli_proxy" (Claude Code / Codex CLI)
    "deep_think_llm": "claude-opus-4-8",  # Research Manager + Decision Manager
    "quick_think_llm": "claude-haiku-4-5-20251001",  # analysts, researchers, trader, risk
    "backend_url": None,
    "temperature": None,                  # None = provider default
    "output_language": "English",

    # Debate / pipeline
    "max_debate_rounds": 1,               # Bull/Bear rounds (each = 2 turns)
    "max_risk_rounds": 1,                 # Risk rounds (each = 3 turns)
    "max_recur_limit": 100,
    "analyst_concurrency_limit": 1,
    "checkpoint_enabled": False,

    # Kalshi (read-only in v1, so prod is the sensible default — real prices)
    "kalshi_env": "prod",                 # "demo" or "prod"
    "kalshi_demo_url": "https://demo-api.kalshi.co/trade-api/v2",
    "kalshi_prod_url": "https://api.elections.kalshi.com/trade-api/v2",

    # Decision / sizing
    "no_trade_band": 0.03,                # min edge (3pp) to act; below -> HOLD
    "kelly_cap": 0.25,                    # max fraction of full Kelly
    "max_stake_pct": 0.05,                # hard cap on recommended stake

    # Data-fetch params
    "news_article_limit": 15,
    "reddit_post_limit": 15,
    "recent_form_games": 10,

    # Multi-sport
    "enabled_sports": ["nba", "nfl", "mlb", "soccer"],
    "sport_stats_vendors": {
        "nba":    "balldontlie",
        "nfl":    "espn",
        "mlb":    "mlb_statsapi",
        "soccer": "football_data",
    },

    # Data-vendor routing (category-level defaults)
    "data_vendors": {
        "market_prices":    "kalshi",
        "sportsbook_odds":  "odds_api",
        "team_stats":       "balldontlie",
        "scores_standings": "espn",
        "injury_news":      "openweb_news",
        "social_sentiment": "reddit",
    },
    # Per-tool overrides (take precedence over category defaults)
    "tool_vendors": {},
})


def kalshi_base_url(config: dict) -> str:
    """Return the active Kalshi base URL for the configured environment."""
    if config.get("kalshi_env", "prod") == "demo":
        return config["kalshi_demo_url"]
    return config["kalshi_prod_url"]
