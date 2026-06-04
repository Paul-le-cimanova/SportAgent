"""Vendor routing for SportAgent data tools.

Agent ``@tool`` functions call ``route_to_vendor(method, *args)`` which resolves
the active vendor for that method via a two-level lookup and dispatches to the
vendor implementation:

    1. ``tool_vendors[method]``      (per-tool override, highest priority)
    2. ``data_vendors[category]``    (category-level default)
    3. hardcoded fallback in TOOL_CATEGORIES

Vendor implementations are registered in ``VENDOR_METHODS``. Each is expected to
fail open (return a string/dict, never raise), so routing stays crash-free.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from sportagent.core.dataflows.config import get_config

logger = logging.getLogger(__name__)


# method -> category. Adding a tool means adding it here + to VENDOR_METHODS.
TOOL_CATEGORIES: Dict[str, str] = {
    # market_prices
    "get_kalshi_market":   "market_prices",
    "get_kalshi_price":    "market_prices",
    "get_kalshi_orderbook": "market_prices",
    # sportsbook_odds
    "get_moneyline":       "sportsbook_odds",
    "get_line_movement":   "sportsbook_odds",
    # team_stats
    "get_team_stats":      "team_stats",
    "get_h2h":             "team_stats",
    "get_recent_form":     "team_stats",
    # scores_standings
    "get_standings":       "scores_standings",
    "get_schedule":        "scores_standings",
    "get_rest_status":     "scores_standings",
    # injury_news
    "get_injury_news":     "injury_news",
    "get_lineup_news":     "injury_news",
    # social_sentiment
    "get_reddit_sentiment": "social_sentiment",
}

# Hardcoded last-resort default per category (used if config is missing one).
_CATEGORY_FALLBACK: Dict[str, str] = {
    "market_prices":    "kalshi",
    "sportsbook_odds":  "odds_api",
    "team_stats":       "balldontlie",
    "scores_standings": "espn",
    "injury_news":      "openweb_news",
    "social_sentiment": "reddit",
}


# method -> {vendor -> callable}. Populated lazily by register_vendor_method()
# from each vendor module to avoid import cycles. Vendor modules call
# register_vendor_method() at import time.
VENDOR_METHODS: Dict[str, Dict[str, Callable[..., Any]]] = {}


def register_vendor_method(method: str, vendor: str, fn: Callable[..., Any]) -> None:
    """Register a vendor implementation for a method.

    Called by vendor modules at import so ``route_to_vendor`` can dispatch.
    """
    VENDOR_METHODS.setdefault(method, {})[vendor] = fn


def get_category_for_method(method: str) -> str:
    """Return the category a method belongs to.

    Raises ValueError for unknown methods (a programming error, not runtime).
    """
    category = TOOL_CATEGORIES.get(method)
    if category is None:
        raise ValueError(f"Unknown data method: {method!r}")
    return category


def resolve_vendor(method: str, config: Optional[Dict] = None) -> str:
    """Resolve the active vendor for ``method`` via tool -> category -> fallback."""
    cfg = config or get_config()
    tool_vendors = cfg.get("tool_vendors", {}) or {}
    if method in tool_vendors:
        return tool_vendors[method]
    category = get_category_for_method(method)
    data_vendors = cfg.get("data_vendors", {}) or {}
    if category in data_vendors:
        return data_vendors[category]
    return _CATEGORY_FALLBACK[category]


def route_to_vendor(method: str, *args: Any, config: Optional[Dict] = None, **kwargs: Any) -> Any:
    """Dispatch ``method`` to its resolved vendor implementation.

    Returns the vendor's result. If the method/vendor pair is unregistered,
    returns a placeholder string (fail-open) so the pipeline never crashes.
    """
    try:
        vendor = resolve_vendor(method, config)
    except ValueError as exc:
        logger.warning("route_to_vendor: %s", exc)
        return f"<routing error: {exc}>"

    impls = VENDOR_METHODS.get(method)
    if not impls:
        return f"<no vendor registered for {method!r} (vendor module not imported?)>"
    fn = impls.get(vendor)
    if fn is None:
        # Fall back to any registered vendor for this method rather than failing.
        fallback_vendor, fn = next(iter(impls.items()))
        logger.warning(
            "route_to_vendor: %s has no impl for vendor %r; using %r",
            method, vendor, fallback_vendor,
        )
    return fn(*args, **kwargs)