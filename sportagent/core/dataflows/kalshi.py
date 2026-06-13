"""Kalshi market-data client (read-only, signed).

Wraps the Kalshi trade-api v2 REST endpoints used in v1:
- ``GET /events``            : discover NBA game events
- ``GET /markets``           : list markets (filterable by event/series/status)
- ``GET /markets/{ticker}``  : single market (price, volume, status, result)
- ``GET /markets/{ticker}/orderbook`` : depth (optional liquidity check)

Every request is RSA-signed (see ``kalshi_auth``). All public functions
**fail open**: on any network/parse error they return a dict with an
``"error"`` key rather than raising, so callers never crash on a dead source.

v1 is read-only — no order placement.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlparse

import requests

from sportagent.core.dataflows.config import get_config
from sportagent.core.dataflows.interface import register_vendor_method
from sportagent.core.dataflows.kalshi_auth import build_auth_headers
from sportagent.default_config import kalshi_base_url

logger = logging.getLogger(__name__)

_TIMEOUT = 10.0
_TRADE_API_PREFIX = "/trade-api/v2"


def _credentials() -> tuple[Optional[str], Optional[str]]:
    """Read Kalshi credentials from the environment (loaded via .env)."""
    import os

    return (
        os.environ.get("KALSHI_ACCESS_KEY_ID"),
        os.environ.get("KALSHI_PRIVATE_KEY_PATH"),
    )


def _signed_get(path_with_query: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Perform a signed GET against the configured Kalshi base URL.

    Args:
        path_with_query: API path *after* the ``/trade-api/v2`` prefix,
            including any query string, e.g. ``/markets?limit=10``.
        config: Optional config override (defaults to runtime config).

    Returns:
        Parsed JSON dict on success, or ``{"error": "..."}`` on failure.
    """
    cfg = config or get_config()
    access_key_id, private_key_path = _credentials()
    if not access_key_id or not private_key_path:
        return {"error": "Kalshi credentials missing (set KALSHI_ACCESS_KEY_ID and KALSHI_PRIVATE_KEY_PATH)"}

    base = kalshi_base_url(cfg)
    # The signed path must include the /trade-api/v2 prefix. base already ends
    # with it, so derive the prefix from base to stay correct for demo/prod.
    base_path = urlparse(base).path.rstrip("/")  # e.g. /trade-api/v2
    full_path = f"{base_path}{path_with_query}"
    url = f"{base}{path_with_query}"

    try:
        headers = build_auth_headers(
            access_key_id=access_key_id,
            private_key_path=private_key_path,
            method="GET",
            path=full_path,
        )
        headers["Accept"] = "application/json"
        resp = requests.get(url, headers=headers, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 — fail open
        logger.warning("Kalshi GET %s failed: %s", path_with_query, exc)
        return {"error": f"{type(exc).__name__}: {exc}"}


# --- Public read API ---------------------------------------------------------


def get_events(
    *,
    series_ticker: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """List events, optionally filtered by series ticker and status.

    Returns the raw Kalshi response (``{"events": [...], "cursor": ...}``) or
    ``{"error": ...}``.
    """
    params: Dict[str, Any] = {"limit": limit}
    if series_ticker:
        params["series_ticker"] = series_ticker
    if status:
        params["status"] = status
    return _signed_get(f"/events?{urlencode(params)}", config)


def get_markets(
    *,
    event_ticker: Optional[str] = None,
    series_ticker: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    cursor: Optional[str] = None,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """List markets, optionally filtered by event/series/status.

    Pass ``cursor`` (from a prior response's ``"cursor"``) to page through
    results — the World Cup bracket exceeds one page. Returns
    ``{"markets": [...], "cursor": ...}`` or ``{"error": ...}``.
    """
    params: Dict[str, Any] = {"limit": limit}
    if event_ticker:
        params["event_ticker"] = event_ticker
    if series_ticker:
        params["series_ticker"] = series_ticker
    if status:
        params["status"] = status
    if cursor:
        params["cursor"] = cursor
    return _signed_get(f"/markets?{urlencode(params)}", config)


def get_market(ticker: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Get a single market by ticker.

    Returns ``{"market": {...}}`` (with yes_bid/yes_ask/last_price/volume/
    open_interest/status/result) or ``{"error": ...}``.
    """
    return _signed_get(f"/markets/{ticker}", config)


def get_orderbook(
    ticker: str, depth: int = 10, config: Optional[Dict] = None
) -> Dict[str, Any]:
    """Get the order book for a market (optional liquidity check)."""
    return _signed_get(f"/markets/{ticker}/orderbook?{urlencode({'depth': depth})}", config)


# --- Convenience extractors --------------------------------------------------


def _to_cents(value: Any) -> Optional[int]:
    """Coerce a Kalshi price field to integer cents (0-100), or None.

    Kalshi returns prices in two shapes depending on field:
    - integer cents (legacy ``last_price``/``yes_bid``/``yes_ask``), and
    - decimal-dollar strings (``last_price_dollars``: ``\"0.6600\"`` etc.).
    Dollar values (``<= 1``) are scaled to cents; values already in the
    1-100 range are taken as cents directly.
    """
    if value is None or value == "":
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num <= 0:
        return None
    # Dollar string like "0.66" → 66 cents; "66" (cents) stays 66.
    cents = round(num * 100) if num <= 1.0 else round(num)
    return int(cents) if 0 < cents <= 100 else None


def extract_price_cents(market: Dict[str, Any]) -> Optional[int]:
    """Pick the best available YES price (cents) from a market dict.

    Handles both the integer-cents fields and the newer ``*_dollars`` string
    fields the live API returns. Prefers last price, then the mid of the YES
    bid/ask. Returns None if no usable price is present.
    """
    if not market or "error" in market:
        return None
    m = market.get("market", market)

    # Last-traded price (dollar-string preferred, then legacy cents).
    last = _to_cents(m.get("last_price_dollars"))
    if last is None:
        last = _to_cents(m.get("last_price"))
    if last is not None:
        return last

    # Fall back to the YES bid/ask midpoint.
    bid = _to_cents(m.get("yes_bid_dollars"))
    if bid is None:
        bid = _to_cents(m.get("yes_bid"))
    ask = _to_cents(m.get("yes_ask_dollars"))
    if ask is None:
        ask = _to_cents(m.get("yes_ask"))
    if bid is not None and ask is not None:
        return int(round((bid + ask) / 2))
    # A single side is still a usable signal.
    return bid if bid is not None else ask


def extract_result(market: Dict[str, Any]) -> Optional[str]:
    """Return settlement result (``"yes"``/``"no"``) or None if unsettled."""
    if not market or "error" in market:
        return None
    m = market.get("market", market)
    result = m.get("result")
    return result if result in ("yes", "no") else None


# --- Registered tool implementations -----------------------------------------


def get_kalshi_price(market_ticker: str, config: Optional[Dict] = None) -> str:
    """Formatted YES contract price (cents) + implied probability for a ticker.

    Fails open: returns a placeholder string on any error/missing price.
    """
    from sportagent.core.agents.utils import probability as prob

    market = get_market(market_ticker, config)
    if "error" in market:
        return f"<Kalshi market {market_ticker} unavailable: {market['error']}>"
    price = extract_price_cents(market)
    if price is None:
        return (
            f"<Kalshi market {market_ticker}: no usable YES price "
            f"(no last_price_dollars and no yes_bid/ask) — use the de-vigged "
            f"sportsbook consensus from the verified-odds snapshot instead>"
        )
    m = market.get("market", market)
    status = m.get("status", "?")
    volume = m.get("volume_fp") or m.get("volume") or "?"
    return (
        f"Kalshi {market_ticker}: YES {price}c → implied "
        f"{prob.implied_prob(price) * 100:.1f}% (status {status}, volume {volume})."
    )


def get_kalshi_market(market_ticker: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Raw single-market dict for a ticker (routing alias for get_market)."""
    return get_market(market_ticker, config)


def get_kalshi_orderbook(market_ticker: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Order book for a market (routing alias for get_orderbook)."""
    return get_orderbook(market_ticker, config=config)


register_vendor_method("get_kalshi_price", "kalshi", get_kalshi_price)
register_vendor_method("get_kalshi_market", "kalshi", get_kalshi_market)
register_vendor_method("get_kalshi_orderbook", "kalshi", get_kalshi_orderbook)
