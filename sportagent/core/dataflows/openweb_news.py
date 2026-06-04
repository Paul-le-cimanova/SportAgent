"""OpenWeb Ninja News fetcher — injury / lineup / beat-writer headlines.

Accessed via the OpenWeb Ninja **direct** API (``api.openwebninja.com``), which
authenticates with a single ``X-API-Key`` header. The user supplies the key via
``OPENWEB_NINJA_API_KEY``; the base URL is overridable via
``OPENWEB_NINJA_NEWS_HOST`` (defaults to ``https://api.openwebninja.com``). We
hit the ``/realtime-news-data/search`` endpoint and degrade gracefully on any
error (returns a placeholder).

Registers ``get_injury_news`` and ``get_lineup_news`` (vendor "openweb_news").
All fetchers fail open — never raise.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import requests

from sportagent.core.dataflows.config import get_config
from sportagent.core.dataflows.interface import register_vendor_method

logger = logging.getLogger(__name__)

_TIMEOUT = 10.0
# OpenWeb Ninja direct API defaults (overridable via OPENWEB_NINJA_NEWS_HOST).
_DEFAULT_BASE_URL = "https://api.openwebninja.com"
_SEARCH_PATH = "/realtime-news-data/search"


def _credentials() -> tuple[Optional[str], str]:
    """Return ``(api_key, base_url)``. Base URL defaults to the direct API."""
    base_url = (
        os.environ.get("OPENWEB_NINJA_NEWS_HOST", "").strip() or _DEFAULT_BASE_URL
    )
    # Tolerate a bare host (no scheme) for backward compatibility.
    if base_url and not base_url.startswith(("http://", "https://")):
        base_url = f"https://{base_url}"
    return (
        os.environ.get("OPENWEB_NINJA_API_KEY"),
        base_url.rstrip("/"),
    )


def _search_news(query: str, limit: int) -> Any:
    """Run a news search via the OpenWeb Ninja direct API, or return an error string."""
    api_key, base_url = _credentials()
    if not api_key:
        return "<openweb news unavailable: set OPENWEB_NINJA_API_KEY>"
    url = f"{base_url}{_SEARCH_PATH}"
    headers = {"X-API-Key": api_key}
    params = {"query": query, "limit": limit, "time_published": "7d"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 — fail open
        logger.warning("OpenWeb Ninja News fetch failed for %r: %s", query, exc)
        return f"<news unavailable: {type(exc).__name__}>"


def _format_articles(payload: Any, query: str) -> str:
    """Render a search payload into a compact article list.

    Tolerant of shape: looks for a list under ``data``/``articles``/``results``
    and pulls title/snippet/source/date from common field names.
    """
    if isinstance(payload, str):  # error placeholder
        return payload
    if not isinstance(payload, dict):
        return f"<unexpected news response for {query!r}>"

    items: List[dict] = (
        payload.get("data")
        or payload.get("articles")
        or payload.get("results")
        or []
    )
    if not items:
        return f"<no news found for {query!r} in the past week>"

    lines = [f"News for {query!r} (past 7 days):"]
    for art in items:
        title = art.get("title") or art.get("headline") or ""
        snippet = (
            art.get("snippet")
            or art.get("description")
            or art.get("body")
            or ""
        )
        source = (
            art.get("source")
            or (art.get("source_name") if isinstance(art.get("source_name"), str) else "")
            or art.get("publisher")
            or "?"
        )
        date = art.get("published_datetime_utc") or art.get("date") or art.get("published") or ""
        snippet = str(snippet).replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:220] + "…"
        lines.append(f"  [{str(date)[:10]} · {source}] {title}")
        if snippet:
            lines.append(f"    {snippet}")
    return "\n".join(lines)


def get_injury_news(team: str, date: Optional[str] = None, limit: int = 15) -> str:
    """Recent injury / availability news for a team."""
    cfg = get_config()
    limit = cfg.get("news_article_limit", limit)
    payload = _search_news(f"{team} NBA injury report status", limit)
    return _format_articles(payload, f"{team} injuries")


def get_lineup_news(team: str, date: Optional[str] = None, limit: int = 15) -> str:
    """Recent starting-lineup / rotation news for a team."""
    cfg = get_config()
    limit = cfg.get("news_article_limit", limit)
    payload = _search_news(f"{team} NBA starting lineup probable", limit)
    return _format_articles(payload, f"{team} lineup")


register_vendor_method("get_injury_news", "openweb_news", get_injury_news)
register_vendor_method("get_lineup_news", "openweb_news", get_lineup_news)