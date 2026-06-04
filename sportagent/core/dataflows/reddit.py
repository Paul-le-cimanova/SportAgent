"""Reddit sentiment fetcher for NBA / sports discussion (free, no key).

Primary path is Reddit's public JSON search (``reddit.com/r/{sub}/search.json``);
when Reddit's WAF returns 403/blocks, falls back to the public Atom/RSS search
feed (``/search.rss``). Returns a formatted plaintext block ready for prompt
injection and degrades gracefully — returns a placeholder string rather than
raising, so callers never special-case missing data.

Registers ``get_reddit_sentiment`` with the routing interface (vendor "reddit").
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sportagent.core.dataflows.interface import register_vendor_method

logger = logging.getLogger(__name__)

_API = "https://www.reddit.com/r/{sub}/search.json?{qs}"
_RSS = "https://www.reddit.com/r/{sub}/search.rss?{qs}"
_UA = "sportagent/0.1 (+https://github.com/sportagent)"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# Default subreddits for NBA discussion, roughly by signal density.
DEFAULT_SUBREDDITS = ("nba", "sportsbook", "nbadiscussion")


def _search_qs(query: str, limit: int) -> str:
    return urlencode({
        "q": query,
        "restrict_sr": "on",
        "sort": "new",
        "t": "week",
        "limit": limit,
    })


def _iso_to_ts(iso_str: Optional[str]) -> Optional[float]:
    if not iso_str:
        return None
    try:
        normalized = iso_str[:-1] + "+00:00" if iso_str.endswith("Z") else iso_str
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return None


def _strip_html(content: str) -> str:
    if not content:
        return ""
    if "<!-- SC_OFF -->" in content and "<!-- SC_ON -->" in content:
        content = content.split("<!-- SC_OFF -->")[1].split("<!-- SC_ON -->")[0]
    text = re.sub(r"<[^>]+>", " ", content)
    return " ".join(html.unescape(text).split())


def _fetch_rss(query: str, sub: str, limit: int, timeout: float) -> List[dict]:
    url = _RSS.format(sub=sub, qs=_search_qs(query, limit))
    req = Request(url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=timeout) as resp:
            root = ET.fromstring(resp.read())
    except (HTTPError, URLError, TimeoutError, ET.ParseError) as exc:
        logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, query, exc)
        return []
    posts = []
    for entry in root.findall("atom:entry", _ATOM_NS)[:limit]:
        title_el = entry.find("atom:title", _ATOM_NS)
        published_el = entry.find("atom:published", _ATOM_NS)
        content_el = entry.find("atom:content", _ATOM_NS)
        posts.append({
            "title": (title_el.text if title_el is not None else "") or "",
            "score": None,
            "num_comments": None,
            "created_utc": _iso_to_ts(published_el.text if published_el is not None else None),
            "selftext": _strip_html(content_el.text if content_el is not None else ""),
            "source": "rss",
        })
    return posts


def _fetch_sub(query: str, sub: str, limit: int, timeout: float) -> List[dict]:
    url = _API.format(sub=sub, qs=_search_qs(query, limit))
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        children = (payload.get("data") or {}).get("children") or []
        return [c.get("data", {}) for c in children if isinstance(c, dict)]
    except (HTTPError, URLError, json.JSONDecodeError, TimeoutError) as exc:
        # The JSON endpoint is frequently WAF-blocked (403); the RSS fallback is
        # the expected path, so this is debug-level noise, not a warning.
        logger.debug(
            "Reddit JSON fetch fell back to RSS for r/%s · %s: %s",
            sub, query, exc,
        )
        return _fetch_rss(query, sub, limit, timeout)


def get_reddit_sentiment(
    query: str,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float = 0.4,
) -> str:
    """Fetch recent Reddit posts mentioning ``query`` (a team/matchup) and
    return them as a formatted plaintext block for sentiment analysis.

    ``inter_request_delay`` keeps us under Reddit's public rate limit.
    """
    blocks = []
    total = 0
    for i, sub in enumerate(subreddits):
        if i > 0:
            time.sleep(inter_request_delay)
        posts = _fetch_sub(query, sub, limit_per_sub, timeout)
        total += len(posts)
        if not posts:
            blocks.append(f"r/{sub}: <no posts found mentioning {query} in the past week>")
            continue
        via_rss = any(p.get("source") == "rss" for p in posts)
        header = f"r/{sub} — {len(posts)} recent posts mentioning {query}"
        header += " (via RSS; scores/comments unavailable):" if via_rss else ":"
        lines = [header]
        for p in posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            score = p.get("score")
            comments = p.get("num_comments")
            created = p.get("created_utc")
            created_str = time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            meta = created_str
            if score is not None and comments is not None:
                meta += f" · {score:>4}↑ · {comments:>3}c"
            body = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(body) > 240:
                body = body[:240] + "…"
            lines.append(f"  [{meta}] {title}" + (f"\n    excerpt: {body}" if body else ""))
        blocks.append("\n".join(lines))

    if total == 0:
        subs = ", ".join(f"r/{s}" for s in subreddits)
        return f"<no Reddit posts found mentioning {query} across {subs} in the past week>"
    return "\n\n".join(blocks)


register_vendor_method("get_reddit_sentiment", "reddit", get_reddit_sentiment)