"""Structured-output helpers with graceful free-text fallback.

The Research Manager, Trader, Decision Manager, and Sentiment Analyst produce
Pydantic-typed output. Providers expose structured output differently and a
given model may fail to satisfy a schema; rather than block the pipeline, we
try ``with_structured_output(schema)`` first and, on any failure, fall back to
a plain LLM call returning the raw text. Either way the caller gets markdown.

``bind_structured`` returns the structured-capable LLM (or None if the provider
doesn't support it). ``invoke_structured_or_freetext`` runs the full attempt →
render → fallback flow and returns ``(markdown, parsed_or_None)``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def bind_structured(llm: Any, schema: Type[BaseModel]) -> Optional[Any]:
    """Return ``llm.with_structured_output(schema)`` or None if unsupported.

    Fails open: a provider/binding that raises leaves us with None so the
    caller falls back to a free-text invocation.
    """
    bind = getattr(llm, "with_structured_output", None)
    if bind is None:
        return None
    try:
        return bind(schema)
    except Exception as exc:  # noqa: BLE001 — fail open to free text
        logger.warning("bind_structured failed for %s: %s", schema.__name__, exc)
        return None


def invoke_structured_or_freetext(
    llm: Any,
    schema: Type[BaseModel],
    messages: Any,
    render: Callable[[BaseModel], str],
) -> Tuple[str, Optional[BaseModel]]:
    """Invoke ``llm`` for structured ``schema`` output, falling back to text.

    Returns ``(markdown, parsed)`` where ``parsed`` is the Pydantic instance on
    success or None when the free-text path was used. The pipeline never blocks:
    any failure degrades to the raw text content of a plain invocation.
    """
    structured = bind_structured(llm, schema)
    if structured is not None:
        try:
            result = structured.invoke(messages)
            if isinstance(result, schema):
                return render(result), result
            # Some providers return a dict; coerce through the model.
            if isinstance(result, dict):
                parsed = schema(**result)
                return render(parsed), parsed
        except Exception as exc:  # noqa: BLE001 — fall back to free text
            logger.warning(
                "Structured invoke failed for %s; falling back to free text: %s",
                schema.__name__, exc,
            )

    # Free-text fallback.
    try:
        result = llm.invoke(messages)
        text = getattr(result, "content", None)
        if text is None:
            text = str(result)
        return text, None
    except Exception as exc:  # noqa: BLE001 — last-resort placeholder
        logger.error("Free-text invoke also failed for %s: %s", schema.__name__, exc)
        return f"<LLM call failed: {exc}>", None