"""LLM client factory — dual-tier (deep/quick) over Anthropic + OpenAI.

``create_llm_client`` returns a LangChain chat model for the given provider and
model name. The graph builds two instances per run: a **deep** model (Research
Manager + Decision Manager) and a **quick** model (everything else), selected by
``deep_think_llm`` / ``quick_think_llm`` in config.

Imports of the provider packages are deferred so importing this module never
requires both SDKs to be installed; only the selected provider's package is
needed at call time.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ANTHROPIC = "anthropic"
_OPENAI = "openai"


def create_llm_client(
    provider: str,
    model: str,
    *,
    temperature: Optional[float] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Create a LangChain chat model for ``provider``/``model``.

    Args:
        provider: ``"anthropic"`` or ``"openai"``.
        model: Provider model name (e.g. ``"claude-opus-4.8"``, ``"gpt-5.5"``).
        temperature: Optional sampling temperature; omitted (provider default)
            when None.
        base_url: Optional API base URL override (e.g. proxy / Azure-style).
        **kwargs: Passed through to the underlying chat-model constructor.

    Returns:
        A LangChain ``BaseChatModel`` supporting ``invoke``, ``bind_tools``, and
        ``with_structured_output``.

    Raises:
        ValueError: for an unknown provider.
        ImportError: if the selected provider's package is not installed.
    """
    key = provider.strip().lower()
    params: dict[str, Any] = dict(kwargs)
    if temperature is not None:
        params["temperature"] = temperature

    if key == _ANTHROPIC:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "langchain-anthropic is required for the Anthropic provider "
                "(pip install langchain-anthropic)."
            ) from exc
        if base_url:
            params["base_url"] = base_url
        return ChatAnthropic(model=model, **params)

    if key == _OPENAI:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "langchain-openai is required for the OpenAI provider "
                "(pip install langchain-openai)."
            ) from exc
        if base_url:
            params["base_url"] = base_url
        return ChatOpenAI(model=model, **params)

    raise ValueError(
        f"Unknown llm_provider {provider!r}; expected 'anthropic' or 'openai'."
    )


def create_deep_and_quick(config: dict) -> tuple[Any, Any]:
    """Build the ``(deep_llm, quick_llm)`` pair from a runtime config dict.

    The deep model serves the Research Manager + Decision Manager; the quick
    model serves all other agents. Both honor ``temperature`` and ``backend_url``
    when present.
    """
    provider = config.get("llm_provider", _ANTHROPIC)
    temperature = config.get("temperature")
    base_url = config.get("backend_url")
    deep = create_llm_client(
        provider,
        config.get("deep_think_llm", "claude-opus-4.8"),
        temperature=temperature,
        base_url=base_url,
    )
    quick = create_llm_client(
        provider,
        config.get("quick_think_llm", "claude-haiku-4.5"),
        temperature=temperature,
        base_url=base_url,
    )
    return deep, quick