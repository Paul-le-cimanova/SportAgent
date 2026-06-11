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
    auth_method: str = "api_key",
    temperature: Optional[float] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Create a LangChain chat model for ``provider``/``model``.

    Args:
        provider: ``"anthropic"`` or ``"openai"``.
        model: Provider model name (e.g. ``"claude-opus-4.8"``, ``"gpt-5.5"``).
        auth_method: ``"api_key"`` (direct SDK, default) or ``"cli_proxy"``
            (shell out to the locally-installed Claude Code / Codex CLI; no
            API key needed).
        temperature: Optional sampling temperature; omitted (provider default)
            when None.
        base_url: Optional API base URL override (e.g. proxy / Azure-style).
        **kwargs: Passed through to the underlying chat-model constructor.

    Returns:
        A LangChain ``BaseChatModel`` supporting ``invoke``, ``bind_tools``, and
        ``with_structured_output``.

    Raises:
        ValueError: for an unknown provider or auth method.
        ImportError: if the selected provider's package is not installed.
        RuntimeError: if ``cli_proxy`` is selected but the CLI is missing.
    """
    key = provider.strip().lower()
    method = (auth_method or "api_key").strip().lower()

    if method == "cli_proxy":
        if key == _ANTHROPIC:
            from sportagent.core.llm_clients.claude_code_proxy import (
                ChatClaudeCodeProxy,
                is_claude_code_available,
            )

            if not is_claude_code_available():
                raise RuntimeError(
                    "llm_auth_method is 'cli_proxy' but the `claude` CLI was not "
                    "found. Install Claude Code (https://claude.com/claude-code) "
                    "or switch to API-key mode with `sportagent setup`."
                )
            return ChatClaudeCodeProxy(model=model, temperature=temperature)
        if key == _OPENAI:
            from sportagent.core.llm_clients.codex_proxy import (
                ChatCodexProxy,
                is_codex_available,
            )

            if not is_codex_available():
                raise RuntimeError(
                    "llm_auth_method is 'cli_proxy' but the `codex` CLI was not "
                    "found. Install Codex (npm install -g @openai/codex) "
                    "or switch to API-key mode with `sportagent setup`."
                )
            return ChatCodexProxy(model=model, temperature=temperature)
        raise ValueError(
            f"cli_proxy auth is not supported for provider {provider!r}; "
            "expected 'anthropic' or 'openai'."
        )
    if method != "api_key":
        raise ValueError(
            f"Unknown llm_auth_method {auth_method!r}; expected 'api_key' or 'cli_proxy'."
        )

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
    auth_method = config.get("llm_auth_method", "api_key")
    temperature = config.get("temperature")
    base_url = config.get("backend_url")
    deep = create_llm_client(
        provider,
        config.get("deep_think_llm", "claude-opus-4.8"),
        auth_method=auth_method,
        temperature=temperature,
        base_url=base_url,
    )
    quick = create_llm_client(
        provider,
        config.get("quick_think_llm", "claude-haiku-4.5"),
        auth_method=auth_method,
        temperature=temperature,
        base_url=base_url,
    )
    return deep, quick
