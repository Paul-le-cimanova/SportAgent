"""Claude Code CLI proxy — LangChain ``BaseChatModel`` that shells out to ``claude``.

Lets users who already have Claude Code installed and authenticated (OAuth
subscription) run SportAgent **without an Anthropic API key**: every LLM call
becomes a ``claude -p`` (print mode) subprocess.

Verified against Claude Code CLI 2.x:
- ``-p/--print``               : non-interactive, print response and exit
- ``--system-prompt <prompt>`` : system prompt for the session
- ``--model <model>``          : model name or alias (e.g. ``opus``/``sonnet``)
- ``--json-schema <schema>``   : structured output validated against a JSON Schema
- ``--tools ""``               : disable all built-in tools (pure completion)
- ``--no-session-persistence`` : don't save the session to disk
- there is **no** ``--max-tokens`` / ``--temperature`` flag (provider defaults)

Limitations vs the direct API:
- No streaming (full response only)
- No native LangChain tool calling (``bind_tools``) — agents that need
  tool-calls should use the API-key path; structured output IS supported via
  ``--json-schema``
- No token accounting
- Slower (subprocess + CLI startup overhead per call)
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from typing import Any, List, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_AUTH_ERROR_HINT = (
    "Your Claude Code session appears to be expired or unauthenticated. "
    "Run `claude` once to log in (or `claude /login`), or switch to API-key "
    "mode with `sportagent setup`."
)


def is_claude_code_available() -> bool:
    """Return True if the ``claude`` CLI is installed and runnable."""
    if not shutil.which("claude"):
        return False
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001 — any failure means "not available"
        return False


def _split_messages(messages: List[BaseMessage]) -> tuple[str, str]:
    """Flatten LangChain messages into ``(system_prompt, conversation)`` strings."""
    system_parts: list[str] = []
    conversation_parts: list[str] = []
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if isinstance(msg, SystemMessage):
            system_parts.append(content)
        elif isinstance(msg, HumanMessage):
            conversation_parts.append(content)
        elif isinstance(msg, AIMessage):
            conversation_parts.append(f"[Previous assistant response]: {content}")
        else:
            conversation_parts.append(content)
    return "\n\n".join(system_parts), "\n\n".join(conversation_parts)


class ChatClaudeCodeProxy(BaseChatModel):
    """LangChain chat model that proxies completions through the Claude Code CLI.

    The user must have Claude Code installed and authenticated. Each call runs
    ``claude -p`` with tools disabled and no session persistence, passing the
    prompt on stdin (avoids ARG_MAX limits on long analyst prompts).
    """

    model: str = "claude-haiku-4-5-20251001"
    temperature: Optional[float] = None  # accepted for interface parity; CLI ignores it
    timeout_seconds: int = 600  # deep-model calls can take minutes
    json_schema: Optional[dict] = None  # set via with_structured_output()

    @property
    def _llm_type(self) -> str:
        return "claude-code-proxy"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model, "llm_type": self._llm_type}

    # ------------------------------------------------------------------ core

    def _build_command(self, system_prompt: str) -> list[str]:
        cmd = [
            "claude",
            "-p",
            "--no-session-persistence",
            "--tools",
            "",
        ]
        if self.model:
            cmd.extend(["--model", self.model])
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])
        if self.json_schema:
            # Structured output is delivered in the `structured_output` field
            # of the JSON result envelope, so we need --output-format json.
            cmd.extend(["--json-schema", json.dumps(self.json_schema)])
            cmd.extend(["--output-format", "json"])
        return cmd

    def _run_cli(self, system_prompt: str, prompt: str) -> str:
        cmd = self._build_command(system_prompt)
        logger.debug("Claude Code CLI call: model=%s schema=%s", self.model, bool(self.json_schema))
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "The `claude` CLI was not found on PATH. Install Claude Code "
                "(https://claude.com/claude-code) or switch to API-key mode "
                "with `sportagent setup`."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Claude Code CLI timed out after {self.timeout_seconds}s. "
                "Deep-model calls can be slow; increase timeout_seconds if needed."
            ) from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout or f"exit code {result.returncode}"
            lowered = detail.lower()
            if any(tok in lowered for tok in ("auth", "login", "credential", "api key", "unauthorized")):
                raise RuntimeError(f"Claude Code CLI auth error: {detail}\n{_AUTH_ERROR_HINT}")
            raise RuntimeError(f"Claude Code CLI error: {detail}")

        stdout = (result.stdout or "").strip()
        if self.json_schema:
            # Parse the JSON result envelope and pull out structured_output.
            try:
                envelope = json.loads(stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Claude Code CLI returned non-JSON envelope: {stdout[:200]!r}"
                ) from exc
            if envelope.get("is_error"):
                raise RuntimeError(f"Claude Code CLI error result: {envelope.get('result')}")
            structured = envelope.get("structured_output")
            if structured is not None:
                return json.dumps(structured)
            # Fall back to the plain-text result field.
            return str(envelope.get("result", "")).strip()
        return stdout

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_prompt, prompt = _split_messages(messages)
        response_text = self._run_cli(system_prompt, prompt)
        message = AIMessage(content=response_text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # ------------------------------------------------- structured output

    def with_structured_output(self, schema: Type[BaseModel], **kwargs: Any) -> Any:
        """Return a runnable that yields parsed ``schema`` instances.

        Uses the CLI's native ``--json-schema`` flag so the model's output is
        validated JSON, then parses it into the Pydantic model.
        """
        return _StructuredProxyRunnable(self, schema)


class _StructuredProxyRunnable:
    """Minimal runnable wrapper: invoke the proxy with a JSON schema, parse out."""

    def __init__(self, llm: ChatClaudeCodeProxy, schema: Type[BaseModel]):
        json_schema = schema.model_json_schema()
        self._llm = llm.model_copy(update={"json_schema": json_schema})
        self._schema = schema

    def invoke(self, messages: Any, config: Any = None, **kwargs: Any) -> BaseModel:
        result = self._llm.invoke(messages)
        text = getattr(result, "content", None) or str(result)
        data = _extract_json(text)
        return self._schema(**data)


def _extract_json(text: str) -> dict:
    """Parse a JSON object out of CLI output (tolerates surrounding prose/fences)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences if present.
    if "```" in text:
        for chunk in text.split("```"):
            chunk = chunk.strip()
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{"):
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    continue
    # Last resort: outermost-brace slice.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON object found in Claude Code CLI output: {text[:200]!r}")