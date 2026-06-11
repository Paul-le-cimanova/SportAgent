"""Codex CLI proxy — LangChain ``BaseChatModel`` that shells out to ``codex``.

Lets users who already have OpenAI's Codex CLI installed and authenticated
(ChatGPT subscription OAuth) run SportAgent **without an OpenAI API key**:
every LLM call becomes a ``codex exec`` (non-interactive) subprocess.

Verified against Codex CLI 0.139.0 (``codex exec --help``):
- ``codex exec [PROMPT]``         : non-interactive ("headless") execution;
  with ``-`` (or no prompt arg) instructions are read from stdin
- ``-m/--model <model>``          : model override (e.g. ``gpt-5.5``)
- ``--skip-git-repo-check``       : allow running outside a git repo
- ``--ephemeral``                 : don't persist session files to disk
- ``--sandbox read-only``         : prevent the agent from writing/executing
- ``--output-last-message <file>``: write only the final assistant message
- ``--output-schema <file>``      : JSON Schema for the final response shape
  (native structured output)
- there is no ``--temperature`` / ``--max-tokens`` flag (provider defaults)

Limitations vs the direct API:
- No streaming (full response only)
- No native LangChain tool calling (``bind_tools``) — agents needing
  tool-calls should use the API-key path; structured output IS supported via
  ``--output-schema``
- No token accounting; slower (subprocess overhead per call)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, List, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_AUTH_ERROR_HINT = (
    "Your Codex CLI session appears to be expired or unauthenticated. "
    "Run `codex login` to re-authenticate, or switch to API-key mode with "
    "`sportagent setup`."
)


def is_codex_available() -> bool:
    """Return True if the ``codex`` CLI is installed and runnable."""
    if not shutil.which("codex"):
        return False
    try:
        result = subprocess.run(
            ["codex", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001 — any failure means "not available"
        return False


def _flatten_messages(messages: List[BaseMessage]) -> str:
    """Flatten LangChain messages into a single prompt string.

    ``codex exec`` takes one prompt, so the system prompt is prepended as a
    clearly-delimited instruction block.
    """
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

    parts: list[str] = []
    if system_parts:
        parts.append("[System instructions]\n" + "\n\n".join(system_parts))
    parts.extend(conversation_parts)
    return "\n\n".join(parts)


class ChatCodexProxy(BaseChatModel):
    """LangChain chat model that proxies completions through the Codex CLI.

    The user must have Codex CLI installed and authenticated (``codex login``).
    Each call runs ``codex exec -`` with the prompt on stdin, a read-only
    sandbox, no session persistence, and reads the final assistant message
    from a temp file (``--output-last-message``) so the answer is clean of
    Codex's progress/log output.
    """

    model: str = "gpt-5.5"
    temperature: Optional[float] = None  # accepted for interface parity; CLI ignores it
    timeout_seconds: int = 600  # deep-model calls can take minutes
    json_schema: Optional[dict] = None  # set via with_structured_output()

    @property
    def _llm_type(self) -> str:
        return "codex-proxy"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model, "llm_type": self._llm_type}

    # ------------------------------------------------------------------ core

    def _build_command(self, output_path: str, schema_path: Optional[str]) -> list[str]:
        cmd = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            "read-only",
        ]
        if self.model:
            cmd.extend(["--model", self.model])
        cmd.extend(["--output-last-message", output_path])
        if schema_path:
            cmd.extend(["--output-schema", schema_path])
        cmd.append("-")  # read the prompt from stdin
        return cmd

    def _run_cli(self, prompt: str) -> str:
        fd, output_path = tempfile.mkstemp(prefix="sportagent-codex-", suffix=".txt")
        os.close(fd)
        schema_path: Optional[str] = None
        try:
            if self.json_schema:
                sfd, schema_path = tempfile.mkstemp(
                    prefix="sportagent-codex-schema-", suffix=".json"
                )
                with os.fdopen(sfd, "w", encoding="utf-8") as fh:
                    json.dump(self.json_schema, fh)

            cmd = self._build_command(output_path, schema_path)
            logger.debug("Codex CLI call: model=%s schema=%s", self.model, bool(self.json_schema))
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
                    "The `codex` CLI was not found on PATH. Install Codex "
                    "(npm install -g @openai/codex) or switch to API-key mode "
                    "with `sportagent setup`."
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Codex CLI timed out after {self.timeout_seconds}s. "
                    "Deep-model calls can be slow; increase timeout_seconds if needed."
                ) from exc

            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                stdout = (result.stdout or "").strip()
                detail = stderr or stdout or f"exit code {result.returncode}"
                lowered = detail.lower()
                if any(
                    tok in lowered
                    for tok in ("auth", "login", "credential", "api key", "unauthorized")
                ):
                    raise RuntimeError(f"Codex CLI auth error: {detail}\n{_AUTH_ERROR_HINT}")
                raise RuntimeError(f"Codex CLI error: {detail}")

            # Prefer the clean last-message file; fall back to stdout.
            try:
                with open(output_path, "r", encoding="utf-8") as fh:
                    last_message = fh.read().strip()
            except OSError:
                last_message = ""
            return last_message or (result.stdout or "").strip()
        finally:
            for path in (output_path, schema_path):
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _flatten_messages(messages)
        response_text = self._run_cli(prompt)
        message = AIMessage(content=response_text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # ------------------------------------------------- structured output

    def with_structured_output(self, schema: Type[BaseModel], **kwargs: Any) -> Any:
        """Return a runnable that yields parsed ``schema`` instances.

        Uses the CLI's native ``--output-schema`` flag so the model's final
        message is schema-shaped JSON, then parses it into the Pydantic model.
        """
        return _StructuredProxyRunnable(self, schema)


def _prepare_strict_schema(schema: dict) -> dict:
    """Make a Pydantic JSON schema acceptable to OpenAI strict structured output.

    OpenAI's ``response_format`` strict mode (used by Codex's
    ``--output-schema``) requires every object node to declare
    ``additionalProperties: false`` and to list **all** properties as
    ``required``. This walks the schema (including ``$defs``) and patches both.
    """

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                props = node.get("properties")
                if isinstance(props, dict):
                    node["required"] = list(props.keys())
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    prepared = json.loads(json.dumps(schema))  # deep copy
    _walk(prepared)
    return prepared


class _StructuredProxyRunnable:
    """Minimal runnable wrapper: invoke the proxy with a JSON schema, parse out."""

    def __init__(self, llm: ChatCodexProxy, schema: Type[BaseModel]):
        json_schema = _prepare_strict_schema(schema.model_json_schema())
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
    raise ValueError(f"No JSON object found in Codex CLI output: {text[:200]!r}")