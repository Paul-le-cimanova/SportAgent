"""Tests for the CLI proxy adapters (Claude Code / Codex) + factory routing.

All subprocess interaction is mocked — no live CLI calls. Live behavior was
verified manually against claude 2.1.87 and codex-cli 0.139.0.
"""

from __future__ import annotations

import json
import subprocess
from unittest import mock

import pytest
from pydantic import BaseModel

from sportagent.core.llm_clients import claude_code_proxy, codex_proxy
from sportagent.core.llm_clients.claude_code_proxy import (
    ChatClaudeCodeProxy,
    is_claude_code_available,
)
from sportagent.core.llm_clients.codex_proxy import (
    ChatCodexProxy,
    _prepare_strict_schema,
    is_codex_available,
)
from sportagent.core.llm_clients.factory import create_llm_client


class _Pick(BaseModel):
    winner: str
    win_prob: float


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# --- detection ---------------------------------------------------------------


def test_claude_detection_missing_binary():
    with mock.patch.object(claude_code_proxy.shutil, "which", return_value=None):
        assert is_claude_code_available() is False


def test_claude_detection_ok():
    with mock.patch.object(claude_code_proxy.shutil, "which", return_value="/usr/bin/claude"), \
         mock.patch.object(claude_code_proxy.subprocess, "run", return_value=_completed("2.1.87")):
        assert is_claude_code_available() is True


def test_codex_detection_missing_binary():
    with mock.patch.object(codex_proxy.shutil, "which", return_value=None):
        assert is_codex_available() is False


def test_codex_detection_ok():
    with mock.patch.object(codex_proxy.shutil, "which", return_value="/usr/bin/codex"), \
         mock.patch.object(codex_proxy.subprocess, "run", return_value=_completed("codex-cli 0.139.0")):
        assert is_codex_available() is True


# --- Claude Code proxy --------------------------------------------------------


def test_claude_proxy_command_shape():
    llm = ChatClaudeCodeProxy(model="haiku")
    cmd = llm._build_command("be terse")
    assert cmd[:2] == ["claude", "-p"]
    assert "--no-session-persistence" in cmd
    assert "--model" in cmd and "haiku" in cmd
    assert "--system-prompt" in cmd and "be terse" in cmd
    # tools disabled for pure completion
    i = cmd.index("--tools")
    assert cmd[i + 1] == ""


def test_claude_proxy_invoke_text():
    llm = ChatClaudeCodeProxy(model="haiku")
    with mock.patch.object(
        claude_code_proxy.subprocess, "run", return_value=_completed("PONG")
    ) as run:
        result = llm.invoke("ping")
    assert result.content == "PONG"
    # the prompt travels via stdin
    assert run.call_args.kwargs["input"] == "ping"


def test_claude_proxy_structured_output_envelope():
    llm = ChatClaudeCodeProxy(model="haiku")
    envelope = json.dumps(
        {"type": "result", "is_error": False, "result": "ok",
         "structured_output": {"winner": "Lakers", "win_prob": 0.7}}
    )
    structured = llm.with_structured_output(_Pick)
    with mock.patch.object(claude_code_proxy.subprocess, "run", return_value=_completed(envelope)) as run:
        pick = structured.invoke("who wins?")
    assert isinstance(pick, _Pick)
    assert pick.winner == "Lakers" and pick.win_prob == 0.7
    # schema mode adds --json-schema and --output-format json
    cmd = run.call_args.args[0]
    assert "--json-schema" in cmd and "--output-format" in cmd


def test_claude_proxy_error_raises():
    llm = ChatClaudeCodeProxy(model="haiku")
    with mock.patch.object(
        claude_code_proxy.subprocess, "run",
        return_value=_completed(stderr="boom", returncode=1),
    ):
        with pytest.raises(RuntimeError, match="Claude Code CLI error"):
            llm.invoke("ping")


def test_claude_proxy_auth_error_hint():
    llm = ChatClaudeCodeProxy(model="haiku")
    with mock.patch.object(
        claude_code_proxy.subprocess, "run",
        return_value=_completed(stderr="Not authenticated, please login", returncode=1),
    ):
        with pytest.raises(RuntimeError, match="auth"):
            llm.invoke("ping")


# --- Codex proxy ---------------------------------------------------------------


def test_codex_proxy_command_shape():
    llm = ChatCodexProxy(model="gpt-5.5")
    cmd = llm._build_command("/tmp/out.txt", None)
    assert cmd[:2] == ["codex", "exec"]
    assert "--skip-git-repo-check" in cmd
    assert "--ephemeral" in cmd
    assert "--output-last-message" in cmd
    assert cmd[-1] == "-"  # prompt from stdin


def test_codex_proxy_invoke_reads_last_message_file():
    llm = ChatCodexProxy(model="gpt-5.5")

    def fake_run(cmd, **kwargs):
        out_path = cmd[cmd.index("--output-last-message") + 1]
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("PONG")
        return _completed("progress noise on stdout")

    with mock.patch.object(codex_proxy.subprocess, "run", side_effect=fake_run):
        result = llm.invoke("ping")
    assert result.content == "PONG"


def test_codex_proxy_structured_output():
    llm = ChatCodexProxy(model="gpt-5.5")
    structured = llm.with_structured_output(_Pick)

    def fake_run(cmd, **kwargs):
        # schema file must be passed
        assert "--output-schema" in cmd
        schema_path = cmd[cmd.index("--output-schema") + 1]
        schema = json.load(open(schema_path))
        assert schema["additionalProperties"] is False
        out_path = cmd[cmd.index("--output-last-message") + 1]
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"winner": "Lakers", "win_prob": 0.7}))
        return _completed()

    with mock.patch.object(codex_proxy.subprocess, "run", side_effect=fake_run):
        pick = structured.invoke("who wins?")
    assert isinstance(pick, _Pick)
    assert pick.winner == "Lakers" and pick.win_prob == 0.7


def test_prepare_strict_schema_patches_objects():
    schema = _prepare_strict_schema(_Pick.model_json_schema())
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == ["win_prob", "winner"]


def test_codex_proxy_error_raises():
    llm = ChatCodexProxy(model="gpt-5.5")
    with mock.patch.object(
        codex_proxy.subprocess, "run",
        return_value=_completed(stderr="boom", returncode=1),
    ):
        with pytest.raises(RuntimeError, match="Codex CLI error"):
            llm.invoke("ping")


# --- JSON extraction -----------------------------------------------------------


@pytest.mark.parametrize("extract", [claude_code_proxy._extract_json, codex_proxy._extract_json])
def test_extract_json_variants(extract):
    assert extract('{"a": 1}') == {"a": 1}
    assert extract('prose\n```json\n{"b": 2}\n```\nafter') == {"b": 2}
    assert extract('text {"c": 3} text') == {"c": 3}
    with pytest.raises((ValueError, json.JSONDecodeError)):
        extract("no json here")


# --- factory routing ------------------------------------------------------------


def test_factory_routes_anthropic_cli_proxy():
    with mock.patch.object(claude_code_proxy, "is_claude_code_available", return_value=True):
        client = create_llm_client("anthropic", "haiku", auth_method="cli_proxy")
    assert isinstance(client, ChatClaudeCodeProxy)
    assert client.model == "haiku"


def test_factory_routes_openai_cli_proxy():
    with mock.patch.object(codex_proxy, "is_codex_available", return_value=True):
        client = create_llm_client("openai", "gpt-5.5", auth_method="cli_proxy")
    assert isinstance(client, ChatCodexProxy)
    assert client.model == "gpt-5.5"


def test_factory_cli_proxy_missing_cli_raises():
    with mock.patch.object(claude_code_proxy, "is_claude_code_available", return_value=False):
        with pytest.raises(RuntimeError, match="claude"):
            create_llm_client("anthropic", "haiku", auth_method="cli_proxy")


def test_factory_unknown_auth_method_raises():
    with pytest.raises(ValueError, match="llm_auth_method"):
        create_llm_client("anthropic", "haiku", auth_method="bogus")


def test_factory_cli_proxy_unknown_provider_raises():
    with pytest.raises(ValueError, match="cli_proxy"):
        create_llm_client("google", "gemini", auth_method="cli_proxy")


# --- onboarding conditional requirements ---------------------------------------


def test_onboarding_cli_proxy_makes_llm_keys_optional():
    from sportagent import onboarding

    env = {"SPORTAGENT_LLM_PROVIDER": "anthropic", "SPORTAGENT_LLM_AUTH_METHOD": "cli_proxy"}
    results = onboarding.check_environment(env=env)
    llm_rows = [r for r in results if r.name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")]
    assert all(not r.required for r in llm_rows)


def test_onboarding_api_key_openai_requires_openai_key_only():
    from sportagent import onboarding

    env = {"SPORTAGENT_LLM_PROVIDER": "openai", "SPORTAGENT_LLM_AUTH_METHOD": "api_key"}
    results = onboarding.check_environment(env=env)
    req = {r.name: r.required for r in results if r.name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
    assert req == {"ANTHROPIC_API_KEY": False, "OPENAI_API_KEY": True}


def test_onboarding_default_is_backward_compatible():
    from sportagent import onboarding

    results = onboarding.check_environment(env={})
    req = {r.name: r.required for r in results if r.name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}
    assert req == {"ANTHROPIC_API_KEY": True, "OPENAI_API_KEY": False}