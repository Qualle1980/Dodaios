"""ClaudeDirectWorker against a FAKE claude CLI: what it runs, and what it reports.

No model is called anywhere here. The "CLI" is a small Python script this test
writes, which records the argv it was given and prints the stream-json records
a real ``claude --print --output-format stream-json`` would print — including
the ``system`` init record that names the model the CLI actually loaded.

The defect being pinned: on 2026-10-28 a mission asked for ``haiku``, Opus ran,
and the telemetry still said haiku.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from jarvis.missions.workers import claude_direct_worker as cdw
from jarvis.missions.workers.claude_direct_worker import ClaudeDirectWorker
from jarvis.missions.workers.model_policy import DEFAULT_WORKER_MODEL, BillingSource

_FAKE_CLI = '''
import json, sys, pathlib
argv = sys.argv[1:]
record = pathlib.Path(sys.argv[0]).with_suffix(".argv.json")
record.write_text(json.dumps(argv), encoding="utf-8")
model = argv[argv.index("--model") + 1] if "--model" in argv else ""
sys.stdin.read()
reported = __FAKE_REPORTED_MODEL__ or model
if __FAKE_REJECT_MODELS__ and model in __FAKE_REJECT_MODELS__:
    print(json.dumps({"type": "result", "subtype": "error_during_execution",
                      "is_error": True, "session_id": "fake",
                      "result": "API Error: the requested model is currently unavailable"}))
    sys.exit(0)
print(json.dumps({"type": "system", "subtype": "init", "model": reported,
                  "session_id": "fake"}))
print(json.dumps({"type": "assistant", "message": {"content": [
    {"type": "tool_use", "name": "Write", "input": {}}]}}))
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "session_id": "fake", "total_cost_usd": 0.0, "num_turns": 1,
                  "result": "done"}))
'''


class _Job:
    def assign(self, pid: int) -> None:
        return None


@pytest.fixture
def fake_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Install a fake claude CLI; returns a builder for its behaviour."""

    def build(*, reported_model: str | None = None, reject: tuple[str, ...] = ()) -> Path:
        script = tmp_path / "fake_claude.py"
        script.write_text(
            _FAKE_CLI.replace(
                "__FAKE_REPORTED_MODEL__", json.dumps(reported_model or "")
            ).replace("__FAKE_REJECT_MODELS__", json.dumps(list(reject))),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            cdw, "_resolve_claude_argv_prefix", lambda: [sys.executable, str(script)]
        )
        monkeypatch.setattr(
            "jarvis.claude_auth.claude_cli_supports_safe_mode", lambda _prefix: False
        )
        # No configured model: the policy default applies unless a test pins one.
        monkeypatch.setattr(cdw, "_configured_claude_model", lambda _primary: "")
        monkeypatch.setattr(cdw, "_resolve_provider_chain", lambda: ())
        return script

    return build


async def _run(
    worker: ClaudeDirectWorker,
    *,
    tmp_path: Path,
    model: str,
    env: dict[str, str] | None = None,
) -> list[Any]:
    worktree = tmp_path / "ws"
    worktree.mkdir(exist_ok=True)
    log_dir = tmp_path / "logs"
    return [
        ev
        async for ev in worker.spawn(
            "write a file",
            worktree=worktree,
            env=env if env is not None else {},
            job=_Job(),
            worker_id="w1",
            log_dir=log_dir,
            model=model,
            allowed_tools="Write",
        )
    ]


def _argv(script: Path) -> list[str]:
    return json.loads(script.with_suffix(".argv.json").read_text(encoding="utf-8"))


def _model_arg(script: Path) -> str:
    argv = _argv(script)
    assert "--model" in argv, f"--model missing from {argv}"
    return argv[argv.index("--model") + 1]


# --- what runs -------------------------------------------------------------


@pytest.mark.asyncio
async def test_haiku_requested_is_haiku_executed(fake_cli, tmp_path: Path) -> None:
    script = fake_cli()
    worker = ClaudeDirectWorker()
    await _run(worker, tmp_path=tmp_path, model="haiku")
    assert _model_arg(script) == "haiku"


@pytest.mark.asyncio
async def test_haiku_requested_never_silently_becomes_opus(
    fake_cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even with Opus configured, an explicit cheap pick is what runs."""
    script = fake_cli()
    monkeypatch.setattr(cdw, "_configured_claude_model", lambda _p: "claude-opus-4-8")
    worker = ClaudeDirectWorker()
    await _run(worker, tmp_path=tmp_path, model="haiku")
    assert _model_arg(script) == "haiku"
    assert "opus" not in " ".join(_argv(script))


@pytest.mark.asyncio
async def test_no_request_and_no_config_does_not_reach_opus(
    fake_cli, tmp_path: Path
) -> None:
    script = fake_cli()
    worker = ClaudeDirectWorker()
    await _run(worker, tmp_path=tmp_path, model="")
    assert _model_arg(script) == DEFAULT_WORKER_MODEL


@pytest.mark.asyncio
async def test_a_stale_codex_login_fallback_does_not_escalate_the_tier(
    fake_cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production path: codex login dead -> ClaudeDirectWorker.

    The factory's decision moves the FAMILY. It approves no tier, and the
    step's codex model id is not a Claude model, so the worker must land on
    the default tier and never on the premium one.
    """
    script = fake_cli()

    class _Primary:
        provider = "openai-codex"
        model = "gpt-5-codex"

    monkeypatch.setattr(cdw, "_resolve_provider_chain", lambda: (_Primary(),))
    monkeypatch.setattr(cdw, "_configured_claude_model", lambda _p: "claude-opus-4-8")
    worker = ClaudeDirectWorker()
    await _run(worker, tmp_path=tmp_path, model="gpt-5-codex")
    assert _model_arg(script) == DEFAULT_WORKER_MODEL


# --- the rejection path ----------------------------------------------------


@pytest.mark.asyncio
async def test_a_rejected_model_retries_one_step_cheaper_with_an_explicit_flag(
    fake_cli, tmp_path: Path
) -> None:
    """``--model`` is never dropped: the retry names a cheaper model itself."""
    script = fake_cli(reject=("claude-opus-4-8",))
    worker = ClaudeDirectWorker()
    events = await _run(worker, tmp_path=tmp_path, model="claude-opus-4-8")
    assert _model_arg(script) == "sonnet"
    assert not any(
        getattr(ev, "is_error", False) for ev in events if ev.__class__.__name__ == "ClaudeResult"
    )


@pytest.mark.asyncio
async def test_an_exhausted_ladder_fails_instead_of_letting_the_cli_choose(
    fake_cli, tmp_path: Path
) -> None:
    script = fake_cli(reject=("haiku",))
    worker = ClaudeDirectWorker()
    events = await _run(worker, tmp_path=tmp_path, model="haiku")
    assert _model_arg(script) == "haiku"  # never retried without the flag
    results = [ev for ev in events if ev.__class__.__name__ == "ClaudeResult"]
    assert results and results[-1].is_error


# --- telemetry -------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_actual_model_comes_from_the_cli_not_from_the_request(
    fake_cli, tmp_path: Path
) -> None:
    """The CLI's own init record is authoritative, even when it disagrees."""
    fake_cli(reported_model="claude-opus-4-8")
    worker = ClaudeDirectWorker()
    await _run(worker, tmp_path=tmp_path, model="haiku")
    assert worker.last_requested_model == "haiku"
    assert worker.last_actual_model == "claude-opus-4-8"


@pytest.mark.asyncio
async def test_an_unproven_model_is_reported_unknown_not_guessed(
    fake_cli, tmp_path: Path
) -> None:
    """A CLI that names no model leaves ``unknown``, never the request."""
    fake_cli(reject=("haiku",))  # errors out before any system record
    worker = ClaudeDirectWorker()
    await _run(worker, tmp_path=tmp_path, model="haiku")
    assert worker.last_actual_model == "unknown"


@pytest.mark.asyncio
async def test_the_billing_source_is_read_from_the_worker_environment(
    fake_cli, tmp_path: Path
) -> None:
    fake_cli()
    worker = ClaudeDirectWorker()
    await _run(
        worker,
        tmp_path=tmp_path,
        model="haiku",
        env={"ANTHROPIC_API_KEY": "sk-ant-api03-test"},
    )
    assert worker.last_billing_source is BillingSource.API_KEY


@pytest.mark.asyncio
async def test_no_credential_means_unknown_billing_not_a_free_run(
    fake_cli, tmp_path: Path
) -> None:
    fake_cli()
    worker = ClaudeDirectWorker()
    await _run(worker, tmp_path=tmp_path, model="haiku", env={})
    assert worker.last_billing_source is BillingSource.UNKNOWN
