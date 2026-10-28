"""Regression for the 2026-06-08 sub-mission instant-fail bug.

Forensic ground truth (`data/missions.db`, missions ``019ea82e-3fac`` /
``019ea82e-8cc5`` / ``019ea830-072f``): every mission died in ~3 s with
``WorkerKilled reason=user`` and ``MissionFailed reason=task_error``. The
verbatim worker error in ``data/jarvis_desktop.log`` was::

    ClaudeDirectWorker: primary provider is grok, expected claude-api
    ClaudeDirectWorker: primary provider is openai-codex, expected claude-api

Root cause: post-Welle-4 the openclaw-backed ``SubJarvisWorker`` was removed
(the ~92% nested-claude hang), so ``jarvis.missions.init._worker_factory``
routes EVERY ``[brain.sub_jarvis].provider`` that is not ``openai-codex`` /
``chatgpt`` (grok, gemini, openai, openrouter, openclaw-claude, AND the unset
default — whose chain resolver falls back to the ``("grok", "grok-4.3")`` stub)
to ``ClaudeDirectWorker``. But the worker still carried a guard that *refused*
to run unless the resolved provider was ``claude-api`` — a fall-through to a
worker that no longer exists. The result: with ``sub_jarvis.provider`` set to
anything but ``claude-api`` / codex, EVERY mission failed instantly.

The fix: ``ClaudeDirectWorker`` is the universal heavy-worker fallback; it now
always runs on the Claude Max OAuth ``claude`` CLI with a *valid claude model*
(never a foreign slug like ``grok-4.3``), only LOGGING when the configured
provider differs (anti-silent-fallback — legible, never a hidden swap).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jarvis.missions.workers import claude_direct_worker as cdw
from jarvis.missions.workers.claude_direct_worker import ClaudeDirectWorker
from jarvis.missions.workers.provider_chain import _FallbackStep


class _FakeStream:
    def __init__(self, *, data: bytes = b"") -> None:
        self._data = data
        self._sent = False
        self._lines = data.splitlines(keepends=True)
        self._line_idx = 0

    async def read(self, n: int = -1) -> bytes:
        if self._sent:
            return b""
        self._sent = True
        return self._data

    async def readline(self) -> bytes:
        # The worker now streams via readline(); serve the data line-by-line.
        if self._line_idx >= len(self._lines):
            return b""
        line = self._lines[self._line_idx]
        self._line_idx += 1
        return line

    def write(self, _b: bytes) -> None:
        pass

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass


class _OkProc:
    """A subprocess that streams one successful result line then exits 0."""

    def __init__(self, result_line: bytes) -> None:
        self.pid = 4242
        self.returncode: int | None = 0
        self.stdin = _FakeStream()
        self.stdout = _FakeStream(data=result_line)
        self.stderr = _FakeStream()

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return (b"", b"")


class _Job:
    def assign(self, _pid: int) -> None:
        pass


_RESULT_LINE = (
    b'{"type":"result","subtype":"success","is_error":false,'
    b'"result":"OK","session_id":"s1"}\n'
)


@pytest.mark.parametrize("provider,model", [
    ("gemini", "gemini-3.1-pro-preview"),  # unset sub_jarvis -> chain stub default
    ("openai", "gpt-5.5-pro"),
    ("openrouter", "some/model"),
])
@pytest.mark.asyncio
async def test_non_claude_provider_runs_on_claude_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str, model: str
) -> None:
    """A non-claude sub_jarvis provider routed here MUST run on Claude, never
    refuse with 'primary provider is X, expected claude-api'."""
    monkeypatch.setattr(
        cdw, "_resolve_provider_chain",
        lambda *a, **k: (_FallbackStep(provider, model),),
    )
    monkeypatch.setattr(
        "jarvis.claude_auth.claude_cli_supports_safe_mode",
        lambda _prefix: False,
    )

    captured: dict[str, Any] = {}

    async def _fake_exec(*a: Any, **_k: Any) -> _OkProc:
        captured["argv"] = list(a)
        return _OkProc(_RESULT_LINE)

    monkeypatch.setattr(cdw.asyncio, "create_subprocess_exec", _fake_exec)

    worker = ClaudeDirectWorker()
    events: list[Any] = []
    async for ev in worker.spawn(
        "do the task",
        worktree=tmp_path,
        env={},
        job=_Job(),
        worker_id="fb",
        log_dir=tmp_path / "logs",
        first_output_timeout_s=5.0,
        timeout_s=5.0,
    ):
        events.append(ev)

    final = events[-1]
    # 1. It must NOT refuse with the provider-mismatch error.
    assert not (
        getattr(final, "is_error", False)
        and "expected claude-api" in (getattr(final, "result", "") or "")
    ), f"worker refused instead of falling back to Claude: {getattr(final, 'result', None)!r}"
    # 2. It must have actually reached the subprocess spawn (no early bail).
    assert "argv" in captured, "worker bailed before create_subprocess_exec"
    argv = captured["argv"]
    # 3. --model must be a real claude model, never the foreign provider slug.
    assert model not in argv, f"foreign provider model leaked into claude argv: {argv}"
    mi = argv.index("--model")
    model_arg = argv[mi + 1]
    assert model_arg.startswith("claude") or model_arg in {"opus", "sonnet", "haiku"}, (
        f"--model must be a claude model, got {model_arg!r}"
    )
    # 4. The run completes normally.
    assert getattr(final, "is_error", None) is False
    assert final.result == "OK"


@pytest.mark.asyncio
async def test_safe_mode_preserves_native_auth_without_loading_user_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Modern Claude keeps Keychain auth while safe mode disables customizations."""
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(
        cdw,
        "_resolve_provider_chain",
        lambda *a, **k: (_FallbackStep("claude-api", "claude-opus-4-8"),),
    )
    monkeypatch.setattr(cdw, "_resolve_claude_argv_prefix", lambda: ["claude"])
    monkeypatch.setattr(
        "jarvis.claude_auth.claude_cli_supports_safe_mode",
        lambda _prefix: True,
    )
    captured: dict[str, Any] = {}

    async def _fake_exec(*args: Any, **kwargs: Any) -> _OkProc:
        captured["argv"] = list(args)
        captured["env"] = dict(kwargs["env"])
        return _OkProc(_RESULT_LINE)

    monkeypatch.setattr(cdw.asyncio, "create_subprocess_exec", _fake_exec)
    env = {
        "HOME": "/Users/tester",
        "USER": "tester",
        "CLAUDE_CONFIG_DIR": str(tmp_path / "isolated-claude"),
    }

    events: list[Any] = []
    async for event in ClaudeDirectWorker().spawn(
        "do the task",
        worktree=tmp_path,
        env=env,
        job=_Job(),
        worker_id="native-auth",
        log_dir=tmp_path / "logs-native-auth",
        first_output_timeout_s=5.0,
        timeout_s=5.0,
    ):
        events.append(event)

    assert events[-1].is_error is False
    assert "--safe-mode" in captured["argv"]
    assert "CLAUDE_CONFIG_DIR" not in captured["env"]
    assert captured["env"]["HOME"] == "/Users/tester"
    assert captured["env"]["USER"] == "tester"


def test_configured_model_never_returns_a_foreign_slug() -> None:
    """``_configured_claude_model`` reads the operator's Claude pin only.

    A foreign chain model (``gemini-3.1-pro-preview``) is not a Claude model
    and must never reach ``claude --model`` — the 2026-06-08 instant-fail bug.
    The resolver answers "" for it and the policy picks the tier instead.
    """
    foreign = cdw._configured_claude_model(
        _FallbackStep("gemini", "gemini-3.1-pro-preview")
    )
    assert "gemini" not in foreign
    # A claude-api primary with a model IS an explicit approval of that tier.
    assert (
        cdw._configured_claude_model(_FallbackStep("claude-api", "claude-opus-4-8"))
        == "claude-opus-4-8"
    )
    # A None primary must not crash; it simply configures nothing.
    assert isinstance(cdw._configured_claude_model(None), str)


def test_the_last_resort_default_is_neither_fable_nor_premium(monkeypatch) -> None:
    """Two rules at once, and they no longer agree with the old constant.

    ``claude-fable-5`` is approved-access-only and the Claude Max subscription
    cannot reach it via the CLI ("Claude Fable 5 is currently unavailable",
    live mission 019ec615), so it may never be the default — that part is the
    maintainer decision of 2026-06-14 and still holds.

    What changed is the other half. The replacement default was
    ``claude-opus-4-8``, the dearest tier, and because ``spawn()`` ignored the
    caller's ``model`` it won over an explicit cheap pick: a mission asking for
    haiku ran on Opus (live 2026-10-28). An absent configuration must resolve
    DOWN, so the default is the middle tier.
    """
    from jarvis.missions.workers.model_policy import (
        DEFAULT_WORKER_MODEL,
        ModelTier,
        resolve_worker_model,
        tier_of,
    )

    assert DEFAULT_WORKER_MODEL != "claude-fable-5"
    assert tier_of(DEFAULT_WORKER_MODEL) is ModelTier.STANDARD

    # With config access broken entirely, nothing is configured and the policy
    # default applies — no crash, no premium tier.
    def _boom(*_a, **_k):
        raise RuntimeError("config unavailable")

    monkeypatch.setattr("jarvis.core.config.load_config", _boom)
    assert cdw._configured_claude_model(None) == ""
    resolved = resolve_worker_model(requested="", configured="").model
    assert resolved == DEFAULT_WORKER_MODEL
    assert tier_of(resolved) < ModelTier.PREMIUM
