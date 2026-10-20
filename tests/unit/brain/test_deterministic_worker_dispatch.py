"""Deterministic dispatch reaches the worker for every talker (phase 4).

An explicit heavy request is identified without a model: ``_should_force_spawn``
matches an explicit trigger and ``_force_spawn_worker`` calls the spawn tool
directly, before any provider chain is built. One gate stood in front of that
path and asked the wrong question. With no worker provider configured it
refused the whole dispatch unless the TALKER was claude-api or
gemini — measured on a default install, chatgpt-web, chatgpt-plan, claude-cli,
ollama and openai were all refused, although the factory resolves an
unconfigured worker to the user's ACTIVE brain and builds a real worker for it.
The refusal pushed an already identified action back onto an LLM routing round
it did not need.

The gate now asks the mission layer's own routing function with the same input
the factory gives it, and refuses only BUG-017's blind Gemini spawn — an
install with no provider configured at all.

Deterministic — no provider call, no network, no model inference.
"""
from __future__ import annotations

from typing import Any

import pytest

from jarvis.brain.manager import BrainManager
from jarvis.core.bus import EventBus
from jarvis.core.config import BrainTierConfig, JarvisConfig
from jarvis.core.protocols import ExecutionContext, ToolResult

_TALKERS = ("chatgpt-web", "chatgpt-plan", "claude-cli", "ollama", "openai")


class _SpawnTool:
    name = "spawn_worker"
    risk_tier = "safe"
    description = "Hand heavy work to a worker"
    schema: dict[str, Any] = {"type": "object", "properties": {}}

    def __init__(self, result: ToolResult | None = None) -> None:
        self.calls: list[dict] = []
        self._result = result or ToolResult(success=True, output="worker finished")

    async def execute(self, args, ctx: ExecutionContext) -> ToolResult:
        self.calls.append(args)
        return self._result


class _RecordingExecutor:
    """Stands in for ToolExecutor: records, runs the tool, changes no policy."""

    def __init__(self, tool: _SpawnTool) -> None:
        self.tool = tool
        self.executions: list[str] = []

    async def execute(self, tool, args, *a: Any, **k: Any) -> ToolResult:
        # The executor is handed the tool itself, not its name.
        self.executions.append(getattr(tool, "name", tool))
        return await self.tool.execute(args, None)  # type: ignore[arg-type]


def _manager(
    primary: str,
    *,
    worker: BrainTierConfig | None = None,
    tool: _SpawnTool | None = None,
) -> tuple[BrainManager, _SpawnTool, _RecordingExecutor]:
    config = JarvisConfig()
    config.brain.primary = primary
    config.brain.worker = worker
    spawn = tool or _SpawnTool()
    executor = _RecordingExecutor(spawn)
    mgr = BrainManager(
        config=config,
        bus=EventBus(),
        tools={spawn.name: spawn},
        tool_executor=executor,  # type: ignore[arg-type]
    )
    mgr._active_name = primary  # type: ignore[attr-defined]
    return mgr, spawn, executor


# --------------------------------------------------------------------------
# The gate.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("primary", _TALKERS)
def test_the_default_worker_route_is_viable_for_any_talker(primary: str) -> None:
    mgr, _, _ = _manager(primary)

    assert mgr._heavy_worker_provider_viable() is True


@pytest.mark.parametrize("primary", ["chatgpt-web", "ollama", "gemini", "claude-api"])
def test_a_gemini_step_model_cannot_divert_a_resolved_provider(primary: str) -> None:
    """A resolved provider is a hard lock: the step model routes nothing.

    An earlier version of this gate refused these configs, reasoning that a
    gemini step model would reach the Gemini API worker. It does not — the
    worker factory resolves the provider first, and only an install with NO
    provider at all can land there (see test_worker_precheck_parity).
    """
    mgr, _, _ = _manager(
        primary, worker=BrainTierConfig(provider="", model="gemini-3-flash")
    )

    assert mgr._heavy_worker_provider_viable() is True


def test_a_configured_worker_is_viable_whatever_the_talker() -> None:
    mgr, _, _ = _manager(
        "chatgpt-web", worker=BrainTierConfig(provider="openai-codex", model="")
    )

    assert mgr._heavy_worker_provider_viable() is True


# --------------------------------------------------------------------------
# The dispatch itself: identified without a model, run without one.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("primary", _TALKERS)
def test_an_explicit_request_is_identified_without_a_model(primary: str) -> None:
    mgr, _, _ = _manager(primary)

    assert mgr._should_force_spawn(
        "spawne einen subagenten und baue den health endpoint"  # i18n-allow: trigger
    ) is True


@pytest.mark.asyncio
async def test_the_worker_runs_and_its_result_comes_back_with_no_model_call() -> None:
    mgr, spawn, executor = _manager("chatgpt-web")
    mgr._get_brain = _no_brain  # type: ignore[assignment]

    answer = await mgr._force_spawn_worker(
        "spawne einen subagenten und baue den health endpoint"  # i18n-allow: trigger
    )

    assert executor.executions == ["spawn_worker"], "the dispatch must go to the worker"
    assert answer is not None and "worker finished" in answer
    assert len(spawn.calls) == 1, "exactly one dispatch, never two"


@pytest.mark.asyncio
async def test_a_failing_worker_is_never_reported_as_success() -> None:
    failing = _SpawnTool(ToolResult(success=False, output=None, error="exit 1"))
    mgr, _, _ = _manager("chatgpt-web", tool=failing)
    mgr._get_brain = _no_brain  # type: ignore[assignment]

    answer = await mgr._force_spawn_worker(
        "spawne einen subagenten und baue den health endpoint"  # i18n-allow: trigger
    )

    assert answer is not None
    assert "worker finished" not in answer
    assert "exit 1" not in answer, "the opaque exit token must never be spoken"


def _no_brain(*_a: Any, **_k: Any):
    raise AssertionError("the deterministic dispatch must not call a model")
