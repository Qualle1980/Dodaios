"""A tool lead that cannot be reached is not a lead (latency, 2026-10-20).

Measured on a machine where Ollama is configured but not installed: a 20 s
answer spent 11 s on the lead — building a tool surface for qwen3.5:4b,
sizing its context, then a request that ended in ``Request timed out`` —
before the ChatGPT seat answered in the remaining 8 s. The health ledger had
already written ``ollama/brain none -> unreachable``; nothing asked it.

The session dead-list cannot cover this: it starts empty after every restart,
so the first turn of every run paid the connect timeout again.

Deterministic — no provider call, no network, no model.
"""
from __future__ import annotations

from typing import Any

import pytest

from jarvis.brain import provider_health_ledger as ledger_mod
from jarvis.brain import provider_test as pt
from jarvis.brain.manager import BrainManager
from jarvis.core.bus import EventBus
from jarvis.core.config import BrainTierConfig, JarvisConfig
from jarvis.core.protocols import ToolResult


@pytest.fixture(autouse=True)
def _ollama_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """These turns assume an installed local runtime (Ollama gates its tool
    capability on the binary since 2026-10-23)."""
    monkeypatch.setattr("jarvis.brain.ollama_runtime.find_binary", lambda: "/usr/bin/ollama")



class _Tool:
    name = "spawn_worker"
    risk_tier = "safe"
    description = "d"
    schema: dict[str, Any] = {"type": "object", "properties": {}}

    async def execute(self, args: Any, ctx: Any) -> ToolResult:
        return ToolResult(success=True, output="")


class _Executor:
    async def execute(self, tool: Any, args: Any, *a: Any, **k: Any) -> ToolResult:
        return ToolResult(success=True, output="")


class _Outcome:
    def __init__(self, status: str) -> None:
        self.status = status


class _Ledger:
    """Answers like the real ledger for the providers it was given."""

    def __init__(self, statuses: dict[str, str]) -> None:
        self._statuses = statuses
        self.asked: list[str] = []

    def effective(self, provider: str, modality: str, **_k: Any) -> _Outcome | None:
        self.asked.append(provider)
        status = self._statuses.get(provider)
        return _Outcome(status) if status else None


def _manager(primary: str = "chatgpt-web") -> BrainManager:
    config = JarvisConfig()
    config.brain.primary = primary
    config.brain.worker = BrainTierConfig(provider="openai-codex", model="")
    mgr = BrainManager(
        config=config, bus=EventBus(), tools={"spawn_worker": _Tool()},
        tool_executor=_Executor(),  # type: ignore[arg-type]
    )
    mgr._active_name = primary  # type: ignore[attr-defined]
    return mgr


@pytest.fixture
def ledger(monkeypatch: pytest.MonkeyPatch):
    def _install(statuses: dict[str, str]) -> _Ledger:
        fake = _Ledger(statuses)
        monkeypatch.setattr(ledger_mod, "get_ledger", lambda: fake)
        return fake

    return _install


def test_an_unreachable_runtime_is_not_chosen_as_the_lead(ledger) -> None:
    mgr = _manager()
    unled = mgr._first_tool_capable_provider("deep")
    assert unled is not None, "premise: this install has a lead at all"

    ledger({unled[0]: pt.UNREACHABLE})

    picked = mgr._first_tool_capable_provider("deep")

    assert picked is None or picked[0] != unled[0]


def test_an_account_problem_does_not_veto_the_lead(ledger) -> None:
    """Only a network answer counts. 'No credits' is the chain walk's business,
    and it reports that honestly — silently skipping it here would hide it."""
    mgr = _manager()
    unled = mgr._first_tool_capable_provider("deep")
    assert unled is not None

    ledger({unled[0]: pt.NO_CREDITS})

    assert mgr._first_tool_capable_provider("deep") == unled


def test_a_healthy_provider_is_untouched(ledger) -> None:
    mgr = _manager()
    unled = mgr._first_tool_capable_provider("deep")
    assert unled is not None

    ledger({unled[0]: pt.OK})

    assert mgr._first_tool_capable_provider("deep") == unled


def test_an_unreadable_ledger_never_vetoes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken probe must not cost the turn its lead."""
    mgr = _manager()
    unled = mgr._first_tool_capable_provider("deep")
    assert unled is not None

    def _boom() -> Any:
        raise RuntimeError("ledger unavailable")

    monkeypatch.setattr(ledger_mod, "get_ledger", _boom)

    assert mgr._first_tool_capable_provider("deep") == unled
