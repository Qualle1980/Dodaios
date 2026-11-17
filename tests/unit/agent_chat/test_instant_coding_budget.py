"""Instant coding-dispatch envelope and Codex-always review."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from jarvis.agent_chat.delegation_acceptance import Receipt
from jarvis.agent_chat.review_routing import ReviewEvidence
from jarvis.brain import IterationBudget
from jarvis.brain.iteration_budget import IterationBudget as Budget
from jarvis.brain.tool_use_loop import ToolUseLoop, _BUDGET_FINAL_DIRECTIVE
from jarvis.core.protocols import BrainDelta, BrainRequest, ToolResult


ROOT = "work/bubble"


def _run(command: str, *, ok: bool = True, out: str = "x", cwd: str = ROOT) -> Receipt:
    return Receipt("RunCommand", {"command": command}, ok, out, cwd)


def _full_evidence() -> ReviewEvidence:
    return ReviewEvidence(
        receipts=(
            _run("git diff", out="diff --git a/calc.py b/calc.py\n+def subtract"),
            _run("python -m pytest -q", out="2 passed in 0.3s"),
        ),
        verdict_text="PASS — change read and tests are green.",
    )


def test_iteration_budget_default_is_50m_input_tokens() -> None:
    """Hard cost backstop only — not a context-saving Instant cap (7c985c1ba)."""
    assert IterationBudget().max_input_tokens_total == 50_000_000
    assert IterationBudget.max_input_tokens_total == 50_000_000


def test_chatgpt_plan_turn_uses_default_max_turns() -> None:
    from jarvis.agent_chat.runner_brain import MAX_TURNS, build_override
    from jarvis.agent_chat.store import AgentChatSession

    session = AgentChatSession(
        session_id="s1",
        title="",
        provider="chatgpt-plan",
        model="gpt-x",
        effort="",
        cwd=".",
        permission_mode="ask",
        vendor_session=None,
        created_ms=0,
        updated_ms=0,
        message_count=0,
        preview="",
        surface="jarvis",
    )
    override = build_override(
        session, brain=None, stance="ask", cwd=Path("."), ref="r"
    )
    assert override.max_turns == MAX_TURNS == 40

    ordinary = AgentChatSession(
        session_id="s2",
        title="",
        provider="openai",
        model="gpt-x",
        effort="",
        cwd=".",
        permission_mode="ask",
        vendor_session=None,
        created_ms=0,
        updated_ms=0,
        message_count=0,
        preview="",
        surface="jarvis",
    )
    other = build_override(
        ordinary, brain=None, stance="ask", cwd=Path("."), ref="r"
    )
    assert other.max_turns == 40


class _ListTool:
    name = "wiki-list"
    schema: dict[str, Any] = {}


class _ExecOK:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, tool: Any, args: dict[str, Any], **_: Any) -> ToolResult:
        self.calls.append(args)
        return ToolResult(success=True, output="- log.md")


class _GreedyBrain:
    """Calls a tool on every round that still offers tools."""

    def __init__(self) -> None:
        self.requests: list[BrainRequest] = []

    async def complete(self, req: BrainRequest) -> AsyncIterator[BrainDelta]:
        self.requests.append(req)
        if req.tools:
            yield BrainDelta(
                tool_call={"id": f"c{len(self.requests)}", "name": "wiki-list", "input": {}}
            )
            yield BrainDelta(finish_reason="tool_use")
            return
        yield BrainDelta(
            content="I have the wiki list; open gaps remain on the tests."
        )
        yield BrainDelta(finish_reason="stop")


@pytest.mark.asyncio
async def test_cap_hit_answers_now_with_gaps_and_no_new_round() -> None:
    """When an explicit round envelope is spent, answer now; no new tool round."""
    brain = _GreedyBrain()
    executor = _ExecOK()
    loop = ToolUseLoop(
        brain,
        {"wiki-list": _ListTool()},
        executor,  # type: ignore[arg-type]
        budget=Budget(max_turns=2, max_input_tokens_total=80_000),
    )
    result = await loop.run([], user_utterance="hand this coding task off and verify")
    assert len(brain.requests) == 3  # two tool rounds + one final
    assert not brain.requests[-1].tools
    joined = " ".join(
        str(getattr(m, "content", "")) for m in brain.requests[-1].messages
    ).lower()
    assert "round budget exhausted" in joined
    assert "gap" in _BUDGET_FINAL_DIRECTIVE.lower()
    assert "do not start another round" in _BUDGET_FINAL_DIRECTIVE.lower()
    assert "wiki" in result.text.lower() or "gap" in result.text.lower()


def test_codex_review_runs_even_when_instant_evidence_is_complete() -> None:
    from jarvis.agent_chat.codex_fallback import attempt_codex_fallback
    from jarvis.agent_chat.review_routing import choose_reviewer, instant_evidence_complete
    from tests.unit.agent_chat.test_codex_fallback import FakeRegistry, FakeTerminal, FakeWorkspace

    evidence = _full_evidence()
    assert instant_evidence_complete(evidence) is True
    assert choose_reviewer(evidence) == "codex"

    registry = FakeRegistry(
        workspaces=[FakeWorkspace("ws-1", terminals=[FakeTerminal("Codex", "codex", "codex-hist")])]
    )

    async def run() -> None:
        result = await attempt_codex_fallback(
            evidence=evidence,
            prompt="Review the handback.",
            registry=registry,
        )
        assert result.action == "sent"
        assert len(registry.sends) == 1

    asyncio.run(run())


def test_codex_unavailable_reports_review_not_run_never_passed() -> None:
    from jarvis.agent_chat.codex_fallback import attempt_codex_fallback
    from tests.unit.agent_chat.test_codex_fallback import FakeRegistry

    registry = FakeRegistry(workspaces=[])

    async def run() -> None:
        result = await attempt_codex_fallback(
            evidence=_full_evidence(),
            prompt="Review the handback.",
            registry=registry,
        )
        assert result.action == "missing"
        assert result.message
        lowered = result.message.lower()
        assert "review not run" in lowered
        assert "passed" not in lowered

    asyncio.run(run())
