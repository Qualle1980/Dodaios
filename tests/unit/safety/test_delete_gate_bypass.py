"""Destructive actions must not be answered by bypass (decision 0002).

On the base commit, ``approval_mode=bypass`` publishes
``ActionApproved(approved_by="bypass")`` for every undecided ask-tier ticket,
including a destructive shell delete. These tests pin the six outcomes the
delete gate must hold after the fix.
"""
from __future__ import annotations

import sys
from typing import Any

import pytest

from jarvis.core.bus import EventBus
from jarvis.core.config import BrainPlausibilityConfig, SafetyConfig
from jarvis.core.events import (
    ActionApprovalRequired,
    ActionApproved,
    ActionExecuted,
)
from jarvis.core.protocols import ExecutionContext, ToolResult, Transcript
from jarvis.safety.approval import ApprovalWorkflow
from jarvis.safety.approval_surface import CONVERSATIONAL, UNATTENDED
from jarvis.safety.command_impact import DESTRUCTIVE, classify_command
from jarvis.safety.explicit_intent import command_confirms_destruction
from jarvis.safety.risk_tier import RiskTierEvaluator
from jarvis.safety.tool_executor import (
    APPROVAL_UNAVAILABLE_PREFIX,
    ToolExecutor,
)

# explicit_intent accepts only an absolute path of the host platform: a Windows
# drive path on win32, a POSIX path elsewhere. A fixed Windows path made the
# explicit-intent test pass on Windows and fail on the Linux runner.
_TARGET = r"C:\tmp\x" if sys.platform == "win32" else "/tmp/x"
_DESTRUCTIVE_CMD = f"rm {_TARGET}"
_DESTRUCTIVE_UTTERANCE = f"delete {_TARGET}"
_UNATTENDED = {"approval_surface": UNATTENDED}
_CONVERSATIONAL = {"approval_surface": CONVERSATIONAL}


class _ShellTool:
    """Fake run_shell-shaped tool with an execution counter (never unittest.mock)."""

    name = "run_shell"
    risk_tier = "monitor"
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def __init__(self) -> None:
        self.calls = 0
        self.last_ctx: ExecutionContext | None = None

    def risk_tier_for_args(self, args: dict[str, Any]) -> str | None:
        command = (args.get("command") or "").strip()
        if not command:
            return None
        if classify_command(command).level == DESTRUCTIVE:
            return "ask"
        return None

    def intent_confirms_args(self, args: dict[str, Any], utterance: str) -> bool:
        command = (args.get("command") or "").strip()
        if not command or classify_command(command).level != DESTRUCTIVE:
            return False
        return command_confirms_destruction(command, utterance)

    async def execute(self, args: dict[str, Any], ctx: ExecutionContext) -> ToolResult:
        self.calls += 1
        self.last_ctx = ctx
        return ToolResult(success=True, output={"ran": args.get("command")})


def _bypass_executor(
    *,
    plausibility_context_fn: Any = None,
) -> tuple[ToolExecutor, EventBus, list[Any]]:
    bus = EventBus()
    events: list[Any] = []

    async def _capture(event: Any) -> None:
        events.append(event)

    bus.subscribe(ActionApprovalRequired, _capture)
    bus.subscribe(ActionApproved, _capture)
    bus.subscribe(ActionExecuted, _capture)
    executor = ToolExecutor(
        bus,
        RiskTierEvaluator(SafetyConfig(approval_mode="bypass")),
        ApprovalWorkflow(bus),
        plausibility_config=BrainPlausibilityConfig(),
        plausibility_context_fn=plausibility_context_fn,
    )
    return executor, bus, events


def _of_type(events: list[Any], cls: type) -> list[Any]:
    return [e for e in events if isinstance(e, cls)]


@pytest.mark.asyncio
async def test_bypass_does_not_execute_destructive_on_unattended() -> None:
    """Headline: bypass on, destructive command, nobody there → must NOT run."""
    executor, _bus, events = _bypass_executor()
    tool = _ShellTool()

    result = await executor.execute(
        tool,
        args={"command": _DESTRUCTIVE_CMD},
        config_snapshot=_UNATTENDED,
    )

    assert tool.calls == 0
    assert result.success is False
    assert (result.error or "").startswith(APPROVAL_UNAVAILABLE_PREFIX)
    assert _of_type(events, ActionApprovalRequired)
    assert not any(
        isinstance(e, ActionApproved) and e.approved_by == "bypass" for e in events
    )


@pytest.mark.asyncio
async def test_person_can_still_approve_destructive_under_bypass() -> None:
    """Same destructive call, conversational surface, authorizer says yes."""
    executor, bus, events = _bypass_executor()
    tool = _ShellTool()

    async def _authorizer(event: ActionApprovalRequired) -> None:
        await bus.publish(
            ActionApproved(
                trace_id=event.trace_id,
                tool_name=event.tool_name,
                approved_by="user:chat",
            )
        )

    bus.subscribe(ActionApprovalRequired, _authorizer)

    result = await executor.execute(
        tool,
        args={"command": _DESTRUCTIVE_CMD},
        config_snapshot=_CONVERSATIONAL,
    )

    assert result.success is True
    assert tool.calls == 1
    assert tool.last_ctx is not None
    assert tool.last_ctx.approved_by == "user:chat"
    executed = _of_type(events, ActionExecuted)
    assert executed
    assert executed[-1].approved_by == "user:chat"
    assert not any(
        isinstance(e, ActionApproved) and e.approved_by == "bypass" for e in events
    )


@pytest.mark.asyncio
async def test_explicit_intent_executes_and_records_approval() -> None:
    """Exact utterance authorizes the delete; audit trail says explicit-intent."""
    executor, _bus, events = _bypass_executor()
    tool = _ShellTool()

    result = await executor.execute(
        tool,
        args={"command": _DESTRUCTIVE_CMD},
        user_utterance=_DESTRUCTIVE_UTTERANCE,
        config_snapshot=_UNATTENDED,
    )

    assert result.success is True
    assert tool.calls == 1
    assert _of_type(events, ActionApprovalRequired)
    approved = _of_type(events, ActionApproved)
    assert any(e.approved_by == "explicit-intent" for e in approved)
    executed = _of_type(events, ActionExecuted)
    assert executed
    assert executed[-1].approved_by == "explicit-intent"
    assert tool.last_ctx is not None
    assert tool.last_ctx.approved_by == "explicit-intent"


@pytest.mark.parametrize(
    "command,utterance",
    [
        (r"rm C:\tmp\other", r"delete C:\tmp\x"),
        (r"rm C:\tmp\x\..", r"delete C:\tmp\x\.."),
        (r"rm C:\tmp\x && rm C:\tmp\y", r"delete C:\tmp\x"),
        ("rm tmp/x", "delete tmp/x"),
        (r"rm C:\tmp\*", r"delete C:\tmp\*"),
    ],
)
@pytest.mark.asyncio
async def test_explicit_intent_near_misses_still_need_approval(
    command: str, utterance: str
) -> None:
    """Near-misses stay refused by explicit_intent and are not auto-run by bypass."""
    assert not command_confirms_destruction(command, utterance)
    executor, _bus, _events = _bypass_executor()
    tool = _ShellTool()

    result = await executor.execute(
        tool,
        args={"command": command},
        user_utterance=utterance,
        config_snapshot=_UNATTENDED,
    )

    assert tool.calls == 0
    assert result.success is False
    assert (result.error or "").startswith(APPROVAL_UNAVAILABLE_PREFIX)


@pytest.mark.asyncio
async def test_plausibility_outranks_explicit_intent_and_bypass() -> None:
    """Low STT confidence keeps the ticket open even with a matching utterance."""
    transcript = Transcript(text=_DESTRUCTIVE_UTTERANCE, language="en", confidence=0.1)
    executor, _bus, events = _bypass_executor(
        plausibility_context_fn=lambda: (transcript, 5.0),
    )
    tool = _ShellTool()

    result = await executor.execute(
        tool,
        args={"command": _DESTRUCTIVE_CMD},
        user_utterance=_DESTRUCTIVE_UTTERANCE,
        config_snapshot=_UNATTENDED,
    )

    assert tool.calls == 0
    assert result.success is False
    assert (result.error or "").startswith(APPROVAL_UNAVAILABLE_PREFIX)
    assert _of_type(events, ActionApprovalRequired)
    assert not any(
        isinstance(e, ActionApproved)
        and e.approved_by in {"bypass", "explicit-intent"}
        for e in events
    )


@pytest.mark.asyncio
async def test_safe_read_still_runs_under_bypass_without_approval() -> None:
    """Regression: a read command is untouched by the destructive carve-out."""
    executor, _bus, events = _bypass_executor()
    tool = _ShellTool()

    result = await executor.execute(
        tool,
        args={"command": "dir"},
        config_snapshot=_UNATTENDED,
    )

    assert result.success is True
    assert tool.calls == 1
    assert not _of_type(events, ActionApprovalRequired)
    executed = _of_type(events, ActionExecuted)
    assert executed
    assert executed[-1].success is True


def test_is_destructive_honours_tool_hook_and_fails_closed() -> None:
    from jarvis.safety.destructive import is_destructive

    class _HookTrue:
        name = "other"
        def is_destructive_for_args(self, args: dict[str, Any]) -> bool:
            return True

    class _HookBroken:
        name = "other"
        def is_destructive_for_args(self, args: dict[str, Any]) -> bool:
            raise RuntimeError("boom")

    class _Plain:
        name = "gmail"

    assert is_destructive(_HookTrue(), {}) is True
    assert is_destructive(_HookBroken(), {}) is True
    assert is_destructive(_Plain(), {"to": "a@b.c"}) is False
    assert is_destructive(_ShellTool(), {"command": ""}) is True
    assert is_destructive(_ShellTool(), {"command": "rm x"}) is True
    assert is_destructive(_ShellTool(), {"command": "dir"}) is False
