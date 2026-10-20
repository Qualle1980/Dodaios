"""A plan-seat talker's tool turn, from the user's words to the tool result.

The unit tests next to this one (``tests/unit/brain/test_plan_seat_tool_lead``)
prove WHICH provider may lead a plan-seat talker's tool turn. This one proves
the lead is actually useful: the led turn reaches the tool, the tool runs, and
its result is what comes back to Jarvis — the path a ChatGPT plan/web talker
needs to hand coding work to Codex CLI / Claude Code.

Fakes only: a scripted local lead, a scripted talker, a recording tool. No
provider call, no network, no model inference.
"""
from __future__ import annotations

from typing import Any

import pytest

from jarvis.brain.manager import BrainManager
from jarvis.core.bus import EventBus
from jarvis.core.config import (
    JarvisConfig,
    SafetyBlacklistConfig,
    SafetyConfig,
    SafetyWhitelistConfig,
)
from jarvis.core.protocols import BrainDelta, ExecutionContext, ToolResult
from jarvis.safety import ApprovalWorkflow, RiskTierEvaluator, ToolExecutor
from tests.fixtures.brain.fake_brain import FakeBrain, tool_call_delta

_TASK = "Add a health endpoint to the gateway"


class _DelegateTool:
    """Stands in for delegate_to_agent / the agentic coding runtimes."""

    name = "delegate_to_agent"
    risk_tier = "safe"
    description = "Hand a coding task to an agent runtime"
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"task": {"type": "string"}},
        "required": ["task"],
    }

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(self, args, ctx: ExecutionContext) -> ToolResult:
        self.calls.append(args)
        return ToolResult(success=True, output=f"agent finished: {args.get('task', '')}")


class _ToollessTalker(FakeBrain):
    """A plan seat: it answers, but it can never emit a tool_call."""

    supports_tools = False

    def can_call_tools(self) -> bool:
        return False


def _executor(bus: EventBus) -> ToolExecutor:
    safety = SafetyConfig(
        default_tier="safe",
        whitelist=SafetyWhitelistConfig(commands=[]),
        blacklist=SafetyBlacklistConfig(commands=[]),
    )
    return ToolExecutor(bus, RiskTierEvaluator(safety), ApprovalWorkflow(bus))


def _infra() -> dict[str, Any]:
    bus = EventBus()
    config = JarvisConfig()
    config.brain.primary = "chatgpt-web"
    tool = _DelegateTool()
    manager = BrainManager(
        config=config,
        bus=bus,
        tools={tool.name: tool},
        tool_executor=_executor(bus),
    )
    talker = _ToollessTalker(text_response="I cannot run that myself.")
    lead = FakeBrain(script=[
        [
            tool_call_delta("delegate_to_agent", {"task": _TASK}, call_id="c1"),
            BrainDelta(finish_reason="tool_use"),
        ],
        [
            BrainDelta(content=f"Done — agent finished: {_TASK}"),
            BrainDelta(finish_reason="stop"),
        ],
    ])
    manager._registry._loaded = True
    manager._registry._classes = {  # type: ignore[attr-defined]
        "chatgpt-web": type(talker), "ollama": type(lead),
    }
    # Deterministic cache keys: no per-provider model, so (name, None).
    manager._fast_model = lambda name: None  # type: ignore[method-assign]
    manager._deep_model = lambda name: None  # type: ignore[method-assign]
    manager._brain_cache[("chatgpt-web", None)] = talker
    manager._brain_cache[("ollama", None)] = lead
    return {"manager": manager, "talker": talker, "lead": lead, "tool": tool}


def test_the_local_lead_is_the_plan_seat_talkers_tool_turn() -> None:
    """The chain a substantive tool turn gets: led by the local runtime."""
    infra = _infra()
    manager: BrainManager = infra["manager"]
    manager._turn_substantive = True  # type: ignore[attr-defined]

    chain = manager._build_fallback_chain("fast")

    assert manager._router_lead_key == ("ollama", None)
    assert chain[0] == ("ollama", None), "the lead must answer first"
    assert ("chatgpt-web", None) in chain, "the seat stays as the conversational fallback"


@pytest.mark.asyncio
async def test_the_led_turn_runs_the_tool_and_returns_its_result() -> None:
    """End of the chain: the tool ran, and Jarvis got the agent's result."""
    infra = _infra()
    manager: BrainManager = infra["manager"]
    manager._build_fallback_chain = lambda level: [("ollama", None)]  # type: ignore[assignment]

    answer = await manager.generate(f"Please {_TASK.lower()}")

    assert infra["tool"].calls == [{"task": _TASK}], "the delegation tool must have run"
    assert f"agent finished: {_TASK}" in answer
    assert not infra["talker"].calls, "the plan seat must not be billed for a led tool turn"


@pytest.mark.asyncio
async def test_a_plain_question_costs_exactly_one_call_on_the_seat() -> None:
    """No extra model call, and no tool, for an ordinary chat turn."""
    infra = _infra()
    manager: BrainManager = infra["manager"]
    manager._build_fallback_chain = lambda level: [("chatgpt-web", None)]  # type: ignore[assignment]

    answer = await manager.generate("What is the capital of France?")

    assert answer == "I cannot run that myself."
    assert len(infra["talker"].calls) == 1
    assert not infra["lead"].calls, "no second provider may be called for a chat turn"
    assert not infra["tool"].calls
