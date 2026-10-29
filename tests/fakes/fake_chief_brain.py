"""A scripted tool-calling chief brain for the typed Jarvis chat.

``FakeBrainManager`` answers with prose and publishes a *pretended* tool
round on the bus. That is enough for timeline tests and useless for the
question this fake exists for: does the chat's chief really inspect a
project with its own hands before it hands work to a coding agent, and does
it really verify what comes back?

So this stand-in runs the product's own machinery instead of imitating it:

* The tools it reaches for are the ones the turn actually offers — the
  ``turn_override``'s ``tools_extra`` (the surface kit's folder tools) merged
  with its own registry ``_tools`` (where ``workspace-orchestrate`` lives).
* Every call goes through a real :class:`~jarvis.safety.tool_executor.ToolExecutor`
  with the turn's own ``tool_context`` as the config snapshot, so the risk
  tier, the chat's approval stance and the ``ActionProposed`` /
  ``ActionExecuted`` bus records are the real ones (AP-3).
* The script is per wave: ``waves[0]`` runs on the person's own message,
  ``waves[1]`` on the delegation continuation, and so on. Each wave names
  its tool calls and the text the chief then answers with.

What stays fake is only the decision: a script replaces the model's
judgement, which is exactly the part a test must pin down.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from jarvis.core.protocols import Tool, ToolResult


@dataclass(frozen=True)
class Call:
    """One tool call the chief makes in a wave."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    #: The chief's own words for why — the rationale the timeline shows.
    why: str = "Checking before deciding."


@dataclass(frozen=True)
class Wave:
    """What the chief does in one turn of the original request."""

    calls: tuple[Call, ...] = ()
    reply: str = ""


@dataclass
class Executed:
    """What a call actually did, for the test to assert on."""

    tool: str
    args: dict[str, Any]
    result: ToolResult
    #: The turn's briefing at the time of the call (the continuation payload).
    system_extra: str = ""


class ScriptedChief:
    """A chief that works its script with the turn's real tools."""

    def __init__(
        self,
        *,
        executor: Any,
        waves: list[Wave],
        tools: dict[str, Tool] | None = None,
        fast_model: str = "fake-chief",
    ) -> None:
        self._executor = executor
        self._waves = waves
        #: The brain's own registry, as a chat turn sees it.
        self._tools: dict[str, Tool] = dict(tools or {})
        self.fast_model = fast_model
        self.reply_language = "auto"
        self.conversation_language = ""
        #: Every generate call: (text, kwargs).
        self.calls: list[tuple[str, dict[str, Any]]] = []
        #: Every tool call that really ran, in order, across all waves.
        self.executed: list[Executed] = []
        self.wave = 0

    # -- what the runner reads -------------------------------------------------

    def _fast_model(self, provider: str) -> str:
        return self.fast_model

    def note_skill_trigger(self, name: str, *, content: str = "", source: str = "") -> None:
        return None

    def tool_names(self) -> list[str]:
        """The names this chief would see, for a test that asserts the surface."""
        return sorted(self._tools)

    async def generate(self, text: str, **kwargs: Any) -> str:
        self.calls.append((text, dict(kwargs)))
        override = kwargs.get("turn_override")
        trace_id: UUID = kwargs.get("trace_id") or uuid4()
        consumer = kwargs.get("text_consumer")
        surface: dict[str, Tool] = {
            **self._tools,
            **dict(getattr(override, "tools_extra", {}) or {}),
        }
        briefing = str(getattr(override, "system_extra", "") or "")
        script = self._waves[self.wave] if self.wave < len(self._waves) else Wave(reply="")
        self.wave += 1
        for call in script.calls:
            tool = surface.get(call.tool)
            if tool is None:
                raise AssertionError(
                    f"the turn never offered {call.tool!r}; it offered {sorted(surface)}"
                )
            result = await self._executor.execute(
                tool,
                dict(call.args),
                # The manager runs a turn's tools under the TURN's trace id;
                # the timeline and the acceptance gate both read it back by it.
                trace_id=trace_id,
                rationale=call.why,
                config_snapshot=dict(getattr(override, "tool_context", {}) or {}),
            )
            self.executed.append(Executed(call.tool, dict(call.args), result, briefing))
        if callable(consumer) and script.reply:
            consumer(script.reply)
            await asyncio.sleep(0)
        if override is not None:
            override.receipt.record(
                provider=getattr(override, "provider", "") or "",
                model=getattr(override, "model", None),
                tokens_in=100,
                tokens_out=20,
                cost_usd=0.0,
                finish_reason="stop",
            )
        return script.reply

    # -- what no chat turn may ever do ----------------------------------------

    async def switch(self, *_a: Any, **_k: Any) -> None:
        raise AssertionError("a chat turn must never switch the live brain")

    def apply_provider_model(self, *_a: Any, **_k: Any) -> bool:
        raise AssertionError("a chat turn must never rewrite the provider config")


__all__ = ["Call", "Executed", "ScriptedChief", "Wave"]
