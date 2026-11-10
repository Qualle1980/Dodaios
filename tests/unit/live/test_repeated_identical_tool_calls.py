"""The memory tools must reach a model WITH their arguments.

2026-11-09 (#137, #138, reverted as #139 and #140): ``memory-search`` was
declared to the live call. The model called it with an EMPTY argument object,
the refusal taught it nothing, so it called again identically -- 177 calls at
3-4 a second, every turn ``transcript=0 chars``. The call never spoke and the
app was unusable.

The cause was one attribute name. ``BrainSupervisorToolGateway._describe``
reads ``tool.schema``; the three tools in ``jarvis/plugins/tool/memory_recall.py``
declared ``input_schema``, which is the name of the DESCRIPTOR field, not of the
tool attribute. ``getattr(tool, "schema", None)`` returned ``None``, so each of
them was described with the empty fallback ``{"type": "object",
"properties": {}}``: the model was told the tool takes no arguments, sent ``{}``
accordingly, and schema validation had no ``required`` left to refuse it on, so
the bare ``"query is missing"`` from the tool body came back instead of the
actionable refusal ``_execute`` already builds.

Both reverted PRs carried green tests asserting exactly which tools get
declared, and neither could see this: a declaration assertion describes the
surface, only a turn shows the behaviour. So the second test here drives the
real session over the native wire.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from jarvis.brain.tool_gateway import BrainSupervisorToolGateway
from jarvis.live import native
from jarvis.live.native import NativeLiveVoiceSession
from jarvis.live.state import LiveLedger
from jarvis.live.tools import LiveTools
from jarvis.plugins.tool.memory_recall import (
    MemoryHistoryTool,
    MemorySearchTool,
    MemoryTopicsTool,
)
from jarvis.realtime.protocol import RealtimeEvent
from tests.fakes.fake_native_voice import (
    FakeNativeConnection,
    FakeToolGateway,
    gemini_like_provider,
    local_voice_like_provider,
)

PROVIDERS = {
    "gemini-like": gemini_like_provider,
    "local-voice-like": local_voice_like_provider,
}
both_providers = pytest.mark.parametrize("make_provider", PROVIDERS.values(), ids=PROVIDERS)

# Hang guard only, never the measured quantity.
_HANG_GUARD_S = 30.0


@pytest.mark.parametrize(
    "tool, argument",
    [(MemorySearchTool, "query"), (MemoryHistoryTool, "topic")],
    ids=["memory-search", "memory-history"],
)
def test_a_memory_tool_reaches_the_model_with_its_required_argument(tool, argument) -> None:
    """The described schema carries the argument, so a model knows what to send."""
    described = BrainSupervisorToolGateway._describe({tool.name: tool()})

    assert len(described) == 1
    schema = described[0].input_schema
    assert argument in schema["properties"], (
        f"{tool.name} was described with {schema} -- a model told a tool takes no "
        "arguments sends none, which is exactly the 2026-11-09 loop"
    )
    assert schema.get("required") == [argument]


def test_no_tool_hides_its_schema_behind_the_descriptor_field_name() -> None:
    """`input_schema` on a tool is silently dropped; name the trap, not the symptom.

    ``_describe`` reads ``schema`` and falls back to an EMPTY schema, so this
    slip costs a model its arguments without any error anywhere. The three
    memory tools paid for it on 2026-11-09.
    """
    offenders = [
        tool.name
        for tool in (MemorySearchTool, MemoryHistoryTool, MemoryTopicsTool)
        if not isinstance(getattr(tool, "schema", None), dict)
    ]

    assert offenders == [], (
        f"{offenders} declare input_schema instead of schema; _describe would "
        "hand the model the empty fallback schema"
    )


def _config():
    return SimpleNamespace(
        brain=SimpleNamespace(reply_language="en", providers={}, realtime=None),
        voice=SimpleNamespace(realtime_tool_declaration_budget_tokens=0),
    )


def _in_call(provider, tmp_path, gateway):
    """A session already in a call: connection, ledger and tools in place."""

    async def send_json(_frame):
        return None

    async def send_binary(_data):
        return None

    session = NativeLiveVoiceSession(
        session_id="native-live",
        send_json=send_json,
        send_binary=send_binary,
        providers=[provider],
        config=_config(),
    )
    connection = FakeNativeConnection(creates_responses_automatically=provider._creates, model="m")
    ledger = LiveLedger(tmp_path / f"{provider.name}.sqlite3")
    session._connection = connection
    session._ledger = ledger
    session._tools = LiveTools(gateway, ledger, "native-live", language="en", backend_model="")
    return session, connection


@both_providers
@pytest.mark.asyncio
async def test_a_turn_refuses_empty_arguments_with_something_to_act_on(
    tmp_path, monkeypatch, make_provider
):
    """A real turn: the model gets the field AND what to send, and nothing ran.

    This is the check both reverted PRs lacked. It runs the call rather than
    asserting the catalog, and it uses the descriptor the real gateway builds
    from the real tool, so a regression in `_describe` fails it.
    """
    monkeypatch.setattr(native, "_TOOL_DEADLINE_S", _HANG_GUARD_S)
    search = MemorySearchTool()
    gateway = FakeToolGateway(BrainSupervisorToolGateway._describe({search.name: search}))
    session, connection = _in_call(make_provider(), tmp_path, gateway)
    try:
        await session._native_event(
            RealtimeEvent(
                type="tool_call",
                call_id="c0",
                tool_name="memory-search",
                tool_args={},
            )
        )
        await asyncio.wait_for(connection.tool_result_sent.wait(), _HANG_GUARD_S)

        _, _, result = connection.tool_results[0]
        assert result["success"] is False and result["executed"] is False
        assert result["retryable"] is True
        # Not a bare status: it names the field and what the tool accepts, which
        # is what a model needs to send a different call instead of the same one.
        assert "query" in result["error"]
        assert "Accepted arguments" in result["error"]
        assert gateway.calls == [], "nothing may run on arguments that did not validate"
    finally:
        session._ledger.close()
