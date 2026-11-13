"""The native live path consults the turn planner before the model answers.

Measured 2026-11-13: spoken B2 (German product question about WebSocket
tickets without HTTPS) completed with function_calls=0 and a textbook
answer. Offline, plan_turn (once F-1 lands) returns product_help /
SELF_KNOWLEDGE. ``rg plan_turn jarvis/live/`` was comment-only: the live
voice path never consulted the planner, so fixing the planner alone cannot
help. Gemini Live also drops ``required_tool``, so the app must run the
lookup itself and hand the result to the model as context.
"""

from __future__ import annotations

import pytest

from jarvis.brain.turn_planner import TurnPath, TurnPlan, TurnReason
from jarvis.core.protocols import SupervisorToolDescriptor, ToolResult
from jarvis.realtime.protocol import RealtimeEvent
from tests.fakes.fake_native_voice import FakeToolGateway, gemini_like_provider

from .test_native_live_fixes import _in_call, _settle

# Exact live B2 transcript (German speech-input fixture).  # i18n-allow
B2_TURN = (
    "Warum gibt uns der Server ohne HTTPS keine WebSocket-Tickets aus?"  # i18n-allow
)
WEATHER_TURN = "Wie ist das Wetter in Berlin heute?"  # i18n-allow: DE weather fixture
CHAT_TURN = "Hallo, wie geht es dir?"  # i18n-allow: DE chat fixture


def _self_knowledge_plan() -> TurnPlan:
    """A planner verdict that requires the built-in product guide.

    Order C lands ``TurnReason.SELF_KNOWLEDGE``; until then CAPABILITY still
    carries ``product_help`` in ``required_capabilities``, which is what the
    live path keys on.
    """
    reason = getattr(TurnReason, "SELF_KNOWLEDGE", TurnReason.CAPABILITY)
    return TurnPlan(
        path=TurnPath.ORCHESTRATOR,
        reasons=frozenset({reason}),
        required_capabilities=("product_help",),
        requires_evidence=True,
    )


def _product_help_gateway() -> FakeToolGateway:
    gateway = FakeToolGateway(
        (
            SupervisorToolDescriptor(
                "product_help",
                "Read Dodaios's built-in user guide.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "slug": {"type": "string"},
                    },
                },
                "safe",
            ),
            SupervisorToolDescriptor(
                "search_web",
                "Search the web.",
                {"type": "object", "properties": {"query": {"type": "string"}}},
                "safe",
            ),
        )
    )
    gateway.results["product_help"] = ToolResult(
        True,
        {
            "matches": [{"slug": "websocket-tickets", "title": "WebSocket tickets"}],
            "guide": {
                "slug": "websocket-tickets",
                "title": "WebSocket tickets",
                "text": "Tickets require HTTPS on the control key path.",
            },
        },
    )
    return gateway


def _product_help_calls(gateway: FakeToolGateway) -> list[tuple[str, dict]]:
    return [call for call in gateway.calls if call[0] == "product_help"]


@pytest.mark.asyncio
async def test_self_knowledge_turn_runs_product_help_before_model_answer(
    tmp_path, monkeypatch
) -> None:
    """A SELF_KNOWLEDGE / product_help plan grounds before the model answers."""
    from jarvis.live import tools as live_tools

    monkeypatch.setattr(
        live_tools,
        "plan_turn",
        lambda text, **_kwargs: _self_knowledge_plan(),
        raising=False,
    )
    gateway = _product_help_gateway()
    session, connection, _frames = _in_call(
        gemini_like_provider(), tmp_path, gateway=gateway
    )

    await session._native_event(
        RealtimeEvent(type="input_transcript", text=B2_TURN, is_final=True)
    )
    await _settle(session)

    calls = _product_help_calls(gateway)
    assert len(calls) == 1, (
        "live path must run product_help itself when the planner requires it "
        f"(got {gateway.calls!r})"
    )
    assert B2_TURN in str(calls[0][1].get("query", ""))
    # The trusted guide must reach the model as application context before it
    # answers unaided (Gemini cannot be forced to call the tool).
    text_inputs = [payload for name, payload in connection.calls if name == "send_text"]
    assert text_inputs, "expected product guide context via send_text"
    joined = "\n".join(str(item) for item in text_inputs)
    assert "websocket-tickets" in joined or "WebSocket tickets" in joined
    assert "product guide" in joined.lower() or "product_help" in joined.lower()


@pytest.mark.asyncio
async def test_weather_question_does_not_run_product_help(tmp_path) -> None:
    """A weather / public-fact turn must not trigger the product guide lookup."""
    gateway = _product_help_gateway()
    session, connection, _frames = _in_call(
        gemini_like_provider(), tmp_path, gateway=gateway
    )

    await session._native_event(
        RealtimeEvent(type="input_transcript", text=WEATHER_TURN, is_final=True)
    )
    await _settle(session)

    assert _product_help_calls(gateway) == []
    assert connection.names().count("send_text") == 0


@pytest.mark.asyncio
async def test_turn_with_no_reasons_changes_nothing(tmp_path) -> None:
    """Plain chat with an empty plan leaves the wire untouched."""
    gateway = _product_help_gateway()
    session, connection, _frames = _in_call(
        gemini_like_provider(), tmp_path, gateway=gateway
    )

    await session._native_event(
        RealtimeEvent(type="input_transcript", text=CHAT_TURN, is_final=True)
    )
    await _settle(session)

    assert _product_help_calls(gateway) == []
    assert "update_session" not in connection.names()
    assert "send_text" not in connection.names()


@pytest.mark.asyncio
async def test_product_help_lookup_runs_at_most_once_per_user_turn(
    tmp_path, monkeypatch
) -> None:
    """No 2026-11-09 loop: one planner grounding per user revision."""
    from jarvis.live import tools as live_tools

    monkeypatch.setattr(
        live_tools,
        "plan_turn",
        lambda text, **_kwargs: _self_knowledge_plan(),
        raising=False,
    )
    gateway = _product_help_gateway()
    session, _connection, _frames = _in_call(
        gemini_like_provider(), tmp_path, gateway=gateway
    )

    await session._native_event(
        RealtimeEvent(type="input_transcript", text=B2_TURN, is_final=True)
    )
    await _settle(session)
    # Same turn / same revision must not re-fire even if the session consults again.
    assert session._tools is not None
    ground = getattr(session._tools, "ground_self_knowledge_once", None)
    assert callable(ground), "LiveTools must own the once-per-turn product_help grounding"
    again = await ground(B2_TURN)
    assert again is None
    assert len(_product_help_calls(gateway)) == 1
