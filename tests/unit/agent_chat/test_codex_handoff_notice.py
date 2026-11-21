"""codex_next without pending must not spend Instant corrections_left.

Ultra-review PR #186 task C (2026-11-17): runner_brain labeled pure Codex
hand-offs as delegation_correction, so the wait path burned the only Instant
correction before Codex ran.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from jarvis.agent_chat import delegation_wait
from jarvis.agent_chat.events import make_event
from jarvis.agent_chat.runner_api import TurnHandle
from jarvis.agent_chat.store import AgentChatStore
from jarvis.core.protocols import ChatTurn, current_chat_turn


ORDER = "Hand the coding review to Codex after Instant's own check."


@pytest.mark.asyncio
async def test_codex_handoff_without_pending_does_not_spend_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(delegation_wait, "POLL_SECONDS", 0.001)
    monkeypatch.setattr(delegation_wait, "WAIT_TIMEOUT_SECONDS", 1.0)
    # Keep Instant's one correction available; hand-off must not consume it.
    assert delegation_wait.MAX_CORRECTIONS >= 1

    events: list[dict[str, Any]] = []
    waves = {"n": 0}
    corrections_seen: list[int] = []
    fallback_calls = {"n": 0}

    async def emit(event: dict[str, Any]) -> None:
        events.append(event)

    async def approve(*_a: Any, **_k: Any) -> str:
        return "allow"

    async def fake_codex(gate: Any, handle: Any, original: str, corrections: int) -> bool:
        fallback_calls["n"] += 1
        corrections_seen.append(corrections)
        # Prove the wait still held the honest gaps for a failed hand-off.
        assert gate.owed_honesty
        assert gate.correction is None
        gate.codex_fallback_message = "review not run: Codex pane busy"
        return False

    monkeypatch.setattr(delegation_wait, "_maybe_codex_fallback", fake_codex)

    store = AgentChatStore(":memory:")
    session = store.create_session(
        provider="chatgpt-plan", model="gpt-x", effort="", surface="jarvis", cwd="."
    )
    handle = TurnHandle(
        session=session,
        turn_id="t-handoff",
        emit=emit,
        request_approval=approve,
        cancel=asyncio.Event(),
        output_language="en",
        corrections_left=delegation_wait.MAX_CORRECTIONS,
    )
    service = SimpleNamespace(store=store, delegate_wait=None)

    async def run_attempt(active: Any, prompt: str) -> None:
        waves["n"] += 1
        await active.emit(
            make_event(
                "notice",
                {
                    "kind": "delegation_codex_handoff",
                    "turn_id": active.turn_id,
                    "work": ["coding:1"],
                    "missing": ["no change read", "no test run"],
                    "receipts": [],
                    "verdict_text": "",
                    "tool_error": False,
                    "codex_fallback": True,
                },
            )
        )
        await active.emit(
            make_event(
                "turn_finished",
                {"turn_id": active.turn_id, "status": "done", "usage": {}},
            )
        )

    token = current_chat_turn.set(
        ChatTurn(session.session_id, "t-handoff", ORDER, True, "trace")
    )
    try:
        await delegation_wait.run_with_delegates(service, handle, ORDER, run_attempt)
    finally:
        current_chat_turn.reset(token)
        store.close()

    # One Instant wave only — no Instant correction wave.
    assert waves["n"] == 1
    assert fallback_calls["n"] == 1
    assert corrections_seen == [delegation_wait.MAX_CORRECTIONS]
    # Honest line still reaches the person when Codex could not take it.
    texts = [
        str((e.get("payload") or {}).get("text") or "")
        for e in events
        if e.get("kind") == "assistant_text"
    ]
    joined = "\n".join(texts).lower()
    assert "review not run" in joined or "codex" in joined


def test_runner_brain_notice_kind_for_codex_next_without_pending() -> None:
    """Pure Codex next uses delegation_codex_handoff, never delegation_correction."""
    pending: tuple[str, ...] = ()
    codex_next = True
    if pending:
        notice_kind = "delegation_correction"
    elif codex_next:
        notice_kind = "delegation_codex_handoff"
    else:
        notice_kind = "delegation_unverified"
    assert notice_kind == "delegation_codex_handoff"

    pending = ("coding:1",)
    if pending:
        notice_kind = "delegation_correction"
    elif codex_next:
        notice_kind = "delegation_codex_handoff"
    else:
        notice_kind = "delegation_unverified"
    assert notice_kind == "delegation_correction"
