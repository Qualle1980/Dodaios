"""coding-session open/send must register turn-owned delegated work.

The General's chat turn only waits through ``register_delegated_work``.
Wake via ``coordinator.track`` is a later background signal and does not
hold the originating turn — without registration the turn closes done
before the coding thread finishes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from jarvis.agent_chat.events import make_event
from jarvis.agent_chat.service import AgentChatService
from jarvis.agent_chat.store import AgentChatStore
from jarvis.core.delegated_work import collect_delegated_work
from jarvis.core.protocols import ChatTurn, current_chat_turn
from jarvis.society import coding_threads
from jarvis.society.coding_threads import CodingThreads, CodingThreadTool

pytestmark = pytest.mark.asyncio


class ScriptedService(AgentChatService):
    """Real service; turns run a scripted runner so no CLI is billed."""

    def __init__(self, store: AgentChatStore) -> None:
        super().__init__(store)
        self.prompts: list[tuple[str, str]] = []
        self.steps: dict[str, list[Any]] = {}
        self.hold: dict[str, asyncio.Event] = {}

    async def send(self, session_id: str, text: str, *args: Any, **kwargs: Any) -> str:
        async def runner(handle: Any, prompt: str) -> None:
            self.prompts.append((session_id, prompt))
            hold = self.hold.get(session_id)
            if hold is not None:
                await hold.wait()
            queue = self.steps.get(session_id) or []
            step = queue.pop(0) if queue else "done"
            for event in step if isinstance(step, list) else [step]:
                if isinstance(event, str):
                    event = make_event("assistant_text", {"text": event})
                event["payload"].setdefault("turn_id", handle.turn_id)
                await handle.emit(event)
            await handle.emit(
                make_event("turn_finished", {"turn_id": handle.turn_id, "status": "done"})
            )

        kwargs.setdefault("control_runner", runner)
        return await super().send(session_id, text, *args, **kwargs)


class Society:
    def __init__(self, service: ScriptedService) -> None:
        self.service = service
        self.halted = False
        self.agents = {
            "nova": SimpleNamespace(agent_id="nova", name="Nova", state="active"),
        }
        self.store = SimpleNamespace(kill_switch=self._kill_switch)
        self.roster = SimpleNamespace(get=self._get)
        self.coding_threads = CodingThreads(self)

    async def _kill_switch(self) -> bool:
        return self.halted

    async def _get(self, agent_id: str) -> Any:
        return self.agents.get(agent_id)

    def chat_service(self) -> ScriptedService:
        return self.service


@pytest.fixture
async def society(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(coding_threads, "SETTLE_S", 0.01)
    monkeypatch.setattr(coding_threads, "RETRY_S", 0.02)

    async def installed() -> dict[str, bool]:
        return {}

    monkeypatch.setattr(coding_threads, "_installed_clis", installed)
    monkeypatch.setattr(coding_threads, "_ensure_project", lambda _folder: None)
    import jarvis.agent_chat.service as chat_service

    monkeypatch.setattr(chat_service, "resolve_runner", lambda *_a, **_k: "claude-cli")
    service = ScriptedService(AgentChatStore(":memory:"))
    rt = Society(service)
    await rt.coding_threads.start()
    owner = service.create_session(provider="claude-api", cwd=str(tmp_path), surface="agent")
    rt.owner = owner.session_id
    yield rt
    await rt.coding_threads.close()


def tool(rt: Society) -> CodingThreadTool:
    return CodingThreadTool(rt, "nova", session_id=rt.owner)


async def test_coding_session_open_registers_delegated_work(
    society: Society, tmp_path: Path
):
    rt = society
    accepted: list[Any] = []
    token = current_chat_turn.set(
        ChatTurn(rt.owner, "owner-turn", "Open a coding thread", True, "trace")
    )
    try:
        with collect_delegated_work(accepted.append):
            result = await tool(rt).execute(
                {
                    "action": "open",
                    "agent": "Claude Code",
                    "folder": str(tmp_path),
                    "prompt": "Add a login page",
                    "model": "Opus 5.5",
                },
                None,
            )
    finally:
        current_chat_turn.reset(token)
    assert result.success, result.error
    coding = [w for w in accepted if str(w.key).startswith("coding-thread:")]
    assert len(coding) == 1
    assert coding[0].name


async def test_coding_thread_probe_waits_until_turn_finished(
    society: Society, tmp_path: Path
):
    rt = society
    hold = asyncio.Event()
    # Hold every new thread until we release it, so the probe sees work in flight.
    original_send = rt.service.send

    async def send_and_hold(session_id: str, text: str, *args: Any, **kwargs: Any) -> str:
        rt.service.hold[session_id] = hold
        return await original_send(session_id, text, *args, **kwargs)

    rt.service.send = send_and_hold  # type: ignore[method-assign]
    accepted: list[Any] = []
    token = current_chat_turn.set(
        ChatTurn(rt.owner, "owner-turn", "Open a coding thread", True, "trace")
    )
    try:
        with collect_delegated_work(accepted.append):
            result = await tool(rt).execute(
                {
                    "action": "open",
                    "agent": "Claude Code",
                    "folder": str(tmp_path),
                    "prompt": "Add a login page",
                    "model": "Opus 5.5",
                },
                None,
            )
    finally:
        current_chat_turn.reset(token)
    assert result.success, result.error
    work = next(w for w in accepted if str(w.key).startswith("coding-thread:"))
    assert await work.probe() is None
    hold.set()
    thread_id = str(result.output["thread_id"])
    for _ in range(200):
        if not rt.service.is_running(thread_id):
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("coding thread never finished")
    done = await work.probe()
    assert done is not None
    assert done["status"]
    assert "report" in done
