"""Instant observe / screen excerpt caps (token efficiency 2026-11-17)."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from jarvis.agentic_ide import agent_transcript, session
from jarvis.agentic_ide.agent_sessions import ResumeHandle
from jarvis.agentic_ide.control import (
    OBSERVE_PAGE_BYTES,
    SCREEN_EXCERPT_CHARS,
    SCREEN_EXCERPT_LINES,
    CodingSessionControl,
)


def _pane(**kwargs):
    return session.Terminal(
        "t1",
        "First",
        "claude",
        "Claude",
        0,
        status="live",
        activity="waiting",
        activity_at=time.time(),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_observe_page_cap_marks_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    term = _pane(resume=ResumeHandle("claude", "recorded", 0.0))
    owner = SimpleNamespace(id="workspace", folder="project")
    registry = SimpleNamespace(
        find_terminal=lambda *args: (owner, term),
        input_token=lambda t: "input-token-1",
    )
    events = [
        {"kind": "assistant_text", "payload": {"text": ("Message %d " % n) + ("x" * 400)}}
        for n in range(40)
    ]
    monkeypatch.setattr(
        agent_transcript,
        "read_timeline",
        lambda *a, **kw: agent_transcript.TimelineRead(events),
    )
    monkeypatch.setattr(agent_transcript, "can_read", lambda agent: True)
    gateway = CodingSessionControl(registry)
    observed = await gateway.run(
        {
            "action": "observe",
            "workspace_id": owner.id,
            "terminal_id": "pane:" + term.history_id,
            "limit": 30,
        }
    )
    assert OBSERVE_PAGE_BYTES == 8_000
    page_bytes = sum(len(json.dumps(e, ensure_ascii=False)) for e in observed["events"])
    assert page_bytes <= OBSERVE_PAGE_BYTES + 200
    assert observed.get("truncated") is True or any(
        e.get("truncated") for e in observed["events"]
    )


@pytest.mark.asyncio
async def test_observe_keeps_input_line_and_newest_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instant review depends on the input token and the newest activity."""
    term = _pane(resume=ResumeHandle("claude", "recorded", 0.0), pty_id="live-pty")
    term.activity = "working"
    term.activity_at = time.time()
    for i in range(30):
        term.transcript.feed(f"old line {i}\r\n")
    # Longer than SCREEN_EXCERPT_CHARS so the Instant per-line cap is exercised.
    term.transcript.feed("NEWEST activity marker " + ("y" * 160) + "\r\n")
    owner = SimpleNamespace(id="workspace", folder="project")
    registry = SimpleNamespace(
        find_terminal=lambda *args: (owner, term),
        input_token=lambda t: "input-token-xyz",
    )
    monkeypatch.setattr(
        agent_transcript,
        "read_timeline",
        lambda *a, **kw: agent_transcript.TimelineRead(
            [{"kind": "assistant_text", "payload": {"text": "latest"}}]
        ),
    )
    monkeypatch.setattr(agent_transcript, "can_read", lambda agent: True)
    gateway = CodingSessionControl(registry)
    observed = await gateway.run(
        {
            "action": "observe",
            "workspace_id": owner.id,
            "terminal_id": "pane:" + term.history_id,
        }
    )
    assert observed["input_token"] == "input-token-xyz"
    assert observed["activity"] == "working"
    excerpt = observed["screen_excerpt"]
    assert SCREEN_EXCERPT_LINES == 8
    assert SCREEN_EXCERPT_CHARS == 120
    assert len(excerpt) <= SCREEN_EXCERPT_LINES
    assert excerpt, "newest activity must remain visible"
    assert all(len(line) <= SCREEN_EXCERPT_CHARS for line in excerpt)
    # Newest marker stays in the excerpt window (pane wrap may split the line).
    assert any("NEWEST" in line for line in excerpt)
