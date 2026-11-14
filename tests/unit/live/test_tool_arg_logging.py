"""Live tool-call logging keeps diagnosis without retaining free-text args.

The desktop log must answer *which tool, which keys, identical or not* at
INFO, while dictated message bodies stay out of the default level. Full
arguments remain reachable at DEBUG through ``safe_preview``.
"""
from __future__ import annotations

import logging

import pytest

from jarvis.core.protocols import SupervisorToolDescriptor
from jarvis.live.state import LiveLedger
from jarvis.live.tools import LiveTools
from tests.fakes.fake_native_voice import FakeToolGateway


_DICTATED = (
    "Please tell Maria that the meeting moved to Thursday at three "
    "and her private note about the clinic stay stays between us."
)


def _catalog() -> tuple[SupervisorToolDescriptor, ...]:
    return (
        SupervisorToolDescriptor(
            "compose_note",
            "Draft a note.",
            {"type": "object"},
            "safe",
        ),
        SupervisorToolDescriptor(
            "workspace-orchestrate",
            "Inspect workspaces.",
            {"type": "object"},
            "safe",
        ),
    )


def _tools(tmp_path) -> tuple[LiveTools, FakeToolGateway]:
    gateway = FakeToolGateway(_catalog())
    ledger = LiveLedger(tmp_path / "live.sqlite3")
    tools = LiveTools(gateway, ledger, "arg-log", language="en", backend_model="")
    tools.user_text = "send Maria a note"
    return tools, gateway


@pytest.mark.asyncio
async def test_info_log_names_tool_and_keys_without_dictated_body(
    tmp_path, caplog: pytest.LogCaptureFixture,
) -> None:
    tools, _gateway = _tools(tmp_path)
    args = {"action": "draft", "body": _DICTATED}
    with caplog.at_level(logging.INFO, logger="jarvis.live.tools"):
        await tools.execute("call-1", "compose_note", args, tools.revision)

    info = "\n".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.INFO
    )
    assert "Live tool call compose_note" in info
    assert "body:str:" in info
    assert "action:str=draft" in info
    assert "digest=" in info
    for word in ("Maria", "Thursday", "clinic", "private", "meeting"):
        assert word not in info
    assert _DICTATED not in info
    assert "Live tool result compose_note" in info


@pytest.mark.asyncio
async def test_identical_and_different_calls_are_distinguishable_at_info(
    tmp_path, caplog: pytest.LogCaptureFixture,
) -> None:
    tools, _gateway = _tools(tmp_path)
    tools.user_text = "check status"
    with caplog.at_level(logging.INFO, logger="jarvis.live.tools"):
        await tools.execute(
            "c1", "workspace-orchestrate", {"action": "status"}, tools.revision,
        )
        await tools.execute(
            "c2", "workspace-orchestrate", {"action": "status"}, tools.revision,
        )
        await tools.execute(
            "c3", "workspace-orchestrate", {"action": "list"}, tools.revision,
        )

    call_lines = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and r.getMessage().startswith("Live tool call ")
    ]
    assert len(call_lines) == 3
    assert call_lines[0] == call_lines[1]
    assert call_lines[0] != call_lines[2]
    assert "digest=" in call_lines[0]
    assert "digest=" in call_lines[2]


@pytest.mark.asyncio
async def test_debug_log_carries_redacted_full_arguments(
    tmp_path, caplog: pytest.LogCaptureFixture,
) -> None:
    tools, _gateway = _tools(tmp_path)
    tools.user_text = "draft a note"
    secret = "sk-proj-AbCdEf0123456789ghijKLmnopQRstuv"
    args = {"message": _DICTATED, "api_key": secret}
    with caplog.at_level(logging.DEBUG, logger="jarvis.live.tools"):
        await tools.execute("call-d", "compose_note", args, tools.revision)

    debug = "\n".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG
    )
    info = "\n".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.INFO
    )
    assert "args=" in debug
    assert "message" in debug
    assert secret not in debug
    assert secret not in info
    assert "<redacted:" in debug
    for word in ("Maria", "Thursday", "clinic"):
        assert word not in info
