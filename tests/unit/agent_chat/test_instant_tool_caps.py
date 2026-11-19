"""Instant / chat path tool-output caps (token efficiency 2026-11-17).

Coding and mission surfaces keep their own limits; these pin the Instant
folder tools and the chatgpt-plan global backstop.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.agent_chat import tools as folder
from jarvis.brain.tool_use_loop import (
    _CHAT_TOOL_RESULT_CHARS,
    _MAX_TOOL_RESULT_CHARS,
    _cap_tool_result_json,
)


def test_instant_read_cap_is_12k_with_truncation_marker(tmp_path: Path) -> None:
    big = "A" * 20_000
    (tmp_path / "big.txt").write_text(big, encoding="utf-8")
    out, err = folder._read({"file_path": "big.txt"}, tmp_path)
    assert not err
    assert len(out) <= folder._READ_CAP_CHARS + 80
    assert folder._READ_CAP_CHARS == 12_000
    assert "truncated" in out.lower()
    assert "offset" in out.lower() or "limit" in out.lower()


def test_instant_run_command_cap_is_8k_head_and_tail(tmp_path: Path) -> None:
    text = "HEAD-" + ("x" * 10_000) + "-TAIL"
    capped = folder._cap_command_output(text)
    assert folder._COMMAND_CAP_CHARS == 8_000
    assert len(capped) <= folder._COMMAND_CAP_CHARS + 80
    assert "truncated" in capped.lower()
    assert capped.startswith("HEAD-")
    assert capped.endswith("-TAIL")


def test_chatgpt_plan_tool_result_cap_is_20k() -> None:
    assert _CHAT_TOOL_RESULT_CHARS == 20_000
    assert _MAX_TOOL_RESULT_CHARS == 100_000
    chat_huge = "Z" * 50_000
    chat = _cap_tool_result_json(chat_huge, limit=_CHAT_TOOL_RESULT_CHARS)
    assert len(chat) <= _CHAT_TOOL_RESULT_CHARS + 120
    assert "truncated" in chat
    default_huge = "Z" * (_MAX_TOOL_RESULT_CHARS + 10_000)
    default = _cap_tool_result_json(default_huge)
    assert len(default) <= _MAX_TOOL_RESULT_CHARS + 120
    assert "truncated" in default


def test_instant_build_override_tool_result_cap_is_12k() -> None:
    """Instant coding turns use a tighter context cap than the chat default."""
    from jarvis.agent_chat.runner_brain import build_override

    session = SimpleNamespace(
        provider="chatgpt-plan",
        model="gpt-test",
        effort="",
        surface="agent",
    )
    override = build_override(
        session,
        brain=SimpleNamespace(),
        stance="build",
        cwd=Path("."),
        ref="t1",
        kit_tools={},
    )
    assert override.tool_context.get("max_tool_result_chars") == 12_000
