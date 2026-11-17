"""Instant / chat path tool-output caps.

Coding and mission surfaces keep their own limits; these pin the Instant
folder tools. Caps mark a truncated sample; they must not starve context
(mandate 2026-08-24 / 7c985c1ba). Honesty markers from #179 stay.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from jarvis.agent_chat import tools as folder
from jarvis.brain.tool_use_loop import (
    _CHAT_TOOL_RESULT_CHARS,
    _MAX_TOOL_RESULT_CHARS,
    _cap_tool_result_json,
)


def test_instant_read_cap_is_60k_with_truncation_marker(tmp_path: Path) -> None:
    big = "A" * 70_000
    (tmp_path / "big.txt").write_text(big, encoding="utf-8")
    out, err = folder._read({"file_path": "big.txt"}, tmp_path)
    assert not err
    assert len(out) <= folder._READ_CAP_CHARS + 80
    assert folder._READ_CAP_CHARS == 60_000
    assert "truncated" in out.lower()
    assert "offset" in out.lower() or "limit" in out.lower()


def test_instant_run_command_cap_is_30k_head_and_tail(tmp_path: Path) -> None:
    text = "HEAD-" + ("x" * 40_000) + "-TAIL"
    capped = folder._cap_command_output(text)
    assert folder._COMMAND_CAP_CHARS == 30_000
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


def test_instant_build_override_has_no_extra_tool_result_cap() -> None:
    """chatgpt-plan uses the shared chat tool-result ceiling, not a tighter override."""
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
    assert "max_tool_result_chars" not in override.tool_context


def test_coding_threads_files_path_inherits_folder_caps() -> None:
    """society coding_threads._do_files calls agent_chat.tools — same caps."""
    import inspect

    from jarvis.society import coding_threads

    source = inspect.getsource(coding_threads.CodingThreadTool._do_files)
    assert "jarvis.agent_chat" in source or "folder_tools" in source
    assert folder._READ_CAP_CHARS == 60_000
    assert folder._COMMAND_CAP_CHARS == 30_000
