"""Runner-enforced post-write verification (fake reads, no real GitHub)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.agent_chat import write_verify as wv
from jarvis.agent_chat.runner_api import _append_write_verify


def test_build_verify_plan_for_assignee_edit():
    plan = wv.build_verify_plan("gh issue edit 217 --add-assignee grok")
    assert plan is not None
    assert plan.command == "gh issue view 217 --json assignees"
    assert plan.expect_assignees == ("grok",)


def test_check_verify_detects_missing_assignee():
    plan = wv.VerifyPlan(
        command="gh issue view 217 --json assignees",
        expect_assignees=("grok",),
        expect_number="217",
        expect_kind="issue",
        label="issue #217",
    )
    result = wv.check_verify_result(
        plan, '{"assignees":[{"login":"someone-else"}]}', is_error=False
    )
    assert result.mismatch is True
    assert "missing grok" in result.message


def test_check_verify_ok_when_assignee_present():
    plan = wv.VerifyPlan(
        command="gh issue view 217 --json assignees",
        expect_assignees=("grok",),
        expect_kind="issue",
        label="issue #217",
    )
    result = wv.check_verify_result(
        plan, '{"assignees":[{"login":"grok"}]}', is_error=False
    )
    assert result.ok is True
    assert result.mismatch is False


@pytest.mark.asyncio
async def test_run_write_verify_with_fake_read():
    async def fake_read(cmd: str) -> tuple[str, bool]:
        assert "issue view 217" in cmd
        return '{"assignees":[]}', False

    result = await wv.run_write_verify(
        write_command="gh issue edit 217 --add-assignee grok",
        write_output="https://github.com/o/r/issues/217",
        write_is_error=False,
        execute_read=fake_read,
    )
    assert result is not None
    assert result.mismatch is True
    appendix = wv.format_verify_appendix(result)
    assert "[write-verify MISMATCH]" in appendix
    assert "missing grok" in appendix


@pytest.mark.asyncio
async def test_run_write_verify_skips_reads():
    called = False

    async def fake_read(cmd: str) -> tuple[str, bool]:
        nonlocal called
        called = True
        return "", False

    result = await wv.run_write_verify(
        write_command="gh issue view 217 --json assignees",
        write_output="",
        write_is_error=False,
        execute_read=fake_read,
    )
    assert result is None
    assert called is False


@pytest.mark.asyncio
async def test_append_write_verify_in_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    async def fake_execute(name: str, args: dict, *, cwd: Path):
        assert name == "RunCommand"
        assert "issue view 99" in str(args.get("command"))
        return '{"assignees":[{"login":"alice"}]}', False

    monkeypatch.setattr("jarvis.agent_chat.runner_api.execute_tool", fake_execute)
    out = await _append_write_verify(
        name="RunCommand",
        args={"command": "gh issue edit 99 --add-assignee bob"},
        output="https://github.com/o/r/issues/99",
        is_error=False,
        cwd=tmp_path,
    )
    assert "[write-verify MISMATCH]" in out
    assert "missing bob" in out
    assert "https://github.com/o/r/issues/99" in out


@pytest.mark.asyncio
async def test_append_write_verify_ok_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    async def fake_execute(name: str, args: dict, *, cwd: Path):
        return '{"assignees":[{"login":"bob"}]}', False

    monkeypatch.setattr("jarvis.agent_chat.runner_api.execute_tool", fake_execute)
    out = await _append_write_verify(
        name="RunCommand",
        args={"command": "gh issue edit 99 --add-assignee bob"},
        output="ok",
        is_error=False,
        cwd=tmp_path,
    )
    assert "[write-verify OK]" in out


@pytest.mark.asyncio
async def test_gated_write_then_verify_mismatch_via_runner_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Write exits 0 through ``_run_gated_tool``; verify read shows MISMATCH."""
    import asyncio
    from typing import Any

    from jarvis.agent_chat.runner_api import TurnHandle, _run_gated_tool
    from jarvis.agent_chat.store import AgentChatSession

    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text("gitdir: ../main/.git/worktrees/wt\n", encoding="utf-8")
    calls: list[str] = []

    async def fake_execute(name: str, args: dict[str, Any], *, cwd: Path):
        cmd = str(args.get("command") or "")
        calls.append(cmd)
        if "issue edit" in cmd:
            return "https://github.com/o/r/issues/217\n[exit 0 · 0.1s]", False
        if "issue view" in cmd:
            return '{"assignees":[{"login":"someone-else"}]}', False
        return "unexpected", True

    monkeypatch.setattr("jarvis.agent_chat.runner_api.execute_tool", fake_execute)

    handle = TurnHandle(
        session=AgentChatSession(
            session_id="s1",
            title="",
            provider="fakeprov",
            model="m",
            effort="",
            cwd=str(wt),
            permission_mode="auto",
            vendor_session=None,
            created_ms=0,
            updated_ms=0,
            message_count=0,
            preview="",
            surface="agent",
            mode="project",
        ),
        turn_id="t1",
        emit=lambda _e: asyncio.sleep(0),
        request_approval=lambda *_a, **_k: asyncio.sleep(0, result="deny"),
        cancel=asyncio.Event(),
    )
    write_args = {"command": "gh issue edit 217 --add-assignee grok"}
    output, is_error = await _run_gated_tool(
        handle, "auto", "c1", "RunCommand", write_args, wt
    )
    assert is_error is False
    assert "https://github.com/o/r/issues/217" in output
    # Same appendix path the runner uses after ``_run_gated_tool``.
    final = await _append_write_verify(
        name="RunCommand",
        args=write_args,
        output=output,
        is_error=is_error,
        cwd=wt,
    )
    assert "[write-verify MISMATCH]" in final
    assert "missing grok" in final
    assert any("issue edit" in c for c in calls)
    assert any("issue view" in c for c in calls)
