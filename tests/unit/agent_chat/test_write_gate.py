"""GitHub/git write gate: chat / project-main / worktree / reads."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from jarvis.agent_chat import write_gate as wg
from jarvis.agent_chat.runner_api import TurnHandle, _run_gated_tool
from jarvis.agent_chat.store import AgentChatSession


def _session(*, mode: str, cwd: str) -> AgentChatSession:
    return AgentChatSession(
        session_id="s1",
        title="",
        provider="fakeprov",
        model="m",
        effort="",
        cwd=cwd,
        permission_mode="auto",
        vendor_session=None,
        created_ms=0,
        updated_ms=0,
        message_count=0,
        preview="",
        surface="agent",
        mode=mode,
    )


def test_command_is_write_detects_gh_and_git():
    assert wg.command_is_write("gh issue edit 217 --add-assignee grok")
    assert wg.command_is_write("gh issue create --title t")
    assert wg.command_is_write("gh issue close 1")
    assert wg.command_is_write("gh issue comment 1 --body hi")
    assert wg.command_is_write("gh pr create --title t --body b")
    assert wg.command_is_write("gh pr merge 12")
    assert wg.command_is_write("gh pr edit 12 --title x")
    assert wg.command_is_write("gh release create v1.0")
    assert wg.command_is_write("gh repo edit --description x")
    assert wg.command_is_write("gh api -X POST /repos/o/r/issues")
    assert wg.command_is_write("gh api --method PATCH /repos/o/r")
    assert wg.command_is_write("git push origin HEAD")
    assert wg.command_is_write("git commit -m msg")


def test_reads_are_not_writes():
    assert not wg.command_is_write("gh issue view 217 --json assignees")
    assert not wg.command_is_write("gh issue list")
    assert not wg.command_is_write("gh pr view 12")
    assert not wg.command_is_write("gh api /repos/o/r/issues/1")
    assert not wg.command_is_write("gh api -X GET /repos/o/r")
    assert not wg.command_is_write("git status")
    assert not wg.command_is_write("git log -1")
    assert not wg.command_is_write("ls")


def test_checkout_detection(tmp_path: Path):
    main = tmp_path / "main"
    main.mkdir()
    (main / ".git").mkdir()
    assert wg.is_main_checkout(main)
    assert not wg.is_linked_worktree(main)

    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text("gitdir: ../main/.git/worktrees/wt\n", encoding="utf-8")
    assert wg.is_linked_worktree(wt)
    assert not wg.is_main_checkout(wt)


def test_four_cases(tmp_path: Path):
    main = tmp_path / "main"
    main.mkdir()
    (main / ".git").mkdir()
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text("gitdir: ../main/.git/worktrees/wt\n", encoding="utf-8")
    write_args = {"command": "gh issue edit 1 --add-assignee x"}
    read_args = {"command": "gh issue view 1 --json assignees"}

    # 1) Chat mode + write → needs OK
    d = wg.evaluate_write_gate(
        tool_name="RunCommand", args=write_args, mode="chat", cwd=str(wt)
    )
    assert d.is_write and d.needs_ok

    # 2) Project + main checkout + write → needs OK
    d = wg.evaluate_write_gate(
        tool_name="RunCommand", args=write_args, mode="project", cwd=str(main)
    )
    assert d.is_write and d.needs_ok

    # 3) Project + linked worktree + write → allowed
    d = wg.evaluate_write_gate(
        tool_name="RunCommand", args=write_args, mode="project", cwd=str(wt)
    )
    assert d.is_write and not d.needs_ok

    # 4) Read always free (chat + main)
    d = wg.evaluate_write_gate(
        tool_name="RunCommand", args=read_args, mode="chat", cwd=str(main)
    )
    assert not d.needs_ok


@pytest.mark.asyncio
async def test_run_gated_tool_asks_in_chat_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    asked: list[tuple[str, str]] = []

    async def ask(call_id: str, name: str, args: dict[str, Any], summary: str) -> str:
        asked.append((name, str(args.get("command"))))
        return "deny"

    async def fake_execute(name: str, args: dict[str, Any], *, cwd: Path):
        return "ran", False

    monkeypatch.setattr("jarvis.agent_chat.runner_api.execute_tool", fake_execute)

    handle = TurnHandle(
        session=_session(mode="chat", cwd=""),
        turn_id="t1",
        emit=lambda _e: asyncio.sleep(0),
        request_approval=ask,
        cancel=asyncio.Event(),
    )
    out, err = await _run_gated_tool(
        handle,
        "auto",
        "c1",
        "RunCommand",
        {"command": "gh issue edit 1 --add-assignee x"},
        tmp_path,
    )
    assert err is True
    assert "write gate" in out.lower() or "Denied" in out
    assert asked and "gh issue edit" in asked[0][1]


@pytest.mark.asyncio
async def test_run_gated_tool_allows_worktree_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text("gitdir: ../main/.git/worktrees/wt\n", encoding="utf-8")
    asked: list[str] = []

    async def ask(call_id: str, name: str, args: dict[str, Any], summary: str) -> str:
        asked.append(name)
        return "deny"

    async def fake_execute(name: str, args: dict[str, Any], *, cwd: Path):
        return "ok-ran", False

    monkeypatch.setattr("jarvis.agent_chat.runner_api.execute_tool", fake_execute)

    handle = TurnHandle(
        session=_session(mode="project", cwd=str(wt)),
        turn_id="t1",
        emit=lambda _e: asyncio.sleep(0),
        request_approval=ask,
        cancel=asyncio.Event(),
    )
    out, err = await _run_gated_tool(
        handle,
        "auto",
        "c1",
        "RunCommand",
        {"command": "gh pr create --title t --body b"},
        wt,
    )
    assert err is False
    assert out == "ok-ran"
    assert asked == []


@pytest.mark.asyncio
async def test_run_gated_tool_read_never_asks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    asked: list[str] = []

    async def ask(call_id: str, name: str, args: dict[str, Any], summary: str) -> str:
        asked.append(name)
        return "deny"

    async def fake_execute(name: str, args: dict[str, Any], *, cwd: Path):
        return "view-ok", False

    monkeypatch.setattr("jarvis.agent_chat.runner_api.execute_tool", fake_execute)

    handle = TurnHandle(
        session=_session(mode="chat", cwd=""),
        turn_id="t1",
        emit=lambda _e: asyncio.sleep(0),
        request_approval=ask,
        cancel=asyncio.Event(),
    )
    out, err = await _run_gated_tool(
        handle,
        "auto",
        "c1",
        "RunCommand",
        {"command": "gh issue view 217 --json assignees"},
        tmp_path,
    )
    assert err is False
    assert out == "view-ok"
    assert asked == []


def test_folder_tool_write_needs_ok(tmp_path: Path):
    from jarvis.agent_chat.folder_tools import folder_tools

    tools = folder_tools(tmp_path, stance="ask", session_mode="chat")
    run = tools["RunCommand"]
    assert run.write_needs_ok({"command": "gh issue edit 1 --add-assignee x"})
    assert not run.write_needs_ok({"command": "gh issue view 1"})


# --- gaps closed after the 2026-11-21 review ---------------------------------


def test_git_global_options_do_not_hide_writes():
    # ``-C <path>`` takes a value; the verb is still ``push``.
    assert wg.command_is_write("git -C /x/y push origin main")
    assert wg.command_is_write("git -C D:/repo commit -m x")
    assert wg.command_is_write("git --git-dir .git push")
    assert wg.command_is_write("git -c user.name=x commit -m m")
    assert wg.command_is_write("git checkout main")
    assert wg.command_is_write("git reset --hard origin/main")
    assert wg.command_is_write("git stash")
    assert wg.command_is_write("git merge feature")
    assert wg.command_is_write("git rebase main")


def test_git_reads_with_global_options_stay_free():
    assert not wg.command_is_write("git -C /x/y status --short")
    assert not wg.command_is_write("git -C /x/y log --oneline -3")
    assert not wg.command_is_write("git --no-pager diff --stat")
    assert not wg.command_is_write("git -C /x/y show --stat HEAD")


def test_curl_and_wget_mutations_are_writes():
    assert wg.command_is_write("curl -X POST https://api.github.com/repos/o/r/issues -d '{}'")
    assert wg.command_is_write("curl --request PATCH https://api.github.com/repos/o/r")
    assert wg.command_is_write("curl -d 'x=1' https://api.github.com/repos/o/r")
    assert wg.command_is_write("curl --data-binary @body.json https://api.github.com/x")
    assert wg.command_is_write("wget --method=DELETE https://api.github.com/x")
    assert wg.command_is_write("Invoke-RestMethod -Method Post -Uri https://api.github.com/x")


def test_curl_get_is_a_read():
    assert not wg.command_is_write("curl https://api.github.com/repos/o/r/issues/1")
    assert not wg.command_is_write("curl -s -H 'Accept: application/json' https://example.com")


def test_scripts_that_call_github_mutating_are_writes():
    cmd = "python -c \"import requests; requests.post('https://api.github.com/repos/o/r/issues')\""
    assert wg.command_is_write(cmd)
    cmd2 = "node -e \"fetch('https://api.github.com/x', {method:'PATCH'})\""
    assert wg.command_is_write(cmd2)


def test_script_reading_github_is_a_read():
    cmd = "python -c \"import requests; print(requests.get('https://api.github.com/repos/o/r').text)\""
    assert not wg.command_is_write(cmd)
