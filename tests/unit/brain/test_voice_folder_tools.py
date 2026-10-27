"""Voice folder bridge tests; independent of paid voice/provider calls."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.brain.voice_folder_tools import scoped_voice_read_tools


class FakeTool:
    name = "Read"
    description = "Read a file."
    schema = {"type": "object"}
    risk_tier = "safe"

    async def execute(self, args, ctx):
        return SimpleNamespace(success=True, output=args["file_path"])


@pytest.mark.asyncio
async def test_read_stays_inside_selected_project(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "note.txt").write_text("hello")
    tool = scoped_voice_read_tools({"Read": FakeTool()}, project)["Read"]
    result = await tool.execute({"file_path": "note.txt"}, None)
    assert result.success is True
    assert result.output == str((project / "note.txt").resolve())


@pytest.mark.asyncio
async def test_voice_read_rejects_parent_escape(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    tool = scoped_voice_read_tools({"Read": FakeTool()}, project)["Read"]
    result = await tool.execute({"file_path": "../private.txt"}, None)
    assert result.success is False
    assert "outside" in result.error


@pytest.mark.asyncio
async def test_voice_read_rejects_symlink_escape(tmp_path: Path):
    project = tmp_path / "project"
    private = tmp_path / "private"
    project.mkdir()
    private.mkdir()
    try:
        (project / "external").symlink_to(private, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink permission unavailable")
    tool = scoped_voice_read_tools({"Read": FakeTool()}, project)["Read"]
    result = await tool.execute({"file_path": "external/password.txt"}, None)
    assert result.success is False


def test_only_four_read_tools_exposed(tmp_path: Path):
    fake = FakeTool()
    offered = scoped_voice_read_tools(
        {name: fake for name in ("Read", "Ls", "Glob", "Grep", "Write", "Edit", "RunCommand")},
        tmp_path,
    )
    assert set(offered) == {"Read", "Ls", "Glob", "Grep"}


def _junction(link: Path, target: Path) -> bool:
    """Make a Windows junction. False when this platform or box cannot."""
    if os.name != "nt":
        return False
    done = subprocess.run(
        ["cmd", "/d", "/s", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
    )
    return done.returncode == 0 and link.exists()


@pytest.mark.asyncio
async def test_grep_does_not_read_through_a_junction(tmp_path: Path):
    """A junction is no link to os.walk, so the walk itself must stay in root."""
    from jarvis.agent_chat.folder_tools import folder_tools

    project = tmp_path / "project"
    private = tmp_path / "private"
    project.mkdir()
    private.mkdir()
    (project / "own.txt").write_text("harmless", encoding="utf-8")
    (private / "password.txt").write_text("SUPERSECRET=42", encoding="utf-8")
    if not _junction(project / "ext", private):
        pytest.skip("junctions unavailable on this platform")

    tool = scoped_voice_read_tools(folder_tools(project, stance="plan"), project)["Grep"]
    result = await tool.execute({"pattern": "SUPERSECRET"}, None)
    output = str(result.output or "")
    assert "SUPERSECRET=42" not in output
    assert "password.txt" not in output


@pytest.mark.asyncio
async def test_glob_does_not_list_through_a_junction(tmp_path: Path):
    from jarvis.agent_chat.folder_tools import folder_tools

    project = tmp_path / "project"
    private = tmp_path / "private"
    project.mkdir()
    private.mkdir()
    (private / "password.txt").write_text("x", encoding="utf-8")
    if not _junction(project / "ext", private):
        pytest.skip("junctions unavailable on this platform")

    tool = scoped_voice_read_tools(folder_tools(project, stance="plan"), project)["Glob"]
    result = await tool.execute({"pattern": "*.txt"}, None)
    assert "password.txt" not in str(result.output or "")


@pytest.mark.asyncio
async def test_grep_does_not_read_through_a_symlinked_file(tmp_path: Path):
    from jarvis.agent_chat.folder_tools import folder_tools

    project = tmp_path / "project"
    private = tmp_path / "private"
    project.mkdir()
    private.mkdir()
    secret = private / "password.txt"
    secret.write_text("SUPERSECRET=42", encoding="utf-8")
    try:
        (project / "leak.txt").symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlink permission unavailable")

    tool = scoped_voice_read_tools(folder_tools(project, stance="plan"), project)["Grep"]
    result = await tool.execute({"pattern": "SUPERSECRET"}, None)
    assert "SUPERSECRET=42" not in str(result.output or "")


def test_resolver_failure_offers_no_folder_tools_but_keeps_the_rest():
    """A broken project resolver must cost the folder tools, not the voice surface."""
    from jarvis.brain.tool_gateway import BrainSupervisorToolGateway

    def boom():
        raise RuntimeError("IDE registry unavailable")

    manager = SimpleNamespace(
        _tools={"sentinel_tool": FakeTool()},
        _config=SimpleNamespace(computer_use=SimpleNamespace(enabled=False)),
    )
    gateway = BrainSupervisorToolGateway(manager, voice_project_resolver=boom)
    tools = gateway._voice_tools()

    assert not {"Read", "Ls", "Glob", "Grep"} & set(tools)
    assert "sentinel_tool" in tools
