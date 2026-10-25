"""Voice folder bridge tests; independent of paid voice/provider calls."""
from __future__ import annotations

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
