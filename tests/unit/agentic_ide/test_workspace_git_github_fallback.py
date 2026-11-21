"""F-2 — workspace-git GitHub failures steer voice to discover/call_tool."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvis.agentic_ide import git_overview, workspace_git
from jarvis.agentic_ide.git_ops import GitError


def test_pull_requests_without_github_link_steers_to_discover(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fake = SimpleNamespace(
        available=True,
        branch="main",
        branches=[],
        pull_requests=[],
        github=SimpleNamespace(
            available=False,
            reason="GitHub is not connected yet.",
            repo="",
        ),
    )
    monkeypatch.setattr(git_overview, "overview", lambda folder: fake)
    payload = workspace_git._pull_requests(str(tmp_path))
    assert payload["github"]["connected"] is False
    assert "discover_tools" in payload["next_step"]
    assert "github" in payload["next_step"].lower()


@pytest.mark.asyncio
async def test_git_error_on_pull_requests_includes_discover_next_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Orchestrator:
        def __init__(self) -> None:
            self.registry = SimpleNamespace(sessions=[])

        def graph(self):
            return {}

        def _workspace(self, args, graph, unused):
            return (
                {"id": "p1"},
                {"id": "w1", "name": "Repo", "folder": "C:/tmp/Repo"},
            )

    def boom(op, folder, args, open_folders):
        raise GitError("Repo is not a git repository.", code="not_repo")

    monkeypatch.setattr(workspace_git, "_perform", boom)

    result = await workspace_git.run(
        _Orchestrator(), {"op": "pull_requests", "workspace": "Repo"}
    )
    assert result["success"] is False
    assert result["code"] == "not_repo"
    assert "discover_tools" in result["next_step"]
