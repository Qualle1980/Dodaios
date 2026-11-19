"""One open coding workspace over a real checkout, as the workspace tools reach it.

``workspace-git`` and ``workspace-test`` talk to the workspace orchestration
gateway, which resolves a workspace selector to a folder and then runs the
product's own git or test module in it. The real orchestrator needs a PTY
registry, a receipt ledger and a project library; none of that is under test
where only the resolution and the run matter. This fake keeps the product's
own ``workspace_git.run`` and ``workspace_tests.run`` and replaces only the
graph they resolve against: one project, one open workspace, one folder.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from jarvis.agentic_ide import workspace_git, workspace_tests


class FakeCheckoutOrchestrator:
    """Resolves exactly one workspace; anything else is refused like the real one."""

    def __init__(
        self,
        folder: str,
        *,
        project_id: str = "p1",
        workspace_id: str = "w1",
        project_name: str = "Bubble",
        workspace_name: str = "main",
        config: Any = None,
        timeout_s: float | None = None,
    ) -> None:
        self.project = {"id": project_id, "name": project_name, "path": str(folder)}
        self.workspace = {
            "id": workspace_id,
            "name": workspace_name,
            "folder": str(folder),
            "status": "open",
            "agents": [],
        }
        # ``workspace_git.run`` lists the open folders a worktree op must spare.
        self.registry = SimpleNamespace(sessions=[])
        self.config = config
        self.timeout_s = timeout_s
        self.calls: list[dict[str, Any]] = []

    def graph(self) -> dict[str, Any]:
        return {
            "projects": [{**self.project, "workspaces": [self.workspace]}],
            "active_workspace_id": self.workspace["id"],
        }

    def _workspace(
        self, args: dict[str, Any], graph: dict[str, Any], agent_ref: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | dict[str, Any]:
        named = str(args.get("workspace_id") or args.get("workspace") or "")
        if named and named not in {self.workspace["id"], self.workspace["name"]}:
            return {"status": "choose", "success": False, "reason": f"No workspace {named}."}
        return self.project, self.workspace

    async def run(self, args: dict[str, Any], *, trace_id: str = "") -> dict[str, Any]:
        self.calls.append(dict(args))
        action = args.get("action")
        if action == "git":
            return await workspace_git.run(self, args)
        if action == "test":
            return await workspace_tests.run(
                self, args, config=self.config, timeout_s=self.timeout_s
            )
        raise ValueError("Unknown workspace orchestration action.")


__all__ = ["FakeCheckoutOrchestrator"]
