"""GitHub is the project state store; a local dict is only a rebuilt cache.

Milestone = project, issue = task with acceptance criteria, label/comment =
status. Progress = accepted / planned, counted from GitHub.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProjectStore(Protocol):
    def create_milestone(self, title: str, description: str = "") -> Any: ...

    def create_issue(
        self,
        milestone_title: str,
        title: str,
        body: str,
        *,
        status: str = "planned",
    ) -> Any: ...

    def get_milestone(self, title: str) -> Any | None: ...

    def get_issue(self, milestone_title: str, number: int) -> Any | None: ...

    def set_status(self, milestone_title: str, number: int, status: str) -> Any: ...

    def set_commit_sha(self, milestone_title: str, number: int, sha: str) -> Any: ...

    def set_worker(
        self,
        milestone_title: str,
        number: int,
        worker: str,
        *,
        subagent_kind: str | None = None,
    ) -> Any: ...

    def bump_fix_round(self, milestone_title: str, number: int) -> int: ...

    def set_review_pass_sha(
        self, milestone_title: str, number: int, sha: str | None
    ) -> Any: ...

    def list_issues(self, milestone_title: str) -> list[Any]: ...

    def progress(self, milestone_title: str) -> dict[str, Any]: ...

    def snapshot(self) -> dict[str, Any]: ...


class LocalCache:
    """Optional local mirror. Always rebuilt from ``store.snapshot()`` on start."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def rebuild(self, store: ProjectStore) -> dict[str, Any]:
        self.data = store.snapshot()
        return self.data
