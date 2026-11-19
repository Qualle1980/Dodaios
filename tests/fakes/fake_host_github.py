"""In-memory GitHub project store for host-controller tests.

Milestone = project, issue = task, labels/status comments = task state.
No network, no ``gh`` CLI, no ``unittest.mock``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TASK_STATUSES = frozenset(
    {
        "planned",
        "in-progress",
        "review",
        "accepted",
        "blocked",
        "waiting-quota",
        "needs-check",
    }
)


@dataclass
class FakeIssue:
    number: int
    title: str
    body: str
    status: str = "planned"
    commit_sha: str | None = None
    fix_rounds: int = 0
    worker: str | None = None
    subagent_kind: str | None = None
    review_pass_sha: str | None = None
    depends_on: list[int] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)


@dataclass
class FakeMilestone:
    number: int
    title: str
    description: str = ""
    issues: dict[int, FakeIssue] = field(default_factory=dict)


class FakeGitHubProjectStore:
    """GitHub-shaped store the host rebuilds from on start."""

    def __init__(self) -> None:
        self.milestones: dict[str, FakeMilestone] = {}
        self._next_milestone = 1
        self._next_issue = 1
        self.actions: list[dict[str, Any]] = []

    def create_milestone(self, title: str, description: str = "") -> FakeMilestone:
        key = title.strip()
        if key in self.milestones:
            raise ValueError(f"milestone already exists: {title}")
        ms = FakeMilestone(
            number=self._next_milestone,
            title=title,
            description=description,
        )
        self._next_milestone += 1
        self.milestones[key] = ms
        self.actions.append({"op": "create_milestone", "title": title})
        return ms

    def update_milestone_description(self, title: str, description: str) -> FakeMilestone:
        ms = self.milestones[title.strip()]
        ms.description = description
        self.actions.append(
            {"op": "update_milestone_description", "title": title, "description": description}
        )
        return ms

    def create_issue(
        self,
        milestone_title: str,
        title: str,
        body: str,
        *,
        status: str = "planned",
    ) -> FakeIssue:
        ms = self.milestones[milestone_title]
        if status not in TASK_STATUSES:
            raise ValueError(f"unknown status: {status}")
        issue = FakeIssue(
            number=self._next_issue,
            title=title,
            body=body,
            status=status,
            labels=[f"status:{status}"],
        )
        self._next_issue += 1
        ms.issues[issue.number] = issue
        self.actions.append(
            {
                "op": "create_issue",
                "milestone": milestone_title,
                "number": issue.number,
                "title": title,
            }
        )
        return issue

    def get_milestone(self, title: str) -> FakeMilestone | None:
        return self.milestones.get(title.strip())

    def get_issue(self, milestone_title: str, number: int) -> FakeIssue | None:
        ms = self.get_milestone(milestone_title)
        if ms is None:
            return None
        return ms.issues.get(number)

    def set_status(self, milestone_title: str, number: int, status: str) -> FakeIssue:
        if status not in TASK_STATUSES:
            raise ValueError(f"unknown status: {status}")
        issue = self._require_issue(milestone_title, number)
        issue.status = status
        issue.labels = [label for label in issue.labels if not label.startswith("status:")]
        issue.labels.append(f"status:{status}")
        issue.comments.append(f"status:{status}")
        self.actions.append(
            {
                "op": "set_status",
                "milestone": milestone_title,
                "number": number,
                "status": status,
            }
        )
        return issue

    def set_commit_sha(self, milestone_title: str, number: int, sha: str) -> FakeIssue:
        issue = self._require_issue(milestone_title, number)
        issue.commit_sha = sha
        self.actions.append(
            {
                "op": "set_commit_sha",
                "milestone": milestone_title,
                "number": number,
                "sha": sha,
            }
        )
        return issue

    def set_worker(
        self,
        milestone_title: str,
        number: int,
        worker: str,
        *,
        subagent_kind: str | None = None,
    ) -> FakeIssue:
        issue = self._require_issue(milestone_title, number)
        issue.worker = worker
        issue.subagent_kind = subagent_kind
        self.actions.append(
            {
                "op": "set_worker",
                "milestone": milestone_title,
                "number": number,
                "worker": worker,
                "subagent_kind": subagent_kind,
            }
        )
        return issue

    def bump_fix_round(self, milestone_title: str, number: int) -> int:
        issue = self._require_issue(milestone_title, number)
        issue.fix_rounds += 1
        self.actions.append(
            {
                "op": "bump_fix_round",
                "milestone": milestone_title,
                "number": number,
                "fix_rounds": issue.fix_rounds,
            }
        )
        return issue.fix_rounds

    def set_review_pass_sha(
        self, milestone_title: str, number: int, sha: str | None
    ) -> FakeIssue:
        issue = self._require_issue(milestone_title, number)
        issue.review_pass_sha = sha
        self.actions.append(
            {
                "op": "set_review_pass_sha",
                "milestone": milestone_title,
                "number": number,
                "sha": sha,
            }
        )
        return issue

    def set_depends_on(
        self, milestone_title: str, number: int, depends_on: list[int]
    ) -> FakeIssue:
        issue = self._require_issue(milestone_title, number)
        issue.depends_on = [int(n) for n in depends_on]
        self.actions.append(
            {
                "op": "set_depends_on",
                "milestone": milestone_title,
                "number": number,
                "depends_on": list(issue.depends_on),
            }
        )
        return issue

    def set_files(
        self, milestone_title: str, number: int, files: list[str]
    ) -> FakeIssue:
        issue = self._require_issue(milestone_title, number)
        issue.files = [str(path) for path in files]
        self.actions.append(
            {
                "op": "set_files",
                "milestone": milestone_title,
                "number": number,
                "files": list(issue.files),
            }
        )
        return issue

    def list_issues(self, milestone_title: str) -> list[FakeIssue]:
        ms = self.get_milestone(milestone_title)
        if ms is None:
            return []
        return sorted(ms.issues.values(), key=lambda item: item.number)

    def ready_issues(self, milestone_title: str) -> list[FakeIssue]:
        from jarvis.host_controller.deps import ready_issue_numbers

        issues = self.list_issues(milestone_title)
        ready = set(ready_issue_numbers(issues))
        return [issue for issue in issues if issue.number in ready]

    def progress(self, milestone_title: str) -> dict[str, Any]:
        issues = self.list_issues(milestone_title)
        planned = len(issues)
        accepted = sum(1 for issue in issues if issue.status == "accepted")
        in_progress = sum(1 for issue in issues if issue.status == "in-progress")
        percent = 0.0 if planned == 0 else (100.0 * accepted / planned)
        return {
            "planned": planned,
            "accepted": accepted,
            "in_progress": in_progress,
            "percent": percent,
        }

    def snapshot(self) -> dict[str, Any]:
        """Serializable GitHub truth used to rebuild a local cache."""
        out: dict[str, Any] = {"milestones": {}}
        for title, ms in self.milestones.items():
            out["milestones"][title] = {
                "number": ms.number,
                "title": ms.title,
                "description": ms.description,
                "issues": {
                    str(num): {
                        "number": issue.number,
                        "title": issue.title,
                        "body": issue.body,
                        "status": issue.status,
                        "commit_sha": issue.commit_sha,
                        "fix_rounds": issue.fix_rounds,
                        "worker": issue.worker,
                        "subagent_kind": issue.subagent_kind,
                        "review_pass_sha": issue.review_pass_sha,
                        "depends_on": list(issue.depends_on),
                        "files": list(issue.files),
                        "labels": list(issue.labels),
                        "comments": list(issue.comments),
                    }
                    for num, issue in ms.issues.items()
                },
            }
        return out

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.milestones.clear()
        self._next_milestone = 1
        self._next_issue = 1
        for title, raw in (snapshot.get("milestones") or {}).items():
            ms = FakeMilestone(
                number=int(raw["number"]),
                title=str(raw["title"]),
                description=str(raw.get("description") or ""),
            )
            for issue_raw in (raw.get("issues") or {}).values():
                issue = FakeIssue(
                    number=int(issue_raw["number"]),
                    title=str(issue_raw["title"]),
                    body=str(issue_raw.get("body") or ""),
                    status=str(issue_raw.get("status") or "planned"),
                    commit_sha=issue_raw.get("commit_sha"),
                    fix_rounds=int(issue_raw.get("fix_rounds") or 0),
                    worker=issue_raw.get("worker"),
                    subagent_kind=issue_raw.get("subagent_kind"),
                    review_pass_sha=issue_raw.get("review_pass_sha"),
                    depends_on=[int(n) for n in (issue_raw.get("depends_on") or [])],
                    files=[str(p) for p in (issue_raw.get("files") or [])],
                    labels=list(issue_raw.get("labels") or []),
                    comments=list(issue_raw.get("comments") or []),
                )
                ms.issues[issue.number] = issue
                self._next_issue = max(self._next_issue, issue.number + 1)
            self.milestones[title] = ms
            self._next_milestone = max(self._next_milestone, ms.number + 1)

    def _require_issue(self, milestone_title: str, number: int) -> FakeIssue:
        issue = self.get_issue(milestone_title, number)
        if issue is None:
            raise KeyError(f"issue #{number} missing in {milestone_title}")
        return issue
