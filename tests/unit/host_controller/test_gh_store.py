"""GhProjectStore: ProjectStore over the gh CLI (scripted runner, no network)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from jarvis.host_controller.gh_store import GhProjectStore, HostIssue, HostMilestone


@dataclass
class ScriptedGhCall:
    args: list[str]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScriptedGhResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class ScriptedGhRunner:
    """Injectable gh runner for unit tests. No network, no unittest.mock."""

    def __init__(self) -> None:
        self.calls: list[ScriptedGhCall] = []
        self._queue: list[ScriptedGhResult] = []
        self.default = ScriptedGhResult(returncode=0, stdout="[]")

    def enqueue(self, *results: ScriptedGhResult) -> None:
        self._queue.extend(results)

    def __call__(self, args: list[str], **kwargs: Any) -> ScriptedGhResult:
        self.calls.append(ScriptedGhCall(args=list(args), kwargs=dict(kwargs)))
        if self._queue:
            return self._queue.pop(0)
        return self.default


def _json(data: Any) -> str:
    return json.dumps(data)


def test_create_and_get_milestone_uses_gh_api_json() -> None:
    runner = ScriptedGhRunner()
    # list milestones (empty), create milestone, list again for get
    runner.enqueue(
        ScriptedGhResult(stdout=_json([])),
        ScriptedGhResult(
            stdout=_json({"number": 3, "title": "demo", "description": "Demo project"})
        ),
        ScriptedGhResult(
            stdout=_json(
                [{"number": 3, "title": "demo", "description": "Demo project", "state": "open"}]
            )
        ),
    )
    store = GhProjectStore(repo="acme/demo", runner=runner)
    created = store.create_milestone("demo", description="Demo project")
    assert isinstance(created, HostMilestone)
    assert created.title == "demo"
    assert created.number == 3
    found = store.get_milestone("demo")
    assert found is not None
    assert found.number == 3
    assert any("milestones" in " ".join(call.args) for call in runner.calls)
    assert all("--json" in call.args or "api" in call.args for call in runner.calls)


def test_create_issue_sets_status_label_and_milestone() -> None:
    runner = ScriptedGhRunner()
    runner.enqueue(
        # ensure label
        ScriptedGhResult(returncode=0, stdout=""),
        # get_milestone (api list)
        ScriptedGhResult(
            stdout=_json(
                [{"number": 3, "title": "demo", "description": "Demo", "state": "open"}]
            )
        ),
        # create issue via api (JSON object)
        ScriptedGhResult(
            stdout=_json(
                {
                    "number": 11,
                    "title": "Task one",
                    "body": "acceptance",
                    "labels": [{"name": "status:planned"}],
                    "milestone": {"title": "demo", "number": 3},
                    "state": "OPEN",
                    "comments": [],
                }
            )
        ),
    )
    store = GhProjectStore(repo="acme/demo", runner=runner)
    issue = store.create_issue("demo", "Task one", "acceptance", status="planned")
    assert isinstance(issue, HostIssue)
    assert issue.number == 11
    assert issue.status == "planned"
    joined = " ".join(" ".join(c.args) for c in runner.calls)
    assert "issue" in joined or "issues" in joined
    assert "status:planned" in joined or "planned" in joined
    assert any("api" in call.args for call in runner.calls)


def test_set_commit_sha_and_worker_persist_in_state_comment() -> None:
    runner = ScriptedGhRunner()
    # get issue (for set_commit_sha), create/update comment, get issue again for set_worker
    issue_payload = {
        "number": 11,
        "title": "Task one",
        "body": "acceptance",
        "labels": [{"name": "status:in-progress"}],
        "milestone": {"title": "demo", "number": 3},
        "state": "OPEN",
        "comments": [
            {
                "id": 99,
                "body": (
                    "<!-- dodaios-host-state\n"
                    "commit_sha:\n"
                    "worker:\n"
                    "subagent_kind:\n"
                    "fix_rounds: 0\n"
                    "-->"
                ),
            }
        ],
    }
    runner.enqueue(
        ScriptedGhResult(stdout=_json(issue_payload)),
        ScriptedGhResult(stdout=_json({"id": 99})),  # update comment for sha
        ScriptedGhResult(stdout=_json(issue_payload)),
        ScriptedGhResult(stdout=_json({"id": 99})),  # update comment for worker
    )
    store = GhProjectStore(repo="acme/demo", runner=runner)
    issue = store.set_commit_sha("demo", 11, "abc123")
    assert issue.commit_sha == "abc123"
    issue2 = store.set_worker("demo", 11, "claude", subagent_kind="Explore")
    assert issue2.worker == "claude"
    assert issue2.subagent_kind == "Explore"
    bodies = " ".join(
        str(call.kwargs) + " ".join(call.args) for call in runner.calls
    )
    assert "dodaios-host-state" in bodies or "abc123" in bodies


def test_bump_fix_round_and_progress_count_accepted() -> None:
    runner = ScriptedGhRunner()
    issue_open = {
        "number": 1,
        "title": "T1",
        "body": "a",
        "labels": [{"name": "status:in-progress"}],
        "milestone": {"title": "demo", "number": 1},
        "state": "OPEN",
        "comments": [
            {
                "id": 1,
                "body": (
                    "<!-- dodaios-host-state\n"
                    "commit_sha: c1\n"
                    "worker: claude\n"
                    "subagent_kind:\n"
                    "fix_rounds: 1\n"
                    "-->"
                ),
            }
        ],
    }
    runner.enqueue(
        ScriptedGhResult(stdout=_json(issue_open)),
        ScriptedGhResult(stdout=_json({"id": 1})),
    )
    store = GhProjectStore(repo="acme/demo", runner=runner)
    bumped = store.bump_fix_round("demo", 1)
    assert bumped == 2

    # progress: list issues under milestone
    runner.enqueue(
        ScriptedGhResult(
            stdout=_json(
                [
                    {
                        "number": 1,
                        "title": "T1",
                        "body": "a",
                        "labels": [{"name": "status:accepted"}],
                        "state": "CLOSED",
                        "comments": [],
                        "milestone": {"title": "demo"},
                    },
                    {
                        "number": 2,
                        "title": "T2",
                        "body": "b",
                        "labels": [{"name": "status:planned"}],
                        "state": "OPEN",
                        "comments": [],
                        "milestone": {"title": "demo"},
                    },
                    {
                        "number": 3,
                        "title": "T3",
                        "body": "c",
                        "labels": [{"name": "status:in-progress"}],
                        "state": "OPEN",
                        "comments": [],
                        "milestone": {"title": "demo"},
                    },
                ]
            )
        )
    )
    progress = store.progress("demo")
    assert progress["planned"] == 3
    assert progress["accepted"] == 1
    assert progress["in_progress"] == 1
    assert progress["percent"] == pytest.approx(100.0 / 3.0)


def test_snapshot_rebuilds_from_github_alone() -> None:
    runner = ScriptedGhRunner()
    runner.enqueue(
        ScriptedGhResult(
            stdout=_json(
                [{"number": 1, "title": "demo", "description": "d", "state": "open"}]
            )
        ),
        ScriptedGhResult(
            stdout=_json(
                [
                    {
                        "number": 7,
                        "title": "T1",
                        "body": "acc",
                        "labels": [{"name": "status:needs-check"}],
                        "state": "OPEN",
                        "comments": [
                            {
                                "id": 5,
                                "body": (
                                    "<!-- dodaios-host-state\n"
                                    "commit_sha: deadbeef\n"
                                    "worker: grok\n"
                                    "subagent_kind:\n"
                                    "fix_rounds: 0\n"
                                    "-->"
                                ),
                            }
                        ],
                        "milestone": {"title": "demo", "number": 1},
                    }
                ]
            )
        ),
    )
    store = GhProjectStore(repo="acme/demo", runner=runner)
    snap = store.snapshot()
    assert "demo" in snap["milestones"]
    issue = snap["milestones"]["demo"]["issues"]["7"]
    assert issue["status"] == "needs-check"
    assert issue["commit_sha"] == "deadbeef"
    assert issue["worker"] == "grok"


@pytest.mark.skipif(
    __import__("os").environ.get("DODAIOS_HOST_LIVE") != "1",
    reason="live GitHub store requires DODAIOS_HOST_LIVE=1",
)
def test_live_gh_project_store_roundtrip() -> None:
    store = GhProjectStore(repo="")  # repo from GH_REPO / cwd
    title = "host-controller-live-probe"
    existing = store.get_milestone(title)
    if existing is None:
        store.create_milestone(title, description="live probe")
    issue = store.create_issue(title, "live task", "acceptance: none", status="planned")
    store.set_status(title, issue.number, "in-progress")
    store.set_commit_sha(title, issue.number, "live000")
    store.set_worker(title, issue.number, "claude", subagent_kind=None)
    again = store.get_issue(title, issue.number)
    assert again is not None
    assert again.commit_sha == "live000"
    assert again.worker == "claude"
    snap = store.snapshot()
    assert title in snap["milestones"]
