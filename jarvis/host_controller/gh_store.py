"""Live GitHub project store via the ``gh`` CLI.

Milestone = project, issue = task. Status lives in ``status:*`` labels.
Commit SHA, worker, subagent kind and fix rounds live in a fixed HTML
comment block on the issue (``<!-- dodaios-host-state ... -->``) so a
restart can rebuild everything from GitHub alone.

Every subprocess uses ``NO_WINDOW_CREATIONFLAGS`` (AP-1). Output is parsed
from ``gh --json`` / ``gh api`` JSON bodies, never from screen text.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jarvis.core.process_utils import NO_WINDOW_CREATIONFLAGS
from jarvis.host_controller.types import TASK_STATUSES

log = logging.getLogger(__name__)

STATE_MARKER_START = "<!-- dodaios-host-state"
STATE_MARKER_END = "-->"

_STATE_RE = re.compile(
    r"<!--\s*dodaios-host-state\s*(.*?)\s*-->",
    re.DOTALL | re.IGNORECASE,
)

GhRunner = Callable[..., Any]


@dataclass
class HostIssue:
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
    state_comment_id: int | None = None
    closed: bool = False


@dataclass
class HostMilestone:
    number: int
    title: str
    description: str = ""
    issues: dict[int, HostIssue] = field(default_factory=dict)


@dataclass
class _Completed:
    returncode: int
    stdout: str
    stderr: str


def _default_gh_runner(args: list[str], **kwargs: Any) -> _Completed:
    timeout = float(kwargs.get("timeout", 60))
    env = dict(os.environ)
    env.setdefault("GH_PROMPT_DISABLED", "1")
    env.setdefault("GH_NO_UPDATE_NOTIFIER", "1")
    if "env" in kwargs and isinstance(kwargs["env"], dict):
        env.update(kwargs["env"])
    binary = shutil.which("gh") or "gh"
    try:
        result = subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=env,
            creationflags=NO_WINDOW_CREATIONFLAGS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _Completed(returncode=1, stdout="", stderr=str(exc))
    return _Completed(
        returncode=int(result.returncode),
        stdout=result.stdout or "",
        stderr=result.stderr or "",
    )


def _parse_state_block(text: str) -> dict[str, str]:
    match = _STATE_RE.search(text or "")
    if not match:
        return {}
    out: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def _render_state_block(
    *,
    commit_sha: str | None,
    worker: str | None,
    subagent_kind: str | None,
    fix_rounds: int,
    review_pass_sha: str | None = None,
    depends_on: list[int] | None = None,
    files: list[str] | None = None,
) -> str:
    deps = ",".join(str(int(n)) for n in (depends_on or []))
    file_list = ",".join(str(p).replace("\\", "/") for p in (files or []))
    return (
        f"{STATE_MARKER_START}\n"
        f"commit_sha: {commit_sha or ''}\n"
        f"worker: {worker or ''}\n"
        f"subagent_kind: {subagent_kind or ''}\n"
        f"fix_rounds: {int(fix_rounds)}\n"
        f"review_pass_sha: {review_pass_sha or ''}\n"
        f"depends_on: {deps}\n"
        f"files: {file_list}\n"
        f"{STATE_MARKER_END}"
    )


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def _parse_str_list(raw: str) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def _status_from_labels(labels: list[str]) -> str:
    for label in labels:
        if label.startswith("status:"):
            status = label.split(":", 1)[1].strip()
            if status in TASK_STATUSES:
                return status
    return "planned"


def _label_names(raw: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(raw, list):
        return names
    for item in raw:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])
    return names


class GhProjectStore:
    """``ProjectStore`` backed by GitHub milestones + issues via ``gh``."""

    def __init__(
        self,
        *,
        repo: str = "",
        runner: GhRunner | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.repo = (repo or os.environ.get("GH_REPO") or "").strip()
        self._runner = runner or _default_gh_runner
        self.timeout_s = timeout_s

    # --- low-level gh -------------------------------------------------------

    def _run(self, args: list[str], **kwargs: Any) -> _Completed:
        env = dict(kwargs.pop("env", {}) or {})
        if self.repo:
            env.setdefault("GH_REPO", self.repo)
        raw = self._runner(list(args), timeout=self.timeout_s, env=env, **kwargs)
        if isinstance(raw, _Completed):
            return raw
        returncode = int(getattr(raw, "returncode", 1))
        stdout = str(getattr(raw, "stdout", "") or "")
        stderr = str(getattr(raw, "stderr", "") or "")
        return _Completed(returncode=returncode, stdout=stdout, stderr=stderr)

    def _run_json(self, args: list[str], **kwargs: Any) -> Any:
        result = self._run(args, **kwargs)
        if result.returncode != 0:
            raise RuntimeError(
                f"gh {' '.join(args)} failed ({result.returncode}): "
                f"{(result.stderr or result.stdout)[:400]}"
            )
        text = (result.stdout or "").strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except ValueError as exc:
            raise RuntimeError(
                f"gh returned non-JSON for {' '.join(args)}: {text[:200]}"
            ) from exc

    def _repo_args(self) -> list[str]:
        if self.repo:
            return ["-R", self.repo]
        return []

    def _api_path(self, suffix: str) -> str:
        if self.repo:
            return f"repos/{self.repo}/{suffix.lstrip('/')}"
        return f"repos/{{owner}}/{{repo}}/{suffix.lstrip('/')}"

    # --- milestones ---------------------------------------------------------

    def create_milestone(self, title: str, description: str = "") -> HostMilestone:
        existing = self.get_milestone(title)
        if existing is not None:
            raise ValueError(f"milestone already exists: {title}")
        payload = self._run_json(
            [
                "api",
                self._api_path("milestones"),
                "-f",
                f"title={title}",
                "-f",
                f"description={description or ''}",
                "-f",
                "state=open",
            ]
        )
        if not isinstance(payload, dict):
            raise RuntimeError("create_milestone: expected JSON object")
        return HostMilestone(
            number=int(payload["number"]),
            title=str(payload.get("title") or title),
            description=str(payload.get("description") or description or ""),
        )

    def get_milestone(self, title: str) -> HostMilestone | None:
        key = title.strip()
        payload = self._run_json(
            [
                "api",
                self._api_path("milestones"),
                "--paginate",
                "-q",
                ".",
            ]
        )
        rows = payload if isinstance(payload, list) else []
        # Some runners return a single page object; tolerate both.
        if isinstance(payload, dict):
            rows = [payload]
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("title") or "").strip() == key:
                return HostMilestone(
                    number=int(row["number"]),
                    title=str(row["title"]),
                    description=str(row.get("description") or ""),
                )
        return None

    # --- issues -------------------------------------------------------------

    def _ensure_status_label(self, status: str) -> None:
        name = f"status:{status}"
        # Best-effort: ignore failure when the label already exists.
        self._run(
            [
                "label",
                "create",
                name,
                "--force",
                *self._repo_args(),
                "--color",
                "0E8A16",
                "--description",
                f"Host controller status {status}",
            ]
        )

    def create_issue(
        self,
        milestone_title: str,
        title: str,
        body: str,
        *,
        status: str = "planned",
    ) -> HostIssue:
        if status not in TASK_STATUSES:
            raise ValueError(f"unknown status: {status}")
        self._ensure_status_label(status)
        milestone = self.get_milestone(milestone_title)
        if milestone is None:
            raise KeyError(f"milestone not found: {milestone_title}")
        state_block = _render_state_block(
            commit_sha=None,
            worker=None,
            subagent_kind=None,
            fix_rounds=0,
            review_pass_sha=None,
        )
        full_body = f"{body.rstrip()}\n\n{state_block}\n"
        # ``gh issue create`` prints a URL only (no --json on current gh).
        # Create through the API so the host always parses structured JSON.
        payload = self._run_json(
            [
                "api",
                self._api_path("issues"),
                "-f",
                f"title={title}",
                "-f",
                f"body={full_body}",
                "-F",
                f"milestone={milestone.number}",
                "-f",
                f"labels[]=status:{status}",
            ]
        )
        if not isinstance(payload, dict):
            listed = self.list_issues(milestone_title)
            for issue in reversed(listed):
                if issue.title == title:
                    return issue
            raise RuntimeError("create_issue: could not parse issue JSON")
        return self._issue_from_payload(payload, default_status=status)

    def get_issue(self, milestone_title: str, number: int) -> HostIssue | None:
        payload = self._run_json(
            [
                "issue",
                "view",
                str(number),
                *self._repo_args(),
                "--json",
                "number,title,body,labels,milestone,state,comments",
            ]
        )
        if not isinstance(payload, dict):
            return None
        ms = payload.get("milestone")
        if isinstance(ms, dict):
            ms_title = str(ms.get("title") or "")
            if ms_title and ms_title.strip() != milestone_title.strip():
                return None
        return self._issue_from_payload(payload)

    def list_issues(self, milestone_title: str) -> list[HostIssue]:
        payload = self._run_json(
            [
                "issue",
                "list",
                *self._repo_args(),
                "--milestone",
                milestone_title,
                "--state",
                "all",
                "--limit",
                "500",
                "--json",
                "number,title,body,labels,milestone,state,comments",
            ]
        )
        if not isinstance(payload, list):
            return []
        issues = [self._issue_from_payload(row) for row in payload if isinstance(row, dict)]
        return sorted(issues, key=lambda item: item.number)

    def set_status(self, milestone_title: str, number: int, status: str) -> HostIssue:
        if status not in TASK_STATUSES:
            raise ValueError(f"unknown status: {status}")
        issue = self._require_issue(milestone_title, number)
        self._ensure_status_label(status)
        # Drop previous status:* labels, add the new one.
        for label in list(issue.labels):
            if label.startswith("status:") and label != f"status:{status}":
                self._run(
                    [
                        "issue",
                        "edit",
                        str(number),
                        *self._repo_args(),
                        "--remove-label",
                        label,
                    ]
                )
        self._run(
            [
                "issue",
                "edit",
                str(number),
                *self._repo_args(),
                "--add-label",
                f"status:{status}",
            ]
        )
        if status == "accepted":
            self._run(
                ["issue", "close", str(number), *self._repo_args(), "--reason", "completed"]
            )
        elif issue.closed and status != "accepted":
            self._run(["issue", "reopen", str(number), *self._repo_args()])
        refreshed = self.get_issue(milestone_title, number)
        if refreshed is None:
            issue.status = status
            issue.labels = [label for label in issue.labels if not label.startswith("status:")]
            issue.labels.append(f"status:{status}")
            return issue
        return refreshed

    def set_commit_sha(self, milestone_title: str, number: int, sha: str) -> HostIssue:
        issue = self._require_issue(milestone_title, number)
        issue.commit_sha = sha
        self._upsert_state_comment(issue)
        return issue

    def set_worker(
        self,
        milestone_title: str,
        number: int,
        worker: str,
        *,
        subagent_kind: str | None = None,
    ) -> HostIssue:
        issue = self._require_issue(milestone_title, number)
        issue.worker = worker
        issue.subagent_kind = subagent_kind
        self._upsert_state_comment(issue)
        return issue

    def bump_fix_round(self, milestone_title: str, number: int) -> int:
        issue = self._require_issue(milestone_title, number)
        issue.fix_rounds = int(issue.fix_rounds or 0) + 1
        self._upsert_state_comment(issue)
        return issue.fix_rounds

    def set_review_pass_sha(
        self, milestone_title: str, number: int, sha: str | None
    ) -> HostIssue:
        issue = self._require_issue(milestone_title, number)
        issue.review_pass_sha = sha
        self._upsert_state_comment(issue)
        return issue

    def update_milestone_description(self, title: str, description: str) -> HostMilestone:
        ms = self.get_milestone(title)
        if ms is None:
            raise KeyError(f"milestone not found: {title}")
        self._run_json(
            [
                "api",
                self._api_path(f"milestones/{ms.number}"),
                "-X",
                "PATCH",
                "-f",
                f"description={description}",
            ]
        )
        ms.description = description
        return ms

    def set_depends_on(
        self, milestone_title: str, number: int, depends_on: list[int]
    ) -> HostIssue:
        issue = self._require_issue(milestone_title, number)
        issue.depends_on = [int(n) for n in depends_on]
        self._upsert_state_comment(issue)
        return issue

    def set_files(
        self, milestone_title: str, number: int, files: list[str]
    ) -> HostIssue:
        issue = self._require_issue(milestone_title, number)
        issue.files = [str(path) for path in files]
        self._upsert_state_comment(issue)
        return issue

    def ready_issues(self, milestone_title: str) -> list[HostIssue]:
        from jarvis.host_controller.deps import ready_issue_numbers

        issues = self.list_issues(milestone_title)
        ready = set(ready_issue_numbers(issues))
        return [issue for issue in issues if issue.number in ready]

    def progress(self, milestone_title: str) -> dict[str, Any]:
        issues = self.list_issues(milestone_title)
        planned = len(issues)
        accepted = sum(
            1
            for issue in issues
            if issue.status == "accepted" or (issue.closed and issue.status == "accepted")
        )
        # Prefer the status label; closed-and-accepted is the durable form.
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
        milestones_raw = self._run_json(
            ["api", self._api_path("milestones"), "--paginate", "-q", "."]
        )
        rows = milestones_raw if isinstance(milestones_raw, list) else []
        if isinstance(milestones_raw, dict):
            rows = [milestones_raw]
        out: dict[str, Any] = {"milestones": {}}
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            issues = self.list_issues(title)
            out["milestones"][title] = {
                "number": int(row["number"]),
                "title": title,
                "description": str(row.get("description") or ""),
                "issues": {
                    str(issue.number): {
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
                    for issue in issues
                },
            }
        return out

    # --- helpers ------------------------------------------------------------

    def _require_issue(self, milestone_title: str, number: int) -> HostIssue:
        issue = self.get_issue(milestone_title, number)
        if issue is None:
            raise KeyError(f"issue #{number} missing in {milestone_title}")
        return issue

    def _issue_from_payload(
        self, payload: dict[str, Any], *, default_status: str | None = None
    ) -> HostIssue:
        labels = _label_names(payload.get("labels"))
        status = _status_from_labels(labels)
        if default_status and status == "planned" and default_status in TASK_STATUSES:
            # Fresh create may not echo labels on every gh version.
            if f"status:{default_status}" not in labels:
                status = default_status
                labels = list(labels) + [f"status:{default_status}"]
        comments_raw = payload.get("comments") or []
        comment_bodies: list[str] = []
        state_comment_id: int | None = None
        state: dict[str, str] = {}
        if isinstance(comments_raw, list):
            for comment in comments_raw:
                if isinstance(comment, str):
                    comment_bodies.append(comment)
                    parsed = _parse_state_block(comment)
                    if parsed:
                        state = parsed
                    continue
                if not isinstance(comment, dict):
                    continue
                body = str(comment.get("body") or "")
                comment_bodies.append(body)
                parsed = _parse_state_block(body)
                if parsed:
                    state = parsed
                    try:
                        state_comment_id = int(comment.get("id"))
                    except (TypeError, ValueError):
                        state_comment_id = None
        # Body may also carry the initial state block (create path).
        body_text = str(payload.get("body") or "")
        if not state:
            state = _parse_state_block(body_text)
        commit_sha = state.get("commit_sha") or None
        if commit_sha == "":
            commit_sha = None
        worker = state.get("worker") or None
        if worker == "":
            worker = None
        subagent_kind = state.get("subagent_kind") or None
        if subagent_kind == "":
            subagent_kind = None
        review_pass_sha = state.get("review_pass_sha") or None
        if review_pass_sha == "":
            review_pass_sha = None
        try:
            fix_rounds = int(state.get("fix_rounds") or 0)
        except ValueError:
            fix_rounds = 0
        depends_on = _parse_int_list(state.get("depends_on") or "")
        files = _parse_str_list(state.get("files") or "")
        closed = str(payload.get("state") or "").upper() in {"CLOSED", "CLOSE"}
        return HostIssue(
            number=int(payload["number"]),
            title=str(payload.get("title") or ""),
            body=body_text,
            status=status,
            commit_sha=commit_sha,
            fix_rounds=fix_rounds,
            worker=worker,
            subagent_kind=subagent_kind,
            review_pass_sha=review_pass_sha,
            depends_on=depends_on,
            files=files,
            labels=labels,
            comments=comment_bodies,
            state_comment_id=state_comment_id,
            closed=closed,
        )

    def _upsert_state_comment(self, issue: HostIssue) -> None:
        body = _render_state_block(
            commit_sha=issue.commit_sha,
            worker=issue.worker,
            subagent_kind=issue.subagent_kind,
            fix_rounds=issue.fix_rounds,
            review_pass_sha=issue.review_pass_sha,
            depends_on=list(issue.depends_on or []),
            files=list(issue.files or []),
        )
        if issue.state_comment_id is not None:
            self._run_json(
                [
                    "api",
                    self._api_path(f"issues/comments/{issue.state_comment_id}"),
                    "-X",
                    "PATCH",
                    "-f",
                    f"body={body}",
                ]
            )
            return
        # Prefer editing the issue body when it already contains the marker.
        if STATE_MARKER_START in (issue.body or ""):
            new_body = _STATE_RE.sub(body, issue.body, count=1)
            self._run(
                [
                    "issue",
                    "edit",
                    str(issue.number),
                    *self._repo_args(),
                    "--body",
                    new_body,
                ]
            )
            issue.body = new_body
            return
        created = self._run_json(
            [
                "api",
                self._api_path(f"issues/{issue.number}/comments"),
                "-f",
                f"body={body}",
            ]
        )
        if isinstance(created, dict) and "id" in created:
            try:
                issue.state_comment_id = int(created["id"])
            except (TypeError, ValueError):
                issue.state_comment_id = None
        issue.comments.append(body)
