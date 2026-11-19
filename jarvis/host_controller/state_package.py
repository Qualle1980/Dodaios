"""Gather a bounded project-state block for Instant's system prompt.

Instant plans from this package, never from assumptions. Hard cap 8 KB.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GitRunner = Callable[[list[str]], str]

STATE_PACKAGE_MAX_BYTES = 8192


def _default_git(args: list[str], *, cwd: Path) -> str:
    import subprocess

    from jarvis.core.process_utils import NO_WINDOW_CREATIONFLAGS

    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
        creationflags=NO_WINDOW_CREATIONFLAGS,
    )
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def _truncate(text: str, max_bytes: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    cut = raw[: max(0, max_bytes - 20)]
    return cut.decode("utf-8", errors="ignore").rstrip() + "\n…[truncated]\n"


def build_state_package(
    *,
    project: str,
    store: Any,
    repo_root: Path | None = None,
    progress_path: Path | None = None,
    open_prs: list[dict[str, Any]] | None = None,
    git_branch: str | None = None,
    git_log: list[str] | None = None,
    git_runner: GitRunner | None = None,
    max_bytes: int = STATE_PACKAGE_MAX_BYTES,
) -> str:
    """Return a markdown state block ≤ ``max_bytes``."""
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    lines: list[str] = [f"# Project state: {project}", ""]

    branch = git_branch
    log_lines = list(git_log or [])
    if branch is None or not log_lines:
        runner = git_runner or (lambda args: _default_git(args, cwd=root))
        if branch is None:
            branch = runner(["rev-parse", "--abbrev-ref", "HEAD"]) or "(unknown)"
        if not log_lines:
            raw = runner(["log", "-5", "--oneline"]) or ""
            log_lines = [row for row in raw.splitlines() if row.strip()]

    lines.append(f"Branch: {branch}")
    if log_lines:
        lines.append("Recent commits:")
        for row in log_lines[:5]:
            lines.append(f"- {row}")
    lines.append("")

    milestone = None
    getter = getattr(store, "get_milestone", None)
    if callable(getter):
        milestone = getter(project)
    if milestone is not None:
        desc = str(getattr(milestone, "description", "") or "").strip()
        lines.append(f"Milestone: {getattr(milestone, 'title', project)}")
        if desc:
            # Keep research notes short in the package.
            lines.append(f"Description: {desc[:500]}")
    issues = []
    lister = getattr(store, "list_issues", None)
    if callable(lister):
        issues = list(lister(project) or [])
    if issues:
        lines.append("Issues:")
        for issue in issues:
            number = getattr(issue, "number", "?")
            title = getattr(issue, "title", "")
            status = getattr(issue, "status", "")
            worker = getattr(issue, "worker", None) or ""
            deps = getattr(issue, "depends_on", None) or []
            files = getattr(issue, "files", None) or []
            bit = f"- #{number} [{status}] {title}"
            if worker:
                bit += f" worker={worker}"
            if deps:
                bit += f" depends_on={list(deps)}"
            if files:
                bit += f" files={list(files)[:6]}"
            lines.append(bit)
    else:
        lines.append("Issues: none")
    lines.append("")

    prs = list(open_prs or [])
    if prs:
        lines.append("Open PRs touching the project:")
        for pr in prs[:8]:
            num = pr.get("number")
            title = pr.get("title") or ""
            head = pr.get("head") or ""
            lines.append(f"- PR #{num}: {title}" + (f" ({head})" if head else ""))
        lines.append("")

    progress_line = ""
    if progress_path is not None:
        path = Path(progress_path)
        if path.is_file():
            try:
                progress_line = path.read_text(encoding="utf-8").strip().splitlines()
                progress_line = next(
                    (row for row in progress_line if row.strip() and not row.startswith("#")),
                    progress_line[0] if progress_line else "",
                )
            except OSError as exc:
                log.debug("host controller: handled (%s)", exc)
                progress_line = ""
    if not progress_line:
        progress_fn = getattr(store, "progress", None)
        if callable(progress_fn):
            prog = progress_fn(project) or {}
            progress_line = (
                f"{float(prog.get('percent') or 0):.1f}% "
                f"({prog.get('accepted', 0)}/{prog.get('planned', 0)} accepted)"
            )
    if progress_line:
        lines.append(f"Last progress: {progress_line}")

    ready_fn = getattr(store, "ready_issues", None)
    if callable(ready_fn):
        ready = list(ready_fn(project) or [])
        if ready:
            lines.append(
                "Ready now: "
                + ", ".join(f"#{getattr(i, 'number', i)}" for i in ready)
            )

    return _truncate("\n".join(lines).rstrip() + "\n", max_bytes)


def format_status_reply(*, project: str, store: Any) -> str:
    """Host-authored STATUS answer from GitHub/store truth."""
    prog = store.progress(project) if hasattr(store, "progress") else {}
    issues = store.list_issues(project) if hasattr(store, "list_issues") else []
    lines = [
        f"Status for {project}: "
        f"{float(prog.get('percent') or 0):.1f}% "
        f"({prog.get('accepted', 0)}/{prog.get('planned', 0)} accepted, "
        f"{prog.get('in_progress', 0)} in progress).",
    ]
    ready = []
    ready_fn = getattr(store, "ready_issues", None)
    if callable(ready_fn):
        ready = list(ready_fn(project) or [])
    if ready:
        lines.append(
            "Ready: " + ", ".join(f"#{getattr(i, 'number', i)}" for i in ready) + "."
        )
    for issue in issues:
        lines.append(
            f"- #{getattr(issue, 'number', '?')} "
            f"[{getattr(issue, 'status', '')}] {getattr(issue, 'title', '')}"
        )
    if not issues:
        lines.append("No issues yet.")
    return "\n".join(lines)


def looks_like_status_request(text: str) -> bool:
    raw = (text or "").strip().lower()
    if not raw:
        return False
    if raw in {"status", "stand", "fortschritt", "progress"}:
        return True
    return raw.startswith("status") or raw.startswith("stand ") or "report progress" in raw
