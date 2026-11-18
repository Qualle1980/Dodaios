"""Instant text entry point for the host controller.

ChatGPT Instant is text-only. ``handle_instant_text`` parses the strict
intent JSON (including Markdown ``\\_`` escapes), validates the project id
against the caller's project, runs the host, and refreshes progress.md +
audit.jsonl in the run folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.host_controller.controller import HostController, HostResult
from jarvis.host_controller.intent import IntentParseError, parse_intent
from jarvis.host_controller.progress import (
    format_progress_markdown,
    summarize_issues,
    write_progress_file,
)
from jarvis.host_controller.store import ProjectStore

DEFAULT_RUN_DIR = Path(
    r"D:\Projects\jarvis-agent-outputs\grok-host-controller-full-2026-11-18"
)


def handle_instant_text(
    text: str,
    project: str,
    *,
    store: ProjectStore | None = None,
    workers: Any = None,
    run_dir: Path | None = None,
    host: HostController | None = None,
    max_runs_per_project: int = 50,
) -> HostResult:
    """Parse Instant output, validate project, execute one intent, write progress.

    ``project`` is the host's active project id. An intent that names a
    different project is rejected before any store mutation.
    """
    project = (project or "").strip()
    if not project:
        return HostResult(ok=False, kind="parse_error", error="project is required")

    run_path = Path(run_dir) if run_dir is not None else DEFAULT_RUN_DIR
    run_path.mkdir(parents=True, exist_ok=True)
    audit_path = run_path / "audit.jsonl"

    if host is None:
        if store is None or workers is None:
            return HostResult(
                ok=False,
                kind="config_error",
                error="store and workers are required when host is not provided",
            )
        host = HostController(
            store=store,
            workers=workers,
            audit_path=audit_path,
            max_runs_per_project=max_runs_per_project,
            progress_path=run_path / "progress.md",
            progress_title=f"host controller ({project})",
        )
    else:
        # Keep audit in the run folder when a pre-built host is reused.
        if getattr(host, "progress_path", None) is None:
            host.progress_path = run_path / "progress.md"
        if getattr(host, "progress_title", None) is None:
            host.progress_title = f"host controller ({project})"

    try:
        intent = parse_intent(text)
    except IntentParseError as exc:
        result = HostResult(ok=False, kind="parse_error", error=str(exc))
        _write_progress_safe(host, project, next_step=f"fix Instant JSON: {exc}")
        return result

    intent_project = str(intent.data.get("project") or "").strip()
    if intent_project != project:
        result = HostResult(
            ok=False,
            kind=intent.kind,
            intent_id=intent.intent_id,
            error=f"intent project {intent_project!r} does not match active project {project!r}",
        )
        host.audit.append(
            intent_id=intent.intent_id,
            action="project_mismatch",
            detail={
                "intent_project": intent_project,
                "active_project": project,
            },
        )
        _write_progress_safe(host, project, next_step="send intent for the active project")
        return result

    # Validate issue ids against the store when present.
    if "issue_number" in intent.data:
        number = intent.data["issue_number"]
        if isinstance(number, int):
            issue = host.store.get_issue(project, number)
            if issue is None and intent.kind != "PLAN":
                result = HostResult(
                    ok=False,
                    kind=intent.kind,
                    intent_id=intent.intent_id,
                    error=f"issue #{number} not found in project {project!r}",
                )
                host.audit.append(
                    intent_id=intent.intent_id,
                    action="issue_missing",
                    detail={"project": project, "issue_number": number},
                )
                _write_progress_safe(host, project, next_step="PLAN or fix issue_number")
                return result

    result = host.handle(text)
    _write_progress_safe(host, project)
    return result


def _write_progress_safe(
    host: HostController,
    project: str,
    *,
    next_step: str = "",
) -> None:
    path = getattr(host, "progress_path", None)
    if path is None:
        return
    try:
        progress = host.store.progress(project)
        issues = host.store.list_issues(project)
        active, blockers = summarize_issues(issues)
        title = getattr(host, "progress_title", None) or f"host controller ({project})"
        markdown = format_progress_markdown(
            title=title,
            progress=progress,
            active=active,
            blockers=blockers,
            next_step=next_step,
        )
        write_progress_file(Path(path), markdown)
    except Exception:  # noqa: BLE001 — progress must not break the Instant turn
        return
