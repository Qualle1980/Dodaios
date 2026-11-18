"""Instant text entry point for the host controller.

ChatGPT Instant is text-only. ``handle_instant_text`` parses the strict
intent JSON (including Markdown ``\\_`` escapes), validates the project id
against the caller's project, runs the host, and refreshes progress.md +
audit.jsonl in the run folder.

User-facing completion wording is built here from host status
(:mod:`jarvis.host_controller.end_state`): "done" only when every task
is accepted, otherwise an in-progress line.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jarvis.host_controller.controller import HostController, HostResult
from jarvis.host_controller.end_state import (
    honest_completion_reply,
    user_end_state_phrase,
)
from jarvis.host_controller.final_report import all_accepted
from jarvis.host_controller.intent import IntentParseError, parse_intent
from jarvis.host_controller.progress import (
    format_progress_markdown,
    summarize_issues,
    write_progress_file,
)
from jarvis.host_controller.store import ProjectStore

log = logging.getLogger(__name__)

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
    code_search: Any | None = None,
    pr_search: Any | None = None,
    memory_bridge: Any | None = None,
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
            report_path=run_path / "report.md",
            code_search=code_search,
            pr_search=pr_search,
            memory_bridge=memory_bridge,
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
        log.warning("host Instant rejected intent: %s", exc)
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


def build_user_reply(
    result: HostResult,
    *,
    store: ProjectStore,
    project: str,
    language: str = "de",
) -> str:
    """Plain-text answer for the user from a host result + store truth.

    Completion claims come from :func:`user_end_state_phrase` (host accepted
    every task), never from Instant prose.
    """
    if not result.ok:
        return (
            f"Nicht ausgeführt: {result.error}"  # i18n-allow
        )
    data = result.data or {}
    progress = store.progress(project)
    issues = list(store.list_issues(project))
    end_phrase = user_end_state_phrase(progress, issues, language=language)

    if result.kind == "PLAN":
        issues_created = data.get("issues", [])
        research = data.get("research") or {}
        base = (
            f"Plan angelegt: {len(issues_created)} Aufgabe(n) "  # i18n-allow
            f"im Projekt {data.get('project', '')}."  # i18n-allow
        )
        hits = list(research.get("code") or []) + list(research.get("prs") or [])
        summary = data.get("research_summary") or ""
        # PLAN is setup, not a completion claim — do not append end-state here.
        if hits and summary:
            return f"{base}\n{summary}"
        return base

    if result.kind == "REPORT_PROGRESS":
        ready = data.get("ready") or []
        ready_txt = f" Bereit: {ready}." if ready else ""  # i18n-allow
        return (
            f"{end_phrase}. "
            f"Stand: {data.get('percent', 0)} % abgenommen "  # i18n-allow
            f"({data.get('accepted', 0)} von {data.get('planned', 0)} Aufgaben), "  # i18n-allow
            f"{data.get('in_progress', 0)} in Arbeit.{ready_txt}"  # i18n-allow
        )

    if result.kind == "ACCEPT_TASK" and data.get("final_report"):
        return (
            f"{end_phrase}. Abschlussbericht geschrieben; warte auf dich."  # i18n-allow
        )

    # ACCEPT of one task (or any other kind) while others remain open must not
    # read as project completion.
    if result.kind == "ACCEPT_TASK" and not all_accepted(progress, issues):
        return f"Ausgeführt: {result.kind}. {end_phrase}."  # i18n-allow
    return f"Ausgeführt: {result.kind}."  # i18n-allow


def scrub_finished_claim(
    text: str,
    *,
    store: ProjectStore,
    project: str,
    language: str = "de",
) -> str | None:
    """If Instant claimed finished while the host is open, return the rewrite.

    Returns ``None`` when the text may stay as Instant wrote it (no finished
    claim, or the host has accepted every task).
    """
    progress = store.progress(project)
    issues = list(store.list_issues(project))
    rewritten = honest_completion_reply(
        text,
        progress=progress,
        issues=issues,
        language=language,
    )
    if rewritten == text:
        return None
    return rewritten


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
    except Exception as exc:  # noqa: BLE001 — progress must not break the Instant turn
        log.warning(
            "host progress.md write failed for project %s: %s",
            project,
            exc,
        )
        return
