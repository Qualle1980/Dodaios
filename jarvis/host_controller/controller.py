"""HostController: Instant intent → gated GitHub actions + worker dispatch.

Dispatch goes through injectable worker adapters (tests use fakes; production
wires ``jarvis.agentic_ide.agent_sessions`` / existing coding CLI paths).
Codex review is independent of the worker that authored the commit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.host_controller.audit import AuditLog
from jarvis.host_controller.budget import ProjectBudget
from jarvis.host_controller.deps import (
    overlapping_files,
    ready_issue_numbers,
    unmet_dependencies,
)
from jarvis.host_controller.final_report import (
    all_accepted,
    format_final_report,
    write_final_report,
)
from jarvis.host_controller.gates import (
    GateBook,
    can_accept,
    next_fix_round_or_block,
    require_commit_for_review,
)
from jarvis.host_controller.intent import IntentParseError, ParsedIntent, parse_intent
from jarvis.host_controller.research import collect_research, format_research_block
from jarvis.host_controller.store import LocalCache, ProjectStore
from jarvis.host_controller.types import MAX_FIX_ROUNDS

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HostResult:
    ok: bool
    kind: str
    intent_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class HostController:
    def __init__(
        self,
        *,
        store: ProjectStore,
        workers: Any,
        audit_path: Path,
        max_runs_per_project: int = 50,
        merge_allowed: bool = True,
        progress_path: Path | None = None,
        progress_title: str | None = None,
        report_path: Path | None = None,
        code_search: Any | None = None,
        pr_search: Any | None = None,
        memory_bridge: Any | None = None,
    ) -> None:
        self.store = store
        self.workers = workers
        self.audit = AuditLog(audit_path)
        self.budget = ProjectBudget(max_runs=max_runs_per_project)
        self.gates = GateBook()
        self.cache = LocalCache()
        self.merge_allowed = merge_allowed
        self.progress_path = Path(progress_path) if progress_path is not None else None
        self.progress_title = progress_title
        self.report_path = Path(report_path) if report_path is not None else None
        self.code_search = code_search
        self.pr_search = pr_search
        self.memory_bridge = memory_bridge
        self.cache.rebuild(store)

    def recover(self) -> list[dict[str, Any]]:
        """Rebuild local cache from GitHub; in-progress without a live process → needs-check."""
        self.cache.rebuild(self.store)
        changed: list[dict[str, Any]] = []
        milestones = self.cache.data.get("milestones") or {}
        for title, raw in milestones.items():
            for issue_raw in (raw.get("issues") or {}).values():
                number = int(issue_raw["number"])
                status = str(issue_raw.get("status") or "")
                if status != "in-progress":
                    continue
                live = False
                probe = getattr(self.workers, "has_live_process", None)
                if callable(probe):
                    live = bool(probe(number))
                if live:
                    continue
                self.store.set_status(title, number, "needs-check")
                row = {"project": title, "issue_number": number, "status": "needs-check"}
                changed.append(row)
                self.audit.append(
                    intent_id=None,
                    action="recover_needs_check",
                    detail=row,
                )
        self.cache.rebuild(self.store)
        return changed

    def handle(self, text: str) -> HostResult:
        try:
            intent = parse_intent(text)
        except IntentParseError as exc:
            log.warning("host rejected intent: %s", exc)
            self.audit.append(
                intent_id=None,
                action="intent_rejected",
                detail={"error": str(exc), "text_preview": (text or "")[:200]},
            )
            return HostResult(ok=False, kind="parse_error", error=str(exc))

        self.audit.append(
            intent_id=intent.intent_id,
            action="intent_received",
            detail={
                "kind": intent.kind,
                "normalised_underscore": intent.normalised_underscore,
            },
        )
        try:
            result = self._dispatch_intent(intent)
        except Exception as exc:  # noqa: BLE001 — host must never crash the chat turn
            log.exception("host intent %s failed", intent.intent_id)
            self.audit.append(
                intent_id=intent.intent_id,
                action="intent_failed",
                detail={"error": str(exc)},
            )
            return HostResult(
                ok=False,
                kind=intent.kind,
                intent_id=intent.intent_id,
                error=str(exc),
            )
        self.audit.append(
            intent_id=intent.intent_id,
            action="intent_handled",
            detail={"ok": result.ok, "kind": result.kind, "error": result.error},
        )
        self.cache.rebuild(self.store)
        return result

    def _dispatch_intent(self, intent: ParsedIntent) -> HostResult:
        handlers = {
            "PLAN": self._plan,
            "ASSIGN_TASK": self._assign,
            "REQUEST_REVIEW": self._review,
            "REQUEST_FIX": self._fix,
            "ACCEPT_TASK": self._accept,
            "BLOCKED": self._blocked,
            "REPORT_PROGRESS": self._progress,
        }
        return handlers[intent.kind](intent)

    def _plan(self, intent: ParsedIntent) -> HostResult:
        data = intent.data
        project = str(data["project"]).strip()
        title = str(data.get("title") or project).strip()
        research = collect_research(
            title=title,
            tasks=list(data.get("tasks") or []),
            code_search=self.code_search,
            pr_search=self.pr_search,
        )
        research_block = format_research_block(research)
        existing = self.store.get_milestone(project)
        if existing is None:
            self.store.create_milestone(project, description=research_block)
        else:
            updater = getattr(self.store, "update_milestone_description", None)
            if callable(updater):
                prior = str(getattr(existing, "description", "") or "").strip()
                merged = f"{prior.rstrip()}\n\n{research_block}" if prior else research_block
                updater(project, merged)
            else:
                # Best-effort: recreate description only when empty.
                if not str(getattr(existing, "description", "") or "").strip():
                    try:
                        existing.description = research_block  # type: ignore[attr-defined]
                    except (AttributeError, TypeError) as exc:
                        # A read-only record cannot carry the research; say so.
                        self.audit.append(
                            intent_id=intent.intent_id,
                            action="research_not_stored",
                            detail={"project": project, "reason": str(exc)},
                        )
        created: list[int] = []
        for task in data["tasks"]:
            body = str(task.get("acceptance") or "").strip()
            issue = self.store.create_issue(
                project,
                str(task["title"]),
                body or "acceptance criteria TBD",
                status="planned",
            )
            deps = task.get("depends_on") or []
            if deps:
                setter = getattr(self.store, "set_depends_on", None)
                if callable(setter):
                    setter(project, int(issue.number), [int(n) for n in deps])
                else:
                    # Only a store without set_depends_on lands here (the fakes).
                    issue.depends_on = [int(n) for n in deps]  # type: ignore[attr-defined]
            created.append(int(issue.number))
        self.audit.append(
            intent_id=intent.intent_id,
            action="plan_created",
            detail={
                "project": project,
                "issues": created,
                "research": research,
            },
        )
        return HostResult(
            ok=True,
            kind="PLAN",
            intent_id=intent.intent_id,
            data={
                "project": project,
                "issues": created,
                "research": research,
                "research_summary": research_block,
            },
        )

    def _assign(self, intent: ParsedIntent) -> HostResult:
        data = intent.data
        project = str(data["project"]).strip()
        number = int(data["issue_number"])
        worker = str(data["worker"])
        allow_subagents = bool(data.get("allow_subagents", False))
        subagent_kind = data.get("subagent_kind")
        files = [str(p) for p in (data.get("files") or []) if str(p).strip()]
        issue = self.store.get_issue(project, number)
        if issue is None:
            return HostResult(
                ok=False,
                kind="ASSIGN_TASK",
                intent_id=intent.intent_id,
                error=f"issue #{number} not found",
            )
        issues = list(self.store.list_issues(project))
        by_number = {int(getattr(item, "number", 0)): item for item in issues}
        unmet = unmet_dependencies(issue, issues_by_number=by_number)
        if unmet:
            return HostResult(
                ok=False,
                kind="ASSIGN_TASK",
                intent_id=intent.intent_id,
                error=f"dependencies not accepted: {unmet}",
                data={"unmet_dependencies": unmet},
            )
        overlap = overlapping_files(
            candidate_files=files or list(getattr(issue, "files", None) or []),
            issues=issues,
            skip_number=number,
        )
        if overlap:
            detail = ", ".join(f"#{num}:{path}" for num, path in overlap)
            return HostResult(
                ok=False,
                kind="ASSIGN_TASK",
                intent_id=intent.intent_id,
                error=f"files overlap in-progress work: {detail}",
                data={"overlap": [{"issue": n, "path": p} for n, p in overlap]},
            )
        if not self.budget.consume(project):
            return HostResult(
                ok=False,
                kind="ASSIGN_TASK",
                intent_id=intent.intent_id,
                error="per-project run budget exhausted",
            )
        if files:
            setter = getattr(self.store, "set_files", None)
            if callable(setter):
                setter(project, number, files)
        self.store.set_worker(
            project,
            number,
            worker,
            subagent_kind=subagent_kind if allow_subagents else None,
        )
        self.store.set_status(project, number, "in-progress")
        prompt = str(data.get("task") or getattr(issue, "title", "") or "")
        try:
            result = self.workers.assign(
                worker=worker,
                issue_number=number,
                prompt=prompt,
                allow_subagents=allow_subagents,
                subagent_kind=subagent_kind if allow_subagents else None,
            )
        except Exception as exc:  # quota or dispatch failure
            if _is_quota(exc):
                self.store.set_status(project, number, "waiting-quota")
                self.audit.append(
                    intent_id=intent.intent_id,
                    action="waiting_quota",
                    detail={"project": project, "issue_number": number},
                )
                return HostResult(
                    ok=False,
                    kind="ASSIGN_TASK",
                    intent_id=intent.intent_id,
                    error="quota exhausted; task waiting-quota (no paid fallback)",
                    data={"status": "waiting-quota"},
                )
            raise
        used_kind = getattr(result, "subagent_kind_used", None) or (
            subagent_kind if allow_subagents else None
        )
        if used_kind:
            self.store.set_worker(project, number, worker, subagent_kind=used_kind)
        sha = getattr(result, "commit_sha", None)
        if sha:
            self.store.set_commit_sha(project, number, str(sha))
            clear_pass = getattr(self.store, "set_review_pass_sha", None)
            if callable(clear_pass):
                clear_pass(project, number, None)
        self.audit.append(
            intent_id=intent.intent_id,
            action="assigned",
            detail={
                "project": project,
                "issue_number": number,
                "worker": worker,
                "subagent_kind": used_kind,
                "commit_sha": sha,
            },
        )
        return HostResult(
            ok=True,
            kind="ASSIGN_TASK",
            intent_id=intent.intent_id,
            data={
                "issue_number": number,
                "worker": worker,
                "subagent_kind": used_kind,
                "commit_sha": sha,
                "status": "in-progress",
            },
        )

    def _review(self, intent: ParsedIntent) -> HostResult:
        data = intent.data
        project = str(data["project"]).strip()
        number = int(data["issue_number"])
        try:
            sha = require_commit_for_review(str(data.get("commit_sha") or ""))
        except ValueError as exc:
            log.warning("host review refused missing commit: %s", exc)
            return HostResult(
                ok=False,
                kind="REQUEST_REVIEW",
                intent_id=intent.intent_id,
                error=str(exc),
            )
        issue = self.store.get_issue(project, number)
        if issue is None:
            return HostResult(
                ok=False,
                kind="REQUEST_REVIEW",
                intent_id=intent.intent_id,
                error=f"issue #{number} not found",
            )
        current = getattr(issue, "commit_sha", None)
        if current and str(current) != sha:
            # Still allow reviewing the claimed SHA, but surface the mismatch.
            log.warning(
                "review SHA %s differs from stored task SHA %s for #%s",
                sha,
                current,
                number,
            )
        if not self.budget.consume(project):
            return HostResult(
                ok=False,
                kind="REQUEST_REVIEW",
                intent_id=intent.intent_id,
                error="per-project run budget exhausted",
            )
        review = self.workers.review(commit_sha=sha, issue_number=number)
        pass_verdict = bool(getattr(review, "pass_verdict", False))
        findings = list(getattr(review, "findings", []) or [])
        reviewed_sha = str(getattr(review, "commit_sha", None) or sha).strip()
        recorded = str(current).strip() if current else ""
        # A PASS counts only when the SHA reviewed equals the SHA the host
        # has recorded for the task (and the SHA Instant asked to review).
        if pass_verdict and reviewed_sha and reviewed_sha != sha:
            pass_verdict = False
            findings = list(findings) + [
                f"review SHA {reviewed_sha} differs from requested SHA {sha}; treated as no review"
            ]
        if pass_verdict and recorded and reviewed_sha and reviewed_sha != recorded:
            pass_verdict = False
            findings = list(findings) + [
                f"review SHA {reviewed_sha} differs from recorded SHA {recorded}; "
                "treated as no review"
            ]
        gate_sha = sha
        if pass_verdict and reviewed_sha:
            gate_sha = reviewed_sha
        self.gates.record_review(
            project,
            number,
            commit_sha=gate_sha,
            pass_verdict=pass_verdict,
            findings=findings,
        )
        # Persist PASS evidence on GitHub so a new Instant turn can ACCEPT.
        set_pass = getattr(self.store, "set_review_pass_sha", None)
        if callable(set_pass):
            set_pass(project, number, gate_sha if pass_verdict else None)
        # Keep the task's commit pointer on the SHA Instant requested for review
        # when the review is bound to it; otherwise leave recorded SHA intact.
        if reviewed_sha == sha or not recorded:
            self.store.set_commit_sha(project, number, sha)
        self.store.set_status(project, number, "review")
        verdict = "PASS" if pass_verdict else "REJECT"
        self.audit.append(
            intent_id=intent.intent_id,
            action="reviewed",
            detail={
                "project": project,
                "issue_number": number,
                "commit_sha": sha,
                "verdict": verdict,
                "findings": findings,
            },
        )
        return HostResult(
            ok=True,
            kind="REQUEST_REVIEW",
            intent_id=intent.intent_id,
            data={
                "verdict": verdict,
                "commit_sha": sha,
                "findings": findings,
            },
        )

    def _fix(self, intent: ParsedIntent) -> HostResult:
        data = intent.data
        project = str(data["project"]).strip()
        number = int(data["issue_number"])
        findings = list(data.get("findings") or [])
        issue = self.store.get_issue(project, number)
        if issue is None:
            return HostResult(
                ok=False,
                kind="REQUEST_FIX",
                intent_id=intent.intent_id,
                error=f"issue #{number} not found",
            )
        if getattr(issue, "status", None) == "blocked":
            return HostResult(
                ok=False,
                kind="REQUEST_FIX",
                intent_id=intent.intent_id,
                error="task is BLOCKED; no further fix rounds",
            )
        current_rounds = int(getattr(issue, "fix_rounds", 0) or 0)
        allowed, _new_rounds, reason = next_fix_round_or_block(current_rounds)
        if not allowed:
            self.store.set_status(project, number, "blocked")
            self.audit.append(
                intent_id=intent.intent_id,
                action="blocked_retry_cap",
                detail={
                    "project": project,
                    "issue_number": number,
                    "fix_rounds": current_rounds,
                    "cap": MAX_FIX_ROUNDS,
                },
            )
            return HostResult(
                ok=False,
                kind="REQUEST_FIX",
                intent_id=intent.intent_id,
                error=reason,
                data={"status": "blocked", "fix_rounds": current_rounds},
            )
        if not self.budget.consume(project):
            return HostResult(
                ok=False,
                kind="REQUEST_FIX",
                intent_id=intent.intent_id,
                error="per-project run budget exhausted",
            )
        worker = str(getattr(issue, "worker", None) or "claude")
        allow_subagents = bool(data.get("allow_subagents", False))
        subagent_kind = data.get("subagent_kind") or getattr(issue, "subagent_kind", None)
        try:
            result = self.workers.fix(
                worker=worker,
                issue_number=number,
                prompt=str(getattr(issue, "title", "") or ""),
                findings=findings,
                allow_subagents=allow_subagents,
                subagent_kind=subagent_kind if allow_subagents else None,
            )
        except Exception as exc:
            if _is_quota(exc):
                self.store.set_status(project, number, "waiting-quota")
                return HostResult(
                    ok=False,
                    kind="REQUEST_FIX",
                    intent_id=intent.intent_id,
                    error="quota exhausted; task waiting-quota (no paid fallback)",
                    data={"status": "waiting-quota"},
                )
            raise
        # Count the round after a real dispatch attempt.
        bumped = self.store.bump_fix_round(project, number)
        sha = getattr(result, "commit_sha", None)
        if sha:
            self.store.set_commit_sha(project, number, str(sha))
            clear_pass = getattr(self.store, "set_review_pass_sha", None)
            if callable(clear_pass):
                clear_pass(project, number, None)
        self.audit.append(
            intent_id=intent.intent_id,
            action="fix_dispatched",
            detail={
                "project": project,
                "issue_number": number,
                "fix_rounds": bumped,
                "commit_sha": sha,
            },
        )
        # Third dispatched fix round exhausts the cap → BLOCKED (no fourth).
        if bumped >= MAX_FIX_ROUNDS:
            self.store.set_status(project, number, "blocked")
            return HostResult(
                ok=False,
                kind="REQUEST_FIX",
                intent_id=intent.intent_id,
                error=f"fix round cap {MAX_FIX_ROUNDS} reached; BLOCKED",
                data={"status": "blocked", "fix_rounds": bumped, "commit_sha": sha},
            )
        self.store.set_status(project, number, "in-progress")
        return HostResult(
            ok=True,
            kind="REQUEST_FIX",
            intent_id=intent.intent_id,
            data={
                "issue_number": number,
                "fix_rounds": bumped,
                "commit_sha": sha,
                "status": "in-progress",
            },
        )

    def _accept(self, intent: ParsedIntent) -> HostResult:
        data = intent.data
        project = str(data["project"]).strip()
        number = int(data["issue_number"])
        sha = str(data["commit_sha"]).strip()
        tests_green = bool(data["tests_green"])
        issue = self.store.get_issue(project, number)
        if issue is None:
            return HostResult(
                ok=False,
                kind="ACCEPT_TASK",
                intent_id=intent.intent_id,
                error=f"issue #{number} not found",
            )
        current_sha = getattr(issue, "commit_sha", None)
        gate_pass = self.gates.pass_for_sha(project, number, sha)
        stored_pass = getattr(issue, "review_pass_sha", None)
        if not gate_pass and stored_pass and str(stored_pass).strip() == sha:
            gate_pass = True
        ok, reason = can_accept(
            gate_pass=gate_pass,
            current_sha=str(current_sha) if current_sha else None,
            claimed_sha=sha,
            tests_green=tests_green,
            merge_allowed=self.merge_allowed,
        )
        if not ok:
            self.audit.append(
                intent_id=intent.intent_id,
                action="accept_refused",
                detail={
                    "project": project,
                    "issue_number": number,
                    "commit_sha": sha,
                    "reason": reason,
                },
            )
            return HostResult(
                ok=False,
                kind="ACCEPT_TASK",
                intent_id=intent.intent_id,
                error=reason,
            )
        self.store.set_status(project, number, "accepted")
        self.audit.append(
            intent_id=intent.intent_id,
            action="accepted",
            detail={"project": project, "issue_number": number, "commit_sha": sha},
        )
        if self.memory_bridge is not None:
            remember = getattr(self.memory_bridge, "remember_decision", None)
            if callable(remember):
                remember(
                    f"Accepted {project} issue #{number} commit {sha} tests_green={tests_green}"
                )
        report_written = False
        progress = self.store.progress(project)
        issues = list(self.store.list_issues(project))
        if self.report_path is not None and all_accepted(progress, issues):
            markdown = format_final_report(project=project, issues=issues, progress=progress)
            write_final_report(self.report_path, markdown)
            report_written = True
            self.audit.append(
                intent_id=intent.intent_id,
                action="final_report",
                detail={"path": str(self.report_path), "percent": progress.get("percent")},
            )
        return HostResult(
            ok=True,
            kind="ACCEPT_TASK",
            intent_id=intent.intent_id,
            data={
                "issue_number": number,
                "status": "accepted",
                "commit_sha": sha,
                "final_report": report_written,
            },
        )

    def _blocked(self, intent: ParsedIntent) -> HostResult:
        data = intent.data
        project = str(data["project"]).strip()
        number = int(data["issue_number"])
        reason = str(data.get("reason") or "blocked")
        issue = self.store.get_issue(project, number)
        if issue is None:
            return HostResult(
                ok=False,
                kind="BLOCKED",
                intent_id=intent.intent_id,
                error=f"issue #{number} not found",
            )
        self.store.set_status(project, number, "blocked")
        self.audit.append(
            intent_id=intent.intent_id,
            action="blocked",
            detail={"project": project, "issue_number": number, "reason": reason},
        )
        return HostResult(
            ok=True,
            kind="BLOCKED",
            intent_id=intent.intent_id,
            data={"issue_number": number, "status": "blocked", "reason": reason},
        )

    def _progress(self, intent: ParsedIntent) -> HostResult:
        project = str(intent.data["project"]).strip()
        progress = self.store.progress(project)
        issues = list(self.store.list_issues(project))
        ready = ready_issue_numbers(issues)
        payload = dict(progress)
        payload["ready"] = ready
        self.audit.append(
            intent_id=intent.intent_id,
            action="progress",
            detail={"project": project, **payload},
        )
        return HostResult(
            ok=True,
            kind="REPORT_PROGRESS",
            intent_id=intent.intent_id,
            data=payload,
        )


def _is_quota(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "quota" in name or "quota" in msg
