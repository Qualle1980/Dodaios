"""Real worker adapters for the Instant host controller.

Wraps existing Claude / Grok / Codex direct workers behind the sync
``assign`` / ``fix`` / ``review`` interface the host already uses.
Unit tests inject scripted drivers; production uses
:func:`build_live_host_workers`. Codex subagents stay OFF (not verified).
Codex review runs read-only against an exact SHA; a PASS whose reviewed
SHA differs from the recorded task SHA is treated as no review.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.core.process_utils import NO_WINDOW_CREATIONFLAGS

log = logging.getLogger(__name__)

Driver = Callable[..., Any]
LiveProbe = Callable[[int], bool]
ShaLookup = Callable[[int], str | None]


@dataclass
class WorkerResult:
    ok: bool
    commit_sha: str | None = None
    message: str = ""
    quota_exhausted: bool = False
    subagent_kind_used: str | None = None
    worktree: str | None = None


@dataclass
class ReviewResult:
    pass_verdict: bool
    commit_sha: str
    findings: list[str] = field(default_factory=list)
    message: str = ""


class QuotaExhausted(Exception):
    """Provider reported its subscription/window is spent."""


class HostWorkerBundle:
    """Dispatch surface matching ``FakeHostWorkers`` for the host controller."""

    def __init__(
        self,
        *,
        work_root: Path,
        repo_root: Path,
        drivers: dict[str, Driver] | None = None,
        live_probe: LiveProbe | None = None,
        recorded_sha_lookup: ShaLookup | None = None,
    ) -> None:
        self.work_root = Path(work_root)
        self.repo_root = Path(repo_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self._drivers = dict(drivers or {})
        self._live_probe = live_probe
        self._recorded_sha_lookup = recorded_sha_lookup
        self._live_issues: set[int] = set()

    def has_live_process(self, issue_number: int) -> bool:
        if self._live_probe is not None:
            return bool(self._live_probe(issue_number))
        return issue_number in self._live_issues

    def mark_live(self, issue_number: int) -> None:
        self._live_issues.add(issue_number)

    def clear_live(self, issue_number: int) -> None:
        self._live_issues.discard(issue_number)

    def assign(
        self,
        *,
        worker: str,
        issue_number: int,
        prompt: str,
        allow_subagents: bool = False,
        subagent_kind: str | None = None,
    ) -> WorkerResult:
        if worker not in {"claude", "grok"}:
            raise ValueError(f"ASSIGN worker must be claude or grok, got {worker!r}")
        # Codex subagents stay OFF; Claude/Grok only when the host allows.
        effective_allow = bool(allow_subagents)
        effective_kind = subagent_kind if effective_allow else None
        driver = self._drivers.get(worker)
        if driver is None:
            raise RuntimeError(f"no driver registered for worker {worker!r}")
        self.mark_live(issue_number)
        try:
            result = driver(
                worker=worker,
                issue_number=issue_number,
                prompt=prompt,
                allow_subagents=effective_allow,
                subagent_kind=effective_kind,
                work_root=self.work_root,
                repo_root=self.repo_root,
                mode="assign",
            )
        except Exception:
            self.clear_live(issue_number)
            raise
        self.clear_live(issue_number)
        return _as_worker_result(result, fallback_kind=effective_kind)

    def fix(
        self,
        *,
        worker: str,
        issue_number: int,
        prompt: str,
        findings: list[str],
        allow_subagents: bool = False,
        subagent_kind: str | None = None,
    ) -> WorkerResult:
        # Codex may author a fix, but its native subagents stay OFF.
        if worker == "codex":
            effective_allow = False
            effective_kind = None
            driver_key = "codex_fix"
        else:
            effective_allow = bool(allow_subagents)
            effective_kind = subagent_kind if effective_allow else None
            driver_key = worker
        driver = self._drivers.get(driver_key) or self._drivers.get(worker)
        if driver is None:
            raise RuntimeError(f"no driver registered for fix worker {worker!r}")
        self.mark_live(issue_number)
        try:
            result = driver(
                worker=worker,
                issue_number=issue_number,
                prompt=prompt,
                findings=list(findings),
                allow_subagents=effective_allow,
                subagent_kind=effective_kind,
                work_root=self.work_root,
                repo_root=self.repo_root,
                mode="fix",
            )
        except Exception:
            self.clear_live(issue_number)
            raise
        self.clear_live(issue_number)
        return _as_worker_result(result, fallback_kind=effective_kind)

    def review(
        self, *, commit_sha: str, issue_number: int, acceptance: str = ""
    ) -> ReviewResult:
        driver = self._drivers.get("codex_review")
        if driver is None:
            raise RuntimeError("no codex_review driver registered")
        raw = driver(
            worker="codex",
            issue_number=issue_number,
            prompt=f"Review commit {commit_sha} read-only. Report PASS or FAIL with findings.",
            commit_sha=commit_sha,
            acceptance=acceptance,
            allow_subagents=False,
            subagent_kind=None,
            work_root=self.work_root,
            repo_root=self.repo_root,
            mode="review",
            read_only=True,
        )
        result = _as_review_result(raw, fallback_sha=commit_sha)
        recorded = None
        if self._recorded_sha_lookup is not None:
            recorded = self._recorded_sha_lookup(issue_number)
        reviewed = (result.commit_sha or "").strip()
        expected = (recorded or commit_sha or "").strip()
        if result.pass_verdict and reviewed and expected and reviewed != expected:
            return ReviewResult(
                pass_verdict=False,
                commit_sha=reviewed,
                findings=list(result.findings)
                + [
                    f"review SHA {reviewed} differs from recorded SHA {expected}; "
                    "treated as no review"
                ],
                message="sha mismatch",
            )
        if result.pass_verdict and reviewed and reviewed != commit_sha.strip():
            return ReviewResult(
                pass_verdict=False,
                commit_sha=reviewed,
                findings=list(result.findings)
                + [
                    f"review SHA {reviewed} differs from requested SHA {commit_sha}; "
                    "treated as no review"
                ],
                message="sha mismatch",
            )
        return result


def _as_worker_result(raw: Any, *, fallback_kind: str | None) -> WorkerResult:
    if isinstance(raw, WorkerResult):
        if raw.quota_exhausted:
            raise QuotaExhausted(raw.message or "quota exhausted")
        if raw.subagent_kind_used is None and fallback_kind:
            return WorkerResult(
                ok=raw.ok,
                commit_sha=raw.commit_sha,
                message=raw.message,
                quota_exhausted=raw.quota_exhausted,
                subagent_kind_used=fallback_kind,
                worktree=raw.worktree,
            )
        return raw
    if getattr(raw, "quota_exhausted", False):
        raise QuotaExhausted(str(getattr(raw, "message", "") or "quota exhausted"))
    return WorkerResult(
        ok=bool(getattr(raw, "ok", True)),
        commit_sha=getattr(raw, "commit_sha", None),
        message=str(getattr(raw, "message", "") or ""),
        subagent_kind_used=getattr(raw, "subagent_kind_used", None) or fallback_kind,
        worktree=getattr(raw, "worktree", None),
    )


def _as_review_result(raw: Any, *, fallback_sha: str) -> ReviewResult:
    if isinstance(raw, ReviewResult):
        return raw
    return ReviewResult(
        pass_verdict=bool(getattr(raw, "pass_verdict", False)),
        commit_sha=str(getattr(raw, "commit_sha", None) or fallback_sha),
        findings=list(getattr(raw, "findings", []) or []),
        message=str(getattr(raw, "message", "") or ""),
    )


def _git(args: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
        creationflags=NO_WINDOW_CREATIONFLAGS,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {(result.stderr or result.stdout)[:300]}"
        )
    return (result.stdout or "").strip()


def _ensure_worktree(repo_root: Path, work_root: Path, issue_number: int) -> Path:
    """Create (or reuse) a dedicated worktree/directory for one issue."""
    dest = work_root / f"issue-{issue_number}-{uuid.uuid4().hex[:8]}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (repo_root / ".git").exists() or (repo_root / ".git").is_file():
        branch = f"host/issue-{issue_number}-{uuid.uuid4().hex[:6]}"
        result = subprocess.run(
            ["git", "worktree", "add", "-b", branch, str(dest), "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
            creationflags=NO_WINDOW_CREATIONFLAGS,
        )
        if result.returncode == 0:
            return dest
        log.warning(
            "git worktree add failed (%s); using plain directory: %s",
            result.returncode,
            (result.stderr or "")[:200],
        )
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _read_head_sha(worktree: Path) -> str | None:
    try:
        sha = _git(["rev-parse", "HEAD"], cwd=worktree)
    except RuntimeError:
        return None
    return sha or None


def _commit_if_dirty(worktree: Path, message: str) -> str | None:
    try:
        status = _git(["status", "--porcelain"], cwd=worktree)
    except RuntimeError:
        return _read_head_sha(worktree)
    if status.strip():
        try:
            _git(["add", "-A"], cwd=worktree)
            env = dict(os.environ)
            env.setdefault("GIT_AUTHOR_NAME", "host-controller")
            env.setdefault("GIT_AUTHOR_EMAIL", "host-controller@local")
            env.setdefault("GIT_COMMITTER_NAME", env["GIT_AUTHOR_NAME"])
            env.setdefault("GIT_COMMITTER_EMAIL", env["GIT_AUTHOR_EMAIL"])
            subprocess.run(
                ["git", "commit", "-m", message, "--allow-empty"],
                cwd=str(worktree),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
                env=env,
                creationflags=NO_WINDOW_CREATIONFLAGS,
            )
        except Exception as exc:  # noqa: BLE001 — still try to report HEAD
            log.warning("auto-commit after worker failed: %s", exc)
    return _read_head_sha(worktree)


#: Tools the host grants a Claude worker, all scoped to its own worktree.
CLAUDE_WORKER_TOOLS = "Read,Edit,Write,Bash,Grep,Glob"


def _claude_extra_args(*, allow_subagents: bool, subagent_kind: str | None) -> tuple[str, ...]:
    if not allow_subagents:
        return ()
    # Claude Code: ``--agents`` selects named agents when the CLI supports it.
    kind = (subagent_kind or "").strip()
    if not kind:
        return ("--agents",)
    return ("--agents", kind)


def _prompt_with_subagent_hint(
    prompt: str,
    *,
    worker: str,
    allow_subagents: bool,
    subagent_kind: str | None,
) -> str:
    if not allow_subagents:
        return prompt
    kind = subagent_kind or "general-purpose"
    if worker == "grok":
        return (
            f"{prompt}\n\n"
            f"Host allows subagents for this task. Prefer spawn_subagent kind={kind!r} "
            "when a bounded specialist helps; record what you used."
        )
    return (
        f"{prompt}\n\n"
        f"Host allows subagents for this task (kind={kind!r}). "
        "Use them only when they help; stay inside this worktree."
    )


async def _drain_worker_events(agen: Any) -> list[Any]:
    events: list[Any] = []
    async for event in agen:
        events.append(event)
    return events


def _run_async(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        # Host is sync; nest when already inside an event loop (rare in tests).
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _events_look_like_quota(events: list[Any], stderr_text: str = "") -> bool:
    blob = " ".join(str(event) for event in events) + " " + (stderr_text or "")
    low = blob.lower()
    return any(
        marker in low
        for marker in ("quota", "usage limit", "rate limit", "too many requests")
    )


def _live_claude_driver(**kwargs: Any) -> WorkerResult:
    from jarvis.missions.workers.claude_direct_worker import ClaudeDirectWorker

    issue_number = int(kwargs["issue_number"])
    repo_root = Path(kwargs["repo_root"])
    work_root = Path(kwargs["work_root"])
    allow_subagents = bool(kwargs.get("allow_subagents"))
    subagent_kind = kwargs.get("subagent_kind")
    prompt = _prompt_with_subagent_hint(
        str(kwargs.get("prompt") or ""),
        worker="claude",
        allow_subagents=allow_subagents,
        subagent_kind=subagent_kind,
    )
    worktree = _ensure_worktree(repo_root, work_root, issue_number)
    log_dir = work_root / f"logs-claude-{issue_number}"
    log_dir.mkdir(parents=True, exist_ok=True)
    worker = ClaudeDirectWorker(backend_fallback=False)
    extra = _claude_extra_args(
        allow_subagents=allow_subagents, subagent_kind=subagent_kind
    )
    env = dict(os.environ)

    async def _run() -> list[Any]:
        agen = worker.spawn(
            prompt,
            worktree=worktree,
            env=env,
            job=_NullJob(),
            worker_id=f"host-claude-{issue_number}",
            log_dir=log_dir,
            extra_args=extra,
            # The CLI is hardened (CLAUDE_CODE_SUBPROCESS_ENV_SCRUB): without an
            # explicit tool list it forces the default mode and denies writes.
            allowed_tools=CLAUDE_WORKER_TOOLS,
            permission_mode="bypassPermissions",
        )
        return await _drain_worker_events(agen)

    events = _run_async(_run())
    if _events_look_like_quota(events):
        return WorkerResult(ok=False, quota_exhausted=True, message="quota exhausted")
    sha = _commit_if_dirty(worktree, f"host: issue #{issue_number}")
    return WorkerResult(
        ok=True,
        commit_sha=sha,
        message="claude done",
        subagent_kind_used=subagent_kind if allow_subagents else None,
        worktree=str(worktree),
    )


def _live_grok_driver(**kwargs: Any) -> WorkerResult:
    from jarvis.missions.workers.grok_build_direct_worker import GrokBuildDirectWorker

    issue_number = int(kwargs["issue_number"])
    repo_root = Path(kwargs["repo_root"])
    work_root = Path(kwargs["work_root"])
    allow_subagents = bool(kwargs.get("allow_subagents"))
    subagent_kind = kwargs.get("subagent_kind")
    prompt = _prompt_with_subagent_hint(
        str(kwargs.get("prompt") or ""),
        worker="grok",
        allow_subagents=allow_subagents,
        subagent_kind=subagent_kind,
    )
    worktree = _ensure_worktree(repo_root, work_root, issue_number)
    log_dir = work_root / f"logs-grok-{issue_number}"
    log_dir.mkdir(parents=True, exist_ok=True)
    worker = GrokBuildDirectWorker()
    env = dict(os.environ)

    async def _run() -> list[Any]:
        agen = worker.spawn(
            prompt,
            worktree=worktree,
            env=env,
            job=_NullJob(),
            worker_id=f"host-grok-{issue_number}",
            log_dir=log_dir,
        )
        return await _drain_worker_events(agen)

    events = _run_async(_run())
    if _events_look_like_quota(events):
        return WorkerResult(ok=False, quota_exhausted=True, message="quota exhausted")
    sha = _commit_if_dirty(worktree, f"host: issue #{issue_number}")
    return WorkerResult(
        ok=True,
        commit_sha=sha,
        message="grok done",
        subagent_kind_used=subagent_kind if allow_subagents else None,
        worktree=str(worktree),
    )


def _live_codex_fix_driver(**kwargs: Any) -> WorkerResult:
    from jarvis.missions.workers.codex_direct_worker import CodexDirectWorker

    issue_number = int(kwargs["issue_number"])
    repo_root = Path(kwargs["repo_root"])
    work_root = Path(kwargs["work_root"])
    findings = list(kwargs.get("findings") or [])
    base_prompt = str(kwargs.get("prompt") or "")
    prompt = (
        f"{base_prompt}\n\nFix the following review findings:\n"
        + "\n".join(f"- {item}" for item in findings)
        + "\nCodex native subagents stay OFF for host-controller fixes."
    )
    worktree = _ensure_worktree(repo_root, work_root, issue_number)
    log_dir = work_root / f"logs-codex-fix-{issue_number}"
    log_dir.mkdir(parents=True, exist_ok=True)
    worker = CodexDirectWorker()
    env = dict(os.environ)

    async def _run() -> list[Any]:
        agen = worker.spawn(
            prompt,
            worktree=worktree,
            env=env,
            job=_NullJob(),
            worker_id=f"host-codex-fix-{issue_number}",
            log_dir=log_dir,
        )
        return await _drain_worker_events(agen)

    events = _run_async(_run())
    if _events_look_like_quota(events):
        return WorkerResult(ok=False, quota_exhausted=True, message="quota exhausted")
    sha = _commit_if_dirty(worktree, f"host: fix issue #{issue_number}")
    return WorkerResult(
        ok=True,
        commit_sha=sha,
        message="codex fix done",
        subagent_kind_used=None,
        worktree=str(worktree),
    )


def _live_codex_review_driver(**kwargs: Any) -> ReviewResult:
    from jarvis.missions.workers.codex_direct_worker import CodexDirectWorker

    try:
        from jarvis.missions.kontrollierer.worker_prompt import READ_ONLY_MARKER as _ROM
    except Exception:  # noqa: BLE001 — marker is optional; sandbox still read-classified
        _ROM = "READ-ONLY TASK:"

    issue_number = int(kwargs["issue_number"])
    commit_sha = str(kwargs.get("commit_sha") or "").strip()
    acceptance = str(kwargs.get("acceptance") or "")
    repo_root = Path(kwargs["repo_root"])
    work_root = Path(kwargs["work_root"])
    worktree = _ensure_worktree(repo_root, work_root, issue_number)
    # Check out the exact SHA when possible so the review is bound to it.
    if commit_sha:
        try:
            _git(["checkout", "--detach", commit_sha], cwd=worktree)
        except RuntimeError as exc:
            log.warning("review checkout %s failed: %s", commit_sha, exc)
    log_dir = work_root / f"logs-codex-review-{issue_number}"
    log_dir.mkdir(parents=True, exist_ok=True)
    prompt = (
        f"{_ROM}\n"
        f"Review git commit {commit_sha} exactly. Do not modify any files.\n"
        "Acceptance criteria of the issue:\n"
        f"{acceptance.strip() or '(none recorded)'}\n"
        "Reply with a single JSON object: "
        '{"verdict":"PASS"|"FAIL","findings":["..."],"commit_sha":"'
        f'{commit_sha}"' + "}. "
        "PASS only if the code at that SHA meets the acceptance criteria above."
    )
    worker = CodexDirectWorker()
    env = dict(os.environ)

    async def _run() -> list[Any]:
        agen = worker.spawn(
            prompt,
            worktree=worktree,
            env=env,
            job=_NullJob(),
            worker_id=f"host-codex-review-{issue_number}",
            log_dir=log_dir,
        )
        return await _drain_worker_events(agen)

    events = _run_async(_run())
    text = " ".join(str(event) for event in events)
    verdict, findings = _parse_review_text(text, fallback_sha=commit_sha)
    reviewed_sha = commit_sha or _read_head_sha(worktree) or ""
    return ReviewResult(
        pass_verdict=verdict,
        commit_sha=reviewed_sha,
        findings=findings,
        message="pass" if verdict else "fail",
    )


def _parse_review_text(text: str, *, fallback_sha: str) -> tuple[bool, list[str]]:
    import json
    import re

    match = re.search(r"\{[^{}]*\"verdict\"[^{}]*\}", text, re.IGNORECASE | re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            verdict_raw = str(data.get("verdict") or "").upper()
            findings = data.get("findings") or []
            if not isinstance(findings, list):
                findings = [str(findings)]
            findings = [str(item) for item in findings]
            return verdict_raw == "PASS", findings
        except ValueError:
            pass
    low = text.lower()
    if "pass" in low and "fail" not in low:
        return True, []
    findings = []
    if text.strip():
        findings.append(text.strip()[:500])
    else:
        findings.append("empty review output")
    return False, findings


class _NullJob:
    """Stand-in Windows job object for host-driven worker spawns."""

    def assign(self, pid: int) -> None:  # noqa: ARG002
        return None


def build_live_host_workers(
    *,
    repo_root: Path,
    work_root: Path,
    recorded_sha_lookup: ShaLookup | None = None,
) -> HostWorkerBundle:
    """Wire Claude / Grok / Codex direct workers for production use."""
    return HostWorkerBundle(
        work_root=work_root,
        repo_root=repo_root,
        drivers={
            "claude": _live_claude_driver,
            "grok": _live_grok_driver,
            "codex_fix": _live_codex_fix_driver,
            "codex_review": _live_codex_review_driver,
        },
        recorded_sha_lookup=recorded_sha_lookup,
    )
