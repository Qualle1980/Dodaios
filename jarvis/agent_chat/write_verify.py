"""Runner-enforced verification after a recognised GitHub/git write.

After a mutating ``RunCommand`` succeeds, the API runner builds a read
command, executes it, and compares the observed state with what the write
claimed. A mismatch is reported in the tool result; the model must not say
the write is done without that check.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Final

from jarvis.agent_chat.write_gate import command_is_write

log = logging.getLogger(__name__)

_ISSUE_EDIT_RE: Final[re.Pattern[str]] = re.compile(
    r"\bgh\s+(?:.+?\s+)?issue\s+edit\s+(\d+)\b", re.IGNORECASE
)
_ISSUE_CREATE_RE: Final[re.Pattern[str]] = re.compile(
    r"\bgh\s+(?:.+?\s+)?issue\s+create\b", re.IGNORECASE
)
_PR_EDIT_RE: Final[re.Pattern[str]] = re.compile(
    r"\bgh\s+(?:.+?\s+)?pr\s+edit\s+(\d+)\b", re.IGNORECASE
)
_PR_CREATE_RE: Final[re.Pattern[str]] = re.compile(
    r"\bgh\s+(?:.+?\s+)?pr\s+(?:create|merge)\b", re.IGNORECASE
)
_ASSIGNEE_RE: Final[re.Pattern[str]] = re.compile(
    r"--add-assignee(?:=|\s+)([^\s]+)", re.IGNORECASE
)
_TITLE_RE: Final[re.Pattern[str]] = re.compile(
    r"--title(?:=|\s+)(?P<q>[\"'])(?P<title>.*?)(?P=q)|--title(?:=|\s+)(?P<title2>\S+)",
    re.IGNORECASE,
)
_URL_ISSUE_RE: Final[re.Pattern[str]] = re.compile(
    r"https://github\.com/[^/\s]+/[^/\s]+/issues/(\d+)"
)
_URL_PR_RE: Final[re.Pattern[str]] = re.compile(
    r"https://github\.com/[^/\s]+/[^/\s]+/pull/(\d+)"
)


@dataclass(frozen=True, slots=True)
class VerifyPlan:
    """A read command that checks the effect of a prior write."""

    command: str
    expect_assignees: tuple[str, ...] = ()
    expect_number: str = ""
    expect_kind: str = ""  # issue | pr | git | generic
    label: str = ""


@dataclass(frozen=True, slots=True)
class VerifyResult:
    ok: bool
    command: str
    output: str
    message: str
    mismatch: bool = False


@dataclass(slots=True)
class PendingVerify:
    """Optional injectable runner for tests (no real GitHub)."""

    run_command: Any = None  # async (command: str) -> tuple[str, bool]


def build_verify_plan(command: str, write_output: str = "") -> VerifyPlan | None:
    """Derive a read command for a recognised write, or ``None`` if unknown."""
    text = command or ""
    if not command_is_write(text):
        return None

    m = _ISSUE_EDIT_RE.search(text)
    if m:
        number = m.group(1)
        assignees = tuple(_ASSIGNEE_RE.findall(text))
        # Assignees-only check: do not also require ``number`` in the JSON
        # (the view may return only the requested fields).
        if assignees:
            return VerifyPlan(
                command=f"gh issue view {number} --json assignees",
                expect_assignees=assignees,
                expect_kind="issue",
                label=f"issue #{number}",
            )
        return VerifyPlan(
            command=f"gh issue view {number} --json number,title,state,assignees",
            expect_number=number,
            expect_kind="issue",
            label=f"issue #{number}",
        )

    if _ISSUE_CREATE_RE.search(text):
        number = ""
        url_m = _URL_ISSUE_RE.search(write_output or "")
        if url_m:
            number = url_m.group(1)
        if number:
            return VerifyPlan(
                command=f"gh issue view {number} --json number,title,state,assignees",
                expect_number=number,
                expect_kind="issue",
                label=f"issue #{number}",
            )
        return VerifyPlan(
            command="gh issue list --limit 1 --json number,title,url",
            expect_kind="issue",
            label="newest issue",
        )

    m = _PR_EDIT_RE.search(text)
    if m:
        number = m.group(1)
        return VerifyPlan(
            command=f"gh pr view {number} --json number,title,state",
            expect_number=number,
            expect_kind="pr",
            label=f"pr #{number}",
        )

    if _PR_CREATE_RE.search(text):
        number = ""
        url_m = _URL_PR_RE.search(write_output or "")
        if url_m:
            number = url_m.group(1)
        if number:
            return VerifyPlan(
                command=f"gh pr view {number} --json number,title,state",
                expect_number=number,
                expect_kind="pr",
                label=f"pr #{number}",
            )
        return VerifyPlan(
            command="gh pr status",
            expect_kind="pr",
            label="pr status",
        )

    # git push / commit
    if re.search(r"\bgit\s+push\b", text, re.IGNORECASE):
        return VerifyPlan(
            command="git status -sb && git log -1 --oneline",
            expect_kind="git",
            label="git push",
        )
    if re.search(r"\bgit\s+commit\b", text, re.IGNORECASE):
        return VerifyPlan(
            command="git log -1 --oneline && git status -sb",
            expect_kind="git",
            label="git commit",
        )

    # Generic gh write: view nothing specific — ask status
    if re.search(r"\bgh\s+", text, re.IGNORECASE):
        return VerifyPlan(
            command="gh status",
            expect_kind="generic",
            label="gh write",
        )
    return None


def check_verify_result(plan: VerifyPlan, output: str, *, is_error: bool) -> VerifyResult:
    """Compare a verify read against the plan's expectations."""
    text = output or ""
    if is_error:
        return VerifyResult(
            ok=False,
            command=plan.command,
            output=text,
            message=(
                f"Verify failed for {plan.label or 'write'}: read command errored. "
                "Do not claim the write succeeded."
            ),
            mismatch=True,
        )

    if plan.expect_assignees:
        missing = _missing_assignees(text, plan.expect_assignees)
        if missing:
            return VerifyResult(
                ok=False,
                command=plan.command,
                output=text,
                message=(
                    f"Verify mismatch for {plan.label}: expected assignee(s) "
                    f"{', '.join(plan.expect_assignees)} but missing {', '.join(missing)}. "
                    "Report the deviation; do not say the assignment is done."
                ),
                mismatch=True,
            )

    if plan.expect_number and plan.expect_kind in {"issue", "pr"}:
        if not _number_present(text, plan.expect_number):
            return VerifyResult(
                ok=False,
                command=plan.command,
                output=text,
                message=(
                    f"Verify mismatch for {plan.label}: expected number "
                    f"{plan.expect_number} not found in read output. "
                    "Report the deviation."
                ),
                mismatch=True,
            )

    return VerifyResult(
        ok=True,
        command=plan.command,
        output=text,
        message=f"Verify ok for {plan.label or 'write'}.",
        mismatch=False,
    )


def format_verify_appendix(result: VerifyResult) -> str:
    """Text appended to the write's tool result for the model."""
    status = "OK" if result.ok else "MISMATCH"
    body = result.output.strip()
    if len(body) > 4000:
        body = body[:4000] + "\n…[truncated]"
    return (
        f"\n\n[write-verify {status}]\n"
        f"$ {result.command}\n"
        f"{body}\n"
        f"{result.message}"
    )


async def run_write_verify(
    *,
    write_command: str,
    write_output: str,
    write_is_error: bool,
    execute_read,
) -> VerifyResult | None:
    """If ``write_command`` was a write and succeeded, run the derived read.

    ``execute_read`` is ``async (command: str) -> tuple[str, bool]``.
    """
    if write_is_error:
        return None
    if not command_is_write(write_command):
        return None
    plan = build_verify_plan(write_command, write_output)
    if plan is None:
        log.debug("write_verify: no plan for %r", write_command)
        return None
    try:
        output, is_error = await execute_read(plan.command)
    except Exception as exc:  # noqa: BLE001 — verification failure is reported, not a crash
        log.warning("write_verify: read failed for %r: %s", plan.command, exc)
        return VerifyResult(
            ok=False,
            command=plan.command,
            output=f"{type(exc).__name__}: {exc}",
            message=(
                f"Verify failed for {plan.label or 'write'}: could not run read. "
                "Do not claim the write succeeded."
            ),
            mismatch=True,
        )
    return check_verify_result(plan, output, is_error=is_error)


def _missing_assignees(output: str, expected: tuple[str, ...]) -> list[str]:
    text = output or ""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        data = None
    present: set[str] = set()
    if isinstance(data, dict):
        raw = data.get("assignees")
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    login = str(item.get("login") or item.get("name") or "").strip()
                    if login:
                        present.add(login.casefold())
                elif item:
                    present.add(str(item).strip().casefold())
        elif isinstance(raw, str) and raw.strip():
            present.add(raw.strip().casefold())
    if not present:
        # Fallback: substring match on raw text
        lower = text.casefold()
        return [a for a in expected if a.casefold() not in lower]
    return [a for a in expected if a.casefold() not in present]


def _number_present(output: str, number: str) -> bool:
    text = output or ""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict) and str(data.get("number") or "") == str(number):
        return True
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and str(item.get("number") or "") == str(number):
                return True
    return str(number) in text


__all__ = [
    "PendingVerify",
    "VerifyPlan",
    "VerifyResult",
    "build_verify_plan",
    "check_verify_result",
    "format_verify_appendix",
    "run_write_verify",
]
