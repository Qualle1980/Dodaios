"""Overwrite the run-folder progress.md after every host state change."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def format_progress_markdown(
    *,
    title: str,
    progress: dict[str, Any],
    active: list[dict[str, Any]] | None = None,
    blockers: list[dict[str, Any]] | None = None,
    next_step: str = "",
) -> str:
    planned = int(progress.get("planned") or 0)
    accepted = int(progress.get("accepted") or 0)
    in_progress = int(progress.get("in_progress") or 0)
    percent = float(progress.get("percent") or 0.0)
    active = list(active or [])
    blockers = list(blockers or [])

    active_bits: list[str] = []
    for row in active:
        num = row.get("issue_number")
        status = row.get("status")
        worker = row.get("worker") or ""
        bit = f"#{num} {status}"
        if worker:
            bit += f" ({worker})"
        active_bits.append(bit)
    active_text = ", ".join(active_bits) if active_bits else "none"

    blocker_bits = [
        f"#{row.get('issue_number')} {row.get('status')}: {row.get('reason') or row.get('status')}"
        for row in blockers
    ]
    blocker_text = "; ".join(blocker_bits) if blocker_bits else "none"

    if not next_step:
        if blockers:
            next_step = "resolve blockers or re-plan"
        elif in_progress:
            # Never label an in-progress project "complete"; the user-facing
            # line is built by end_state.running_user_phrase.
            next_step = "wait for worker commit, then REQUEST_REVIEW"
        elif accepted < planned or planned <= 0:
            next_step = "ASSIGN_TASK for the next planned issue"
        else:
            # Counts say every planned task is accepted (host gate).
            next_step = "project complete"

    from jarvis.host_controller.end_state import running_user_phrase

    # When work is still open, surface the same in-progress line the
    # chat path uses so progress.md never reads as finished early.
    end_line = ""
    if planned > 0 and (accepted < planned or in_progress > 0):
        end_line = f" End state: {running_user_phrase(percent)}."

    return (
        f"# Progress: {title}\n\n"
        f"{percent:.1f}% ({accepted}/{planned} accepted), "
        f"active: {active_text}. "
        f"Blockers: {blocker_text}. "
        f"Next: {next_step}.{end_line}\n"
    )


def write_progress_file(path: Path, markdown: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def summarize_issues(issues: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for issue in issues:
        status = str(getattr(issue, "status", "") or "")
        row = {
            "issue_number": int(getattr(issue, "number", 0) or 0),
            "status": status,
            "worker": getattr(issue, "worker", None),
            "reason": None,
        }
        if status in {"in-progress", "review", "needs-check", "waiting-quota"}:
            active.append(row)
        if status == "blocked":
            row["reason"] = "blocked"
            blockers.append(row)
        if status == "waiting-quota":
            blockers.append({**row, "reason": "waiting-quota"})
    return active, blockers
