"""Task dependency readiness and in-progress file-overlap checks."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def issue_depends_on(issue: Any) -> list[int]:
    raw = getattr(issue, "depends_on", None) or []
    out: list[int] = []
    for item in raw:
        try:
            out.append(int(item))
        except (TypeError, ValueError) as exc:
            log.debug("host controller: fallback taken (%s)", exc)
            continue
            continue
    return out


def issue_files(issue: Any) -> list[str]:
    raw = getattr(issue, "files", None) or []
    return [str(path).replace("\\", "/").strip() for path in raw if str(path).strip()]


def unmet_dependencies(issue: Any, *, issues_by_number: dict[int, Any]) -> list[int]:
    unmet: list[int] = []
    for dep in issue_depends_on(issue):
        other = issues_by_number.get(dep)
        if other is None:
            unmet.append(dep)
            continue
        if str(getattr(other, "status", "") or "") != "accepted":
            unmet.append(dep)
    return unmet


def ready_issue_numbers(issues: list[Any]) -> list[int]:
    by_number = {int(getattr(issue, "number", 0)): issue for issue in issues}
    ready: list[int] = []
    for issue in issues:
        status = str(getattr(issue, "status", "") or "")
        if status != "planned":
            continue
        if unmet_dependencies(issue, issues_by_number=by_number):
            continue
        ready.append(int(getattr(issue, "number", 0)))
    return ready


def overlapping_files(
    *,
    candidate_files: list[str],
    issues: list[Any],
    skip_number: int | None = None,
) -> list[tuple[int, str]]:
    """Return (issue_number, path) pairs that collide with in-progress work."""
    wanted = {path.replace("\\", "/").strip() for path in candidate_files if path}
    if not wanted:
        return []
    hits: list[tuple[int, str]] = []
    for issue in issues:
        number = int(getattr(issue, "number", 0) or 0)
        if skip_number is not None and number == skip_number:
            continue
        status = str(getattr(issue, "status", "") or "")
        if status not in {"in-progress", "review", "needs-check"}:
            continue
        for path in issue_files(issue):
            if path in wanted:
                hits.append((number, path))
    return hits
