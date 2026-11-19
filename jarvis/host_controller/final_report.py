"""Write report.md when every planned task is accepted."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def all_accepted(progress: dict[str, Any], issues: list[Any]) -> bool:
    planned = int(progress.get("planned") or 0)
    accepted = int(progress.get("accepted") or 0)
    if planned <= 0 or accepted < planned:
        return False
    return all(str(getattr(issue, "status", "") or "") == "accepted" for issue in issues)


def format_final_report(
    *,
    project: str,
    issues: list[Any],
    progress: dict[str, Any],
    left_out: list[str] | None = None,
    test_results: str = "see audit / worker logs",
    open_risks: list[str] | None = None,
) -> str:
    lines = [
        f"# Final report: {project}",
        "",
        f"Progress: {float(progress.get('percent') or 0):.1f}% "
        f"({progress.get('accepted', 0)}/{progress.get('planned', 0)} accepted).",
        "",
        "## Accepted",
    ]
    for issue in issues:
        if str(getattr(issue, "status", "") or "") != "accepted":
            continue
        sha = getattr(issue, "commit_sha", None) or "(no sha)"
        worker = getattr(issue, "worker", None) or "?"
        lines.append(
            f"- #{getattr(issue, 'number', '?')} {getattr(issue, 'title', '')} "
            f"— commit `{sha}` worker={worker}"
        )
    if left_out:
        lines.append("")
        lines.append("## Left out")
        for item in left_out:
            lines.append(f"- {item}")
    else:
        lines.append("")
        lines.append("## Left out")
        lines.append("- none")
    lines.append("")
    lines.append("## Tests")
    lines.append(test_results)
    lines.append("")
    lines.append("## Open risks")
    risks = list(open_risks or ["Host waited for the user after 100%."])
    for risk in risks:
        lines.append(f"- {risk}")
    lines.append("")
    lines.append("Host stopped and is waiting for the user.")
    lines.append("")
    return "\n".join(lines)


def write_final_report(path: Path, markdown: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
