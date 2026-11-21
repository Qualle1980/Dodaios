"""Mark a command's output as a sample when the command itself capped it.

``gh pr list --limit 3`` returned three merged rows and the model said there
were no open PRs; thirteen drafts were open (measured 2026-11-15). The output
carried nothing that said "this is three of many". Every shell tool that hands
a model command output runs it through :func:`annotate_command_sample`, and
the honesty backstop (``jarvis.brain.action_honesty``) reads the same marker.

Lives in ``jarvis.core`` so tool plugins and the brain share one definition.
Pure regex, no model call.
"""

from __future__ import annotations

import re

_SAMPLE_LIMIT_RE = re.compile(
    r"(?:"
    r"--limit(?:\s*=\s*|\s+)(\d+)|"
    r"--max(?:-count|-results)?(?:\s*=\s*|\s+)(\d+)|"
    r"\|\s*head\s+(?:-n\s+)?(\d+)|"
    r"(?<![-\w])-n\s+(\d+)"
    r")",
    re.I,
)


def sample_limit_from_command(command: str) -> int | None:
    """Return N when ``command`` deliberately caps the result set, else None."""
    match = _SAMPLE_LIMIT_RE.search(str(command or ""))
    if match is None:
        return None
    for group in match.groups():
        if group is not None:
            return int(group)
    return None


def format_sample_annotation(n: int) -> str:
    """Suffix appended to a tool result that is only a limited sample."""
    return f"(sample of {int(n)}; not the complete set)"


def annotate_command_sample(output: str, command: str) -> str:
    """Append a sample marker when the command itself limited the result set."""
    text = str(output or "")
    limit = sample_limit_from_command(command)
    if limit is None:
        return text
    marker = format_sample_annotation(limit)
    if marker in text:
        return text
    if not text.strip():
        return marker
    return text.rstrip() + "\n" + marker


__all__ = [
    "annotate_command_sample",
    "format_sample_annotation",
    "sample_limit_from_command",
]
