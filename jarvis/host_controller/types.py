"""Shared constants for the Instant host controller."""

from __future__ import annotations

from typing import Final, Literal

IntentKind = Literal[
    "PLAN",
    "ASSIGN_TASK",
    "REQUEST_REVIEW",
    "REQUEST_FIX",
    "ACCEPT_TASK",
    "BLOCKED",
    "REPORT_PROGRESS",
]

INTENT_KINDS: Final[frozenset[str]] = frozenset(
    {
        "PLAN",
        "ASSIGN_TASK",
        "REQUEST_REVIEW",
        "REQUEST_FIX",
        "ACCEPT_TASK",
        "BLOCKED",
        "REPORT_PROGRESS",
    }
)

TaskStatus = Literal[
    "planned",
    "in-progress",
    "review",
    "accepted",
    "blocked",
    "waiting-quota",
    "needs-check",
]

TASK_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "planned",
        "in-progress",
        "review",
        "accepted",
        "blocked",
        "waiting-quota",
        "needs-check",
    }
)

#: Same discipline as ``MAX_CRITIC_LOOPS`` (ADR-0009): three fix rounds, then stop.
MAX_FIX_ROUNDS: Final[int] = 3

ALLOWED_WORKERS: Final[frozenset[str]] = frozenset({"claude", "grok", "codex"})

#: Keys allowed on each intent kind (plus shared ``intent_id`` / ``kind``).
INTENT_KEYS: Final[dict[str, frozenset[str]]] = {
    "PLAN": frozenset({"intent_id", "kind", "project", "title", "tasks"}),
    "ASSIGN_TASK": frozenset(
        {
            "intent_id",
            "kind",
            "project",
            "issue_number",
            "worker",
            "task",
            "allow_subagents",
            "subagent_kind",
        }
    ),
    "REQUEST_REVIEW": frozenset(
        {"intent_id", "kind", "project", "issue_number", "commit_sha"}
    ),
    "REQUEST_FIX": frozenset(
        {
            "intent_id",
            "kind",
            "project",
            "issue_number",
            "findings",
            "allow_subagents",
            "subagent_kind",
        }
    ),
    "ACCEPT_TASK": frozenset(
        {
            "intent_id",
            "kind",
            "project",
            "issue_number",
            "commit_sha",
            "tests_green",
        }
    ),
    "BLOCKED": frozenset(
        {"intent_id", "kind", "project", "issue_number", "reason"}
    ),
    "REPORT_PROGRESS": frozenset({"intent_id", "kind", "project"}),
}
