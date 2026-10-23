"""Lanes: which executor does a part of an order, cheapest and fastest first.

Every part of an order lands in one lane. A lane is an ordered chain of
executors; the first one that is connected wins. Only subscriptions and the
conductor's own seat appear in a chain, never a metered key (background
policy, AP-21/AP-22), so a lane whose chain has no connected executor is
unavailable instead of silently billing a key.

The first pass is deterministic and costs nothing: the user's own words
decide (``task_kind``) plus whether they name a file. The conductor's plan
may move a part to another lane; it can never add an executor.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Literal

from ..task_kind import task_kind

Lane = Literal["answer", "lookup", "change", "deep"]
Executor = Literal["conductor", "codex", "claude-code"]

#: Executor families, for picking an independent reviewer.
FAMILY: Final[dict[Executor, str]] = {
    "conductor": "openai",
    "codex": "openai",
    "claude-code": "anthropic",
}


@dataclass(frozen=True)
class LaneSpec:
    """One lane: its executors in order, plus how hard they should think."""

    chain: tuple[Executor, ...]
    effort: Literal["low", "medium", "high"]
    sandbox: Literal["none", "read-only", "workspace-write"]
    reviewed: bool


LANES: Final[dict[Lane, LaneSpec]] = {
    # No worker at all: the conductor answers from what it knows.
    "answer": LaneSpec(("conductor",), "low", "none", reviewed=False),
    # Read or search files: nothing may change, so the cheapest reader wins.
    "lookup": LaneSpec(("codex", "claude-code"), "low", "read-only", reviewed=False),
    "change": LaneSpec(("codex", "claude-code"), "medium", "workspace-write", reviewed=True),
    "deep": LaneSpec(("claude-code", "codex"), "high", "workspace-write", reviewed=True),
}

_NAMES_A_FILE_RE: Final[re.Pattern[str]] = re.compile(
    r"[\w.-]+[/\x5c][\w./\x5c-]+"
    r"|\b[\w-]+\.(?:py|md|txt|json|toml|ya?ml|[jt]sx?|css|html|ps1|sh|rs|go|java|cs|cpp|c|h)\b",
    re.IGNORECASE,
)

# Work that is large by its own description. German entries are runtime-input
# classifiers.
_DEEP_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:refactor\w*|architecture|redesign|migrat\w+|rewrite|across\s+the\s+(?:repo|codebase)|"
    r"umbau\w*|architektur|neu\s+schreiben|ganze[nm]?\s+(?:repo|projekt|code))\b",  # i18n-allow
    re.IGNORECASE,
)


# A knowledge question by its opening words ("what is the difference between
# rebase and merge?"): the nouns inside it are no order, however verb-like.
_KNOWLEDGE_QUESTION_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:what\s+(?:is|are|does)|why|how\s+(?:does|do|is)|explain|"
    r"was\s+(?:ist|sind|bedeutet)|warum|wieso|"  # i18n-allow: DE runtime-input classifier
    r"wie\s+funktioniert|erkläre?|erklaere?)\b"  # i18n-allow
    r"[^?]*\?\s*$",
    re.IGNORECASE,
)


def names_a_file(request: str) -> bool:
    """Whether the user's own words name a path or a file."""
    return bool(_NAMES_A_FILE_RE.search(request or ""))


def first_lane(request: str) -> Lane:
    """The lane the user's own words point to, decided without a model."""
    if _KNOWLEDGE_QUESTION_RE.match(request or "") and not names_a_file(request):
        return "answer"
    kind = task_kind(request)
    if kind == "change":
        return "deep" if _DEEP_RE.search(request or "") else "change"
    return "lookup" if names_a_file(request) else "answer"


def pick_executor(lane: Lane, connected: Iterable[str]) -> Executor | None:
    """The first executor of the lane's chain that is connected, else ``None``.

    ``conductor`` is always connected: it is the seat running this code.
    """
    available = set(connected) | {"conductor"}
    for executor in LANES[lane].chain:
        if executor in available:
            return executor
    return None


def pick_reviewer(builder: Executor, connected: Iterable[str]) -> Executor | None:
    """A connected executor of the other family, else ``None`` (no fake independence)."""
    available = set(connected)
    for candidate in ("claude-code", "codex"):
        if candidate in available and FAMILY[candidate] != FAMILY[builder]:
            return candidate  # type: ignore[return-value]
    return None


__all__ = [
    "FAMILY", "LANES", "Executor", "Lane", "LaneSpec",
    "first_lane", "names_a_file", "pick_executor", "pick_reviewer",
]
