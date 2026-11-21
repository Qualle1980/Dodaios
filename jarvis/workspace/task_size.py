"""Map an Instant task size to a coding-CLI model and effort from the live list.

Instant picks pane model/effort on ``workspace-orchestrate`` create. This
helper is the shared size → id table so the tool description and the code
stay aligned. Ids come from ``jarvis.agentic_ide.model_lists`` (live CLI
catalogs), never from invented names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jarvis.agentic_ide.model_lists import (
    CLI_CLAUDE,
    CLI_CODEX,
    CLI_GROK,
    ModelEntry,
    ModelListStore,
    shared_store,
)

TASK_SIZES: Final[tuple[str, ...]] = ("simple", "normal", "hard")

_AGENT_TO_CLI: Final[dict[str, str]] = {
    "claude": CLI_CLAUDE,
    "claude-cli": CLI_CLAUDE,
    "codex": CLI_CODEX,
    "codex-cli": CLI_CODEX,
    "grok-build": CLI_GROK,
    "grok": CLI_GROK,
    "grok-cli": CLI_GROK,
}


@dataclass(frozen=True, slots=True)
class SizePick:
    """One create pick: a live catalog model id and an effort level (or empty)."""

    model: str = ""
    effort: str = ""


def _score_simple(entry: ModelEntry) -> tuple[int, str]:
    """Prefer fast / haiku / luna / build-fast ids for simple work."""
    slug = entry.id.lower()
    rank = 50
    if "haiku" in slug or "build-fast" in slug or "fast" in slug:
        rank = 0
    elif "luna" in slug or "mini" in slug:
        rank = 1
    elif "sonnet" in slug or "terra" in slug:
        rank = 30
    elif "opus" in slug or "sol" in slug or "astra" in slug:
        rank = 40
    return rank, entry.id


def _score_normal(entry: ModelEntry) -> tuple[int, str]:
    slug = entry.id.lower()
    rank = 50
    if "sonnet" in slug or "terra" in slug or slug.startswith("grok-4.7") and "fast" not in slug:
        rank = 0
    elif "haiku" in slug or "build-fast" in slug:
        rank = 20
    elif "opus" in slug or "sol" in slug:
        rank = 10
    return rank, entry.id


def _score_hard(entry: ModelEntry) -> tuple[int, str]:
    slug = entry.id.lower()
    rank = 50
    if "opus" in slug or "sol" in slug or "astra" in slug:
        rank = 0
    elif "terra" in slug or "sonnet" in slug:
        rank = 5
    elif "fable" in slug:
        rank = 8
    elif "haiku" in slug or "fast" in slug or "luna" in slug:
        rank = 40
    return rank, entry.id


def _effort_for(size: str, entry: ModelEntry) -> str:
    if not entry.efforts:
        return ""
    if size == "simple":
        return "low" if "low" in entry.efforts else (entry.default_effort or "")
    if size == "normal":
        return "medium" if "medium" in entry.efforts else (entry.default_effort or "")
    # hard
    for level in ("high", "xhigh", "max", "ultra"):
        if level in entry.efforts:
            return level
    return entry.default_effort or (entry.efforts[-1] if entry.efforts else "")


def pick_for_size(
    agent: str,
    size: str,
    *,
    store: ModelListStore | None = None,
) -> SizePick:
    """Live catalog model and effort for ``agent`` at ``size``, or empty picks.

    ``size`` is one of :data:`TASK_SIZES`. Unknown agent or size, an empty
    live list, or a list the account did not confirm, returns an empty
    :class:`SizePick` — callers must not invent a fallback id. An unconfirmed
    (cached) id may be stale; the CLI then refuses it, so the CLI default is
    used instead.
    """
    cli = _AGENT_TO_CLI.get((agent or "").strip())
    size_key = (size or "").strip().lower()
    if cli is None or size_key not in TASK_SIZES:
        return SizePick()
    lists = store or shared_store()
    result = lists.ensure(cli)
    if not result.models or not result.confirmed:
        return SizePick()
    scorers = {"simple": _score_simple, "normal": _score_normal, "hard": _score_hard}
    chosen = min(result.models, key=scorers[size_key])
    return SizePick(model=chosen.id, effort=_effort_for(size_key, chosen))


__all__ = ["TASK_SIZES", "SizePick", "pick_for_size"]
