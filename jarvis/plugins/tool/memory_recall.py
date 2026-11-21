"""The three read tools of the personal memory vault (ADR-0044, chunk 3).

- ``memory-search`` — what do you know about X.
- ``memory-history`` — how a fact changed; answers "where did I live before?".
- ``memory-topics`` — which topics the vault holds anything about, with counts.

``memory-topics`` is the one upstream does not have. Search only helps someone
who already knows what to search for, and the entity index that would answer
"what do you know about me?" existed in the database with nothing exposing it.
It is a tool rather than a line in the prompt on purpose: a catalogue in the
prompt is paid for on every turn - twice over in a live call, where the server
holds the context - and it would be a second copy of sentences that are already
short.

**No memory ids in what the model sees.** An id is stable so that a changed
fact keeps its history, which makes it opaque (``mem_<stamp>_<hex>``), and a
model that is shown one will eventually read it out in a call. Editing by id is
the memory view's job, not the model's, so these tools take a topic.

All three are read-only, safe-tier and local: one FTS query, no model call, so
they are safe on the voice critical path (AP-9, AP-11).
"""
from __future__ import annotations

import logging
from typing import Any

from jarvis.core.protocols import ToolResult

log = logging.getLogger(__name__)

#: Most memories one search answers with.
SEARCH_LIMIT = 8
#: Most versions one history answers with.
HISTORY_LIMIT = 20
#: Most topics listed at once, busiest first.
TOPIC_LIMIT = 40


def _vault() -> Any:
    """The open vault, or ``None`` when it is switched off or unavailable."""
    from jarvis.core import config as cfg
    from jarvis.memory.vault import runtime

    return runtime.active(cfg.load_config())


def _line(memory: Any) -> str:
    """One memory as a line: the sentence, then what qualifies it."""
    details: list[str] = []
    if memory.entities:
        details.append(", ".join(memory.entities))
    # An imported memory's start date is when it was imported, not when it
    # became true, so it would read as a false precision.
    if memory.valid_from and memory.origin != "imported":
        details.append(f"since {memory.valid_from[:10]}")
    if memory.valid_to:
        details.append(f"until {memory.valid_to[:10]}")
    if memory.pinned:
        details.append("pinned")
    if memory.origin != "user":
        details.append("not stated outright")
    suffix = f" ({'; '.join(details)})" if details else ""
    return f"- {memory.text}{suffix}"


class MemorySearchTool:
    """Keyword search over what the assistant knows about the user."""

    name: str = "memory-search"
    risk_tier: str = "safe"
    description: str = (
        "Search the user's long-term memory for what is known about a person, "
        "place, project, preference or standing instruction. Use it when the "
        "user asks what you know about something, or refers to a fact about "
        "their life you do not already have in front of you. Give one to four "
        "keywords, not a sentence."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "One to four keywords, e.g. 'commit language' or 'Lena job'.",
            }
        },
        "required": ["query"],
    }

    async def execute(self, args: dict[str, Any], ctx: Any) -> ToolResult:  # noqa: ARG002
        import asyncio

        query = " ".join(str(args.get("query") or "").split())
        if not query:
            return ToolResult(success=False, output=None, error="query is missing")
        vault = await asyncio.to_thread(_vault)
        if vault is None:
            return ToolResult(success=True, output="The long-term memory is switched off.")
        hits = await asyncio.to_thread(vault.search, query, limit=SEARCH_LIMIT)
        if not hits:
            return ToolResult(success=True, output=f"Nothing remembered about {query!r}.")
        # Surfaced, which is not the same as having helped: `use_count` never
        # decides what the always-on block holds (ADR-0044 principle 7).
        await asyncio.to_thread(vault.mark_used, [hit.memory.id for hit in hits])
        lines = [_line(hit.memory) for hit in hits]
        log.info("memory-search: %d hit(s)", len(hits))
        return ToolResult(success=True, output="\n".join(lines))


class MemoryHistoryTool:
    """Every version of what is known about a topic, oldest first."""

    name: str = "memory-history"
    risk_tier: str = "safe"
    description: str = (
        "Show how what you know about a topic changed over time, including "
        "facts that are no longer true. Use it for questions about the past - "
        "'where did I live before', 'what did I use to drive', 'when did that "
        "change'. Give the person, place or thing it is about."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The person, place, project or thing, e.g. 'Lena' or 'Berlin'.",
            }
        },
        "required": ["topic"],
    }

    async def execute(self, args: dict[str, Any], ctx: Any) -> ToolResult:  # noqa: ARG002
        import asyncio

        topic = " ".join(str(args.get("topic") or "").split())
        if not topic:
            return ToolResult(success=False, output=None, error="topic is missing")
        vault = await asyncio.to_thread(_vault)
        if vault is None:
            return ToolResult(success=True, output="The long-term memory is switched off.")
        memories = await asyncio.to_thread(vault.about, topic)
        if not memories:
            # The topic may not be a recorded entity; a keyword search still
            # finds the sentence, it just cannot order it as a timeline.
            hits = await asyncio.to_thread(vault.search, topic, limit=SEARCH_LIMIT)
            if not hits:
                return ToolResult(success=True, output=f"Nothing remembered about {topic!r}.")
            lines = [_line(hit.memory) for hit in hits]
            return ToolResult(
                success=True,
                output=f"No timeline for {topic!r}; what is known:\n" + "\n".join(lines),
            )
        now = await asyncio.to_thread(vault.now)
        lines: list[str] = []
        for memory in memories[:HISTORY_LIMIT]:
            mark = "" if memory.is_current(now) else " [no longer true]"
            lines.append(f"{_line(memory)}{mark}")
        log.info("memory-history: %d version(s)", len(lines))
        return ToolResult(success=True, output="\n".join(lines))


class MemoryTopicsTool:
    """What the vault holds anything about at all, with counts."""

    name: str = "memory-topics"
    risk_tier: str = "safe"
    description: str = (
        "List the topics the user's long-term memory holds anything about, and "
        "how many kinds of memory there are. Use it when the user asks what you "
        "know or remember about them in general, before searching for a "
        "specific thing. Takes no arguments."
    )
    schema: dict[str, Any] = {"type": "object", "properties": {}}

    async def execute(self, args: dict[str, Any], ctx: Any) -> ToolResult:  # noqa: ARG002
        import asyncio

        vault = await asyncio.to_thread(_vault)
        if vault is None:
            return ToolResult(success=True, output="The long-term memory is switched off.")
        stats = await asyncio.to_thread(vault.stats)
        if not stats["current"]:
            return ToolResult(success=True, output="The long-term memory is still empty.")
        entities = await asyncio.to_thread(vault.entities)
        busiest = sorted(entities, key=lambda e: (-int(e["uses"] or 0), str(e["name"])))
        kinds = ", ".join(
            f"{kind} {count}" for kind, count in sorted(stats["by_kind"].items())
        )
        lines = [f"{stats['current']} memories: {kinds}."]
        if busiest:
            named = ", ".join(
                f"{e['name']} ({e['uses']})" for e in busiest[:TOPIC_LIMIT] if e["uses"]
            )
            if named:
                lines.append(f"Topics: {named}.")
            if len(busiest) > TOPIC_LIMIT:
                lines.append(f"({len(busiest) - TOPIC_LIMIT} more topics.)")
        lines.append("Use memory-search for what is known about one of them.")
        return ToolResult(success=True, output="\n".join(lines))
