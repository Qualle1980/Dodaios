"""The personal memory read tools are declared to a live call, not cut.

The 2026-11-09 loop was a schema field (fixed in #142), but a correct schema
still reached no live model: ``_DIRECT_TOOL_PREFERENCE`` ranked the three
ADR-0044 read tools nowhere, so with twelve direct slots and a byte budget
the mid-weight steering tools consumed ahead of them, every live call was
given a vault door it could not see.

The budget loop uses ``continue``, not ``break``: a single oversized tool is
skipped and the rest still fit. A test that offers one heavy competitor
therefore passes on unfixed code. These cases crowd the catalog with many
mid-weight tools instead, which is what the live session actually carries.
"""

from __future__ import annotations

from jarvis.live.native import _DIRECT_TOOL_LIMIT, _fit_declarations
from jarvis.live.tools import SUBJECT_ALIASES

MEMORY_TOOLS = ("memory-search", "memory-history", "memory-topics")


def _catalog_tool(name: str, *, padding: int = 400) -> dict:
    """A catalog declaration shaped the way the live session emits one."""
    return {
        "name": name,
        "description": f"{name}: " + "x" * padding,
        "parameters": {"type": "object", "properties": {}},
    }


def _crowded_catalog() -> list[dict]:
    """The memory tools behind more mid-weight competitors than there are slots."""
    competitors = [
        "find-app-action", "run-app-action", "search_web", "open_app",
        "screen_snapshot", "google_calendar", "gmail", "youtube_music",
        "spotify", "home_assistant", "product_help", "wiki-recall",
        "wiki-page-read", "wiki-list", "run-shell", "create-skill",
        "inspect-pointer", "spawn-worker",
    ]
    # Catalog order alone would put the memory tools last: 'm' sorts after
    # most of the names above, and the fit is stable by rank, then by order.
    return [_catalog_tool(n) for n in competitors] + [_catalog_tool(n) for n in MEMORY_TOOLS]


def test_memory_read_tools_survive_a_realistic_declaration_budget() -> None:
    declarations = _crowded_catalog()
    # Deliberately too small for the whole catalog: this is the live condition.
    budget = sum(len(str(d)) for d in declarations) // 3
    fitted = _fit_declarations(declarations, budget)
    names = {d["description"].split(": ", 1)[0] for d in fitted}
    assert len(fitted) <= _DIRECT_TOOL_LIMIT
    missing = [tool for tool in MEMORY_TOOLS if tool not in names]
    assert not missing, (
        f"a live call is not given {missing}; the vault door is invisible to it"
    )


def test_a_spoken_memory_question_has_a_lexical_bridge() -> None:
    """The names say 'memory'; people ask what you know about them."""
    for tool in MEMORY_TOOLS:
        assert SUBJECT_ALIASES.get(tool), f"{tool} has no subject alias"
