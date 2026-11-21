"""NAMED_TOOLS keep their order under the live declaration byte budget.

Live 2026-11-14: show_on_screen (NAMED_TOOLS position 4 of 7) lost its slot
109 bytes short while other named tools kept theirs. Cause: ``_PRIORITY_TOOLS``
was a frozenset, so packing ranked promised tools by CORE_PRIORITY (which
lists only ``navigate`` of that tuple) and by name. Mid-tuple tools could
lose while neighbours stayed.

This catalog is sized like a real busy install (~147 tools, 24000 B budget)
so the defect is deterministic without depending on which plugins happen to
be installed (same trap as test_catalog_packing_respects_rank.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jarvis.live.coordination import NAMED_TOOLS
from jarvis.live.tools import (
    _CATALOG_BYTE_BUDGET,
    _PRIORITY_TOOLS,
    _PRIORITY_TOOLS_ORDER,
    LiveTools,
)


@dataclass(frozen=True)
class _Descriptor:
    name: str
    description: str = "does something"
    input_schema: dict[str, Any] = field(default_factory=dict)
    risk_tier: str = "safe"


class _Gateway:
    def __init__(self, descriptors: list[_Descriptor]) -> None:
        self._descriptors = descriptors

    def catalog(self) -> list[_Descriptor]:
        return list(self._descriptors)


def _bridge(descriptors: list[_Descriptor]) -> LiveTools:
    bridge = object.__new__(LiveTools)
    bridge.gateway = _Gateway(descriptors)
    bridge._names = {}
    bridge.backend_model = "gemini-live"
    bridge.language = "de"
    bridge._catalog_surface_reported = None
    return bridge


def _schema_tool(name: str, *, description_chars: int, properties: int) -> _Descriptor:
    """A catalog entry whose wire size is dominated by schema + description."""
    return _Descriptor(
        name=name,
        description=("tool " + name + " ") * max(1, description_chars // (len(name) + 6)),
        input_schema={
            "type": "object",
            "properties": {
                f"arg_{index}": {
                    "type": "string",
                    "description": f"argument {index} for {name}",
                }
                for index in range(properties)
            },
            "required": [f"arg_{index}" for index in range(min(2, properties))],
        },
    )


def _busy_catalog() -> list[_Descriptor]:
    """~147 tools at real-install scale, with every promised name present."""
    catalog: list[_Descriptor] = []
    # Named / promised tools with sizes in the measured live range.
    catalog.append(_schema_tool("workspace-orchestrate", description_chars=400, properties=40))
    catalog.append(_schema_tool("workspace-git", description_chars=200, properties=12))
    catalog.append(_schema_tool("show_on_screen", description_chars=180, properties=6))
    catalog.append(_schema_tool("navigate", description_chars=120, properties=4))
    catalog.append(_schema_tool("ask_agent", description_chars=160, properties=8))
    catalog.append(_schema_tool("mission-amend", description_chars=140, properties=6))
    catalog.append(_schema_tool("find-app-action", description_chars=100, properties=4))
    catalog.append(_schema_tool("run-app-action", description_chars=100, properties=5))
    catalog.append(_schema_tool("memory-search", description_chars=200, properties=3))
    catalog.append(_schema_tool("memory-history", description_chars=180, properties=3))
    catalog.append(_schema_tool("memory-topics", description_chars=120, properties=2))
    catalog.append(_schema_tool("search_web", description_chars=150, properties=4))
    # Fillers to the live-measured catalog size (~147). Heavier than a bare
    # name so the 24000 B budget is under real pressure.
    for index in range(135):
        catalog.append(
            _schema_tool(f"filler-{index:03d}", description_chars=80, properties=3)
        )
    assert len(catalog) >= 147
    return catalog


def _declared_names(descriptors: list[_Descriptor]) -> list[str]:
    bridge = _bridge(descriptors)
    definitions = LiveTools.declarations(bridge)
    return [bridge._names.get(d["name"], d["name"]) for d in definitions]


def test_priority_tools_order_embeds_named_tools_in_order() -> None:
    """The ordered tuple is the source of truth; the frozenset is membership."""
    assert _PRIORITY_TOOLS == frozenset(_PRIORITY_TOOLS_ORDER)
    named_positions = [_PRIORITY_TOOLS_ORDER.index(name) for name in NAMED_TOOLS]
    assert named_positions == sorted(named_positions), (
        "NAMED_TOOLS order was lost inside _PRIORITY_TOOLS_ORDER"
    )
    assert list(NAMED_TOOLS) == [
        name for name in _PRIORITY_TOOLS_ORDER if name in set(NAMED_TOOLS)
    ]


def test_named_tools_survive_a_full_size_catalog_in_order() -> None:
    """Every NAMED_TOOLS member is declared; relative order is preserved.

    Raising the budget is not the fix under test: the catalog and
    ``_CATALOG_BYTE_BUDGET`` stay at the live-measured scale.
    """
    catalog = _busy_catalog()
    assert len(catalog) >= 147
    assert _CATALOG_BYTE_BUDGET == 24_000

    declared = _declared_names(catalog)
    declared_set = set(declared)

    catalog_named = [name for name in NAMED_TOOLS if name != "task_status"]
    missing = [name for name in catalog_named if name not in declared_set]
    assert not missing, (
        "NAMED_TOOLS lost under the 24000 B budget: "
        f"{missing}. Order must keep them; do not raise the budget."
    )
    assert "show_on_screen" in declared_set

    declared_named = [name for name in declared if name in set(NAMED_TOOLS)]
    expected_named = [name for name in NAMED_TOOLS if name in declared_set]
    assert declared_named == expected_named, (
        "NAMED_TOOLS declaration order drifted: "
        f"declared={declared_named}, expected={expected_named}"
    )
