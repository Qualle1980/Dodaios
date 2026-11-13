"""Measure LiveTools catalog wire sizes against the real Brain Manager.

No live turn, no paid provider calls. Builds the same BrainManager +
SupervisorToolGateway the desktop boot registers, then walks the voice
catalog in declarations() sort order.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def main(out_path: Path) -> int:
    # Use the installed Dodaios package (editable main checkout). Do not put a
    # worktree on sys.path: jarvis.core.entry_points then finds 0 tool plugins
    # and the manager keeps only the self-mod trio.
    from jarvis.core.entry_points import invalidate

    invalidate()

    from jarvis.brain.factory import _phase2_full_brain
    from jarvis.brain.tool_shortlist import CORE_PRIORITY, canonical
    from jarvis.core.runtime_refs import get_supervisor_tool_gateway
    from jarvis.live.tools import (
        LiveTools,
        _CATALOG_BYTE_BUDGET,
        _DIRECT_TOOLS,
        _PRIORITY_TOOLS,
        _wire_size,
    )

    # Real BrainManager path (registers BrainSupervisorToolGateway + workspace tools).
    manager = _phase2_full_brain()
    gateway = get_supervisor_tool_gateway()
    if gateway is None:
        print("ERROR: supervisor tool gateway was not registered", file=sys.stderr)
        return 1
    manager_type = type(manager).__name__
    tool_count = len(getattr(manager, "_tools", {}) or {})
    print(f"gateway={type(gateway).__name__} manager={manager_type} manager_tools={tool_count}")

    live = LiveTools(
        gateway,
        ledger=None,
        session_id="catalog-size-measure",
        language="de",
        backend_model="gemini-live",
    )
    catalog = list(live.catalog())
    print(f"voice_catalog descriptors: {len(catalog)}")

    rank = {name: index for index, name in enumerate(CORE_PRIORITY)}
    unranked = len(rank)
    ordered = sorted(
        catalog,
        key=lambda d: (
            d.name not in _PRIORITY_TOOLS,
            rank.get(canonical(d.name), unranked),
            d.name,
        ),
    )

    rows: list[dict[str, object]] = []
    for descriptor in ordered:
        if descriptor.name in _DIRECT_TOOLS:
            definition = {
                "type": "function",
                "name": descriptor.name,
                "description": descriptor.description,
                "parameters": descriptor.input_schema,
            }
        else:
            alias = live._alias(descriptor.name)
            definition = {
                "type": "function",
                "name": alias,
                "description": f"{descriptor.name}: {descriptor.description}",
                "parameters": descriptor.input_schema,
            }
        # declarations() charges +2 for the JSON array separator between items.
        wire_bytes = _wire_size(definition) + 2
        canon = canonical(descriptor.name)
        rows.append(
            {
                "name": descriptor.name,
                "wire_bytes": wire_bytes,
                "in_priority_group": descriptor.name in _PRIORITY_TOOLS,
                "canonical_rank": rank.get(canon, ""),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "wire_bytes", "in_priority_group", "canonical_rank"],
        )
        writer.writeheader()
        writer.writerows(rows)

    declarations = live.declarations(defer_catalog=False)
    declared_catalog = [
        live._names.get(item["name"], item["name"])
        for item in declarations
        if item["name"] not in {"end_call", "discover_tools", "call_tool", "confirm_action", "task_status"}
    ]
    # builtins are always present; catalog tools that fit are the rest minus directs already counted.
    fit_count = len(declared_catalog)
    names_in_order = [row["name"] for row in rows]
    try:
        memory_search_pos = names_in_order.index("memory-search") + 1  # 1-based
    except ValueError:
        memory_search_pos = None

    print(f"wrote {len(rows)} rows -> {out_path}")
    print(f"budget={_CATALOG_BYTE_BUDGET}")
    print(f"declarations_fit_catalog_tools={fit_count}")
    print(f"memory-search_sort_position_1based={memory_search_pos}")
    if memory_search_pos is not None:
        print(f"memory-search_wire_bytes={rows[memory_search_pos - 1]['wire_bytes']}")
        print(f"memory-search_in_priority_group={rows[memory_search_pos - 1]['in_priority_group']}")
        print(f"memory-search_canonical_rank={rows[memory_search_pos - 1]['canonical_rank']}")
    print(f"memory-search_in_declarations={('memory-search' in declared_catalog)}")
    # Summary line for the handoff.
    print("---SUMMARY---")
    print(
        f"total={len(rows)} fit@{_CATALOG_BYTE_BUDGET}={fit_count} "
        f"memory-search_rank={memory_search_pos}"
    )
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("live-catalog-sizes.csv")
    raise SystemExit(main(target))
