"""Measure LiveTools catalog wire sizes against the real Brain Manager.

No live turn, no paid provider calls. Builds the same BrainManager +
SupervisorToolGateway the desktop boot registers, then walks four catalog
surfaces the live and text paths actually send:

A) declarations(defer_catalog=False) under the 24_000-byte budget
B) declarations(defer_catalog=True) plus the discover_tools name index
C) native._fit_declarations with _DIRECT_TOOL_LIMIT = 12
D) text-brain plan_surface / BrainManager._shortlist_tools

Each CSV block and the summary carry provenance so a worktree on sys.path
cannot silently measure the self-mod trio.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import types
from datetime import date
from pathlib import Path


_MIN_TOOL_ENTRY_POINTS = 10
_SESSION_TOOL_NAMES = frozenset(
    {"end_call", "discover_tools", "call_tool", "confirm_action", "task_status"}
)


def _git_provenance(root: Path) -> tuple[str, str]:
    """Commit SHA and branch of the tree that owns this script."""
    commit = "unknown"
    branch = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        branch = subprocess.check_output(
            ["git", "-C", str(root), "branch", "--show-current"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        pass
    return commit, branch


def _wire_bytes(value: object) -> int:
    return len(json.dumps(value).encode("utf-8"))


def main(out_path: Path) -> int:
    # Use the installed Dodaios package (editable main checkout). Do not put a
    # worktree on sys.path: jarvis.core.entry_points then finds 0 tool plugins
    # and the manager keeps only the self-mod trio.
    from jarvis.core.entry_points import entry_points, invalidate

    invalidate()

    tool_eps = entry_points(group="jarvis.tool")
    entry_point_count = len(tool_eps)
    if entry_point_count < _MIN_TOOL_ENTRY_POINTS:
        print(
            f"ERROR: only {entry_point_count} jarvis.tool entry points "
            f"(need >= {_MIN_TOOL_ENTRY_POINTS}). A worktree on sys.path "
            "makes the manager keep only the self-mod trio; aborting rather "
            "than writing a silently wrong measurement.",
            file=sys.stderr,
        )
        return 2

    import jarvis
    from jarvis.brain.factory import _phase2_full_brain
    from jarvis.brain.tool_shortlist import CORE_PRIORITY, FIND_TOOLS_NAME, canonical, plan_surface
    from jarvis.core.runtime_refs import get_supervisor_tool_gateway
    from jarvis.live.native import _DIRECT_TOOL_LIMIT, _SESSION_TOOLS, _fit_declarations
    from jarvis.live.tools import (
        LiveTools,
        _CATALOG_BYTE_BUDGET,
        _DIRECT_TOOLS,
        _PRIORITY_TOOLS,
        _wire_size,
    )
    from jarvis.brain.manager import _TOOL_SURFACE_BUDGET_TOKENS

    script_root = Path(__file__).resolve().parents[1]
    commit, branch = _git_provenance(script_root)
    jarvis_file = getattr(jarvis, "__file__", "unknown")
    measured_on = date.today().isoformat()

    # Real BrainManager path (registers BrainSupervisorToolGateway + workspace tools).
    manager = _phase2_full_brain()
    gateway = get_supervisor_tool_gateway()
    if gateway is None:
        print("ERROR: supervisor tool gateway was not registered", file=sys.stderr)
        return 1
    manager_type = type(manager).__name__
    tool_count = len(getattr(manager, "_tools", {}) or {})
    print(f"gateway={type(gateway).__name__} manager={manager_type} manager_tools={tool_count}")
    print(
        f"provenance commit={commit} branch={branch} jarvis={jarvis_file} "
        f"entry_points={entry_point_count} date={measured_on} budget={_CATALOG_BYTE_BUDGET}"
    )

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

    # --- A: full catalog (defer_catalog=False), per-tool rows ---
    rows_a: list[dict[str, object]] = []
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
        rows_a.append(
            {
                "surface": "A_full_catalog",
                "name": descriptor.name,
                "wire_bytes": wire_bytes,
                "in_priority_group": descriptor.name in _PRIORITY_TOOLS,
                "canonical_rank": rank.get(canon, ""),
                "note": "",
            }
        )

    declarations_a = live.declarations(defer_catalog=False)
    wire_a = _wire_size(declarations_a)
    declared_a = len(declarations_a)
    declared_catalog_a = [
        live._names.get(item["name"], item["name"])
        for item in declarations_a
        if item["name"] not in _SESSION_TOOL_NAMES
    ]
    fit_count = len(declared_catalog_a)
    names_in_order = [row["name"] for row in rows_a]
    try:
        memory_search_pos = names_in_order.index("memory-search") + 1  # 1-based
    except ValueError:
        memory_search_pos = None

    # --- B: deferred live (defer_catalog=True) ---
    declarations_b = live.declarations(defer_catalog=True)
    wire_b = _wire_size(declarations_b)
    declared_b = len(declarations_b)
    names = sorted(descriptor.name for descriptor in catalog)
    name_index = " Available tool names: " + ", ".join(names)
    name_index_bytes = len(name_index.encode("utf-8"))
    discover_desc = ""
    for item in declarations_b:
        if item.get("name") == "discover_tools":
            discover_desc = str(item.get("description", ""))
            break
    rows_b: list[dict[str, object]] = [
        {
            "surface": "B_deferred_live",
            "name": item.get("name", ""),
            "wire_bytes": _wire_size(item),
            "in_priority_group": "",
            "canonical_rank": "",
            "note": "builtin" if item.get("name") in _SESSION_TOOL_NAMES else "direct",
        }
        for item in declarations_b
    ]
    rows_b.append(
        {
            "surface": "B_deferred_live",
            "name": "__name_index__",
            "wire_bytes": name_index_bytes,
            "in_priority_group": "",
            "canonical_rank": "",
            "note": f"names={len(names)} discover_desc_bytes={len(discover_desc.encode('utf-8'))}",
        }
    )

    # --- C: native voice _fit_declarations, limit 12 ---
    # Same stripping native._tool_declarations applies; large char budget so the
    # count limit binds, matching _DIRECT_TOOL_LIMIT = 12.
    native_source = [
        {key: value for key, value in declaration.items() if key != "type"}
        for declaration in live.declarations(defer_catalog=False)
    ]
    safe = frozenset(
        descriptor.name
        for descriptor in catalog
        if str(getattr(descriptor, "risk_tier", "")) == "safe"
    )
    fitted_c = _fit_declarations(native_source, budget_chars=10**9, fillers=safe)
    wire_c = _wire_bytes(list(fitted_c))
    catalog_order_c: list[str] = []
    for declaration in fitted_c:
        name = str(declaration.get("name", ""))
        if name in _SESSION_TOOLS:
            continue
        # Catalog tools may be aliased; canonical name is the description prefix
        # (same rule as native._declared_name).
        desc = str(declaration.get("description", ""))
        if ": " in desc:
            catalog_order_c.append(desc.split(": ", 1)[0])
        else:
            catalog_order_c.append(name)
    declared_c = len(catalog_order_c)
    rows_c: list[dict[str, object]] = [
        {
            "surface": "C_native_voice",
            "name": name,
            "wire_bytes": "",
            "in_priority_group": "",
            "canonical_rank": index + 1,
            "note": f"direct_slot limit={_DIRECT_TOOL_LIMIT}",
        }
        for index, name in enumerate(catalog_order_c)
    ]
    for declaration in fitted_c:
        name = str(declaration.get("name", ""))
        if name not in _SESSION_TOOLS:
            continue
        rows_c.append(
            {
                "surface": "C_native_voice",
                "name": name,
                "wire_bytes": _wire_bytes(declaration),
                "in_priority_group": "",
                "canonical_rank": "",
                "note": "session_builtin",
            }
        )

    # --- D: text-brain shortlist / plan_surface ---
    manager_tools = dict(getattr(manager, "_tools", {}) or {})
    d_note = ""
    declared_d: int | str
    order_d: list[str] = []
    deferred_d = 0
    rows_d: list[dict[str, object]] = []
    if not manager_tools:
        declared_d = "absent"
        d_note = (
            "absent, because BrainManager._tools was empty "
            "(no planning / tool-selection surface to measure)"
        )
    else:
        # Capability: BrainManager._shortlist_tools -> plan_surface with the
        # hosted-model cost budget (_TOOL_SURFACE_BUDGET_TOKENS). Empty prompt
        # and history, roomy window so the cost budget binds.
        brain = types.SimpleNamespace(
            context_window=128_000,
            max_tools=0,
            name="catalog-size-measure",
        )
        if hasattr(manager, "_shortlist_tools"):
            # Exercise the same BrainManager path a text turn uses, then measure
            # the pure plan_surface split under the hosted cost budget.
            manager._shortlist_tools(
                manager_tools,
                brain,
                system_prompt="",
                history=(),
            )
            plan = plan_surface(
                manager_tools,
                budget_tokens=_TOOL_SURFACE_BUDGET_TOKENS,
            )
            order_d = list(plan.direct)
            declared_d = len(plan.direct)
            deferred_d = len(plan.deferred)
            d_note = (
                f"BrainManager._shortlist_tools / plan_surface "
                f"budget_tokens={_TOOL_SURFACE_BUDGET_TOKENS} "
                f"deferred={deferred_d} finder={FIND_TOOLS_NAME}"
            )
            for index, name in enumerate(order_d):
                tool = manager_tools[name]
                rows_d.append(
                    {
                        "surface": "D_text_brain",
                        "name": name,
                        "wire_bytes": _wire_size(
                            {
                                "name": name,
                                "description": getattr(tool, "description", "") or "",
                                "input_schema": getattr(tool, "schema", None) or {},
                            }
                        ),
                        "in_priority_group": "",
                        "canonical_rank": index + 1,
                        "note": "direct",
                    }
                )
            for name in plan.deferred:
                tool = manager_tools[name]
                rows_d.append(
                    {
                        "surface": "D_text_brain",
                        "name": name,
                        "wire_bytes": _wire_size(
                            {
                                "name": name,
                                "description": getattr(tool, "description", "") or "",
                                "input_schema": getattr(tool, "schema", None) or {},
                            }
                        ),
                        "in_priority_group": "",
                        "canonical_rank": "",
                        "note": "deferred",
                    }
                )
        else:
            declared_d = "absent"
            d_note = (
                "absent, because BrainManager._shortlist_tools is missing"
            )

    provenance_fields = {
        "commit": commit,
        "branch": branch,
        "jarvis": jarvis_file,
        "entry_points": entry_point_count,
        "date": measured_on,
        "budget": _CATALOG_BYTE_BUDGET,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "surface",
        "name",
        "wire_bytes",
        "in_priority_group",
        "canonical_rank",
        "note",
        "commit",
        "branch",
        "jarvis",
        "entry_points",
        "date",
        "budget",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        # Provenance header row (empty name) so the CSV alone carries origin.
        writer.writerow(
            {
                "surface": "PROVENANCE",
                "name": "",
                "wire_bytes": "",
                "in_priority_group": "",
                "canonical_rank": "",
                "note": "measurement provenance",
                **provenance_fields,
            }
        )
        for block in (rows_a, rows_b, rows_c, rows_d):
            for row in block:
                writer.writerow({**row, **provenance_fields})

    print(f"wrote {len(rows_a) + len(rows_b) + len(rows_c) + len(rows_d)} rows -> {out_path}")
    print(f"budget={_CATALOG_BYTE_BUDGET}")
    print(f"declarations_fit_catalog_tools={fit_count}")
    print(f"memory-search_sort_position_1based={memory_search_pos}")
    if memory_search_pos is not None:
        print(f"memory-search_wire_bytes={rows_a[memory_search_pos - 1]['wire_bytes']}")
        print(f"memory-search_in_priority_group={rows_a[memory_search_pos - 1]['in_priority_group']}")
        print(f"memory-search_canonical_rank={rows_a[memory_search_pos - 1]['canonical_rank']}")
    print(f"memory-search_in_declarations={('memory-search' in declared_catalog_a)}")

    order_c_str = ",".join(catalog_order_c) if catalog_order_c else "(none)"
    if isinstance(declared_d, str):
        d_line = f"D text-brain     : {d_note}"
    else:
        d_line = f"D text-brain     : declared={declared_d}  deferred={deferred_d}  budget_tokens={_TOOL_SURFACE_BUDGET_TOKENS}"

    print("---SUMMARY---")
    print(
        f"total={len(rows_a)} fit@{_CATALOG_BYTE_BUDGET}={fit_count} "
        f"memory-search_rank={memory_search_pos}"
    )
    print(f"A full-catalog   : declared={declared_a}  wire_bytes={wire_a}  budget={_CATALOG_BYTE_BUDGET}")
    print(
        f"B deferred-live  : declared={declared_b}  wire_bytes={wire_b}  "
        f"name_index_bytes={name_index_bytes}"
    )
    print(
        f"C native-voice   : declared={declared_c}  limit={_DIRECT_TOOL_LIMIT}  "
        f"order={order_c_str}"
    )
    print(d_line)
    print(
        f"provenance: commit={commit}  jarvis={jarvis_file}  "
        f"entry_points={entry_point_count}"
    )
    # Keep wire_c referenced for completeness in verbose logs.
    print(f"C_wire_bytes={wire_c}")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("live-catalog-sizes.csv")
    raise SystemExit(main(target))
