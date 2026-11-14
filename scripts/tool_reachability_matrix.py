#!/usr/bin/env python3
"""Honest reachability matrix: every registry tool × four talk surfaces.

Surfaces
  native_live   — NativeLiveVoiceSession: LiveTools.declarations() then
                  optional _fit_declarations under a token budget
  live_session  — LiveVoiceSession / subscription: declarations(defer_catalog=True)
  typed_chat    — BrainManager router set + find_tools shortlist
  cli           — coding CLI seat (runner_cli): Dodaios registry tools are not
                  declared; the seat uses the provider CLI's own tools

Statuses (exactly three)
  direct          — schema declared to the model on that surface
  discovery-only  — reachable only via discover_tools+call_tool (live) or
                    find_tools (chat)
  absent          — not reachable; reason names the dropping line

Never reads jarvis.toml for the catalog (AP-23). Plugin/MCP expansions that
need a live connection are listed as absent with reason "plugin not connected"
or "mcp server not connected" when the virtual loader yields nothing.

Usage:
  D:\\Projects\\Dodaios\\.venv\\Scripts\\python.exe scripts/tool_reachability_matrix.py
  D:\\Projects\\Dodaios\\.venv\\Scripts\\python.exe scripts/tool_reachability_matrix.py --csv path.csv --md path.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

SURFACES = ("native_live", "live_session", "typed_chat", "cli")


@dataclass(frozen=True)
class Cell:
    status: str  # direct | discovery-only | absent
    reason: str


@dataclass
class Row:
    name: str
    cells: dict[str, Cell]
    in_router: bool
    in_voice: bool
    in_manager: bool


def _load_manager_tools() -> dict[str, Any]:
    """ROUTER_TOOLS entry points that construct, keyed by tool.name.

    Mirrors factory filtering without spinning MissionManager / virtual
    loaders that need a live bus. Virtual loaders (cli-tools, plugin-tools,
    mcp-tools) that need runtime state are skipped and noted separately.
    """
    from jarvis.brain.factory import ROUTER_TOOLS
    from jarvis.core.entry_points import entry_points

    tools: dict[str, Any] = {}
    skipped_virtual: list[str] = []
    for ep in entry_points(group="jarvis.tool"):
        if ep.name not in ROUTER_TOOLS:
            continue
        if ep.name in {"cli-tools", "plugin-tools", "mcp-tools", "spawn-worker"}:
            skipped_virtual.append(ep.name)
            continue
        try:
            tool = ep.load()()
        except Exception:  # noqa: BLE001 — constructor needs runtime; skip
            continue
        name = str(getattr(tool, "name", "") or "").strip()
        if name:
            tools[name] = tool
    # Record the virtual loaders as named placeholders so the matrix can say
    # why their expansions are absent on a cold offline build.
    for slug in skipped_virtual:
        tools[f"__virtual__:{slug}"] = SimpleNamespace(
            name=slug,
            description=f"virtual loader {slug}",
            schema={"type": "object", "properties": {}},
            risk_tier="monitor",
            is_action_tool=False,
            execute=None,
        )
    return tools


def _gateway(manager_tools: dict[str, Any]) -> Any:
    from jarvis.brain.tool_gateway import BrainSupervisorToolGateway
    from jarvis.brain.workspace_tool import WorkspaceOrchestrationTool

    class _FakeWorkspace:
        async def run(self, args: dict[str, Any], *, trace_id: str = "") -> dict[str, Any]:
            return {"status": "ok"}

    concrete = {
        name: tool
        for name, tool in manager_tools.items()
        if not str(name).startswith("__virtual__:")
        and callable(getattr(tool, "execute", None))
    }
    manager = SimpleNamespace(_tools=concrete, _config=SimpleNamespace(computer_use=None))
    return BrainSupervisorToolGateway(
        manager,
        workspace_tool=WorkspaceOrchestrationTool(_FakeWorkspace()),
    )


def _live_bridge(gateway: Any) -> Any:
    from jarvis.live.tools import LiveTools

    return LiveTools(
        gateway,
        ledger=None,
        session_id="reachability-matrix",
        language="en",
        backend_model="matrix",
    )


def _declared_canonical(bridge: Any, *, defer: bool) -> set[str]:
    from jarvis.live.tools import _BUILTIN_TOOLS, _DIRECT_TOOLS

    declared: set[str] = set()
    for item in bridge.declarations(defer_catalog=defer):
        wire = str(item.get("name") or "")
        canonical = bridge._names.get(wire, wire)
        if canonical.startswith("jarvis_"):
            # Prefer the mapped catalog name when present.
            canonical = bridge._names.get(wire, canonical)
        declared.add(str(canonical))
        declared.add(wire)
    declared |= set(_BUILTIN_TOOLS) | set(_DIRECT_TOOLS)
    return declared


def _native_fitted(bridge: Any, budget_tokens: int = 0) -> set[str]:
    """What native voice would declare. budget_tokens=0 means unbounded fit."""
    from jarvis.live.native import (
        _DIRECT_TOOL_PREFERENCE,
        _SESSION_TOOLS,
        _fit_declarations,
        _declared_name,
    )

    declarations = [
        {key: value for key, value in declaration.items() if key != "type"}
        for declaration in bridge.declarations()
    ]
    if budget_tokens <= 0:
        return { _declared_name(d) if d.get("name") not in _SESSION_TOOLS else str(d.get("name"))
                 for d in declarations } | set(_SESSION_TOOLS)

    safe = frozenset(
        descriptor.name
        for descriptor in bridge.catalog()
        if str(getattr(descriptor, "risk_tier", "")) == "safe"
    )
    fitted = _fit_declarations(declarations, budget_tokens * 4, fillers=safe)
    names: set[str] = set()
    for d in fitted:
        name = str(d.get("name") or "")
        if name in _SESSION_TOOLS:
            names.add(name)
        else:
            names.add(_declared_name(d))
    return names


def _chat_plan(manager_tools: dict[str, Any], gateway: Any) -> tuple[set[str], set[str], set[str]]:
    """Returns (direct, deferred, callable_registry)."""
    from jarvis.brain.tool_shortlist import FIND_TOOLS_NAME, plan_surface, tool_tokens

    concrete = {
        name: tool
        for name, tool in manager_tools.items()
        if not str(name).startswith("__virtual__:")
        and callable(getattr(tool, "execute", None))
    }
    # Chat also gets the addressed workspace substitution.
    concrete = gateway.with_addressed_workspace_tool(concrete)
    # Typical cost budget from BrainManager._TOOL_SURFACE_BUDGET_TOKENS path:
    # use plan_surface with the same CORE_PRIORITY and a generous but finite
    # budget so deferred is non-empty when the catalog is large.
    budget = 8_000
    plan = plan_surface(concrete, budget_tokens=budget, max_direct=0)
    direct = set(plan.direct)
    deferred = set(plan.deferred)
    # find_tools itself is direct when anything is deferred.
    if deferred:
        direct.add(FIND_TOOLS_NAME)
    return direct, deferred, set(concrete)


def _all_names(
    manager_tools: dict[str, Any],
    voice_names: set[str],
    chat_callable: set[str],
) -> list[str]:
    from jarvis.brain.factory import ROUTER_TOOLS
    from jarvis.brain.tool_gateway import _VOICE_SUPERSEDED_TOOLS
    from jarvis.core.entry_points import entry_points
    from jarvis.live.tools import _BUILTIN_TOOLS

    names: set[str] = set()
    names |= {str(n) for n in ROUTER_TOOLS}
    names |= set(voice_names)
    names |= set(chat_callable)
    names |= set(_BUILTIN_TOOLS)
    names |= {"workspace-orchestrate", "workspace-git"}
    for ep in entry_points(group="jarvis.tool"):
        names.add(ep.name)
        try:
            tool = ep.load()()
        except Exception:  # noqa: BLE001
            continue
        n = str(getattr(tool, "name", "") or "").strip()
        if n:
            names.add(n)
    # Drop internal placeholders.
    names = {n for n in names if n and not n.startswith("__virtual__:")}
    # Superseded stay in the matrix so absent reasons are visible.
    names |= {n for n in _VOICE_SUPERSEDED_TOOLS if "-" in n or "_" in n}
    return sorted(names)


def _cell(
    *,
    name: str,
    surface: str,
    voice: set[str],
    native_direct: set[str],
    session_direct: set[str],
    chat_direct: set[str],
    chat_deferred: set[str],
    chat_callable: set[str],
    router: set[str],
    superseded: set[str],
) -> Cell:
    from jarvis.brain.tool_shortlist import canonical
    from jarvis.live.tools import _BUILTIN_TOOLS, _DIRECT_TOOLS

    cn = canonical(name)
    in_voice = name in voice or any(canonical(v) == cn for v in voice)
    in_router = name in router or any(canonical(r) == cn for r in router)

    if surface == "cli":
        # Coding CLI seats declare the provider CLI's own tools. Dodaios
        # registry tools are not on that wire unless an MCP bridge adds them
        # (out of scope for this offline matrix).
        return Cell(
            "absent",
            "jarvis/agent_chat/runner_cli.py: coding CLI seat uses provider tools, not BrainManager._tools",
        )

    if surface == "live_session":
        from jarvis.live.tools import _PRIORITY_TOOLS

        if name in _BUILTIN_TOOLS or name in _DIRECT_TOOLS:
            return Cell(
                "direct",
                "jarvis/live/tools.py:declarations defer_catalog=True always declares builtins+_DIRECT_TOOLS",
            )
        if name in _PRIORITY_TOOLS or any(canonical(p) == cn for p in _PRIORITY_TOOLS):
            if in_voice:
                return Cell(
                    "direct",
                    "jarvis/live/tools.py:declarations(defer_catalog=True) packs _PRIORITY_TOOLS as compact direct declarations",
                )
        if in_voice:
            return Cell(
                "discovery-only",
                "jarvis/live/tools.py:declarations(defer_catalog=True) — in voice catalog but outside _PRIORITY_TOOLS/_DIRECT_TOOLS; reachable via discover_tools+call_tool",
            )
        if name in superseded or any(canonical(s) == cn for s in superseded):
            return Cell(
                "absent",
                "jarvis/brain/tool_gateway.py:_VOICE_SUPERSEDED_TOOLS popped in _voice_tools()",
            )
        if in_router:
            return Cell(
                "absent",
                "jarvis/brain/tool_gateway.py:_voice_tools() — tool not present in manager._tools copy after voice filter/additions (or virtual loader not expanded)",
            )
        return Cell(
            "absent",
            "not in voice catalog and not in ROUTER_TOOLS entry-point set for this offline build",
        )

    if surface == "native_live":
        if name in _BUILTIN_TOOLS or name in _DIRECT_TOOLS or name in native_direct:
            # Builtin / direct / fitted
            if name in _BUILTIN_TOOLS or name in _DIRECT_TOOLS:
                return Cell(
                    "direct",
                    "jarvis/live/tools.py + native.py:_SESSION_TOOLS / _DIRECT_TOOLS",
                )
            if name in native_direct or any(canonical(d) == cn for d in native_direct):
                return Cell(
                    "direct",
                    "jarvis/live/tools.py:declarations() packing (_PRIORITY_TOOLS / CORE_PRIORITY) and/or native._fit_declarations",
                )
        if in_voice:
            return Cell(
                "discovery-only",
                "jarvis/live/tools.py:declarations() — in voice catalog but outside packed/fitted direct set; reachable via discover_tools+call_tool",
            )
        if name in superseded or any(canonical(s) == cn for s in superseded):
            return Cell(
                "absent",
                "jarvis/brain/tool_gateway.py:_VOICE_SUPERSEDED_TOOLS popped in _voice_tools()",
            )
        if in_router:
            return Cell(
                "absent",
                "jarvis/brain/tool_gateway.py:_voice_tools() — not in voice catalog (not in manager tools or not added)",
            )
        return Cell(
            "absent",
            "not in voice catalog for this offline build",
        )

    if surface == "typed_chat":
        if name in chat_direct or any(canonical(d) == cn for d in chat_direct):
            return Cell(
                "direct",
                "jarvis/brain/tool_shortlist.py:plan_surface direct set (CORE_PRIORITY / keep)",
            )
        if name in chat_deferred or any(canonical(d) == cn for d in chat_deferred):
            return Cell(
                "discovery-only",
                "jarvis/brain/manager.py:_shortlist_tools — deferred behind find_tools",
            )
        if name in chat_callable or any(canonical(c) == cn for c in chat_callable):
            return Cell(
                "direct",
                "in chat callable set with nothing deferred for this budget",
            )
        if name in _BUILTIN_TOOLS or name in {
            "computer", "take_appshot", "screen_snapshot", "remember",
            "product_help", "update_soul", "runtime_location", "ask_agent",
        }:
            return Cell(
                "absent",
                "jarvis/brain/factory.py:ROUTER_TOOLS — voice-only tool not on the chat router set",
            )
        if name in superseded or any(canonical(s) == cn for s in superseded):
            # computer-use IS on the router; raw primitives are not.
            if in_router:
                return Cell(
                    "direct" if name in chat_direct else (
                        "discovery-only" if name in chat_deferred else "absent"
                    ),
                    "router tool; see shortlist" if in_router else "superseded primitive",
                )
            return Cell(
                "absent",
                "not in ROUTER_TOOLS — live coordinate primitive / superseded vehicle",
            )
        if in_router:
            return Cell(
                "absent",
                "jarvis/brain/factory.py virtual loader not expanded offline (cli-tools/plugin-tools/mcp-tools) or constructor requires runtime",
            )
        return Cell(
            "absent",
            "jarvis/brain/factory.py:ROUTER_TOOLS — entry point not allowed on router tier",
        )

    return Cell("absent", f"unknown surface {surface}")


def build_matrix() -> list[Row]:
    from jarvis.brain.factory import ROUTER_TOOLS
    from jarvis.brain.tool_gateway import _VOICE_SUPERSEDED_TOOLS
    from jarvis.brain.tool_shortlist import canonical

    manager_tools = _load_manager_tools()
    gateway = _gateway(manager_tools)
    bridge = _live_bridge(gateway)
    voice = {d.name for d in bridge.catalog()}
    session_direct = _declared_canonical(bridge, defer=True)
    # Unbounded native declarations (default provider budget 0).
    native_direct = _native_fitted(bridge, budget_tokens=0)
    chat_direct, chat_deferred, chat_callable = _chat_plan(manager_tools, gateway)
    router = set(ROUTER_TOOLS) | {
        str(getattr(t, "name", n))
        for n, t in manager_tools.items()
        if not str(n).startswith("__virtual__:")
    }
    superseded = set(_VOICE_SUPERSEDED_TOOLS)

    rows: list[Row] = []
    for name in _all_names(manager_tools, voice, chat_callable):
        cells = {
            surface: _cell(
                name=name,
                surface=surface,
                voice=voice,
                native_direct=native_direct,
                session_direct=session_direct,
                chat_direct=chat_direct,
                chat_deferred=chat_deferred,
                chat_callable=chat_callable,
                router=router,
                superseded=superseded,
            )
            for surface in SURFACES
        }
        rows.append(
            Row(
                name=name,
                cells=cells,
                in_router=name in router or any(canonical(r) == canonical(name) for r in router),
                in_voice=name in voice or any(canonical(v) == canonical(name) for v in voice),
                in_manager=name in manager_tools
                or any(
                    canonical(str(getattr(t, "name", k))) == canonical(name)
                    for k, t in manager_tools.items()
                ),
            )
        )
    return rows


def _summarize(rows: list[Row]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for surface in SURFACES:
        counts = {"direct": 0, "discovery-only": 0, "absent": 0}
        for row in rows:
            counts[row.cells[surface].status] += 1
        out[surface] = counts
    return out


def _write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "in_router", "in_voice", "in_manager"]
    for surface in SURFACES:
        fields.extend([f"{surface}_status", f"{surface}_reason"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = {
                "name": row.name,
                "in_router": row.in_router,
                "in_voice": row.in_voice,
                "in_manager": row.in_manager,
            }
            for surface in SURFACES:
                cell = row.cells[surface]
                item[f"{surface}_status"] = cell.status
                item[f"{surface}_reason"] = cell.reason
            writer.writerow(item)


def _write_md(rows: list[Row], path: Path, summary: dict[str, dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Tool reachability matrix")
    lines.append("")
    lines.append("Built offline from entry points + voice gateway + shortlist.")
    lines.append("Statuses: `direct` | `discovery-only` | `absent`.")
    lines.append("")
    lines.append("## Summary counts")
    lines.append("")
    lines.append("| Surface | direct | discovery-only | absent |")
    lines.append("|---|---:|---:|---:|")
    for surface in SURFACES:
        c = summary[surface]
        lines.append(
            f"| {surface} | {c['direct']} | {c['discovery-only']} | {c['absent']} |"
        )
    lines.append("")
    lines.append("## Per-tool cells")
    lines.append("")
    lines.append(
        "| Tool | native_live | live_session | typed_chat | cli |"
    )
    lines.append("|---|---|---|---|---|")
    for row in rows:
        cells = " | ".join(row.cells[s].status for s in SURFACES)
        lines.append(f"| `{row.name}` | {cells} |")
    lines.append("")
    lines.append("## Absent / discovery-only reasons (compact)")
    lines.append("")
    for row in rows:
        interesting = [
            (s, row.cells[s])
            for s in SURFACES
            if row.cells[s].status != "direct"
        ]
        if not interesting:
            continue
        lines.append(f"### `{row.name}`")
        for surface, cell in interesting:
            lines.append(f"- **{surface}** `{cell.status}`: {cell.reason}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "_tool-reachability-matrix.csv",
    )
    parser.add_argument(
        "--md",
        type=Path,
        default=REPO_ROOT / "_tool-reachability-matrix.md",
    )
    parser.add_argument(
        "--json-summary",
        type=Path,
        default=None,
        help="Optional path for a machine-readable summary dict.",
    )
    args = parser.parse_args(argv)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    rows = build_matrix()
    summary = _summarize(rows)
    _write_csv(rows, args.csv)
    _write_md(rows, args.md, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"rows={len(rows)}")
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")
    if args.json_summary is not None:
        args.json_summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
