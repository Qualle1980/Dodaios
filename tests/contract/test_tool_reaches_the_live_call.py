"""One contract across the five offline places a voice tool is registered.

A tool can be registered, in the router catalog, in the voice catalog, named
by the direct preference, inside the byte budget — and the model still never
calls it. On 2026-11-09 three tools were declared with a required ``query``
field and the model sent ``args: {}`` 177 times. On 2026-11-11 the memory
reads were in the voice catalog but the 24000-byte cut dropped them at
541/504/360 B. Every one of those landed with green tests, because each
existing test covers exactly ONE place.

The six places, and what this file asserts:

1. Registration — ``jarvis/brain/tool_gateway.py`` (the tool exists for the
   gateway: entry points, ``_voice_tools``, or the addressed ``workspace_tool``).
2. Router catalog — ``ROUTER_TOOLS`` in ``jarvis/brain/factory.py``, except
   ``workspace-orchestrate``, which is deliberately outside that set
   (ADR-0011) and is instead wired as the gateway's addressed workspace tool.
3. Voice catalog — ``LiveTools`` / ``voice_catalog()`` in
   ``jarvis/live/tools.py`` and the gateway.
4. Direct preference and aliases — ``_DIRECT_TOOL_PREFERENCE`` in
   ``jarvis/live/native.py`` and ``SUBJECT_ALIASES`` in ``jarvis/live/tools.py``
   (``NAMED_TOOLS`` is deliberately not a place of its own).
5. Byte budget — membership in ``_PRIORITY_TOOLS`` and presence in the set
   ``LiveTools.declarations()`` actually emits under ``_CATALOG_BYTE_BUDGET``.
6. The turn itself — whether the model actually calls the tool. Place 6 is
   covered only by a live turn; this file does not fake it.

**Sibling checks.** ``tests/contract/test_live_tool_registration_parity.py``
(#145) spans the same five places for vault reads and NAMED_TOOLS promises.
This file is the per-tool contract for the three tools already lost in
production (``jarvis_memory_search``, ``search_web``,
``workspace-orchestrate``), with a named assertion per place, required-argument
checks on the live declaration, and an explicit refusal to fake place 6.

**What this proves and what it does not.** The declaration check builds the
catalog the way the live path does (real entry points + the addressed
workspace tool + ``BrainSupervisorToolGateway.voice_catalog`` →
``LiveTools.declarations``). ``tests/unit/live/test_catalog_packing_respects_rank.py``
explains why a contract over the real entry points can still pass on unfixed
packing code: this environment installs about 48 of 146 tools, and the ones
evicted live may not be among them. So place 5 here proves the tool is
promised (``_PRIORITY_TOOLS``) and that it is inside the declared set for the
installed catalog under the real 24000-byte budget. It does not by itself prove
the skip-past-heavy packing defect on a full 146-tool install; that needs the
sized synthetic catalog in the packing unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from jarvis.brain.factory import ROUTER_TOOLS
from jarvis.brain.tool_gateway import BrainSupervisorToolGateway
from jarvis.brain.tool_shortlist import canonical
from jarvis.brain.workspace_tool import WorkspaceOrchestrationTool
from jarvis.core.entry_points import entry_points
from jarvis.live.native import _DIRECT_TOOL_PREFERENCE
from jarvis.live.tools import (
    SUBJECT_ALIASES,
    LiveTools,
    _PRIORITY_TOOLS,
)


@dataclass(frozen=True)
class _ToolUnderContract:
    """One tool that must survive every offline place.

    ``label`` is the name the order and the live declaration use
    (``jarvis_memory_search`` is the alias ``LiveTools._alias`` emits for
    ``memory-search``). ``canonical_name`` is the gateway / catalog name.
    """

    label: str
    canonical_name: str
    #: Name as it appears in ``ROUTER_TOOLS``, or ``None`` when the tool is
    #: deliberately outside that set (addressed workspace tool, ADR-0011).
    router_name: str | None
    #: Exact spelling expected in ``_DIRECT_TOOL_PREFERENCE``.
    preference_name: str
    #: Exact spelling expected in ``_PRIORITY_TOOLS``.
    priority_name: str
    #: When True, place 4 also requires a non-empty ``SUBJECT_ALIASES`` entry.
    needs_subject_alias: bool
    #: Required argument names the live declaration must carry (the 2026-11-09
    #: empty-schema loop). Empty means the tool honestly takes no arguments.
    required_args: tuple[str, ...]


#: Lost in production at different places: schema/args (memory), budget
#: (memory + search_web), and preference/packing (workspace-orchestrate).
_TOOLS: tuple[_ToolUnderContract, ...] = (
    _ToolUnderContract(
        label="jarvis_memory_search",
        canonical_name="memory-search",
        router_name="memory-search",
        preference_name="memory-search",
        priority_name="memory-search",
        needs_subject_alias=True,
        required_args=("query",),
    ),
    _ToolUnderContract(
        label="search_web",
        canonical_name="search_web",
        router_name="search-web",
        preference_name="search_web",
        priority_name="search_web",
        needs_subject_alias=False,
        required_args=("query",),
    ),
    _ToolUnderContract(
        label="workspace-orchestrate",
        canonical_name="workspace-orchestrate",
        router_name=None,
        preference_name="workspace-orchestrate",
        priority_name="workspace-orchestrate",
        needs_subject_alias=False,
        required_args=(),
    ),
)


class _FakeWorkspaceGateway:
    async def run(self, args: dict[str, Any], *, trace_id: str = "") -> dict[str, Any]:
        return {"status": "ok"}


def _registered_tools() -> dict[str, Any]:
    """Plugin tools that construct with no arguments, keyed by ``tool.name``.

    Matches the factory's registration key (``tool.name``), not the entry-point
    slug — ``search-web`` registers as ``search_web``.
    """
    tools: dict[str, Any] = {}
    for entry_point in entry_points(group="jarvis.tool"):
        try:
            tool = entry_point.load()()
        except Exception:  # noqa: BLE001, S112
            continue
        name = str(getattr(tool, "name", "") or "").strip()
        if name:
            tools[name] = tool
    return tools


def _live_bridge(tools: dict[str, Any] | None = None) -> LiveTools:
    """A ``LiveTools`` over the real gateway path, including the workspace tool."""
    installed = dict(tools if tools is not None else _registered_tools())
    gateway = BrainSupervisorToolGateway(
        SimpleNamespace(_tools=installed, _config=None),
        workspace_tool=WorkspaceOrchestrationTool(_FakeWorkspaceGateway()),
    )
    return LiveTools(
        gateway,
        ledger=None,
        session_id="six-place-contract",
        language="en",
        backend_model="gemini-live",
    )


def _voice_by_name(bridge: LiveTools) -> dict[str, Any]:
    return {descriptor.name: descriptor for descriptor in bridge.catalog()}


def _declared_by_canonical(bridge: LiveTools) -> dict[str, dict]:
    """Map canonical tool name → the declaration the call is given."""
    declarations = bridge.declarations()
    by_name: dict[str, dict] = {}
    for declaration in declarations:
        wire_name = str(declaration.get("name", ""))
        canonical_name = bridge._names.get(wire_name, wire_name)
        by_name[canonical_name] = declaration
        by_name[wire_name] = declaration
    return by_name


@pytest.fixture(scope="module")
def bridge() -> LiveTools:
    return _live_bridge()


@pytest.mark.parametrize("tool", _TOOLS, ids=lambda t: t.label)
def test_tool_reaches_the_live_call_across_five_places(
    tool: _ToolUnderContract,
    bridge: LiveTools,
) -> None:
    """Places 1–5 for one tool. Place 6 is a live turn only — see module docstring."""
    voice = _voice_by_name(bridge)
    declared = _declared_by_canonical(bridge)
    router = {canonical(name) for name in ROUTER_TOOLS}
    installed = _registered_tools()

    # Place 1 — registration at the source the gateway reads.
    if tool.canonical_name == "memory-search":
        assert tool.canonical_name in bridge.gateway._voice_tools(), (
            f"place 1 (registration): {tool.canonical_name} is not produced by "
            "BrainSupervisorToolGateway._voice_tools()"
        )
    elif tool.canonical_name == "search_web":
        assert tool.canonical_name in installed, (
            f"place 1 (registration): {tool.canonical_name} is not loadable "
            "from jarvis.tool entry points"
        )
    else:
        assert isinstance(bridge.gateway._workspace_tool, WorkspaceOrchestrationTool), (
            f"place 1 (registration): {tool.canonical_name} is not wired as "
            "the gateway workspace_tool"
        )

    # Place 2 — router catalog, or the addressed-workspace substitute.
    if tool.router_name is not None:
        assert canonical(tool.router_name) in router, (
            f"place 2 (router catalog): {tool.router_name} is missing from "
            "ROUTER_TOOLS (jarvis/brain/factory.py)"
        )
    else:
        # Deliberately outside ROUTER_TOOLS (ADR-0011); place 2 is the
        # addressed workspace_tool wire on the gateway.
        assert tool.canonical_name == WorkspaceOrchestrationTool.name
        assert canonical(tool.canonical_name) not in router, (
            f"place 2: {tool.canonical_name} must stay outside ROUTER_TOOLS "
            "(ADR-0011); it is the addressed workspace_tool instead"
        )

    # Place 3 — voice catalog surface used by LiveTools.
    assert tool.canonical_name in voice, (
        f"place 3 (voice catalog): {tool.canonical_name} missing from "
        "LiveTools.catalog() / voice_catalog()"
    )

    # Place 4 — direct preference, and aliases when the name has a lexical gap.
    assert tool.preference_name in _DIRECT_TOOL_PREFERENCE, (
        f"place 4 (direct preference): {tool.preference_name} missing from "
        "_DIRECT_TOOL_PREFERENCE (jarvis/live/native.py)"
    )
    if tool.needs_subject_alias:
        aliases = SUBJECT_ALIASES.get(tool.canonical_name)
        assert aliases, (
            f"place 4 (aliases): {tool.canonical_name} has no SUBJECT_ALIASES "
            "entry (jarvis/live/tools.py)"
        )

    # Place 5a — promised under the byte-budget priority set.
    assert tool.priority_name in _PRIORITY_TOOLS, (
        f"place 5 (byte budget / priority): {tool.priority_name} missing from "
        "_PRIORITY_TOOLS (jarvis/live/tools.py)"
    )

    # Place 5b — actually inside the declared set under the real 24000 budget.
    assert tool.canonical_name in declared or tool.label in declared, (
        f"place 5 (byte budget / declared set): {tool.canonical_name} was not "
        "in LiveTools.declarations() under _CATALOG_BYTE_BUDGET; the packing "
        "cut dropped it"
    )

    # Arguments on the live declaration (the 2026-11-09 empty-schema loop).
    declaration = declared.get(tool.canonical_name) or declared.get(tool.label)
    assert declaration is not None
    parameters = declaration.get("parameters") or {}
    properties = parameters.get("properties") or {}
    required = list(parameters.get("required") or [])
    descriptor = voice[tool.canonical_name]
    gateway_schema = descriptor.input_schema or {}
    gateway_required = list(gateway_schema.get("required") or [])
    gateway_properties = gateway_schema.get("properties") or {}

    if tool.required_args or gateway_required:
        assert properties, (
            f"place 5 (arguments): {tool.canonical_name} was declared with an "
            "empty parameter schema; that is the 2026-11-09 args:{{}} loop"
        )
        expected_required = tool.required_args or tuple(gateway_required)
        for field in expected_required:
            assert field in properties, (
                f"place 5 (arguments): {tool.canonical_name} declaration is "
                f"missing property {field!r}"
            )
            assert field in required, (
                f"place 5 (arguments): {tool.canonical_name} declaration does "
                f"not mark {field!r} required"
            )
            assert properties.get(field), (
                f"place 5 (arguments): {tool.canonical_name}.{field} is empty"
            )
        # Gateway and declaration must agree on required fields when both set.
        if gateway_required:
            assert set(gateway_required) <= set(required) | set(properties), (
                f"place 5 (arguments): gateway requires {gateway_required} but "
                f"the live declaration required={required} properties="
                f"{sorted(properties)}"
            )
        if gateway_properties:
            for field in expected_required:
                assert field in gateway_properties, (
                    f"place 1/5: gateway schema for {tool.canonical_name} "
                    f"lacks {field!r}"
                )
