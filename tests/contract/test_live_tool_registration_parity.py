"""The five places that must agree before a model is given a tool.

A voice tool is not registered once. It is registered FIVE times, in five
files, and nothing in this repo checked that the five agree:

1. ``ROUTER_TOOLS``                 -- jarvis/brain/factory.py
2. ``_voice_tools()`` / the catalog -- jarvis/brain/tool_gateway.py
3. ``_DIRECT_TOOL_PREFERENCE``      -- jarvis/live/native.py
4. ``SUBJECT_ALIASES``              -- jarvis/live/tools.py
5. ``_PRIORITY_TOOLS``              -- jarvis/live/tools.py (the byte budget)

Each of the five has already been the single missing one:

- 2026-11-07: the catalog was sorted by NAME, so the budget cut at "m" and
  took search-web and wiki-recall with it.
- 2026-11-09: the three memory tools declared ``input_schema`` where the
  describer reads ``schema``, so the model was told they take no arguments
  and called one 177 times with ``args: {}`` (#142).
- 2026-11-08 / #145: the same tools were absent from ``_DIRECT_TOOL_PREFERENCE``
  and from ``SUBJECT_ALIASES``, so the corrected schema still never reached a
  live call -- place four was two places.
- 2026-11-11: they were in every list and the BYTE BUDGET still dropped them,
  because the packing loop skipped a tool that did not fit and kept filling
  from further down the order. ``workspace-orchestrate`` lost its slot that
  way, and a voice turn could not delegate at all.

Four fixes, four green test suites, and each time the next place was still
wrong -- because every existing test covers exactly one of the five in
isolation. This file is the one that spans them.
"""

from __future__ import annotations

from typing import Any

import pytest

from jarvis.brain.factory import ROUTER_TOOLS
from jarvis.brain.tool_gateway import BrainSupervisorToolGateway
from jarvis.brain.tool_shortlist import canonical
from jarvis.core.entry_points import entry_points
from jarvis.live.coordination import NAMED_TOOLS
from jarvis.live.native import _DIRECT_TOOL_PREFERENCE, _SESSION_TOOLS
from jarvis.live.tools import SUBJECT_ALIASES, LiveTools, _PRIORITY_TOOLS

#: ADR-0044's read side. The questions these answer ("what do you know about
#: me", "where did I live before") are among the most frequent spoken turns,
#: and every one of the four failures above cost them a live call.
_VAULT_READS = ("memory-search", "memory-history", "memory-topics")


class _Manager:
    def __init__(self, tools: dict[str, Any]) -> None:
        self._tools = dict(tools)
        self._config = None


def _registered_tools() -> dict[str, Any]:
    """Every plugin tool that builds with no arguments, from the entry points.

    A fake catalog cannot overflow the real byte budget, which is the whole
    point of the declaration assertions below.
    """
    tools: dict[str, Any] = {}
    for entry_point in entry_points(group="jarvis.tool"):
        try:
            tools[entry_point.name] = entry_point.load()()
        except Exception:  # noqa: BLE001, S112
            # Needs constructor arguments: not part of this measurement, and
            # therefore not one of the names asserted on.
            continue
    return tools


def _declared_names() -> set[str]:
    tools = _registered_tools()
    if not tools:
        pytest.skip("no plugin tool is installed in this environment")
    live = LiveTools(
        BrainSupervisorToolGateway(_Manager(tools)),
        ledger=None,
        session_id="test",
        language="de",
        backend_model="gemini-live",
    )
    declarations = live.declarations()
    return {live._names.get(d["name"], d["name"]) for d in declarations}


def _canonical(names: object) -> set[str]:
    return {canonical(str(name)) for name in names}  # type: ignore[union-attr]


# --- static agreement between the five places -------------------------------


@pytest.mark.parametrize("tool", NAMED_TOOLS)
def test_a_tool_the_rules_name_is_promised_by_the_byte_budget(tool: str) -> None:
    """The conversation rules tell the model to use these tools by name.

    Being ranked is not being declared: only ``_PRIORITY_TOOLS`` survives the
    cut. A model told to use a tool it was never given says the tool does not
    exist.
    """
    assert tool in _PRIORITY_TOOLS, (
        f"{tool} is named in the conversation rules (NAMED_TOOLS) but is not "
        "in _PRIORITY_TOOLS, so the byte budget may drop it"
    )


@pytest.mark.parametrize("tool", [n for n in NAMED_TOOLS if n not in _SESSION_TOOLS])
def test_a_tool_the_rules_name_is_preferred_on_the_native_path(tool: str) -> None:
    """``_DIRECT_TOOL_PREFERENCE`` is the native path's own ordering.

    Session built-ins are excluded by that list's own construction: they are
    declared whatever the budget, so they need no preference slot.
    """
    assert tool in _DIRECT_TOOL_PREFERENCE, (
        f"{tool} is named in the conversation rules but absent from "
        "_DIRECT_TOOL_PREFERENCE (jarvis/live/native.py)"
    )


@pytest.mark.parametrize("tool", _VAULT_READS)
def test_the_vault_reads_are_registered_in_all_five_places(tool: str) -> None:
    """The exact failure of 2026-11-08 and of #145: present in some, not all."""
    # ROUTER_TOOLS spells its names with hyphens and canonical() normalises
    # them to underscores, so both sides go through it.
    assert canonical(tool) in _canonical(ROUTER_TOOLS), f"{tool} missing from ROUTER_TOOLS"
    assert tool in _DIRECT_TOOL_PREFERENCE, f"{tool} missing from _DIRECT_TOOL_PREFERENCE"
    assert tool in SUBJECT_ALIASES, (
        f"{tool} has no SUBJECT_ALIASES entry: nobody asks for a 'memory', "
        "they ask what you know about them, so a names-only index never "
        "reaches it"
    )
    assert tool in _PRIORITY_TOOLS, f"{tool} missing from _PRIORITY_TOOLS"


# --- the declaration itself, against the real catalog -----------------------


def test_every_promised_tool_is_actually_declared() -> None:
    """The assertion the 2026-11-11 eviction would have failed.

    Three tools were evicted that session while every list above was correct,
    because the packing loop skipped what did not fit instead of stopping.
    This is the only check that reads what the call was GIVEN.
    """
    installed = _registered_tools()
    declared = _declared_names()
    promised = {
        name for name in _PRIORITY_TOOLS
        if name in installed or canonical(name) in _canonical(installed)
    }
    if not promised:
        pytest.skip("no promised tool is installed in this environment")
    missing = sorted(
        name for name in promised
        if name not in declared and canonical(name) not in _canonical(declared)
    )
    assert not missing, (
        "promised tools were not declared to the call: "
        f"{', '.join(missing)}. The byte budget evicted them; raise it, "
        "shrink a schema, or stop promising them -- do not let a model be "
        "told to use a tool it was not given."
    )
