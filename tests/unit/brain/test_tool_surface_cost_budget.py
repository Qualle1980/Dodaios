"""A roomy window is not a reason to send every tool schema every turn.

Live measurement (2026-10-30, the maintainer's install): 24 voice turns,
712_650 input tokens, 29_694 per turn, against 101 tokens of answer. The
SMALLEST turn of the day was 27_174, so the weight was not the conversation.
Summed over the registered `jarvis.tool` entry points, the tool schemas alone
are ~15_000 tokens -- the largest single item in every request.

`_shortlist_tools` already defers tools, and `find_tools` already reaches the
deferred ones, but all three triggers were about *fitting*: a small window, a
provider's tool-count cap, or a caller's declared role. `chatgpt-plan` declares
128_000 tokens, so there was "room to spare" and nothing was ever deferred.

Room to spare is not free. A hosted model with a large window still bills every
schema, on every turn, whether or not the model looks at it. Cost is now a
trigger of its own, and the one that fires on an ordinary cloud turn.
"""
from __future__ import annotations

from typing import Any

from jarvis.brain.manager import _TOOL_SURFACE_BUDGET_TOKENS, BrainManager
from jarvis.brain.tool_shortlist import FIND_TOOLS_NAME, tool_tokens


class _FakeBrain:
    def __init__(self, window: int) -> None:
        self.context_window = window
        self.name = "fake"


def _tool(name: str, *, size: int = 60) -> Any:
    class _T:
        pass

    t = _T()
    t.name = name
    t.description = "d" * size
    t.schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    return t


def _mgr() -> BrainManager:
    m = BrainManager.__new__(BrainManager)  # bypass heavy __init__
    m._tools = {}
    m._evidence_required_tool = None
    m._skill_turn_match = None
    return m


def _shown(surface: dict, deferred: frozenset) -> dict:
    return {n: t for n, t in surface.items() if n not in deferred and n != FIND_TOOLS_NAME}


def test_a_cloud_window_no_longer_buys_the_whole_catalogue() -> None:
    """The live shape: a 128k window and a catalogue far over the budget."""
    tools = {f"t{i}": _tool(f"t{i}", size=400) for i in range(60)}
    surface, deferred = _mgr()._shortlist_tools(
        tools, _FakeBrain(128_000), system_prompt="x" * 28_000, history=None
    )

    assert deferred, "a catalogue over the budget must not be sent whole"
    shown = _shown(surface, deferred)
    assert sum(tool_tokens(n, t) for n, t in shown.items()) <= _TOOL_SURFACE_BUDGET_TOKENS

    # Deferred, never dropped: every tool is still present and still callable,
    # and the finder is there to reveal the rest.
    assert set(tools) <= set(surface)
    assert FIND_TOOLS_NAME in surface


def test_a_catalogue_inside_the_budget_is_left_alone() -> None:
    """No finder, no indirection, no behaviour change for a modest surface."""
    tools = {f"t{i}": _tool(f"t{i}", size=40) for i in range(5)}
    surface, deferred = _mgr()._shortlist_tools(
        tools, _FakeBrain(128_000), system_prompt="x" * 1_000, history=None
    )
    assert surface is tools
    assert not deferred


def test_a_pinned_tool_is_still_shown_over_the_budget() -> None:
    """The budget must not be able to hide the tool a gate mandated."""
    m = _mgr()
    m._evidence_required_tool = "t59"
    tools = {f"t{i}": _tool(f"t{i}", size=400) for i in range(60)}
    surface, deferred = m._shortlist_tools(
        tools, _FakeBrain(128_000), system_prompt="x" * 28_000, history=None
    )
    assert "t59" not in deferred


def test_a_small_window_still_trims_harder_than_the_budget() -> None:
    """The window remains the binding limit when it is the smaller one."""
    tools = {f"t{i}": _tool(f"t{i}", size=400) for i in range(60)}
    small, small_deferred = _mgr()._shortlist_tools(
        tools, _FakeBrain(8_000), system_prompt="x" * 4_000, history=None
    )
    roomy, roomy_deferred = _mgr()._shortlist_tools(
        tools, _FakeBrain(128_000), system_prompt="x" * 4_000, history=None
    )
    assert len(small_deferred) >= len(roomy_deferred)
    assert len(_shown(small, small_deferred)) <= len(_shown(roomy, roomy_deferred))
