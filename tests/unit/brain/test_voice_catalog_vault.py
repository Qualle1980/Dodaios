"""A call can read the long-term vault.

Live 2026-11-07 21:22, asked whether it could reach the owner's Obsidian, the
model said no. That answer was HONEST: ``wiki-recall`` is in the router tool
list (``jarvis/brain/factory.py``) and appeared nowhere in the voice catalog,
which that session reported as "20 of 20 tools". The text brain could search
the vault; a call could not.

So voice gets the tool the same way it already gets ``remember``,
``product_help``, ``update_soul`` and ``show_on_screen``: named explicitly in
``_voice_tools()``, because the shared set does not carry it.
"""

from __future__ import annotations

from typing import Any

from jarvis.brain.tool_gateway import BrainSupervisorToolGateway


class _Manager:
    def __init__(self, tools: dict[str, Any] | None = None) -> None:
        self._tools = dict(tools or {})
        self._config = None


def _gateway(**kwargs: Any) -> BrainSupervisorToolGateway:
    return BrainSupervisorToolGateway(_Manager(), **kwargs)


def test_voice_can_search_the_vault() -> None:
    names = {d.name for d in _gateway().voice_catalog()}
    assert "wiki-recall" in names


def test_the_vault_search_is_built_once() -> None:
    """A fresh VaultSearch per call would pay the index build every time.

    Measured 2026-11-07: a cold instance over the six configured sources cost
    724.5 ms, of which 430.8 ms was index building (#119 moved that to a boot
    warm-up). Rebuilding it per tool lookup would hand that cost back.
    """
    gateway = _gateway()
    first = {d.name: d for d in gateway.voice_catalog()}["wiki-recall"]
    second = {d.name: d for d in gateway.voice_catalog()}["wiki-recall"]
    assert first.name == second.name

    tools_a = gateway._voice_tools()
    tools_b = gateway._voice_tools()
    assert tools_a["wiki-recall"] is tools_b["wiki-recall"]


def test_the_tool_carries_the_external_sources() -> None:
    """The lessons the workers write live in read-only external sources.

    Without them a call searches a 12-file vault and finds nothing the user
    cares about -- which is how this looked like "no Obsidian access".
    """
    tool = _gateway()._voice_tools()["wiki-recall"]
    search = tool._search
    assert hasattr(search, "_external")


def test_a_failure_to_build_it_does_not_empty_the_catalog() -> None:
    """A vault that cannot be resolved must cost one tool, not the call."""
    import jarvis.plugins.tool.wiki_recall as wiki_recall

    original = wiki_recall._build_search_instance

    def boom() -> Any:
        raise RuntimeError("no vault here")

    wiki_recall._build_search_instance = boom
    try:
        names = {d.name for d in _gateway().voice_catalog()}
    finally:
        wiki_recall._build_search_instance = original
    assert "wiki-recall" not in names
    assert "remember" in names, "the rest of the voice catalog must survive"

def test_voice_can_read_the_personal_memory_vault() -> None:
    """ADR-0044's read tools, in a call and not only in the chat.

    The same trap as `wiki-recall` above: `_voice_tools()` is hand-curated, so
    a tool the text brain picks up from the entry points reaches a call only
    when it is named there. "What do you know about me?" has to be answerable
    by voice.
    """
    names = {d.name for d in _gateway().voice_catalog()}

    assert {"memory-search", "memory-history", "memory-topics"} <= names


def test_the_memory_tools_are_read_only_and_safe_tier() -> None:
    """They run on the voice critical path: one FTS query, no model call."""
    catalog = {d.name: d for d in _gateway().voice_catalog()}

    for name in ("memory-search", "memory-history", "memory-topics"):
        assert catalog[name].risk_tier == "safe"
        assert catalog[name].is_action_tool is False
