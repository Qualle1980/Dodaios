"""The General (jarvis surface) offers only read and ask hands.

Plan S2 / chain readiness 2026-11-17: gating Write/Edit/RunCommand as ``ask``
is not enough. A model that never sees those names cannot call them. The
offered schema for a jarvis-surface chat turn must omit mutating folder hands,
shell aliases, computer use, skill/coding spawn paths, the society browser,
and write-side wiki/contact/profile tools. Handoff to coding panes stays via
``workspace-orchestrate``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.agent_chat.runner_brain import build_override, kit_payload
from jarvis.agent_chat.store import AgentChatSession
from jarvis.agent_chat.surface_kits import _jarvis_tools, kit_for
from jarvis.brain.manager import BrainManager

#: Exact names the General must never see. The RED test reports each hit.
_FORBIDDEN: frozenset[str] = frozenset(
    {
        "Write",
        "Edit",
        "RunCommand",
        "run_shell",
        "run-shell",
        "computer_use",
        "run-skill",
        "parallel",
        "coding-session",
        "society_browser",
        "contact-upsert",
        "call-contact",
        "update_profile",
    }
)

#: Read / ask hands the chat must keep when they are present. ``wiki-ingest`` is
#: here on purpose: Ahmad's Obsidian memory is written from the chat (2026-11-17).
_REQUIRED_READS: frozenset[str] = frozenset(
    {
        "wiki-ingest",
        "Read",
        "Ls",
        "Glob",
        "Grep",
        "search_web",
        "wiki-recall",
        "memory-search",
        "memory-topics",
        "memory-history",
        "wiki-list",
        "wiki-page-read",
        "contact-lookup",
        "find_tools",
        "workspace-orchestrate",
    }
)


def _session(*, surface: str = "jarvis", cwd: str = ".", permission_mode: str = "ask") -> AgentChatSession:
    return AgentChatSession(
        session_id="general-s1",
        title="",
        provider="gemini",
        model="gemini-3.5-flash-lite",
        effort="",
        cwd=cwd,
        permission_mode=permission_mode,
        vendor_session=None,
        created_ms=0,
        updated_ms=0,
        message_count=0,
        preview="",
        surface=surface,
    )


def _forbidden_in(names: set[str]) -> set[str]:
    """Return every forbidden name that appears in ``names`` (exact match)."""
    return set(names) & set(_FORBIDDEN)


class _Stub:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = name
        self.schema = {"type": "object"}
        self.risk_tier = "ask"


def test_jarvis_tools_schema_omits_mutating_hands(tmp_path: Path) -> None:
    tools = _jarvis_tools(None, None, SimpleNamespace(cwd=str(tmp_path), permission_mode="ask"))
    assert "Read" in tools
    assert "Ls" in tools
    assert "Glob" in tools
    assert "Grep" in tools
    hits = _forbidden_in(set(tools))
    assert hits == set(), f"General _jarvis_tools still offers forbidden tools: {sorted(hits)}"


def test_jarvis_tool_filter_names_each_forbidden_hand() -> None:
    """Synthetic merged set: every forbidden name must be stripped; reads stay."""
    jarvis_filter = kit_for("jarvis").tool_filter
    assert jarvis_filter is not None
    merged = {name: _Stub(name) for name in sorted(_FORBIDDEN | _REQUIRED_READS)}
    offered = jarvis_filter(merged)  # type: ignore[arg-type]
    hits = _forbidden_in(set(offered))
    assert hits == set(), f"General filter left forbidden tools: {sorted(hits)}"
    missing = sorted(name for name in _REQUIRED_READS if name not in offered)
    assert missing == [], f"General filter dropped required read/ask hands: {missing}"


async def test_kit_payload_for_jarvis_omits_mutating_hands(tmp_path: Path) -> None:
    tools, _extra = await kit_payload(_session(cwd=str(tmp_path)), SimpleNamespace(_config=object()))
    assert tools is not None
    assert "Read" in tools
    hits = _forbidden_in(set(tools))
    assert hits == set(), f"General kit_payload still offers forbidden tools: {sorted(hits)}"
    assert "coding-session" not in tools


def test_build_override_tools_extra_for_jarvis_omits_mutating_hands(tmp_path: Path) -> None:
    override = build_override(
        _session(cwd=str(tmp_path)),
        brain=None,
        stance="ask",
        cwd=tmp_path,
        ref="r",
    )
    assert "Read" in override.tools_extra
    hits = _forbidden_in(set(override.tools_extra))
    assert hits == set(), f"General tools_extra still offers forbidden tools: {sorted(hits)}"


def test_offered_jarvis_surface_strips_brain_shell_aliases(tmp_path: Path) -> None:
    """Brain plugins may still register mutating tools; the General must not see them."""
    kit = kit_for("jarvis")
    kit_tools = _jarvis_tools(None, None, SimpleNamespace(cwd=str(tmp_path), permission_mode="ask"))
    override = build_override(
        _session(cwd=str(tmp_path)),
        brain=None,
        stance="ask",
        cwd=tmp_path,
        ref="r",
        kit_tools=kit_tools,
    )
    brain_tools = {name: _Stub(name) for name in ("wiki-recall", "run_shell", "run-shell", "computer_use", "wiki-ingest")}
    offered = BrainManager._apply_turn_override_tools(
        SimpleNamespace(),  # type: ignore[arg-type]
        brain_tools,  # type: ignore[arg-type]
        override,
    )
    assert "Read" in offered or "wiki-recall" in offered
    hits = _forbidden_in(set(offered))
    assert hits == set(), f"Merged General offer still has forbidden tools: {sorted(hits)}"
    assert kit.tool_filter is not None or kit.session_tool_filter is not None
