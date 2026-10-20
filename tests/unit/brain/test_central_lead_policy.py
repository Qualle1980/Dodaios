"""The central lead policy, its rollout flag, and the shadow comparison.

Phase 3 of the orchestrator rebuild: the "who leads this turn" decision moved
out of the chain assembly into one pure function. These tests pin the function
itself, then prove the flag's three modes — ``shadow`` (decide and compare,
behaviour unchanged), ``on`` (the policy steers), ``off`` (not evaluated).

Deterministic — no provider call, no network, no model.
"""
from __future__ import annotations

import logging

from jarvis.brain.manager import BrainManager
from jarvis.brain.routing_policy import TurnFacts, decide_lead
from jarvis.core.bus import EventBus
from jarvis.core.config import JarvisConfig

_LEAD = ("ollama", "qwen3.5:4b")


def _facts(**over: object) -> TurnFacts:
    base: dict[str, object] = {
        "active": "chatgpt-web",
        "active_can_call_tools": False,
        "substantive": True,
        "needs_tools": False,
        "intelligent_router": True,
    }
    base.update(over)
    return TurnFacts(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The policy itself.
# --------------------------------------------------------------------------

def test_a_tool_capable_talker_is_never_led() -> None:
    calls: list[int] = []

    def _candidate():
        calls.append(1)
        return _LEAD

    decision = decide_lead(_facts(active_can_call_tools=True), _candidate)

    assert not decision.leads
    assert not calls, "a candidate search costs brains: never run it unused"


def test_the_router_lead_falls_through_to_the_talker() -> None:
    decision = decide_lead(_facts(), lambda: _LEAD)

    assert decision.lead == _LEAD
    assert decision.falls_through, "the talker must keep the voice it was picked for"


def test_an_action_turn_with_the_router_off_is_answered_by_the_lead() -> None:
    decision = decide_lead(
        _facts(intelligent_router=False, substantive=True, needs_tools=True),
        lambda: _LEAD,
    )

    assert decision.lead == _LEAD
    assert not decision.falls_through


def test_a_plain_turn_is_not_led_and_costs_no_candidate_search() -> None:
    calls: list[int] = []

    def _candidate():
        calls.append(1)
        return _LEAD

    decision = decide_lead(
        _facts(intelligent_router=False, substantive=False, needs_tools=False),
        _candidate,
    )

    assert not decision.leads
    assert not calls


def test_no_lead_when_nothing_may_lead_this_talker() -> None:
    """A plan seat with no permitted candidate: honest toolless turn."""
    decision = decide_lead(_facts(), lambda: None)

    assert not decision.leads
    assert "may lead" in decision.reason


def test_the_talker_itself_is_not_its_own_lead() -> None:
    decision = decide_lead(_facts(), lambda: ("chatgpt-web", "x"))

    assert not decision.leads


# --------------------------------------------------------------------------
# The rollout flag.
# --------------------------------------------------------------------------

class _ToolCapableBrain:
    supports_tools = True


def _manager(mode: str) -> BrainManager:
    config = JarvisConfig()
    config.brain.primary = "chatgpt-web"
    config.brain.routing.central_lead_policy = mode  # type: ignore[assignment]
    mgr = BrainManager(config=config, bus=EventBus(), tools={})
    mgr._active_name = "chatgpt-web"  # type: ignore[attr-defined]
    mgr._active_can_call_tools = lambda: False  # type: ignore[assignment]
    mgr._fast_model = lambda name: "qwen3.5:4b" if name == "ollama" else None  # type: ignore[method-assign]
    mgr._deep_model = lambda name: "qwen3.5:4b" if name == "ollama" else None  # type: ignore[method-assign]
    mgr._registry._loaded = True
    mgr._registry._classes = {"chatgpt-web": object, "ollama": object}  # type: ignore[attr-defined]
    mgr._brain_cache[("ollama", "qwen3.5:4b")] = _ToolCapableBrain()
    mgr._turn_substantive = True  # type: ignore[attr-defined]
    return mgr


def test_shadow_mode_changes_nothing_and_reports_no_divergence(caplog) -> None:
    mgr = _manager("shadow")
    with caplog.at_level(logging.WARNING, logger="jarvis.brain.manager"):
        chain = mgr._build_fallback_chain("fast")

    assert mgr._router_lead_key == _LEAD
    assert [name for name, _ in chain][:2] == ["ollama", "chatgpt-web"]
    assert not [r for r in caplog.records if "SHADOW divergence" in r.getMessage()], (
        "the central policy must reproduce today's routing exactly"
    )


def test_the_policy_steers_the_turn_when_switched_on() -> None:
    mgr = _manager("on")

    chain = mgr._build_fallback_chain("fast")

    assert mgr._router_lead_key == _LEAD
    assert [name for name, _ in chain][:2] == ["ollama", "chatgpt-web"]


def test_switched_off_the_policy_is_not_evaluated() -> None:
    mgr = _manager("off")
    mgr._first_tool_capable_provider = (  # type: ignore[assignment]
        lambda level, **kw: _LEAD
    )

    mode, decision = mgr._central_lead("fast", True)

    assert (mode, decision) == ("off", None)
    # The legacy branches still route the turn, unchanged.
    assert mgr._build_fallback_chain("fast")[0] == _LEAD


def test_an_unknown_mode_falls_back_to_shadow() -> None:
    mgr = _manager("nonsense")

    mode, _ = mgr._central_lead("fast", True)

    assert mode == "shadow"


def test_a_real_divergence_is_logged_loudly(caplog) -> None:
    """The alarm itself: the two sides answering differently must be visible."""
    from jarvis.brain.routing_policy import LeadDecision

    mgr = _manager("shadow")
    decision = LeadDecision(("gemini", "g"), True, "a drifted policy")

    with caplog.at_level(logging.WARNING, logger="jarvis.brain.manager"):
        mgr._audit_central_lead("shadow", decision, _LEAD)

    assert any("SHADOW divergence" in r.getMessage() for r in caplog.records)
