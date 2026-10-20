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
    searches: list[int] = []

    def _lookup():
        searches.append(1)
        return _LEAD

    mode, decision = mgr._central_lead(True, _lookup)

    assert (mode, decision) == ("off", None)
    assert not searches, "off must not even cost a candidate search"
    # The legacy branches still route the turn, unchanged.
    assert mgr._build_fallback_chain("fast")[0] == _LEAD


def test_an_unknown_mode_falls_back_to_shadow() -> None:
    mgr = _manager("nonsense")

    mode, _ = mgr._central_lead(True, lambda: _LEAD)

    assert mode == "shadow"


def test_a_real_divergence_is_logged_loudly(caplog) -> None:
    """The alarm itself: the two sides answering differently must be visible."""
    from jarvis.brain.routing_policy import LeadDecision

    mgr = _manager("shadow")
    decision = LeadDecision(("gemini", "g"), True, "a drifted policy")

    with caplog.at_level(logging.WARNING, logger="jarvis.brain.manager"):
        mgr._audit_central_lead("shadow", decision, _LEAD, True)

    assert any("SHADOW divergence" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------
# One candidate search per turn (finding a lead instantiates brains).
# --------------------------------------------------------------------------

def _count_searches(mgr: BrainManager) -> list[int]:
    """Counts candidate searches without changing what they return."""
    original = mgr._first_tool_capable_provider
    searches: list[int] = []

    def _counted(level, **kw):
        searches.append(1)
        return original(level, **kw)

    mgr._first_tool_capable_provider = _counted  # type: ignore[assignment]
    return searches


def test_shadow_mode_searches_for_a_candidate_exactly_once() -> None:
    """The policy and the branches beside it ran a search each: 2 instead of 1."""
    mgr = _manager("shadow")
    searches = _count_searches(mgr)

    chain = mgr._build_fallback_chain("fast")

    assert len(searches) == 1, f"expected one candidate search, got {len(searches)}"
    assert chain[0] == _LEAD


def test_a_tool_capable_talker_costs_no_candidate_search_in_any_mode() -> None:
    """An ordinary chat turn on a tool-capable talker: no lead, no search."""
    for mode in ("off", "shadow", "on"):
        mgr = _manager(mode)
        mgr._active_can_call_tools = lambda: True  # type: ignore[assignment]
        searches = _count_searches(mgr)

        chain = mgr._build_fallback_chain("fast")

        assert not searches, mode
        assert mgr._router_lead_key is None, mode
        assert chain[0][0] == "chatgpt-web", mode


# --------------------------------------------------------------------------
# "on" is authoritative.
# --------------------------------------------------------------------------

def test_on_with_no_lead_does_not_fall_back_to_the_legacy_branches() -> None:
    """A no-lead decision must END the decision, not pass it to the old path.

    The legacy branches would find a perfectly good lead here, which is the
    point: in "on" the policy alone decides, so its refusal has to hold.
    """
    from jarvis.brain.routing_policy import LeadDecision

    mgr = _manager("on")
    mgr._central_lead = (  # type: ignore[assignment]
        lambda intelligent, candidate: (
            "on", LeadDecision(None, False, "policy says no")
        )
    )

    chain = mgr._build_fallback_chain("fast")

    assert mgr._router_lead_key is None
    assert all(name != "ollama" for name, _ in chain), chain
    assert chain[0][0] == "chatgpt-web"


def test_on_with_a_lead_installs_exactly_that_lead() -> None:
    mgr = _manager("on")

    chain = mgr._build_fallback_chain("fast")

    assert mgr._router_lead_key == _LEAD
    assert chain[0] == _LEAD
    assert [name for name, _ in chain].count("ollama") == 1, "no double lead"


def test_an_action_turn_with_the_router_off_does_not_fall_through() -> None:
    """The other fall-through shape: the lead answers the whole turn."""
    mgr = _manager("on")
    mgr._config.brain.routing.intelligent_router = False
    mgr._turn_substantive = False  # type: ignore[attr-defined]
    mgr._turn_needs_tools = True  # type: ignore[attr-defined]

    chain = mgr._build_fallback_chain("fast")

    assert chain[0] == _LEAD
    assert mgr._router_lead_key is None, (
        "no fall-through: the lead owns this answer, the talker does not speak"
    )


def test_on_keeps_the_plan_seat_off_every_paid_provider() -> None:
    """The seat guard is untouched by who decides the lead."""
    mgr = _manager("on")
    mgr._registry._classes["gemini"] = object  # type: ignore[attr-defined]
    mgr._fast_model = (  # type: ignore[method-assign]
        lambda name: "qwen3.5:4b" if name == "ollama" else "paid-model"
    )
    mgr._brain_cache[("gemini", "paid-model")] = _ToolCapableBrain()

    chain = mgr._build_fallback_chain("fast")

    assert [name for name, _ in chain] == ["ollama", "chatgpt-web"]


# --------------------------------------------------------------------------
# The shadow comparison covers more than the provider name.
# --------------------------------------------------------------------------

def _divergence(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if "SHADOW divergence" in r.getMessage()
    ]


def test_a_drifted_model_on_the_same_provider_is_a_divergence(caplog) -> None:
    from jarvis.brain.routing_policy import LeadDecision

    mgr = _manager("shadow")
    decision = LeadDecision(("ollama", "some-other-model"), True, "drifted")

    with caplog.at_level(logging.WARNING, logger="jarvis.brain.manager"):
        mgr._audit_central_lead("shadow", decision, _LEAD, True)

    assert _divergence(caplog), "a different model is a different lead"
    assert "model:" in _divergence(caplog)[0]


def test_a_drifted_fall_through_is_a_divergence(caplog) -> None:
    """Same lead, but the talker would no longer get the last word."""
    from jarvis.brain.routing_policy import LeadDecision

    mgr = _manager("shadow")
    decision = LeadDecision(_LEAD, False, "drifted")

    with caplog.at_level(logging.WARNING, logger="jarvis.brain.manager"):
        mgr._audit_central_lead("shadow", decision, _LEAD, True)

    assert _divergence(caplog)
    assert "falls through" in _divergence(caplog)[0]


def test_a_missing_lead_on_one_side_is_a_divergence(caplog) -> None:
    from jarvis.brain.routing_policy import LeadDecision

    mgr = _manager("shadow")

    with caplog.at_level(logging.WARNING, logger="jarvis.brain.manager"):
        mgr._audit_central_lead(
            "shadow", LeadDecision(None, False, "none"), _LEAD, False
        )

    assert _divergence(caplog)
    assert "lead present:" in _divergence(caplog)[0]


def test_an_identical_decision_logs_nothing(caplog) -> None:
    from jarvis.brain.routing_policy import LeadDecision

    mgr = _manager("shadow")

    with caplog.at_level(logging.WARNING, logger="jarvis.brain.manager"):
        mgr._audit_central_lead("shadow", LeadDecision(_LEAD, True, "same"), _LEAD, True)

    assert not _divergence(caplog)


def test_the_divergence_warning_carries_only_routing_shape(caplog) -> None:
    """Provider and model names are routing shape; the turn text is not logged."""
    from jarvis.brain.routing_policy import LeadDecision

    mgr = _manager("shadow")

    with caplog.at_level(logging.WARNING, logger="jarvis.brain.manager"):
        mgr._audit_central_lead(
            "shadow", LeadDecision(("gemini", "g"), True, "drifted"), _LEAD, True
        )

    message = _divergence(caplog)[0]
    assert "gemini" in message and "ollama" in message
    assert "qwen3.5:4b" in message
