"""One place that decides who LEADS a turn.

Phase 3 of the orchestrator rebuild. "Who answers this turn" used to be
decided by branches inside :meth:`BrainManager._build_fallback_chain` only,
mixed into the chain assembly. Three things read the same facts afterwards —
the chain walk, the tool fall-through gate and the seat filter — and every fix
had to re-derive the intent from the chain's shape. Phase 2 was exactly that
failure: a lead was chosen, silently dropped by a filter further down, and the
turn ran toolless with nothing reporting a decision had been reverted.

This module holds the decision itself: frozen facts in, a frozen decision out,
with the reason it was taken. It performs no I/O, reads no config, and never
calls a model — so a caller can evaluate it in SHADOW mode (decide, compare,
log, act on the old path) before it is allowed to steer a turn.

The candidate lead is passed as a lookup rather than a value because finding
one instantiates brains: a turn whose talker can call tools must not pay for
a lead it will not use. The lookup is called at most once.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

#: A (provider, model) pair, as the fallback chain uses throughout.
Candidate = tuple[str, str | None]

#: The modes the config flag accepts. ``shadow`` decides and logs without
#: steering the turn; ``off`` does not even decide.
MODES = ("off", "shadow", "on")


@dataclass(frozen=True, slots=True)
class TurnFacts:
    """Everything the lead decision is allowed to depend on."""

    #: The talker the user picked, and whether it can emit tool_calls now.
    active: str
    active_can_call_tools: bool
    #: Whether this turn is substantive (not an ack, not smalltalk) and
    #: whether a deterministic detector saw an action/tool intent.
    substantive: bool
    needs_tools: bool
    #: The intelligent-router kill switch (brain.routing.intelligent_router).
    intelligent_router: bool


@dataclass(frozen=True, slots=True)
class LeadDecision:
    """Who leads, whether the talker still gets the last word, and why."""

    lead: Candidate | None
    #: True when the lead exists only to pick a tool: if it picks none, the
    #: turn falls through to the talker so the user keeps the voice they
    #: chose. False when the lead answers the whole turn.
    falls_through: bool
    reason: str

    @property
    def leads(self) -> bool:
        return self.lead is not None


_NO_LEAD_TALKER_CAPABLE = "the talker picks its own tools"
_NO_LEAD_NO_CANDIDATE = "no reachable provider may lead this talker"
_NO_LEAD_NOT_WANTED = "a plain turn: nothing to lead"
_LEAD_ROUTER = "intelligent router: the lead picks the tool, the talker keeps the voice"
_LEAD_ACTION = "action intent with the router off: the lead answers the whole turn"


def decide_lead(
    facts: TurnFacts, candidate: Callable[[], Candidate | None]
) -> LeadDecision:
    """The lead for one turn.

    Args:
        facts: the turn's capability and intent facts.
        candidate: returns the preferred provider that may lead this talker,
            already filtered for capability, health and seat policy by the
            caller. Called at most once, and only when a lead is wanted.
    """
    if facts.active_can_call_tools:
        return LeadDecision(None, False, _NO_LEAD_TALKER_CAPABLE)

    # The talker cannot emit tool_calls. Either the router leads every
    # substantive turn and falls through when it picks no tool, or — with the
    # kill switch off — only a deterministic action intent delegates, and then
    # the lead owns the whole answer.
    if facts.intelligent_router and facts.substantive:
        falls_through, reason = True, _LEAD_ROUTER
    elif facts.needs_tools:
        falls_through, reason = False, _LEAD_ACTION
    else:
        return LeadDecision(None, False, _NO_LEAD_NOT_WANTED)

    pick = candidate()
    if pick is None or pick[0] == facts.active:
        return LeadDecision(None, False, _NO_LEAD_NO_CANDIDATE)
    return LeadDecision(pick, falls_through, reason)


__all__ = [
    "MODES",
    "Candidate",
    "LeadDecision",
    "TurnFacts",
    "decide_lead",
]
