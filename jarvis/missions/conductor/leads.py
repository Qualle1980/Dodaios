"""Domain leads: the roles the conductor hands a part of an order to.

Four leads, not ten: Planning, Coding, Quality, Memory. A lead is a ROLE in
the conductor's own team, never a process and never a second orchestrator —
the existing loop keeps planning the order and keeps naming who acts next
(ADR-0011 leaves the router a pure dispatcher; ADR-0043 keeps one conductor).
The team roster of named specialists is Magentic-One's shape, already adapted
in :mod:`.loop` and :mod:`.ledger`, where the MIT notice lives: the only
change here is that the names the model may say are roles with a remit
instead of raw CLI names.

What a lead actually adds over a bare executor list:

* **A remit.** The plan reads "Quality: judge finished work against the
  evidence", not "claude-code: large refactors". The model picks by what the
  step needs, not by which binary it remembers.
* **Write access follows the role, not the phrasing.** Only Coding may change
  files. A read-only lead's instruction is marked read-only whatever words
  the conductor chose — live 2026-10-24 a read order ran with write access
  because a step was phrased as "Run `wc -l` ...".
* **Independence is real or absent.** Quality resolves to a connected
  executor of ANOTHER family than the one that built the work
  (:func:`.lanes.pick_reviewer`). With one family connected there is no
  reviewer, and the lead is unavailable rather than a reviewer in name only.

Which executor serves a lead is still the lanes' deterministic, cost-ordered
decision (:func:`.lanes.pick_executor`). Whether that executor can really run
is decided where it already is: the spawn returns ``SETUP_FAILED`` or
``WAITING_CAPACITY`` and halts the order honestly.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Literal

from .lanes import Executor, Lane, first_lane, pick_executor, pick_reviewer

Lead = Literal["planning", "coding", "quality", "memory"]


@dataclass(frozen=True)
class LeadSpec:
    """One lead: its remit, the lane it works in, and what it may touch."""

    #: The lane whose executor chain, effort and sandbox the lead inherits.
    #: ``None`` for Coding: it follows the order's own lane, so a small fix
    #: stays cheap and a refactor still gets the capable seat.
    lane: Lane | None
    #: What the lead is for, as the plan and progress prompts read it.
    role: str
    #: Whether this lead may create, modify or delete files.
    writes: bool
    #: Whether this lead must come from another executor family than the one
    #: whose work it judges.
    independent: bool


LEADS: Final[dict[Lead, LeadSpec]] = {
    "planning": LeadSpec(
        lane="answer",
        role="answers from what is already known: knowledge questions, "
             "summaries, plans and decisions. No worker, no file access, "
             "cheapest and fastest.",
        writes=False,
        independent=False,
    ),
    "memory": LeadSpec(
        lane="lookup",
        role="finds what already exists before anything is built: reads and "
             "searches files, earlier results and recorded lessons. Reads "
             "only, never changes anything.",
        writes=False,
        independent=False,
    ),
    "coding": LeadSpec(
        lane=None,
        role="the only lead that changes the repository: writes code, fixes "
             "and features, small or large.",
        writes=True,
        independent=False,
    ),
    "quality": LeadSpec(
        lane="lookup",
        role="judges finished work against its evidence: reads the diff, "
             "runs the tests, names what is missing. Always someone other "
             "than whoever built the work, and changes nothing itself.",
        writes=False,
        independent=True,
    ),
}

#: The same roster by plain name: what the conductor says is a string, and
#: anything that is not a lead is an executor the user named (or nothing).
_BY_NAME: Final[dict[str, LeadSpec]] = {str(name): spec for name, spec in LEADS.items()}

#: The lane a writing lead falls back to when the order names no scale.
_DEFAULT_WRITE_LANE: Final[Lane] = "change"


def roster(request: str) -> dict[str, str]:
    """The team the conductor conducts over: every lead and its remit.

    All four are offered. A lead that turns out to have nobody behind it is
    caught at resolution (:func:`resolve_executor` returns ``None``), so the
    order fails or halts for a named reason instead of the plan quietly
    losing a role it was built around.
    """
    del request  # the remits are fixed; the request decides the lane, not the roster
    return {name: spec.role for name, spec in LEADS.items()}


def may_write(lead: str) -> bool:
    """Whether ``lead`` is allowed to change files at all."""
    spec = _BY_NAME.get(lead)
    return bool(spec and spec.writes)


def lane_for(lead: str, request: str) -> Lane:
    """The lane ``lead`` works in for this request.

    Coding follows the order's own deterministic lane so the scale of the
    work keeps deciding the seat; the request can only move it between the
    two writing lanes, never into a read-only one.
    """
    spec = _BY_NAME.get(lead)
    if spec is None:
        raise KeyError(lead)
    if spec.lane is not None:
        return spec.lane
    lane = first_lane(request)
    return lane if lane in ("change", "deep") else _DEFAULT_WRITE_LANE


def resolve_executor(
    name: str,
    request: str,
    *,
    builder: Executor | None = None,
    connected: Iterable[str] = ("codex", "claude-code"),
) -> Executor | None:
    """Who serves ``name``, or ``None`` when nobody can.

    ``name`` is a lead, or an executor the user named themselves — that name
    travels the same path instead of a second branch in the caller. An
    independent lead needs a connected executor of another family than
    ``builder``; without one it is unavailable, never a stand-in.
    """
    spec = _BY_NAME.get(name)
    if spec is None:
        available = set(connected) | {"conductor"}
        return name if name in available else None  # type: ignore[return-value]
    if spec.independent:
        if builder is None:
            return None
        return pick_reviewer(builder, connected)
    return pick_executor(lane_for(name, request), connected)


__all__ = ["LEADS", "Lead", "LeadSpec", "lane_for", "may_write", "resolve_executor", "roster"]
