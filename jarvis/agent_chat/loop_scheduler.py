"""Per-agent task slots for the Instant coding loop.

Ahmad's concurrency rule (2026-11-17): one Claude coding task, one Grok Build
coding task, and one Codex review may run together — at most three workers.
Instant itself is the orchestrator and is not a fourth coding slot.

Pure logic: no pane I/O. Callers ``create`` a fresh pane after ``Acquired``,
and never ``send`` a new task into a busy pane.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Literal

CoderSlot = Literal["claude", "grok-build"]
SlotName = Literal["claude", "grok-build", "codex-review"]
AgentPref = Literal["claude", "grok-build", "codex-review", "either"]
Kind = Literal["code", "review"]

_CODER_SLOTS: tuple[SlotName, ...] = ("claude", "grok-build")
_ALL_SLOTS: tuple[SlotName, ...] = ("claude", "grok-build", "codex-review")

#: CLI / pane agent id → scheduler slot. Unknown CLIs are not gated here.
_CLI_TO_SLOT: dict[str, SlotName] = {
    "claude": "claude",
    "grok-build": "grok-build",
    "grok": "grok-build",
    "codex": "codex-review",
}


def slot_for_cli(cli: str | None) -> SlotName | None:
    """The scheduler slot for a coding CLI name, or ``None`` when ungated."""
    key = str(cli or "").strip().casefold()
    return _CLI_TO_SLOT.get(key)


@dataclass(frozen=True)
class Acquired:
    """The caller holds ``agent`` and may start the work (fresh create)."""

    handle: str
    agent: SlotName
    kind: Kind
    status: Literal["acquired"] = "acquired"


@dataclass(frozen=True)
class Queued:
    """Preferred slot is full; job waits in FIFO order."""

    handle: str
    agent: AgentPref
    kind: Kind
    position: int
    message: str
    status: Literal["queued"] = "queued"


@dataclass(frozen=True)
class Rejected:
    """Agent/kind combination is not a valid loop job."""

    reason: str
    status: Literal["rejected"] = "rejected"


AcquireResult = Acquired | Queued | Rejected


@dataclass
class _Pending:
    handle: str
    agent_pref: AgentPref
    kind: Kind


@dataclass
class _Held:
    handle: str
    agent: SlotName
    kind: Kind


def _waiting_message(agent: AgentPref, ahead: int) -> str:
    if agent == "either":
        label = "coder"
    else:
        label = agent
    return f"waiting for the {label} slot ({ahead} ahead)"


class LoopScheduler:
    """One-slot-per-agent FIFO coordinator for coding and review jobs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held: dict[SlotName, _Held] = {}
        self._queue: list[_Pending] = []

    def reset(self) -> None:
        """Drop every hold and queued job (tests only)."""
        with self._lock:
            self._held.clear()
            self._queue.clear()

    def try_acquire(self, agent: str, kind: str) -> AcquireResult:
        """Take a free matching slot, or enqueue, or refuse a bad combination."""
        pref = str(agent or "").strip().casefold()
        job_kind = str(kind or "").strip().casefold()
        validated = self._validate(pref, job_kind)
        if isinstance(validated, Rejected):
            return validated
        agent_pref, kind_lit = validated
        with self._lock:
            free = self._pick_free(agent_pref, kind_lit)
            if free is not None:
                handle = uuid.uuid4().hex
                self._held[free] = _Held(handle=handle, agent=free, kind=kind_lit)
                return Acquired(handle=handle, agent=free, kind=kind_lit)
            ahead = self._ahead_count(agent_pref, kind_lit)
            handle = uuid.uuid4().hex
            self._queue.append(
                _Pending(handle=handle, agent_pref=agent_pref, kind=kind_lit)
            )
            return Queued(
                handle=handle,
                agent=agent_pref,
                kind=kind_lit,
                position=ahead,
                message=_waiting_message(agent_pref, ahead),
            )

    def release(self, handle: str) -> Acquired | None:
        """Free a held slot (or drop a queued ticket). Start the next match.

        Returns the newly ``Acquired`` job when a queued waiter took the slot,
        else ``None``.
        """
        key = str(handle or "").strip()
        if not key:
            return None
        with self._lock:
            for pending in list(self._queue):
                if pending.handle == key:
                    self._queue.remove(pending)
                    return None
            freed: SlotName | None = None
            for slot, held in list(self._held.items()):
                if held.handle == key:
                    del self._held[slot]
                    freed = slot
                    break
            if freed is None:
                return None
            return self._start_next_locked(freed)

    def status(self) -> dict[str, object]:
        """Slot busy flags and per-agent queue lengths for the status line."""
        with self._lock:
            slots = {
                name: {
                    "busy": name in self._held,
                    "handle": self._held[name].handle if name in self._held else None,
                }
                for name in _ALL_SLOTS
            }
            lengths: dict[str, int] = {name: 0 for name in _ALL_SLOTS}
            lengths["either"] = 0
            for pending in self._queue:
                key = pending.agent_pref
                lengths[key] = lengths.get(key, 0) + 1
            return {"slots": slots, "queue_lengths": lengths}

    def _validate(
        self, agent: str, kind: str
    ) -> tuple[AgentPref, Kind] | Rejected:
        if kind not in {"code", "review"}:
            return Rejected(reason=f"unknown kind {kind!r}")
        if kind == "review":
            if agent != "codex-review":
                return Rejected(
                    reason="review jobs only take the codex-review slot"
                )
            return "codex-review", "review"
        if agent not in {"claude", "grok-build", "either"}:
            return Rejected(
                reason="coding jobs need claude, grok-build, or either"
            )
        return agent, "code"  # type: ignore[return-value]

    def _pick_free(self, agent_pref: AgentPref, kind: Kind) -> SlotName | None:
        if kind == "review":
            return None if "codex-review" in self._held else "codex-review"
        if agent_pref == "either":
            for slot in _CODER_SLOTS:
                if slot not in self._held:
                    return slot
            return None
        if agent_pref in _CODER_SLOTS and agent_pref not in self._held:
            return agent_pref  # type: ignore[return-value]
        return None

    def _ahead_count(self, agent_pref: AgentPref, kind: Kind) -> int:
        count = 0
        for pending in self._queue:
            if pending.kind != kind:
                continue
            if agent_pref == "either":
                if pending.agent_pref in {"either", "claude", "grok-build"}:
                    count += 1
            elif pending.agent_pref in {agent_pref, "either"}:
                count += 1
            elif agent_pref == "codex-review" and pending.agent_pref == "codex-review":
                count += 1
        return count

    def _start_next_locked(self, freed: SlotName) -> Acquired | None:
        for index, pending in enumerate(self._queue):
            if not self._matches(pending, freed):
                continue
            del self._queue[index]
            self._held[freed] = _Held(
                handle=pending.handle, agent=freed, kind=pending.kind
            )
            return Acquired(
                handle=pending.handle, agent=freed, kind=pending.kind
            )
        return None

    def _matches(self, pending: _Pending, slot: SlotName) -> bool:
        if slot == "codex-review":
            return pending.kind == "review" and pending.agent_pref == "codex-review"
        if pending.kind != "code":
            return False
        return pending.agent_pref in {slot, "either"}


#: Process-wide scheduler for Instant coding hand-offs.
SCHEDULER = LoopScheduler()


__all__ = [
    "Acquired",
    "AgentPref",
    "Kind",
    "LoopScheduler",
    "Queued",
    "Rejected",
    "SCHEDULER",
    "SlotName",
    "slot_for_cli",
    "try_acquire",
    "release",
    "status",
]


def try_acquire(agent: str, kind: str) -> AcquireResult:
    return SCHEDULER.try_acquire(agent, kind)


def release(handle: str) -> Acquired | None:
    return SCHEDULER.release(handle)


def status() -> dict[str, object]:
    return SCHEDULER.status()
