"""Turn a judged candidate fact into a memory (ADR-0044, chunk 2).

The wiki's Stage-1 extractor already produces what the vault stores: one
atomic, self-contained sentence, with the subjects it is about, the turn it
came from, a verbatim excerpt of what the user said, and a grading of the
evidence. Stage 2 then judges it against what is already known and answers
``add`` / ``update`` / ``noop`` / ``invalidate``.

So the vault needs **no model call of its own**. It files the same judged
candidate, at the moment the wiki write for it has landed - which means the
secret guard and the page validation have already passed. One trigger, one
journal, one daily call cap, two sinks.

What this does NOT do is guess. ``update`` and ``invalidate`` name a wiki
*page*; the vault needs the id of the memory being replaced or closed, and the
page-shaped judge never saw a memory id. Picking the nearest one would silently
close a fact that is still true, so those candidates are reported as deferred
and nothing is written. Letting the judge see memory ids is its own step.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from jarvis.memory.vault.models import WriteResult

log = logging.getLogger(__name__)

#: Stage-1 kind -> vault kind. The extractor names twelve things a fact can be
#: about; the vault has eight. Where a candidate kind has no counterpart, it
#: becomes ``profile`` - "who the user is" - because the alternative is
#: dropping a fact the user stated.
KIND_MAP: dict[str, str] = {
    "identity": "profile",
    "preference": "preference",
    # A described habit is part of who someone is, not a one-off event.
    "activity": "profile",
    "person": "relationship",
    "relationship": "relationship",
    "project": "project",
    # A decision is made inside ongoing work and is read back with it.
    "decision": "project",
    "event": "episode",
    # An asset and a place both answer "where things are".
    "asset": "reference",
    "place": "reference",
    "organization": "profile",
    "other": "profile",
}

#: Where the vault files a kind it has never seen.
FALLBACK_KIND = "profile"

#: Stage-1 ``basis`` -> vault ``origin``. ``explicit`` is the user asserting a
#: fact; ``behavioral`` is the extractor reading one out of what they
#: described. The second is an inference, and the store caps an inference's
#: confidence, so this mapping is what keeps a derived fact from entering as
#: certain.
ORIGIN_MAP: dict[str, str] = {
    "explicit": "user",
    "behavioral": "inferred",
}

#: Where the vault files evidence it cannot grade.
FALLBACK_ORIGIN = "inferred"

#: Decisions that name a wiki page rather than a memory, so the vault cannot
#: act on them without guessing which memory is meant.
NEEDS_MEMORY_ID = ("update", "invalidate")

#: Longest excerpt kept as the evidence for a memory.
MAX_EVIDENCE_CHARS = 500


class Candidate(Protocol):
    """The part of a journal row the vault reads. A ``JournalRow`` satisfies it."""

    id: int
    fact: str
    kind: str
    subjects: tuple[str, ...]
    evidence_turn_id: str
    evidence_excerpt: str
    session_id: str
    source_label: str
    basis: str


@dataclass(frozen=True)
class FilingResult:
    """What filing one candidate did.

    ``action`` is the vault's own word (``added``, ``existing``, ``refused``,
    ``blocked``) or ``deferred`` when the decision needs a memory id the judge
    could not give, or ``off`` when the vault is switched off.
    """

    action: str
    reason: str = ""
    memory_id: str = ""

    @property
    def wrote(self) -> bool:
        return self.action == "added"


def vault_kind(candidate_kind: str) -> str:
    """The vault kind a Stage-1 kind becomes."""
    return KIND_MAP.get((candidate_kind or "").strip().lower(), FALLBACK_KIND)


def vault_origin(basis: str) -> str:
    """The origin a Stage-1 evidence grading becomes."""
    return ORIGIN_MAP.get((basis or "").strip().lower(), FALLBACK_ORIGIN)


def source_of(candidate: Candidate) -> dict[str, Any]:
    """Where a memory came from, kept with it so a claim can be traced back."""
    return {
        "via": "wiki candidate journal",
        "candidate_id": candidate.id,
        "session_id": candidate.session_id or "",
        "turn_id": candidate.evidence_turn_id or "",
        "label": candidate.source_label or "",
        "basis": (candidate.basis or "").strip().lower(),
        "evidence": (candidate.evidence_excerpt or "")[:MAX_EVIDENCE_CHARS],
    }


def file_candidate(vault: Any, candidate: Candidate, decision: str) -> FilingResult:
    """File one judged candidate. Writes only for ``add``; never guesses.

    ``vault`` is a :class:`~jarvis.memory.vault.store.MemoryVault`. Raises
    nothing the caller has to handle: a store error is the caller's to log, and
    everything this function decides against comes back as a result.
    """
    verdict = (decision or "").strip().lower()
    if verdict == "noop":
        return FilingResult("deferred", reason="the judge kept nothing")
    if verdict in NEEDS_MEMORY_ID:
        # Deliberate: see the module docstring. Nothing is written rather than
        # closing a memory the judge never saw.
        return FilingResult(
            "deferred",
            reason=f"{verdict} names a page, not a memory; the judge saw no memory ids",
        )
    if verdict != "add":
        return FilingResult("deferred", reason=f"unknown decision {verdict!r}")

    fact = " ".join((candidate.fact or "").split())
    if not fact:
        return FilingResult("deferred", reason="the candidate has no fact")

    result: WriteResult = vault.remember(
        fact,
        vault_kind(candidate.kind),
        entities=tuple(candidate.subjects or ()),
        origin=vault_origin(candidate.basis),
        source=source_of(candidate),
    )
    return FilingResult(
        result.action,
        reason=result.reason,
        memory_id=result.memory.id if result.memory is not None else "",
    )
