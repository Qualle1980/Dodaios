"""A provider test may claim only the depth it actually reached.

Measured live on 2026-10-30: ``jarvisctl brain test chatgpt-web`` answered
``status ok`` for the browser bridge that had just lost 11 of 11 real turns
(``chatgpt_submitted_turn_failed``), while ``chatgpt-plan`` answered ``ok``
too. One word, two very different truths — so the word carried no information
and no acceptance could rest on it.

The cause is here, not in the bridge. ``BrainHealthChecker._drain_first_delta``
consumed the stream with ``async for _delta in stream: break``. A stream that
yields NOTHING runs that loop zero times, raises nothing, and falls through to
``ok=True``: silence was success. A stream of empty acknowledgement chunks did
the same.

These tests pin the two halves of the repair:

1. A turn that produced no answer is not a passing turn.
2. A result says WHICH of the five depths it reached, so "the login works"
   can never again be read as "tool calls work".

Every test here is offline against a fake brain. None of them may be read as
evidence that a live provider works -- that is exactly the confusion being
fixed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from jarvis.brain.healthcheck import BrainHealthChecker
from jarvis.brain.provider_test import (
    PROOF_STAGES,
    PROVEN_ANSWERED,
    PROVEN_AUTHENTICATED,
    PROVEN_END_TO_END,
    PROVEN_NOTHING,
    PROVEN_TOOL_CALL,
    ProviderTestResult,
    at_least,
)
from jarvis.core.protocols import BrainDelta, BrainRequest


class _Brain:
    """A brain whose stream is whatever the test hands it."""

    name = "fake-brain"
    context_window = 8192
    supports_tools = False
    supports_vision = False

    def __init__(self, deltas: tuple[BrainDelta, ...], *, model: str = "m") -> None:
        self._deltas = deltas
        self.model = model

    async def complete(self, req: BrainRequest) -> AsyncIterator[BrainDelta]:
        for delta in self._deltas:
            yield delta

    def estimate_cost(self, req: BrainRequest) -> float:  # pragma: no cover
        return 0.0


class _Registry:
    def __init__(self, deltas: tuple[BrainDelta, ...]) -> None:
        self._deltas = deltas

    def instantiate(self, name: str, **kwargs: Any) -> Any:
        return _Brain(self._deltas, **kwargs)


async def _probe(deltas: tuple[BrainDelta, ...]) -> Any:
    checker = BrainHealthChecker(_Registry(deltas))  # type: ignore[arg-type]
    return await checker.probe("fake", "m", timeout_s=5.0)


# ── 1. Silence is not success ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_stream_that_yields_nothing_is_not_a_passing_turn() -> None:
    """The exact shape that made the browser bridge report ``ok``."""
    result = await _probe(())
    assert not result.ok
    assert result.error, "a failure has to say why"


@pytest.mark.asyncio
async def test_a_stream_of_empty_acknowledgements_is_not_a_passing_turn() -> None:
    """A bridge that echoes chunks carrying no answer has not answered."""
    result = await _probe(
        (BrainDelta(content=""), BrainDelta(content=None), BrainDelta(finish_reason="stop"))
    )
    assert not result.ok


@pytest.mark.asyncio
async def test_real_content_still_passes() -> None:
    """The repair must not turn a working provider red."""
    result = await _probe((BrainDelta(content="hi", finish_reason="stop"),))
    assert result.ok
    assert result.error is None


@pytest.mark.asyncio
async def test_content_after_an_empty_first_chunk_still_passes() -> None:
    """Many providers open with an empty role/ack chunk before any text.

    The old code broke on the FIRST delta, so this shape was indistinguishable
    from silence. Draining to the first chunk that carries something keeps it
    honest in both directions.
    """
    result = await _probe(
        (BrainDelta(content=""), BrainDelta(content="hello"), BrainDelta(finish_reason="stop"))
    )
    assert result.ok


@pytest.mark.asyncio
async def test_a_tool_call_counts_as_an_answer() -> None:
    """A turn whose only output is a tool call did answer."""
    result = await _probe((BrainDelta(tool_call={"name": "search_web"}),))
    assert result.ok


# ── 2. The five depths ───────────────────────────────────────────────────────


def test_the_ladder_is_ordered_weakest_first() -> None:
    assert PROOF_STAGES == (
        PROVEN_NOTHING,
        "reachable",
        PROVEN_AUTHENTICATED,
        PROVEN_ANSWERED,
        PROVEN_TOOL_CALL,
        PROVEN_END_TO_END,
    )


@pytest.mark.parametrize(
    ("reached", "claimed", "expected"),
    [
        (PROVEN_ANSWERED, PROVEN_AUTHENTICATED, True),
        (PROVEN_ANSWERED, PROVEN_ANSWERED, True),
        (PROVEN_AUTHENTICATED, PROVEN_ANSWERED, False),
        (PROVEN_AUTHENTICATED, PROVEN_TOOL_CALL, False),
        (PROVEN_END_TO_END, PROVEN_TOOL_CALL, True),
        (PROVEN_NOTHING, PROVEN_AUTHENTICATED, False),
    ],
)
def test_a_shallower_proof_never_satisfies_a_deeper_claim(
    reached: str, claimed: str, expected: bool
) -> None:
    assert at_least(reached, claimed) is expected


def test_an_unknown_stage_is_never_a_proof() -> None:
    """A typo or a future stage name must fail closed, not pass."""
    assert at_least("made-up", PROVEN_AUTHENTICATED) is False
    assert at_least(PROVEN_END_TO_END, "made-up") is False


# ── 3. What a result reports, per status ─────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("ok", PROVEN_ANSWERED),
        ("no_credits", PROVEN_AUTHENTICATED),
        ("rate_limited", PROVEN_AUTHENTICATED),
        ("model_unavailable", PROVEN_AUTHENTICATED),
        ("bad_key", "reachable"),
        ("unreachable", PROVEN_NOTHING),
        ("not_configured", PROVEN_NOTHING),
        ("error", PROVEN_NOTHING),
    ],
)
def test_a_status_reports_the_depth_it_proves(status: str, expected: str) -> None:
    """A refusal is still evidence, and says how far it got.

    "Out of credits" and "no access to that model" are answers a provider can
    only give once it has accepted the credential, so they prove authentication
    while proving nothing about a turn. A rejected key proves only that
    something answered.
    """
    result = ProviderTestResult("p", status)
    assert result.proof == expected


def test_an_explicit_depth_wins_over_the_status() -> None:
    """A site whose ``ok`` means less than an answer can say so."""
    result = ProviderTestResult("p", "ok", proven=PROVEN_AUTHENTICATED)
    assert result.proof == PROVEN_AUTHENTICATED
    assert at_least(result.proof, PROVEN_ANSWERED) is False


def test_a_connected_login_does_not_prove_a_turn() -> None:
    """The regression in words.

    A subscription CLI reports a connected login without sending anything. That
    is real evidence of authentication and no evidence at all about turns or
    tools, which is precisely the distinction the live ``ok`` erased.
    """
    assert at_least(PROVEN_AUTHENTICATED, PROVEN_ANSWERED) is False
    assert at_least(PROVEN_AUTHENTICATED, PROVEN_TOOL_CALL) is False
    assert at_least(PROVEN_AUTHENTICATED, PROVEN_END_TO_END) is False
