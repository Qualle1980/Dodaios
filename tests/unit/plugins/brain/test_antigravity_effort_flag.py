"""agy must NOT get --effort beside --model: the level lives inside the id.

This contract reversed, so both halves are recorded here.

Live 2026-07-26, agy ~1.0.x: the binary REQUIRED the pair and aborted without
it, writing `invalid model selection (--model "gemini-3.5-flash" --effort "")`
to stdout where the caller read it as the model's answer.

Live 2026-11-05, agy 1.5.1: the opposite. Effort is now spelled in the model id
(`gemini-3.8-flash-low` / `-medium` / `-high`) and passing the flag as well is
an error:

    error: invalid model selection (--model "gemini-3.8-flash-low"
    --effort "medium"): --model gemini-3.8-flash-low conflicts with
    --effort=medium

A run with a valid id and no flag succeeded in 7.2 s. So the ladder maps onto
the id suffix and the flag is never sent — see
`tests/unit/brain/test_antigravity_model_discovery.py` for the resolution.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvis.core.protocols import BrainRequest
from jarvis.plugins.brain import antigravity


def _agy() -> SimpleNamespace:
    return SimpleNamespace(kind="agy", argv_prefix=["agy"])


def _gemini() -> SimpleNamespace:
    return SimpleNamespace(kind="gemini", argv_prefix=["gemini"])


def test_agy_never_gets_an_effort_beside_its_model() -> None:
    argv = antigravity._build_argv(_agy(), "prompt", "gemini-3.8-flash-low")

    assert "--effort" not in argv, "agy 1.5.1 refuses the flag beside a levelled id"
    assert argv[argv.index("--model") + 1] == "gemini-3.8-flash-low"


def test_a_refused_model_is_recognised_as_a_model_problem() -> None:
    """A rejected model must be told apart from a dead subscription.

    Only the first retries with a different id; the second must surface.
    """
    for complaint in (
        'invalid model selection (--model "gemini-3.5-flash" --effort "")',
        "model gemini-3.5-flash is not recognized as a known model",
        "--model gemini-3.8-flash-low conflicts with --effort=medium",
    ):
        assert antigravity._is_model_rejection(complaint), complaint

    for outage in (
        "Google CLI failed with exit status 1.",
        "could not reach the subscription",
        "please sign in again",
    ):
        assert not antigravity._is_model_rejection(outage), outage


def test_the_gemini_cli_is_left_alone() -> None:
    """The Gemini CLI never took this flag either; its path is untouched."""
    argv = antigravity._build_argv(_gemini(), "prompt", "gemini-3.5-flash")

    assert "--effort" not in argv


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        # agy has no "off"; the caller's intent maps to the lowest it offers.
        ("none", "low"),
        # Anything unknown or unset falls back rather than passing debris on.
        (None, "medium"),
        ("nonsense", "medium"),
    ],
)
def test_effort_follows_the_caller_and_never_goes_empty(
    requested: str | None, expected: str
) -> None:
    req = BrainRequest(messages=(), reasoning_effort=requested)  # type: ignore[arg-type]

    assert antigravity._agy_effort(req) == expected


def test_a_request_without_the_field_still_yields_a_valid_effort() -> None:
    assert antigravity._agy_effort(None) == "medium"
    assert antigravity._agy_effort(SimpleNamespace()) == "medium"
