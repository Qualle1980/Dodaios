"""What a delegated worker is told — and what it must never be told (phase 7).

The brief a worker receives is assembled from the request plus recent
conversation. Three things have to hold at once: the context a task needs
arrives, the context it does not need is bounded, and a credential never
leaves the process this way.

The last one was measured on 2026-10-20 and did not hold: an ``sk-…`` key
typed in the conversation reached the mission prompt verbatim, and from there
a third-party CLI's input, its log, and the stored mission event. A secret is
an ordinary sentence to every layer above this one, so the brief is redacted
on the way out. That is a safety net, not the guarantee — credentials belong
in ``get_secret`` and never in a prompt at all (AGENTS.md §3/AP-2).

Deterministic — no worker, no model, no network.
"""
from __future__ import annotations

from jarvis.plugins.tool.spawn_worker import (
    _MAX_CONTEXT_HINT_CHARS,
    _MAX_CONTEXT_HINTS,
    _bounded_context_hints,
    _build_mission_prompt,
)

_TASK = "add a health endpoint to the gateway"
_KEY = "sk-ant-api03-A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0"
_TOKEN = "ghp_" + "a1b2c3d4e5" * 3 + "a1b2c3"  # 36 chars after the prefix


# --------------------------------------------------------------------------
# Secrets.
# --------------------------------------------------------------------------

def test_a_key_from_the_conversation_never_reaches_the_worker() -> None:
    hints = [f"Earlier turn — User: 'my key is {_KEY}' | Assistant: 'noted'"]

    prompt = _build_mission_prompt(_TASK, "add an endpoint", "", hints)

    assert _KEY not in prompt
    assert "<REDACTED>" in prompt


def test_a_key_in_the_request_itself_is_redacted_too() -> None:
    """The utterance is passed verbatim by design — verbatim, but not a key."""
    prompt = _build_mission_prompt(f"deploy with token {_TOKEN}", "", "", None)

    assert _TOKEN not in prompt


def test_the_bounded_hints_are_already_clean() -> None:
    """Hints also ride on bus events for the UI, so they are scrubbed there."""
    cleaned = _bounded_context_hints([f"the token is {_TOKEN}"])

    assert _TOKEN not in cleaned[0]


# --------------------------------------------------------------------------
# The context a task does need.
# --------------------------------------------------------------------------

def test_ordinary_project_context_survives_untouched() -> None:
    """Over-redaction is its own failure: it removes the task's meaning."""
    hints = [
        "Earlier turn — User: 'the gateway lives in jarvis/mobile_gateway' | "
        "Assistant: 'noted'",
    ]

    prompt = _build_mission_prompt(_TASK, "add an endpoint", "", hints)

    assert "jarvis/mobile_gateway" in prompt
    assert "<REDACTED>" not in prompt


def test_a_task_without_context_gets_no_context_block() -> None:
    prompt = _build_mission_prompt(_TASK, "add an endpoint", "", None)

    assert _TASK in prompt
    assert "Supporting context" not in prompt


# --------------------------------------------------------------------------
# Context that is too big, or not text at all.
# --------------------------------------------------------------------------

def test_an_oversized_conversation_is_bounded() -> None:
    hints = [f"turn {i}: " + "x" * (_MAX_CONTEXT_HINT_CHARS * 2) for i in range(200)]

    bounded = _bounded_context_hints(hints)

    assert len(bounded) == _MAX_CONTEXT_HINTS
    assert all(len(hint) <= _MAX_CONTEXT_HINT_CHARS for hint in bounded)


def test_unusable_memory_entries_are_dropped_not_rendered() -> None:
    """A malformed store must not put ``None`` or a dict into a worker brief."""
    bounded = _bounded_context_hints([None, 42, {"a": 1}, "", "   ", "real context"])

    assert bounded == ["real context"]


def test_control_characters_are_flattened() -> None:
    bounded = _bounded_context_hints(["a\x00b\x07c\nd"])

    assert bounded == ["a b c d"]
