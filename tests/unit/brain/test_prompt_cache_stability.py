"""Prompt-cache prefix stability (slice 3, 2026-11-17)."""

from __future__ import annotations

import json

from jarvis.core.protocols import BrainMessage, BrainRequest
from jarvis.plugins.brain import chatgpt_plan


def _tools(*names: str) -> tuple[dict, ...]:
    return tuple(
        {
            "name": name,
            "description": f"Tool {name}",
            "input_schema": {"type": "object", "properties": {}},
        }
        for name in names
    )


def test_same_prefix_same_cache_key_and_serialized_tools() -> None:
    req_a = BrainRequest(
        system="Be brief.\n\nStanding rule.",
        messages=[BrainMessage(role="user", content="first")],
        tools=_tools("zeta", "alpha"),
    )
    req_b = BrainRequest(
        system="Be brief.\n\nStanding rule.",
        messages=[BrainMessage(role="user", content="second question")],
        tools=_tools("alpha", "zeta"),  # different insertion order
    )
    body_a = chatgpt_plan.build_body(req_a, "gpt-a")
    body_b = chatgpt_plan.build_body(req_b, "gpt-a")
    assert body_a["prompt_cache_key"] == body_b["prompt_cache_key"]
    assert body_a["instructions"] == body_b["instructions"]
    # Tool order on the wire must be name-sorted so the rendered prefix matches.
    names_a = [t["name"] for t in body_a["tools"]]
    names_b = [t["name"] for t in body_b["tools"]]
    assert names_a == sorted(names_a)
    assert names_a == names_b
    assert json.dumps(body_a["tools"], sort_keys=True) == json.dumps(
        body_b["tools"], sort_keys=True
    )


def test_instruction_change_changes_cache_key() -> None:
    base = BrainRequest(
        system="Be brief.",
        messages=[BrainMessage(role="user", content="hi")],
        tools=_tools("alpha"),
    )
    changed = BrainRequest(
        system="Be brief.\n\nNew standing rule.",
        messages=[BrainMessage(role="user", content="hi")],
        tools=_tools("alpha"),
    )
    assert (
        chatgpt_plan.build_body(base, "gpt-a")["prompt_cache_key"]
        != chatgpt_plan.build_body(changed, "gpt-a")["prompt_cache_key"]
    )


def test_continuation_stays_out_of_instructions() -> None:
    """Wave continuations must not rewrite the cached instruction prefix."""
    from jarvis.agent_chat.delegation_wait import CONTINUE_TURN, continuation_prompt

    briefing = continuation_prompt(
        [{"name": "Claude Code", "status": "completed", "report": "wrote hallo.txt"}],
        "Create hallo.txt",
    )
    # The user/turn text carries the briefing; instructions stay the stable system.
    assert CONTINUE_TURN
    assert "Create hallo.txt" in briefing or "hallo" in briefing.lower()
    req = BrainRequest(
        system="Stable system only.",
        messages=[
            BrainMessage(role="user", content=f"{briefing}\n\n{CONTINUE_TURN}"),
        ],
        tools=_tools("coding-session"),
    )
    body = chatgpt_plan.build_body(req, "gpt-a")
    assert body["instructions"] == "Stable system only."
    assert "hallo" in json.dumps(body["input"]).lower()
