"""Brain provenance on the latency log: which provider/model really answered.

Covers the pure event builder in ``BrainManager`` and the writer joining it to
the matching turn row. No provider is called.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from uuid import uuid4

from jarvis.brain.manager import _build_turn_provenance
from jarvis.core.events import LatencyTurnComplete
from jarvis.telemetry.latency_log import LatencyLogWriter

_CHAIN = [("anthropic", "claude-haiku-4-5"), ("chatgpt-web", None)]


def test_success_records_provider_models_and_tokens() -> None:
    tid = uuid4()
    ev = _build_turn_provenance(
        tid, _CHAIN, "anthropic",
        {"idx": 0, "requested_model": "claude-haiku-4-5",
         "responded_model": "claude-haiku-4-5-20251001",
         "usage": {"input_tokens": 120, "output_tokens": 40, "cache_hit_tokens": 7}},
        [],
    )
    assert ev.trace_id == tid
    assert (ev.provider, ev.success, ev.fallback_used) == ("anthropic", True, False)
    assert ev.requested_model == "claude-haiku-4-5"
    assert ev.responded_model == "claude-haiku-4-5-20251001"
    assert (ev.tokens_in, ev.tokens_out, ev.tokens_cached) == (120, 40, 7)
    assert ev.chatgpt_bridge_used is False
    assert ev.attempts == (("anthropic", "claude-haiku-4-5", "ok"),)


def test_fallback_to_chatgpt_bridge_without_tokens() -> None:
    ev = _build_turn_provenance(
        uuid4(), _CHAIN, "chatgpt-web",
        {"idx": 1, "requested_model": None, "responded_model": None, "usage": None},
        [("anthropic", "claude-haiku-4-5", "rate_limit", "HTTP 429")],
    )
    assert ev.success and ev.fallback_used and ev.chatgpt_bridge_used
    assert ev.error_kind is None
    assert ev.requested_model is None and ev.responded_model is None
    assert (ev.tokens_in, ev.tokens_out, ev.tokens_cached) == (None, None, None)
    assert ev.attempts == (
        ("anthropic", "claude-haiku-4-5", "rate_limit"),
        ("chatgpt-web", None, "ok"),
    )


def test_whole_chain_failure() -> None:
    ev = _build_turn_provenance(
        uuid4(), _CHAIN, None, None,
        [("anthropic", "claude-haiku-4-5", "auth", "secret-ish detail"),
         ("chatgpt-web", "", "call_fail", "boom")],
    )
    assert ev.success is False and ev.provider is None
    assert ev.error_kind == "call_fail"
    assert ev.fallback_used is False and ev.chatgpt_bridge_used is False
    assert ev.tokens_in is None
    # Error detail text never reaches the event.
    assert "secret-ish detail" not in repr(ev)


async def test_writer_joins_provenance_to_matching_turn(tmp_path: Path) -> None:
    writer = LatencyLogWriter(tmp_path / "latency.jsonl")
    turn_a, turn_b = uuid4(), uuid4()
    try:
        await writer._on_provenance(_build_turn_provenance(
            turn_a, _CHAIN, "anthropic",
            {"idx": 0, "requested_model": "m-a", "responded_model": None, "usage": None},
            [],
        ))
        await writer._on_provenance(_build_turn_provenance(
            turn_b, _CHAIN, "chatgpt-web",
            {"idx": 1, "requested_model": None, "responded_model": None, "usage": None},
            [("anthropic", "m-a", "rate_limit", "HTTP 429")],
        ))
        unrelated = uuid4()
        for tid in (turn_b, unrelated, turn_a):
            await writer._on_event(LatencyTurnComplete(
                trace_id=tid, anchor_ns=1, stages_ms={"brain_last_token": 5.0},
            ))
        deadline = time.monotonic() + 2.0
        # The writer flushes on its own thread; there is no event to await.
        while writer.written < 3 and time.monotonic() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
    finally:
        writer.close()
    lines = (tmp_path / "latency.jsonl").read_text("utf-8").splitlines()
    rows = {r["turn_id"]: r for r in (json.loads(x) for x in lines if x)}
    assert rows[turn_a.hex]["brain"]["provider"] == "anthropic"
    assert rows[turn_a.hex]["brain"]["requested_model"] == "m-a"
    assert rows[turn_b.hex]["brain"]["provider"] == "chatgpt-web"
    assert rows[turn_b.hex]["brain"]["chatgpt_bridge_used"] is True
    assert rows[turn_b.hex]["brain"]["fallback_used"] is True
    assert rows[unrelated.hex]["brain"] is None
