"""search_web's answer instruction follows the turn's delivery.

The spoken instruction ("one or two SPOKEN sentences, never name a
title/url/source/date") is right for voice and wrong for a scheduled written
digest — every search-based automation fought it. A turn that declares
``ExecutionContext.config["delivery"] == "written"`` gets the written
variant; every other context (including one without the key) keeps the
spoken one. Also pins that the dispatcher → loop path actually delivers the
``tool_context`` keys into the ExecutionContext.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from jarvis.core.protocols import ExecutionContext
from jarvis.plugins.tool import search_web as sw


def _ctx(config: dict[str, Any] | None) -> ExecutionContext:
    return ExecutionContext(
        trace_id=uuid4(), user_utterance="q", config=config or {}, memory_read=None,
    )


def test_spoken_by_default() -> None:
    assert sw._answer_instruction_for(_ctx({})) is sw._ANSWER_INSTRUCTION
    assert sw._answer_instruction_for(_ctx({"delivery": "spoken"})) is sw._ANSWER_INSTRUCTION


def test_written_delivery_gets_written_instruction() -> None:
    text = sw._answer_instruction_for(_ctx({"delivery": "written"}))
    assert text is sw._WRITTEN_ANSWER_INSTRUCTION
    assert "SPOKEN" not in text
    assert "source" in text.lower()


def test_foreign_context_shape_degrades_to_spoken() -> None:
    class _Odd:
        config = None

    assert sw._answer_instruction_for(_Odd()) is sw._ANSWER_INSTRUCTION  # type: ignore[arg-type]


def test_tool_context_reaches_execution_context() -> None:
    """ToolUseLoop merges the dispatcher's tool_context under the per-turn keys."""
    from jarvis.brain.tool_use_loop import ToolUseLoop

    loop = ToolUseLoop(
        object(), {}, object(),  # type: ignore[arg-type]
        tool_context={"delivery": "written", "output_language": "must-lose"},
    )
    merged = {**loop._tool_context, "output_language": "de", "voice_confirm": False}
    assert merged == {"delivery": "written", "output_language": "de", "voice_confirm": False}


class TestQueryVariantsAcceptBothShapes:
    """The required field is `query`, so that is where a batching model puts
    its list.

    `jarvis/brain/manager.py` instructs the model: "For web research: put your
    2-3 query variants into ONE search_web call (its 'queries' parameter)".
    The schema marks `query` -- not `queries` -- as required, so a model that
    batches reaches for `query` and sends a list. Live 2026-11-07: a question
    about Crunchyroll prices died on `'list' object has no attribute 'strip'`,
    and the assistant told the owner it had no internet access. It does; the
    retry one second later with a plain string succeeded.
    """

    def test_a_plain_string_is_one_variant(self) -> None:
        assert sw._query_variants({"query": " crunchyroll preise "}) == ["crunchyroll preise"]

    def test_a_list_in_the_required_field_is_not_a_crash(self) -> None:
        got = sw._query_variants({"query": ["crunchyroll preise", "crunchyroll cost"]})
        assert got == ["crunchyroll preise", "crunchyroll cost"]

    def test_both_fields_merge_without_duplicates(self) -> None:
        got = sw._query_variants(
            {"query": ["a question", "A QUESTION"], "queries": ["another one", "a question"]}
        )
        assert got == ["a question", "another one"]

    def test_the_variant_cap_still_holds(self) -> None:
        got = sw._query_variants({"query": [f"q{i}" for i in range(10)]})
        assert len(got) == sw._MAX_QUERY_VARIANTS

    def test_nothing_usable_is_empty_not_an_exception(self) -> None:
        assert sw._query_variants({}) == []
        assert sw._query_variants({"query": None, "queries": []}) == []
        assert sw._query_variants({"query": ["", "   "]}) == []
