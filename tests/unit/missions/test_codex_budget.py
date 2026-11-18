"""Per-run Codex budget: tokens and shell calls stop one run, never retried."""

from __future__ import annotations

from jarvis.missions.workers.codex_budget import (
    CODEX_RUN_TOKEN_CAP,
    CODEX_RUN_TOOL_CAP,
    CodexRunBudget,
)


def test_fresh_budget_has_no_stop_reason() -> None:
    assert CodexRunBudget().stop_reason() is None


def test_tokens_sum_input_and_output_over_turns() -> None:
    b = CodexRunBudget(token_cap=1_000)
    b.observe_turn_usage({"input_tokens": 300, "output_tokens": 100})
    b.observe_turn_usage({"input_tokens": 400, "output_tokens": 50})
    assert b.tokens == 850
    assert b.turns == 2
    assert b.stop_reason() is None


def test_token_cap_is_crossed_by_a_single_turn() -> None:
    b = CodexRunBudget(token_cap=1_000)
    b.observe_turn_usage({"input_tokens": 900, "output_tokens": 200})
    reason = b.stop_reason()
    assert reason is not None
    assert "tokens" in reason and "1000" in reason


def test_cap_at_exactly_the_limit_does_not_stop() -> None:
    b = CodexRunBudget(token_cap=1_000)
    b.observe_turn_usage({"input_tokens": 600, "output_tokens": 400})
    assert b.stop_reason() is None


def test_missing_or_bad_usage_counts_as_zero_but_still_a_turn() -> None:
    b = CodexRunBudget(token_cap=10)
    b.observe_turn_usage(None)
    b.observe_turn_usage({"input_tokens": "lots", "output_tokens": True})
    assert b.tokens == 0
    assert b.turns == 2
    assert b.stop_reason() is None


def test_shell_calls_are_counted_only_for_command_execution() -> None:
    b = CodexRunBudget(tool_cap=2)
    b.observe_tool_item({"type": "command_execution"})
    b.observe_tool_item({"type": "agent_message"})
    b.observe_tool_item({"type": "file_change"})
    b.observe_tool_item(None)
    assert b.tools == 1
    assert b.stop_reason() is None


def test_tool_cap_stops_the_run() -> None:
    b = CodexRunBudget(tool_cap=2)
    for _ in range(3):
        b.observe_tool_item({"type": "command_execution"})
    reason = b.stop_reason()
    assert reason is not None
    assert "shell calls" in reason


def test_default_caps_are_the_documented_values() -> None:
    b = CodexRunBudget()
    assert b.token_cap == CODEX_RUN_TOKEN_CAP == 1_500_000
    assert b.tool_cap == CODEX_RUN_TOOL_CAP == 120


def test_the_321_call_session_would_have_been_stopped() -> None:
    # Shape of the measured session: about 2.9 million tokens and 155 shell calls.
    b = CodexRunBudget()
    b.observe_turn_usage({"input_tokens": 2_800_000, "output_tokens": 100_000})
    for _ in range(155):
        b.observe_tool_item({"type": "command_execution"})
    assert b.stop_reason() is not None
