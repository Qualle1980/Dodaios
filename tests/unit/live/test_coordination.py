"""Every live engine gets the same rules for talking, looking and handing over."""

from __future__ import annotations

import inspect
import re

from jarvis.core.agent_brief import AGENT_BRIEF_RULE
from jarvis.live import native
from jarvis.live.config import LiveConfig
from jarvis.live.coordination import (
    MEMORY_RULE,
    NAMED_TOOLS,
    ON_SCREEN_RULE,
    TEAM_TALK_RULE,
    WORK_STATUS_RULE,
    conversation_rules,
)
from jarvis.live.tools import _LOOKUP_TOOLS

_RULES = (
    WORK_STATUS_RULE
    + MEMORY_RULE
    + ON_SCREEN_RULE
    + TEAM_TALK_RULE
    + conversation_rules("your tools")
)
# Hyphenated or underscored words in the rules that are plain English, not tools.
_NOT_TOOLS = {"multi-step", "trade-offs", "mission_id", "long-term"}
#: The two sets the catalog guarantees room for, whatever else it drops:
#: NAMED_TOOLS by being in the priority group, the memory reads by sorting
#: ahead of it (#137). A rule may name a tool from either — being told to use
#: a tool that was not declared is what makes a model say it does not exist.
_ALWAYS_DECLARED = set(NAMED_TOOLS) | set(_LOOKUP_TOOLS)


def test_every_tool_the_rules_name_is_listed_and_every_listed_tool_is_named():
    named_in_text = set(re.findall(r"\b[a-z]+(?:[-_][a-z]+)+\b", _RULES)) - _NOT_TOOLS
    assert named_in_text <= _ALWAYS_DECLARED
    for name in NAMED_TOOLS:
        assert re.search(rf"\b{re.escape(name)}\b", _RULES), name


def test_the_rules_spell_out_the_argument_a_small_model_has_to_pass():
    # "workspace-orchestrate status" read as an argument named status (live 2026-10-14).
    assert "workspace-orchestrate with action status" in WORK_STATUS_RULE
    assert "workspace-orchestrate with action watch" in WORK_STATUS_RULE
    assert "workspace-git, with op state" in WORK_STATUS_RULE


def _wire(**profile) -> dict:
    return LiveConfig(configured=True, backend_model="chosen-model", **profile).session_config(
        language="en", tools=[]
    )


def test_speaking_model_is_told_how_to_handle_a_request():
    rules = conversation_rules("the backend")
    spoken = _wire()["instructions"]
    assert rules in spoken
    for behaviour in (
        "Talk it through",
        "Take a quick look",
        "Hand it over",
        "one focused question at a time",
        "start no implementation until they approve",
        "preferences for the whole task",
        "Never read out tables, lists, diffs, code or logs",
        "never say it cannot be changed",
    ):
        assert behaviour in rules, behaviour


def test_a_clear_order_is_not_held_back_as_planning():
    assert "A clear request to do something now is not planning" in conversation_rules("x")


def test_rules_name_whatever_carries_out_the_work():
    assert "the backend" in conversation_rules("the backend")
    assert "the backend" not in conversation_rules("your tools")
    assert "your tools" in conversation_rules("your tools")


def test_subscription_calls_speak_under_the_same_rules():
    wire = _wire(auth_mode="chatgpt_subscription", subscription_backend_model="chosen-model")
    assert conversation_rules("the backend") in wire["instructions"]


def test_the_executing_model_knows_the_status_board_on_every_engine():
    assert WORK_STATUS_RULE in _wire()["delegation"]["responses"]["instructions"]
    source = inspect.getsource(native.NativeLiveVoiceSession)
    assert "WORK_STATUS_RULE" in source
    assert 'conversation_rules("your tools")' in source


def test_a_brief_asks_the_agent_to_report_status_decisions_and_blockers():
    rule = AGENT_BRIEF_RULE.lower()
    assert "report back what changed" in rule
    assert "any decision the user has to make" in rule
    assert "when blocked, report that too" in rule


def test_the_prompts_leave_room_for_tools_and_history_on_small_wires():
    # One session.started message carries both prompts, the tool declarations
    # (24 kB budget) and the history seed inside a 64 KiB limit.
    wire = _wire()
    prompts = wire["instructions"] + wire["delegation"]["responses"]["instructions"]
    assert len(prompts.encode("utf-8")) < 20_000


def test_a_call_is_told_its_standing_memory_is_only_a_summary():
    """Otherwise it answers "I know nothing" from a partial block.

    Live 2026-11-08 and again 2026-11-09, asked in German what it knew about a
    named friend, a call said it knew nothing while the vault held the answer
    verbatim. Measured on the real vault 2026-11-09: the standing block carries
    5 of 54 memories, because it is budgeted to 2400 characters and ordered by
    kind, and five new `profile` memories at confidence 1.0 had pushed out the
    `relationship` memory that answered exactly that question.

    Having the tool is not enough (that was #137). The model also has to know
    that what it was handed is partial, and which store answers a question
    about the user.
    """
    assert "summary of their memory, not all of it" in MEMORY_RULE
    # The instruction that turns a shrug into a lookup.
    assert "BEFORE" in MEMORY_RULE
    assert "memory-search" in MEMORY_RULE
    # wiki-recall is a DIFFERENT store; it ranks ahead of memory_search in
    # CORE_PRIORITY, and the live model reached for it every time.
    assert "wiki-recall is a different store" in MEMORY_RULE


def test_the_native_live_engine_is_given_the_memory_rule():
    """A rule only helps the engine that actually sends it.

    Three prompt builds exist and the memory block had to be added to each one
    separately (#131). `native` is the gemini-live path, the one the 2026-11-09
    session ran on.
    """
    source = inspect.getsource(native)
    assert "MEMORY_RULE" in source
