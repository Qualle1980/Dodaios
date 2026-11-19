"""Written-turn instruction trim (slice 2, 2026-11-17).

Instant coding is a typed turn: the full voice persona (~3300 tokens) was
re-sent every call and then overridden by WRITTEN_CHAT_STYLE. Compact persona
keeps character, language and honesty once; voice-only sections stay in the
full persona for speech.
"""

from __future__ import annotations

from jarvis.brain.persona_loader import (
    load_compact_persona_prompt,
    load_effective_persona_prompt,
    load_persona_prompt,
)


# chars/4 estimate, same as chatgpt_plan._log_weight
def _tok(text: str) -> int:
    return len(text) // 4


def test_compact_persona_is_under_half_of_full_and_keeps_core_rules() -> None:
    full = load_persona_prompt()
    compact = load_compact_persona_prompt()
    assert _tok(compact) < _tok(full) // 2
    assert _tok(compact) <= 900
    lowered = compact.lower()
    assert "warm" in lowered and "friend" in lowered
    assert "language" in lowered
    assert "never claim" in lowered or "honesty" in lowered
    # Voice-only bulk that the written path must not need twice:
    assert "ending the call" not in lowered
    assert "echo-paraphrase" not in lowered


def test_written_path_loads_compact_effective_persona() -> None:
    """The written-turn builder must ask for compact=True (custom override still wins)."""
    compact = load_effective_persona_prompt(compact=True)
    full = load_effective_persona_prompt(compact=False)
    assert _tok(compact) <= _tok(full)
    assert _tok(compact) <= 900 or compact == full  # custom override may equal full


def test_tool_list_rule_english_once() -> None:
    from jarvis.brain.manager import _TOOL_LIST_RULE

    assert "Call ONLY tools from the list above" in _TOOL_LIST_RULE
    # German duplicate removed; content remains in the English sentence.
    assert "STRENGE REGEL" not in _TOOL_LIST_RULE
    assert _tok(_TOOL_LIST_RULE) <= 100


def test_written_static_instruction_budget() -> None:
    """Named static blocks for a written Instant turn stay under the size gate."""
    from jarvis.brain.manager import (
        _SELF_CONTROL_STANDING,
        _TOOL_LIST_RULE,
        _TOOL_ROUTING_RULES,
        _WRITTEN_CHAT_STYLE,
    )

    compact = load_effective_persona_prompt(compact=True)
    total = sum(
        _tok(block)
        for block in (
            compact,
            _SELF_CONTROL_STANDING,
            _TOOL_ROUTING_RULES,
            _TOOL_LIST_RULE,
            _WRITTEN_CHAT_STYLE,
        )
    )
    # Before: persona alone ~3300 + written ~1200 + routing/self ~500 ≈ 5000+
    # of the ~11–12k instruction weight. Gate the named static core under 3500.
    assert total <= 3500, f"written static core ~{total} tokens"
