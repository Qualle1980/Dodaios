"""Spoken output never carries a code block, closed or cut off.

Live 2026-10-23: a truncated mission summary left its fence open and the
announcement handed raw Python to TTS.
"""
from __future__ import annotations

from jarvis.brain.output_filter import scrub_for_voice


def test_an_unclosed_fence_is_dropped_to_the_end() -> None:
    text = (
        "Gefunden: budget.py\n\n"  # i18n-allow
        '```python\n"""Token-bucket"""\nimport asyncio\nx = `y`'
    )
    cleaned = scrub_for_voice(text, language="de").cleaned

    assert "import" not in cleaned
    assert "`" not in cleaned
    assert cleaned.startswith("Gefunden: budget.py")  # i18n-allow


def test_a_closed_fence_with_backticks_inside_is_dropped() -> None:
    text = "Siehe ```a = `b` ``` und dann fertig."  # i18n-allow
    cleaned = scrub_for_voice(text, language="de").cleaned

    assert "a = " not in cleaned
    assert "fertig" in cleaned  # i18n-allow
