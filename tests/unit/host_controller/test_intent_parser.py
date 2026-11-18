"""Intent parser: strict schema, \\_ normalisation, reject bad escapes."""

from __future__ import annotations

import logging

import pytest

from jarvis.host_controller.intent import IntentParseError, parse_intent


def test_underscore_escape_inside_string_is_normalised_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw = (
        '{"intent_id":"i1","kind":"ASSIGN\\_TASK","project":"demo",'
        '"issue_number":1,"worker":"claude","task":"test\\_x.py"}'
    )
    with caplog.at_level(logging.INFO):
        intent = parse_intent(raw)
    assert intent.kind == "ASSIGN_TASK"
    assert intent.data["task"] == "test_x.py"
    assert any("normalised" in record.getMessage().lower() for record in caplog.records)
    assert any("\\_" in record.getMessage() or "underscore" in record.getMessage().lower()
               for record in caplog.records)


def test_hex_escape_inside_string_is_rejected() -> None:
    raw = (
        '{"intent_id":"i2","kind":"ASSIGN_TASK","project":"demo",'
        '"issue_number":1,"worker":"claude","task":"evil\\x41"}'
    )
    with pytest.raises(IntentParseError, match=r"\\\\x|bad escape|unsupported escape"):
        parse_intent(raw)


def test_unknown_kind_is_rejected() -> None:
    raw = '{"intent_id":"i3","kind":"DELETE_REPO","project":"demo"}'
    with pytest.raises(IntentParseError, match="unknown intent kind"):
        parse_intent(raw)


def test_unknown_key_is_rejected() -> None:
    raw = (
        '{"intent_id":"i4","kind":"REPORT_PROGRESS","project":"demo",'
        '"shell":"rm -rf /"}'
    )
    with pytest.raises(IntentParseError, match="unknown key"):
        parse_intent(raw)


def test_raw_text_is_never_a_shell_command() -> None:
    with pytest.raises(IntentParseError):
        parse_intent("rm -rf / && echo pwned")
