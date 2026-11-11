"""Failed live tool receipts must force the model to speak the failure."""

from __future__ import annotations

import logging

from jarvis.live.session import LiveVoiceSession
from jarvis.live.tools import annotate_failed_tool_result


def test_failed_receipt_carries_a_speak_failure_mandate() -> None:
    annotated = annotate_failed_tool_result(
        {"success": False, "error": "permission denied"}
    )
    assert annotated["speak_failure"] is True
    assert "Speak this failure to the user now" in annotated["error"]
    assert "Do not claim the action succeeded" in annotated["error"]


def test_successful_receipt_is_unchanged() -> None:
    payload = {"success": True, "output": {"ok": True}}
    assert annotate_failed_tool_result(payload) is payload


def test_pending_background_receipt_is_not_treated_as_a_failure() -> None:
    payload = {
        "success": False,
        "pending": True,
        "error": "still running",
    }
    assert annotate_failed_tool_result(payload) is payload


def test_annotate_is_idempotent() -> None:
    once = annotate_failed_tool_result({"success": False, "error": "boom"})
    twice = annotate_failed_tool_result(once)
    assert twice["error"].count("Speak this failure") == 1


def test_spoken_success_after_failed_tools_is_reported(caplog) -> None:
    session = object.__new__(LiveVoiceSession)
    session._captions = {
        "assistant": "Your full name was successfully entered into the documents.",
        "user": "",
    }
    session._tools = None
    session._turn_failed_tools = {"wiki_ingest"}
    with caplog.at_level(logging.WARNING, logger="jarvis.live.session"):
        session._report_capability_denial()
    assert "success claimed over voice after failed tools" in caplog.text
    assert "wiki_ingest" in caplog.text


def test_spoken_failure_ack_is_not_reported(caplog) -> None:
    session = object.__new__(LiveVoiceSession)
    session._captions = {
        "assistant": "A tool action failed; I could not save that.",
        "user": "",
    }
    session._tools = None
    session._turn_failed_tools = {"wiki_ingest"}
    with caplog.at_level(logging.WARNING, logger="jarvis.live.session"):
        session._report_capability_denial()
    assert "success claimed over voice after failed tools" not in caplog.text
