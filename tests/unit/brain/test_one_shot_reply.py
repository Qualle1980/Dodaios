"""A thank-you does not need ChatGPT (latency, 2026-10-20).

Measured in the running app: "danke" took 7.8 s, because the turn travelled
the whole provider chain like any other. The intent router already knew it was
a closed-set one-shot; nothing used that.

What must NOT change: the matcher is anchored to the whole utterance, so an
opener in front of a real question is not a one-shot, and a language with no
line falls through to the model rather than being answered in the wrong one.

Deterministic — no provider call, no clock drift (the time is injected).
"""
from __future__ import annotations

import datetime as _dt

import pytest

from jarvis.brain.intent_router import classify, one_shot_kind
from jarvis.brain.one_shot_reply import one_shot_reply

_NOON = _dt.datetime(2026, 10, 20, 14, 7)


@pytest.mark.parametrize(
    "text,kind",
    [
        ("danke", "thanks"),  # i18n-allow: simulated German utterance
        ("hallo", "greeting"),  # i18n-allow
        ("ok", "ack"),
        ("thanks", "thanks"),
        ("what time is it", "clock"),
        ("welches datum ist heute", "date"),  # i18n-allow
    ],
)
def test_the_closed_set_is_recognised(text: str, kind: str) -> None:
    assert one_shot_kind(text) == kind


@pytest.mark.parametrize(
    "text",
    [
        "danke, und wie geht das jetzt weiter?",  # i18n-allow
        "nenne drei farben",  # i18n-allow
        "ok, oeffne notepad",  # i18n-allow
    ],
)
def test_an_opener_in_front_of_a_real_request_is_not_one(text: str) -> None:
    assert one_shot_kind(text) is None
    assert one_shot_reply(text, "de") is None


def test_a_thank_you_is_answered_without_a_model() -> None:
    assert one_shot_reply("danke", "de") is not None  # i18n-allow


def test_the_clock_answer_is_computed_not_guessed() -> None:
    assert one_shot_reply("wie spaet ist es", "de", now=_NOON) == "Es ist 14:07 Uhr."  # i18n-allow


def test_the_date_answer_is_computed_not_guessed() -> None:
    assert one_shot_reply("welches datum ist heute", "de", now=_NOON) == (  # i18n-allow
        "Heute ist 20.10.2026."  # i18n-allow
    )


def test_a_language_without_a_line_falls_through_to_the_model() -> None:
    """A German answer to a Portuguese greeting is worse than a slow one."""
    assert one_shot_reply("ola", "pt") is None


def test_the_same_utterance_answers_the_same_way() -> None:
    assert one_shot_reply("danke", "de") == one_shot_reply("danke", "de")  # i18n-allow


def test_these_turns_were_already_classified_trivial() -> None:
    """The premise: the router knew, the chain just never asked."""
    for text in ("danke", "hallo", "ok"):  # i18n-allow
        assert classify(text).reason == "one-shot"
