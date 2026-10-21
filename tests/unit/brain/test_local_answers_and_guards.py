"""Local date/clock/math answers, the model-resolution cache, and the
production-config guard."""
from __future__ import annotations

import datetime as dt
import os
import tempfile
from pathlib import Path

import pytest

from jarvis.brain.intent_router import one_shot_kind
from jarvis.brain.one_shot_reply import one_shot_reply

FIXED = dt.datetime(2026, 10, 21, 14, 5)  # a Wednesday


# ---- P3: date / weekday ----------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Welcher Tag ist heute?",
        "Welcher Wochentag ist heute?",
        "Welches Datum ist heute?",
        "Der wievielte ist heute?",
        "Was haben wir heute für einen Tag?",
        # the live utterance that cost 11_670 tokens: addressed + greeted
        "Hallo Jarvis, was haben wir heute für einen Tag?",
        "Jarvis, welcher Tag ist heute?",
        "What day is it?",
    ],
)
def test_date_questions_are_answered_locally(text: str) -> None:
    assert one_shot_kind(text) == "date"
    answer = one_shot_reply(text, "de", now=FIXED)
    assert answer is not None
    assert "Mittwoch" in answer, "a day question is answered with the weekday"
    assert "21.10.2026" in answer


def test_clock_survives_being_addressed() -> None:
    assert one_shot_reply("Jarvis, wie spät ist es?", "de", now=FIXED) == "Es ist 14:05 Uhr."


def test_addressing_cannot_turn_a_real_request_into_a_one_shot() -> None:
    """Only the greeting and the wake name are stripped, never content."""
    assert one_shot_kind("Hallo Jarvis, erkläre mir Python ausführlich.") is None
    assert one_shot_kind("Jarvis, analysiere den Build") is None


# ---- P3: arithmetic --------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Was ist 10 plus 10?", "10 plus 10 ist 20."),
        ("Was ist 7 mal 6?", "7 mal 6 ist 42."),
        ("Was ist 5 minus 8?", "5 minus 8 ist -3."),
        ("Wieviel ist 100 geteilt durch 4?", "100 geteilt durch 4 ist 25."),
        ("Was ist 2.5 plus 2.5?", "2.5 plus 2.5 ist 5."),
        ("10 + 10", "10 plus 10 ist 20."),
    ],
)
def test_arithmetic_is_answered_locally(text: str, expected: str) -> None:
    assert one_shot_kind(text) == "math"
    assert one_shot_reply(text, "de", now=FIXED) == expected


def test_division_by_zero_goes_to_a_model() -> None:
    """This module computes; explaining why that is undefined is not its job."""
    assert one_shot_reply("Was ist 10 durch 0?", "de", now=FIXED) is None


def test_open_ended_calculation_still_reaches_the_model() -> None:
    assert one_shot_kind("Rechne mir die Marge aus") is None
    assert one_shot_kind("Was ist der Durchschnitt der letzten drei Monate?") is None


def test_math_never_evaluates_the_utterance() -> None:
    """No utterance may execute anything: parsed numbers only, never eval."""
    assert one_shot_reply("Was ist 2 plus __import__('os').system('echo hi')", "de") is None


def test_an_unknown_reply_language_falls_through() -> None:
    assert one_shot_reply("Was ist 10 plus 10?", "pt", now=FIXED) is None


# ---- P2: the model resolution cache ----------------------------------------

def test_the_model_is_resolved_from_the_account_then_cached() -> None:
    """Nothing is hardcoded; only the per-turn round trip is removed."""
    from jarvis.plugins.brain import chatgpt_plan as cp

    calls: list[int] = []

    class _Auth:
        def models(self) -> list[dict[str, str]]:
            calls.append(1)
            return [{"slug": "model-from-account"}, {"slug": "other"}]

    cp.reset_model_cache()
    auth = _Auth()
    assert cp._resolved_default_model(auth) == "model-from-account"
    assert cp._resolved_default_model(auth) == "model-from-account"
    assert cp._resolved_default_model(auth) == "model-from-account"
    assert len(calls) == 1, "the account list is asked once, not once per turn"
    cp.reset_model_cache()


def test_an_empty_model_list_is_an_error_not_a_guessed_slug() -> None:
    from jarvis.plugins.brain import chatgpt_plan as cp
    from jarvis.plugins.brain.chatgpt_plan_auth import ChatGPTPlanError

    class _Empty:
        def models(self) -> list[dict[str, str]]:
            return []

    cp.reset_model_cache()
    with pytest.raises(ChatGPTPlanError):
        cp._resolved_default_model(_Empty())
    cp.reset_model_cache()


def test_the_provider_reports_the_model_it_actually_used() -> None:
    from jarvis.plugins.brain.chatgpt_plan import ChatGPTPlanBrain

    assert ChatGPTPlanBrain().last_model is None, "nothing claimed before a turn"


# ---- P5: tests must not reach the production config ------------------------

def test_a_non_production_write_leaves_the_global_layers_alone() -> None:
    """A setter called with its own TOML must not touch config-soll or ENV.

    Those two layers take no path and cannot be redirected, so without this
    guard a suite run rewrote the user's pinned provider — observed
    2026-10-21.
    """
    from jarvis.core import config_writer as w

    soll = Path("scripts/config-soll.json")
    before_soll = soll.read_text(encoding="utf-8") if soll.exists() else None
    before_env = os.environ.get("JARVIS__BRAIN__PRIMARY")

    with tempfile.TemporaryDirectory() as tmpdir:
        toml = Path(tmpdir) / "jarvis.toml"
        toml.write_text('[brain]\nprimary = "before"\n', encoding="utf-8")
        w.set_brain_primary("must-not-leak", path=toml)
        assert "must-not-leak" in toml.read_text(encoding="utf-8")

    if before_soll is not None:
        assert soll.read_text(encoding="utf-8") == before_soll
        assert "must-not-leak" not in soll.read_text(encoding="utf-8")
    assert os.environ.get("JARVIS__BRAIN__PRIMARY") == before_env


def test_the_guard_defaults_to_production() -> None:
    """A thread that has not written a config yet must not be locked out."""
    from jarvis.core import config_writer as w

    w._write_target.__dict__.pop("is_production", None)
    assert w._target_is_production() is True
