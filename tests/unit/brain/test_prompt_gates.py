"""The system prompt only carries what this turn can actually use."""
from __future__ import annotations

import pytest

from jarvis.brain.personal_context import wants_personal_context

# ---- the personal-context gate ---------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Wie spät ist es?",
        "Welches Datum ist heute?",
        "Welcher Tag ist heute?",
        "Welcher Wochentag ist heute?",
        "Hallo Jarvis, was haben wir heute für einen Tag?",
        "Was ist 10 plus 10?",
        "Was ist 7 mal 6?",
        "Danke.",
        "Tschüss",
        "What time is it?",
        "Thanks",
    ],
)
def test_impersonal_turns_drop_the_personal_corpus(text: str) -> None:
    assert wants_personal_context(text) is False


@pytest.mark.parametrize(
    "text",
    [
        # strong personal markers beat an impersonal shape
        "Wie spät ist es bei meiner Schwester?",
        "Wie heißt meine Schwester?",
        "Wer ist Ahmad?",
        "Gib mir die Nummer von Peter",
        "Weißt du noch, was ich gesagt habe?",
        "Wie ist die Adresse von meinem Chef?",
        # anything unclassified keeps the context
        "Wie geht es dir heute?",
        "Erzähl mir etwas Interessantes.",
        "Erkläre mir Python ausführlich.",
        "Was hältst du davon?",
        "",
        "   ",
    ],
)
def test_everything_else_keeps_the_personal_corpus(text: str) -> None:
    """The default is INCLUDE: a missing name costs the answer, not just tokens."""
    assert wants_personal_context(text) is True


# ---- both gates reach the prompt builder -----------------------------------

class _Block:
    """Stands in for a renderer whose output must be gated."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def render_for_prompt(self, **_kw: object) -> str:
        self.calls += 1
        return self.text


@pytest.fixture
def mgr():
    pytest.importorskip("jarvis.brain.factory")
    from jarvis.brain.factory import _legacy_full_brain

    return _legacy_full_brain()


def test_people_and_contacts_are_omitted_for_an_impersonal_turn(mgr) -> None:
    people = _Block("PEOPLE-INDEX-MARKER")
    contacts = _Block("CONTACT-INDEX-MARKER")
    mgr._people = people
    mgr._contacts = contacts

    mgr._turn_personal_context = True
    with_corpus = mgr._build_system_prompt()
    assert "PEOPLE-INDEX-MARKER" in with_corpus
    assert "CONTACT-INDEX-MARKER" in with_corpus

    mgr._turn_personal_context = False
    without = mgr._build_system_prompt()
    assert "PEOPLE-INDEX-MARKER" not in without
    assert "CONTACT-INDEX-MARKER" not in without
    assert len(without) < len(with_corpus)


def test_tool_blocks_are_omitted_when_nothing_can_call_a_tool(mgr) -> None:
    mgr._turn_can_use_tools = lambda: True
    with_tools = mgr._build_system_prompt()
    mgr._turn_can_use_tools = lambda: False
    without = mgr._build_system_prompt()
    assert len(without) < len(with_tools), "the tool contract must cost something"


def test_the_tool_gate_keeps_the_contract_when_a_lead_is_installed(mgr) -> None:
    """A tool-capable lead reads those blocks, so they must stay."""
    mgr._router_lead_key = ("claude-api", "some-model")
    assert mgr._turn_can_use_tools() is True


def test_the_tool_gate_errs_toward_keeping_the_contract(mgr) -> None:
    """A broken probe must not silently strip the tool rules."""

    def boom() -> bool:
        raise RuntimeError("registry exploded")

    mgr._active_can_call_tools = boom
    assert mgr._turn_can_use_tools() is True


def test_the_personal_gate_defaults_to_include(mgr) -> None:
    """A prompt built without a classified utterance keeps everything."""
    for attr in ("_turn_personal_context",):
        if hasattr(mgr, attr):
            delattr(mgr, attr)
    assert mgr._turn_wants_personal_context() is True
