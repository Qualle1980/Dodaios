"""The live session measures denials it spoke while tools were attached."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from jarvis.live.session import LiveVoiceSession


@dataclass(frozen=True)
class _Descriptor:
    name: str


class _Gateway:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def catalog(self) -> list[_Descriptor]:
        return [_Descriptor(name) for name in self._names]


class _Tools:
    def __init__(self, names: list[str]) -> None:
        self.gateway = _Gateway(names)

    def catalog(self) -> list[_Descriptor]:
        return self.gateway.catalog()


class _BrokenTools:
    def catalog(self) -> list[_Descriptor]:
        raise RuntimeError("gateway is gone")


def _session(spoken: str, tools: object) -> LiveVoiceSession:
    session = object.__new__(LiveVoiceSession)
    session._captions = {"assistant": spoken, "user": ""}
    session._tools = tools
    return session


DENIAL = "Ich kann nicht im Internet suchen."  # i18n-allow: DE output


def test_a_contradicted_denial_is_reported(caplog) -> None:
    session = _session(DENIAL, _Tools(["search_web", "end_call"]))
    with caplog.at_level(logging.WARNING, logger="jarvis.live.session"):
        session._report_capability_denial()
    assert "capability denied over voice while 2 tools were attached" in caplog.text


def test_an_honest_denial_is_not_reported(caplog) -> None:
    session = _session(DENIAL, _Tools(["end_call", "take_appshot"]))
    with caplog.at_level(logging.WARNING, logger="jarvis.live.session"):
        session._report_capability_denial()
    assert caplog.text == ""


def test_an_ordinary_answer_is_not_reported(caplog) -> None:
    plain = "Im Internet steht, es regnet."  # i18n-allow: DE output
    session = _session(plain, _Tools(["search_web"]))
    with caplog.at_level(logging.WARNING, logger="jarvis.live.session"):
        session._report_capability_denial()
    assert caplog.text == ""


def test_an_unreadable_catalog_does_not_break_the_turn(caplog) -> None:
    """A measurement may never be the reason a turn ends badly -- but it says so."""
    session = _session(DENIAL, _BrokenTools())
    with caplog.at_level(logging.DEBUG, logger="jarvis.live.session"):
        session._report_capability_denial()
    assert "could not read the catalog" in caplog.text
