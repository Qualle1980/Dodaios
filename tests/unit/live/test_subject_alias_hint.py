"""The names-only index must not hide a tool whose subject is not in its name.

Live 2026-11-07 20:35, translated: the model said it had no direct access to
the owner's Obsidian — while ``wiki-recall`` sat in the catalog, reachable, and
``discover()`` searches descriptions, so a single ``discover_tools("obsidian")``
would have found it. The model never asked, because under ``defer_catalog`` it
reads tool NAMES only and the word "Obsidian" lives in the description.

Calendar and Gmail worked in the same turn: "Kalender" leads to
``google_calendar`` by name. From "Obsidian" to ``wiki-recall`` there is no
lexical path at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jarvis.live.tools import SUBJECT_ALIASES, LiveTools


@dataclass(frozen=True)
class _Descriptor:
    name: str
    description: str = "does something"
    input_schema: dict[str, Any] = field(default_factory=dict)
    risk_tier: str = "safe"


class _Gateway:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def catalog(self) -> list[_Descriptor]:
        return [_Descriptor(name) for name in self._names]


def _bridge(names: list[str]) -> LiveTools:
    bridge = object.__new__(LiveTools)
    bridge.gateway = _Gateway(names)
    return bridge


def _discover_description(names: list[str]) -> str:
    declarations = LiveTools.declarations(_bridge(names), defer_catalog=True)
    discover = next(d for d in declarations if d["name"] == "discover_tools")
    return str(discover["description"])


def test_obsidian_leads_to_the_tool_that_serves_it() -> None:
    text = _discover_description(["wiki-recall", "gmail", "google_calendar"])
    assert "obsidian" in text.lower()
    assert "wiki-recall" in text


def test_the_hint_is_absent_when_the_tool_is() -> None:
    """A hint for a tool nobody has is a promise the session cannot keep."""
    text = _discover_description(["gmail", "google_calendar"])
    assert "obsidian" not in text.lower()


def test_the_plain_name_index_is_still_there() -> None:
    text = _discover_description(["wiki-recall", "gmail"])
    assert "Available tool names: " in text
    names_part = text.split("Available tool names: ", 1)[1]
    # The names themselves stay clean: an annotated name would be passed back
    # as a tool name and fail to dispatch.
    first_line = names_part.split(".")[0]
    assert "(" not in first_line


def test_every_alias_names_a_real_capability_word() -> None:
    """The map is a lexical bridge, not a place to invent capabilities."""
    for tool, words in SUBJECT_ALIASES.items():
        assert words, f"{tool} has an empty alias set"
        for word in words:
            assert word == word.casefold(), f"{word} is not folded"
            assert word not in tool, f"{word} is already in the name {tool}"
