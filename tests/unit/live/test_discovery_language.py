"""The on-demand tool index answers the language the user actually speaks.

The live catalog declares 15 of 147 tools (live log 2026-11-15 10:06:08); the
other 132 are reachable ONLY through ``discover_tools``. Discovery is therefore
the single door to 90 % of what the app can do, and a query it cannot score is
a capability the model reports as missing.

``discover`` ranks keyword overlap against English tool names and English
descriptions. Measured offline on 2026-11-16 over a 25-tool catalog, three of
eight everyday German questions returned ZERO tools where the same question in
English returned between three and twenty-one. That is live defect D4, the
wrong refusal, and F-2 recorded the same shape: the model tried
``workspace-git`` twice for a GitHub question and then gave up.
"""

from __future__ import annotations

from jarvis.core.protocols import SupervisorToolDescriptor
from jarvis.live.discovery import discover


def _tool(name: str, description: str) -> SupervisorToolDescriptor:
    return SupervisorToolDescriptor(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {}},
        risk_tier="safe",
    )


#: An English catalog, the way every tool in this repo is written (AGENTS.md:
#: everything committed is English). The user speaks German to it.
CATALOG = [
    _tool("google_calendar", "Read and create events in the user's calendar."),
    _tool("gmail", "Send and read email messages for the user."),
    _tool("screenshot", "Capture a picture of the current screen."),
    _tool("memory-search", "Search what is known about the user."),
    _tool("youtube_music", "Play music and control playback."),
    _tool("task_status", "Report which background tasks are running."),
    _tool("open-app", "Open an application or a browser window."),
]


def test_a_german_question_reaches_the_tool_its_english_twin_reaches() -> None:
    """Same intent, two languages, same tool on top.

    Without a bridge the German words share no stem with the English name or
    description, every tool scores zero, and ``discover`` returns nothing at
    all -- so the model answers that the app cannot do it.
    """
    pairs = [
        ("Was steht heute in meinem Kalender", "google_calendar"),  # i18n-allow: DE input
        ("Schick eine Nachricht an meine Frau", "gmail"),  # i18n-allow: DE input
        ("Mach ein Bild vom Bildschirm", "screenshot"),  # i18n-allow: DE input
        ("Welche Aufgaben laufen gerade", "task_status"),  # i18n-allow: DE input
        ("Spiel Musik ab", "youtube_music"),  # i18n-allow: DE input
    ]
    for question, expected in pairs:
        result = discover(CATALOG, question)
        assert result["tools"], f"German question found no tool at all: {question!r}"
        assert result["tools"][0]["name"] == expected, (
            f"{question!r} -> {result['tools'][0]['name']}, expected {expected}"
        )


def test_an_unmatched_query_shows_what_exists_instead_of_nothing() -> None:
    """A model that is shown nothing concludes the capability does not exist.

    Advice to "try fewer English keywords" is not data. Browsing the index is,
    and the names it returns dispatch directly through the exact-name path.
    """
    result = discover(CATALOG, "Kannst du mir beim Hausbau helfen")  # i18n-allow: DE input

    assert result["total"] == len(CATALOG), "the index must still be visible"
    assert result["tools"], "an unmatched query returned no tools to look at"
    assert {tool["name"] for tool in result["tools"]} <= {t.name for t in CATALOG}


def test_an_exact_name_still_wins_alone() -> None:
    """The bridge must not drag neighbours into an exact lookup (existing rule)."""
    result = discover(CATALOG, "gmail")

    assert [tool["name"] for tool in result["tools"]] == ["gmail"]
