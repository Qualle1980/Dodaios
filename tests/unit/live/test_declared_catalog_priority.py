"""What a call is actually GIVEN, measured on the declarations themselves.

Live 2026-11-08: every lookup question failed. The user asked, in German,
what the assistant knew about a named friend, and was told it knew nothing
about him -- while the vault held the answer. Quoted verbatim, because the
exact wording is the evidence:
"Was weisst du ueber Christian?" -> "Ich weiss nichts ueber Christian".  # i18n-allow
The
memory tools were registered as entry points, they were in ``voice_catalog()``,
and ``tests/unit/brain/test_voice_catalog_vault.py`` was green -- because that
test builds a gateway over an EMPTY tool dict and only ever sees the handful of
tools ``_voice_tools()`` names explicitly.

The drop happened one layer further down. ``LiveTools.declarations()`` fits the
catalog into ``_CATALOG_BYTE_BUDGET`` and used to order it BY NAME, so the cut
landed around "m" and silently took everything after it. Measured on this
catalog before the fix: 25 of 49 declared, and memory-search, memory-topics,
memory-history, wiki-recall, remember and search-web were all absent.

So this test asserts against ``declarations()`` with a REAL catalog, which is
the only surface that answers "did the model get this tool".
"""

from __future__ import annotations

from typing import Any

import pytest

from jarvis.brain.tool_gateway import BrainSupervisorToolGateway
from jarvis.core.entry_points import entry_points
from jarvis.live.tools import LiveTools

#: The reads a question about the user needs. Each one failed live 2026-11-08.
_LOOKUP_TOOLS = (
    "memory-search",
    "memory-topics",
    "memory-history",
    "wiki-recall",
)


class _Manager:
    def __init__(self, tools: dict[str, Any]) -> None:
        self._tools = dict(tools)
        self._config = None


def _registered_tools() -> dict[str, Any]:
    """Every plugin tool that builds with no arguments, from the entry points.

    Not a fake: a fake catalog cannot show that the real one overflows the
    byte budget, which is the entire failure this test exists for.
    """
    tools: dict[str, Any] = {}
    for entry_point in entry_points(group="jarvis.tool"):
        try:
            tools[entry_point.name] = entry_point.load()()
        except Exception:  # noqa: BLE001, S112
            # A tool needing constructor arguments is not part of this
            # measurement; it is not skipped silently, it is simply not one
            # of the names asserted below.
            continue
    return tools


def _declared_names() -> set[str]:
    tools = _registered_tools()
    if not any(name in tools for name in _LOOKUP_TOOLS):
        pytest.skip("no lookup tool is installed in this environment")
    live = LiveTools(
        BrainSupervisorToolGateway(_Manager(tools)),
        ledger=None,
        session_id="test",
        language="de",
        backend_model="gemini-live",
    )
    declarations = live.declarations()
    # Catalog tools are declared under an alias; _names maps it back.
    return {live._names.get(d["name"], d["name"]) for d in declarations}


@pytest.mark.parametrize("tool", _LOOKUP_TOOLS)
def test_a_call_is_given_the_tool_that_answers_a_question_about_the_user(tool: str) -> None:
    """A model not given a tool says it knows nothing; it does not go looking."""
    declared = _declared_names()
    assert tool in declared, (
        f"{tool} was not declared to the call. The byte budget in "
        "LiveTools.declarations() dropped it, so the model could not look "
        "anything up and answered that it knows nothing."
    )


def test_the_catalog_is_ordered_by_priority_not_by_name() -> None:
    """The alphabet is not a priority order.

    Sorted by name, a budget that fits about half the catalog keeps a..m and
    drops n..z. That is what took remember, search-web, update_soul and
    wiki-recall away from the live path.
    """
    declared = _declared_names()
    late_in_the_alphabet = {"wiki-recall", "search-web", "remember"} & set(_registered_tools())
    assert late_in_the_alphabet <= declared, (
        "tools late in the alphabet were dropped, which means the catalog is "
        "being cut by name order again"
    )
