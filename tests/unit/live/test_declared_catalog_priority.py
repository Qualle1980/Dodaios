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
from jarvis.live.tools import _LOOKUP_TOOLS as _SOURCE_LOOKUP_TOOLS
from jarvis.live.tools import LiveTools

#: The reads a question about the user needs. Each one failed live 2026-11-08.
#: Taken from the source, never copied: two catalogs that each keep their own
#: copy of one list drift, and then a green test proves nothing about what the
#: call was given (the lesson #105/#108/#110 already paid for).
_LOOKUP_TOOLS = tuple(sorted(_SOURCE_LOOKUP_TOOLS))


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


def test_a_heavy_steering_group_does_not_starve_the_memory_reads() -> None:
    """A big app-steering tool must not cost the call its memory.

    Live 2026-11-09, AFTER the CORE_PRIORITY ordering landed: the catalog
    still declared only **13 of 146 tools in 23977 of 24000 bytes**, and the
    memory reads were never called in that session. The ordering did not help
    because ``_PRIORITY_TOOLS`` sorts AHEAD of CORE_PRIORITY, and that group
    is heavy -- measured from the real classes: workspace-orchestrate 8637 B,
    workspace-git 2467 B, navigate 1477 B, ask_agent 1220 B,
    show_on_screen 725 B, find-app-action 702 B, run-app-action 605 B. The
    steering group alone can spend the budget before a read is reached.

    The fixtures above cannot see this: they vary how MANY tools exist, while
    the cut is by BYTES. This one puts a single steering tool heavy enough to
    end the budget in front of the reads, which is the live shape. The exact
    live sizes need not be reproduced -- the invariant is that no steering
    weight may take the reads away.
    """
    from jarvis.live.tools import _CATALOG_BYTE_BUDGET, _PRIORITY_TOOLS
    from tests.fakes.fake_heavy_tool import FakeHeavyTool  # noqa: PLC0415

    tools = _registered_tools()
    installed = [tool for tool in sorted(_LOOKUP_TOOLS) if tool in tools]
    if not installed:
        pytest.skip("no lookup tool is installed in this environment")
    # Not ONE huge tool: the fitting loop drops an oversized declaration with
    # a bare `continue` and carries on, so a single 22 KB tool costs the
    # catalog nothing. The budget is spent the way it is spent live -- by
    # SEVERAL mid-weight steering tools adding up. Each fake here is a real
    # _PRIORITY_TOOLS name, together just over the budget.
    steering = sorted(_PRIORITY_TOOLS)
    each = _CATALOG_BYTE_BUDGET // len(steering) + 200
    for name in steering:
        tools[name] = FakeHeavyTool(name, each)
    live = LiveTools(
        BrainSupervisorToolGateway(_Manager(tools)),
        ledger=None,
        session_id="test",
        language="de",
        backend_model="gemini-live",
    )
    declared = {live._names.get(d["name"], d["name"]) for d in live.declarations()}
    missing = [tool for tool in installed if tool not in declared]
    assert not missing, (
        f"the steering group's weight dropped {missing} from the declaration. "
        "A model not given a memory tool answers that it knows nothing; it "
        "does not go looking (live 2026-11-08, and again 2026-11-09)."
    )
