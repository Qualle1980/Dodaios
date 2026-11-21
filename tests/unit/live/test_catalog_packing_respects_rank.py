"""The byte budget must not hand a high-ranked tool's slot to a lighter one.

Measured on a live gemini-live call 2026-11-11: the declaration carried 13 of
146 tools in 23977 of 24000 bytes. Three tools that the catalog ranks ahead of
what was declared were absent, each at 360-541 bytes, while ``screen_snapshot``
(275 B, rank 16) still fit. The third eviction was ``workspace-orchestrate``,
and that is why the voice turn that was asked to delegate work could not do it
at all.

The cause is one word. The packing loop dropped a tool that did not fit with a
bare ``continue`` and kept filling the remaining bytes from further down the
order, so weight silently outranked rank.

**Why this test builds its own catalog instead of using the installed one.**
``tests/contract/test_live_tool_registration_parity.py`` asserts the same
promise against the real entry points, and it passes on the unfixed code: this
environment installs 48 of the 146 tools, and the ones that were actually
evicted live are not among them. A regression test that only fails when the
right plugins happen to be installed is not a regression test -- the same trap
recorded on 2026-11-09, when a budget test passed against unfixed code because
one oversized item was harmless. The catalog below is sized so the defect is
deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jarvis.live import tools as live_tools
from jarvis.live.tools import _CATALOG_BYTE_BUDGET, LiveTools


@dataclass(frozen=True)
class _Descriptor:
    name: str
    description: str = "does something"
    input_schema: dict[str, Any] = field(default_factory=dict)
    risk_tier: str = "safe"


class _Gateway:
    def __init__(self, descriptors: list[_Descriptor]) -> None:
        self._descriptors = descriptors

    def catalog(self) -> list[_Descriptor]:
        return list(self._descriptors)


def _bridge(descriptors: list[_Descriptor]) -> LiveTools:
    bridge = object.__new__(LiveTools)
    bridge.gateway = _Gateway(descriptors)
    bridge._names = {}
    bridge.backend_model = "gemini-live"
    bridge.language = "de"
    bridge._catalog_surface_reported = None
    return bridge


def _heavy(name: str, size: int) -> _Descriptor:
    """A descriptor whose wire size is dominated by its description."""
    return _Descriptor(name=name, description="x" * size)


def _declared(descriptors: list[_Descriptor]) -> list[str]:
    bridge = _bridge(descriptors)
    definitions = LiveTools.declarations(bridge)
    return [bridge._names.get(d["name"], d["name"]) for d in definitions]


def test_a_promised_tool_is_declared_even_when_it_is_heavy() -> None:
    """``workspace-orchestrate``, the eviction that broke delegation.

    It is named by the conversation rules, so it is in ``_PRIORITY_TOOLS``. A
    catalog stuffed with light filler must not take its slot.
    """
    filler = [_heavy(f"filler-{index:02d}", 300) for index in range(40)]
    catalog = [_heavy("workspace-orchestrate", 2000), *filler]
    declared = _declared(catalog)
    assert "workspace-orchestrate" in declared, (
        "a promised tool lost its slot to lighter filler: the packing loop is "
        "skipping what does not fit instead of keeping rank"
    )


def test_packing_stops_instead_of_skipping_past_a_heavy_tool() -> None:
    """Rank order is only respected if the loop STOPS at the first miss.

    ``heavy`` and ``light`` are both unpromised, and ``heavy`` comes first in
    the order fed to the packer. If ``light`` is declared while ``heavy`` is
    not, the budget preferred weight over rank -- the 2026-11-11 defect.
    """
    # Fill the budget to just under the top, then offer one heavy tool and one
    # light one behind it.
    bulk = [_heavy(f"aaa-{index:03d}", 1000) for index in range(_CATALOG_BYTE_BUDGET // 1100)]
    catalog = [*bulk, _heavy("zzz-heavy", 4000), _heavy("zzz-light", 50)]
    declared = _declared(catalog)
    assert not ("zzz-light" in declared and "zzz-heavy" not in declared), (
        "a light tool behind a heavy one was declared while the heavy one was "
        "dropped: the packer skipped instead of stopping, so weight outranked "
        "order"
    )


def _wide_schema(properties: int, name_length: int) -> dict[str, Any]:
    """A schema whose bulk is structure, not prose, so shortening cannot shrink it.

    ``compact_declaration`` drops descriptions and value limits but keeps every
    argument name and type, so a schema built this way survives it at nearly
    full size. That is the only way a promised tool can still be undeclarable.
    """
    return {
        "type": "object",
        "properties": {
            f"{'a' * name_length}{index}": {"type": "string"}
            for index in range(properties)
        },
    }


def test_an_oversized_promised_tool_is_declared_in_short(caplog) -> None:
    """A promised tool too heavy to declare in full is declared in short.

    ``workspace-orchestrate`` weighs 8662 of the 24000 bytes and lost its slot
    the moment the memory reads were pinned ahead of it (measured live
    2026-11-12). Dropping it costs the voice turn its delegation; shortening it
    costs only prose, because ``discover_tools`` still returns the full schema.
    A silent drop is the defect, and so is a drop that had a cheaper option.
    """
    oversized = _heavy("workspace-orchestrate", _CATALOG_BYTE_BUDGET * 2)
    with caplog.at_level("INFO", logger=live_tools.log.name):
        declared = _declared([oversized])
    assert "workspace-orchestrate" in declared, (
        "a promised tool that does not fit in full was dropped although its "
        "short declaration fits: shortening comes before dropping"
    )
    assert any(
        "workspace-orchestrate" in record.getMessage() and "short" in record.getMessage()
        for record in caplog.records
    ), "a promised tool was shortened without a word in the log"


def test_a_promised_tool_that_cannot_fit_even_in_short_is_reported(caplog) -> None:
    """AP-30: silence about a broken promise is the defect, not the drop.

    Shortening is the first answer, not the only one. A tool whose bulk is
    argument structure survives shortening at nearly full size, so the budget
    still cannot carry it -- and then it must leave a record. Being told to use
    a tool it was never given makes a model answer that the tool does not
    exist, and three sessions were spent looking for that cause in the prompt
    instead of in the budget.
    """
    unshrinkable = _Descriptor(
        name="workspace-orchestrate",
        description="x" * (_CATALOG_BYTE_BUDGET * 2),
        input_schema=_wide_schema(properties=400, name_length=200),
    )
    with caplog.at_level("ERROR", logger=live_tools.log.name):
        declared = _declared([unshrinkable])
    assert "workspace-orchestrate" not in declared
    assert any(
        "workspace-orchestrate" in record.getMessage() for record in caplog.records
    ), "a promised tool was dropped without a word in the log"
