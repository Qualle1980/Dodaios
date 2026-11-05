"""Parity guard: the TS ``ProviderProofStage`` union mirrors the Python SSOT.

The proof ladder crosses Python -> Pydantic -> TS -> UI, so it owes the
five-layer anti-drift check (AP-4). Order matters here in a way a status
vocabulary does not: the rungs are compared by index, so a union that holds the
right names in the wrong order would silently let a shallower proof satisfy a
deeper claim.
"""

from __future__ import annotations

import re
from pathlib import Path

from jarvis.brain.provider_test import PROOF_STAGES

_TS_FILE = (
    Path(__file__).resolve().parents[4]
    / "jarvis"
    / "ui"
    / "web"
    / "frontend"
    / "src"
    / "hooks"
    / "useProviders.ts"
)


def _parse_ordered_union(name: str, source: str) -> list[str]:
    """The members of ``export type <name> = | "a" | "b";`` in source order."""
    match = re.search(rf"export type {name} =(.*?);", source, re.DOTALL)
    assert match, f"{name} not found in {_TS_FILE.name}"
    return re.findall(r'"([^"]+)"', match.group(1))


def test_the_ts_union_has_the_same_rungs_in_the_same_order() -> None:
    members = _parse_ordered_union("ProviderProofStage", _TS_FILE.read_text(encoding="utf-8"))
    assert members == list(PROOF_STAGES)


def test_the_response_model_carries_the_stage() -> None:
    """A field the UI cannot read would make the honest depth invisible."""
    from jarvis.ui.web.provider_routes import ProviderTestResponse

    assert "proof" in ProviderTestResponse.model_fields
