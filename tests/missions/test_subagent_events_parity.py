"""Python <-> TypeScript parity for the two helper-request events (AP-4).

The missions board reads these payloads, so every field the backend emits must
be declared in `missions.ts` and nothing may be declared that is not emitted.
"""
from __future__ import annotations

import re
from pathlib import Path

from jarvis.missions.events import SubagentsDecided, SubagentsRequested

_REPO = Path(__file__).resolve().parents[2]
_TS = _REPO / "jarvis" / "ui" / "web" / "frontend" / "src" / "types" / "missions.ts"


def _ts_fields(interface: str) -> set[str]:
    text = _TS.read_text(encoding="utf-8")
    m = re.search(rf"export interface {interface} extends BasePayload \{{(.*?)\n\}}", text, re.S)
    assert m, f"{interface} interface not found in missions.ts"
    return set(re.findall(r"^\s*([a-z_]+)\??:", m.group(1), re.M))


def test_subagents_requested_python_ts_parity() -> None:
    py = set(SubagentsRequested.model_fields)
    ts = _ts_fields("SubagentsRequested")
    assert py - ts == set(), f"fields missing in missions.ts: {py - ts}"
    assert ts - py == set(), f"fields missing in events.py: {ts - py}"


def test_subagents_decided_python_ts_parity() -> None:
    py = set(SubagentsDecided.model_fields)
    ts = _ts_fields("SubagentsDecided")
    assert py - ts == set(), f"fields missing in missions.ts: {py - ts}"
    assert ts - py == set(), f"fields missing in events.py: {ts - py}"


def test_both_events_are_in_the_typescript_unions() -> None:
    text = _TS.read_text(encoding="utf-8")
    for name in ("SubagentsRequested", "SubagentsDecided"):
        assert f'| "{name}"' in text, f"{name} missing from EventType"
        assert f"| {name};" in text or f"| {name}\n" in text, f"{name} missing from Payload"
