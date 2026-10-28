"""Python <-> TypeScript parity for the worker model/billing telemetry (AP-4).

``WorkerSpawned.model`` alone could not tell a request from a fact, which is
how an Opus run was reported as haiku (live 2026-10-28). The split fields are
only useful if every layer that reads the event declares them, so the frontend
types must stay in step with the payloads.
"""
from __future__ import annotations

import re
from pathlib import Path

from jarvis.missions.events import WorkerDraftReady, WorkerSpawned
from jarvis.missions.workers.model_policy import BillingSource

_REPO = Path(__file__).resolve().parents[2]
_TS = _REPO / "jarvis" / "ui" / "web" / "frontend" / "src" / "types" / "missions.ts"


def _ts_fields(interface: str) -> set[str]:
    text = _TS.read_text(encoding="utf-8")
    m = re.search(
        rf"export interface {interface} extends BasePayload \{{(.*?)\n\}}", text, re.S
    )
    assert m, f"{interface} interface not found in missions.ts"
    return set(re.findall(r"^\s*([a-z_]+)\??:", m.group(1), re.M))


def test_worker_spawned_fields_python_ts_parity() -> None:
    assert set(WorkerSpawned.model_fields) == _ts_fields("WorkerSpawned")


def test_worker_draft_ready_fields_python_ts_parity() -> None:
    assert set(WorkerDraftReady.model_fields) == _ts_fields("WorkerDraftReady")


def test_the_telemetry_fields_exist_under_their_intended_names() -> None:
    for field in ("model", "requested_model", "actual_model", "billing_source"):
        assert field in WorkerSpawned.model_fields
    for field in ("actual_model", "billing_source"):
        assert field in WorkerDraftReady.model_fields


def test_unproven_is_the_default_for_both_actual_fields() -> None:
    """A payload built without evidence says ``unknown``, never a guess."""
    assert WorkerSpawned.model_fields["actual_model"].default == "unknown"
    assert WorkerDraftReady.model_fields["actual_model"].default == "unknown"
    assert WorkerSpawned.model_fields["billing_source"].default == "unknown"
    assert WorkerDraftReady.model_fields["billing_source"].default == "unknown"


def test_billing_source_values_match_the_frontend_union() -> None:
    text = _TS.read_text(encoding="utf-8")
    m = re.search(r"export type BillingSource =([^;]+);", text)
    assert m, "BillingSource union not found in missions.ts"
    ts_values = set(re.findall(r'"([a-z_]+)"', m.group(1)))
    assert ts_values == {source.value for source in BillingSource}
