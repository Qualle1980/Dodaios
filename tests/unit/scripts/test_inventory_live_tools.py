"""F-5 — offline live-tool inventory marks missing evidence as unverified."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "inventory_live_tools.py"


def _load():
    import sys

    spec = importlib.util.spec_from_file_location("inventory_live_tools", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses look the module up in sys.modules while the class body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inventory():
    return _load()


def test_alias_forms_cover_kebab_and_jarvis_prefix(inventory) -> None:
    forms = inventory._alias_forms("workspace-orchestrate")
    assert "workspace-orchestrate" in forms
    assert "workspace_orchestrate" in forms
    assert "jarvis_workspace_orchestrate" in forms


def test_build_inventory_never_marks_broken(inventory) -> None:
    rows = inventory.build_inventory(log_path=None)
    assert rows
    assert all(row.execution_status == "unverified" for row in rows)
    assert "broken" not in {row.execution_status for row in rows}
    assert any(row.name == "product_help" and row.in_voice_catalog for row in rows)
    assert any(row.name == "runtime_location" and row.in_voice_catalog for row in rows)


def test_zero_log_hits_stay_unverified(inventory, tmp_path: Path) -> None:
    log_path = tmp_path / "quiet.log"
    log_path.write_text("boot ok\n", encoding="utf-8")
    rows = inventory.build_inventory(log_path=log_path)
    assert all(row.execution_status == "unverified" for row in rows)
    assert all((row.log_hits or 0) == 0 for row in rows)


def test_positive_log_hit_is_seen_not_broken(inventory, tmp_path: Path) -> None:
    log_path = tmp_path / "busy.log"
    log_path.write_text("calling jarvis_product_help now\n", encoding="utf-8")
    rows = {row.name: row for row in inventory.build_inventory(log_path=log_path)}
    assert rows["product_help"].execution_status == "seen_in_log"
    assert rows["product_help"].log_hits and rows["product_help"].log_hits > 0
    # A tool with no log line stays unverified, never broken.
    other = next(row for name, row in rows.items() if name != "product_help")
    assert other.execution_status == "unverified"
