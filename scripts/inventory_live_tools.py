#!/usr/bin/env python3
"""Offline inventory of live/voice tools (F-5).

Builds a freeze matrix over:
  - gateway voice catalog + router catalog
  - live declaration packing (what fits the byte/count budget)
  - discover index (every catalog name is searchable)
  - contract-test mentions
  - optional log greps for executions

Never marks a tool broken because a log grep missed an alias. Missing
execution evidence is ``unverified``, not ``broken``.

Usage (from a worktree, with the Dodaios venv):

  D:\\Projects\\Dodaios\\.venv\\Scripts\\python.exe scripts/inventory_live_tools.py
  D:\\Projects\\Dodaios\\.venv\\Scripts\\python.exe scripts/inventory_live_tools.py --log path\\to\\app.log
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ToolRow:
    name: str
    in_voice_catalog: bool = False
    in_router_catalog: bool = False
    named_or_priority: bool = False
    declared_in_budget: bool = False
    discoverable: bool = False
    contract_tests: list[str] = field(default_factory=list)
    log_hits: int | None = None  # None = not scanned
    execution_status: str = "unverified"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "in_voice_catalog": self.in_voice_catalog,
            "in_router_catalog": self.in_router_catalog,
            "named_or_priority": self.named_or_priority,
            "declared_in_budget": self.declared_in_budget,
            "discoverable": self.discoverable,
            "contract_tests": ";".join(self.contract_tests) or "",
            "log_hits": "" if self.log_hits is None else self.log_hits,
            "execution_status": self.execution_status,
        }


def _gateway() -> Any:
    from jarvis.brain.tool_gateway import BrainSupervisorToolGateway

    manager = SimpleNamespace(_tools={}, _config=SimpleNamespace(computer_use=None))
    return BrainSupervisorToolGateway(manager)


def _voice_and_router_names(gateway: Any) -> tuple[set[str], set[str]]:
    from jarvis.brain.factory import ROUTER_TOOLS

    voice = {d.name for d in gateway.voice_catalog()}
    router = {d.name for d in gateway.catalog()} | set(ROUTER_TOOLS)
    return voice, router


def _priority_names() -> set[str]:
    from jarvis.live.coordination import NAMED_TOOLS
    from jarvis.live.tools import _PRIORITY_TOOLS

    return set(NAMED_TOOLS) | set(_PRIORITY_TOOLS)


def _declared_names(gateway: Any) -> set[str]:
    """Names that fit the current live declaration budget (read-only)."""
    import tempfile

    from jarvis.live.state import LiveLedger
    from jarvis.live.tools import LiveTools

    ledger_path = Path(tempfile.mkdtemp()) / "inventory-ledger.json"
    bridge = LiveTools(
        gateway,
        LiveLedger(ledger_path),
        "inventory-freeze",
        language="en",
        backend_model="inventory",
    )
    declared: set[str] = set()
    for item in bridge.declarations():
        raw = item.get("name") or ""
        # Declarations use jarvis_<alias>; map back through the bridge table.
        canonical = bridge._names.get(raw, raw)
        if canonical.startswith("jarvis_"):
            canonical = canonical.removeprefix("jarvis_")
        declared.add(str(canonical))
        declared.add(str(raw))
    return declared


def _discoverable_names(gateway: Any) -> set[str]:
    from jarvis.live.discovery import discover

    catalog = list(gateway.voice_catalog())
    # Empty query browses; walk pages so every catalog name is checked.
    seen: set[str] = set()
    offset = 0
    while True:
        page = discover(catalog, "", offset=offset)
        for tool in page.get("tools") or []:
            seen.add(str(tool["name"]))
        next_offset = page.get("next_offset")
        if next_offset is None:
            break
        offset = int(next_offset)
    return seen


def _contract_mentions(names: Iterable[str]) -> dict[str, list[str]]:
    """Map tool name -> contract test files that mention it."""
    contract_dir = REPO_ROOT / "tests" / "contract"
    if not contract_dir.is_dir():
        return {}
    files = list(contract_dir.glob("test_*.py"))
    # Pre-read once; match whole-word-ish including kebab/snake.
    contents = {
        path.relative_to(REPO_ROOT).as_posix(): path.read_text(encoding="utf-8", errors="replace")
        for path in files
    }
    hits: dict[str, list[str]] = {name: [] for name in names}
    for name in names:
        pattern = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")
        for rel, text in contents.items():
            if pattern.search(text):
                hits[name].append(rel)
    return hits


def _alias_forms(name: str) -> set[str]:
    """Canonical name plus the live declaration alias forms."""
    forms = {name, name.replace("-", "_"), name.replace("_", "-")}
    forms.add("jarvis_" + re.sub(r"[^A-Za-z0-9_]", "_", name))
    return forms


def _scan_log_hits(log_path: Path, names: Iterable[str]) -> dict[str, int]:
    """Count log lines that mention a tool or one of its aliases.

    A zero count is still ``unverified`` at the row level — logs may use a
    form this scanner does not know. Never promote zero to ``broken``.
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
    counts: dict[str, int] = {}
    for name in names:
        total = 0
        for form in _alias_forms(name):
            total += len(re.findall(re.escape(form), text))
        counts[name] = total
    return counts


def build_inventory(*, log_path: Path | None = None) -> list[ToolRow]:
    gateway = _gateway()
    voice, router = _voice_and_router_names(gateway)
    priority = _priority_names()
    declared = _declared_names(gateway)
    discoverable = _discoverable_names(gateway)
    all_names = sorted(voice | router | priority)
    contracts = _contract_mentions(all_names)
    log_hits = _scan_log_hits(log_path, all_names) if log_path is not None else None

    rows: list[ToolRow] = []
    for name in all_names:
        row = ToolRow(
            name=name,
            in_voice_catalog=name in voice,
            in_router_catalog=name in router,
            named_or_priority=name in priority,
            declared_in_budget=name in declared or any(
                form in declared for form in _alias_forms(name)
            ),
            discoverable=name in discoverable,
            contract_tests=contracts.get(name, []),
        )
        if log_hits is not None:
            row.log_hits = log_hits.get(name, 0)
            # Positive evidence only. Zero stays unverified (alias risk).
            row.execution_status = "seen_in_log" if row.log_hits > 0 else "unverified"
        else:
            row.execution_status = "unverified"
        rows.append(row)
    return rows


def _print_summary(rows: list[ToolRow]) -> None:
    voice = sum(1 for r in rows if r.in_voice_catalog)
    declared = sum(1 for r in rows if r.declared_in_budget)
    contracted = sum(1 for r in rows if r.contract_tests)
    seen = sum(1 for r in rows if r.execution_status == "seen_in_log")
    unverified = sum(1 for r in rows if r.execution_status == "unverified")
    print(f"tools={len(rows)} voice_catalog={voice} declared_in_budget={declared}")
    print(f"contract_mentioned={contracted} log_seen={seen} unverified={unverified}")
    print(
        "Note: execution_status never says 'broken'. "
        "Missing log evidence is always 'unverified'."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional app log to scan for execution evidence (aliases counted).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write the matrix as CSV to this path.",
    )
    args = parser.parse_args(argv)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    rows = build_inventory(log_path=args.log)
    _print_summary(rows)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].as_dict()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row.as_dict())
        print(f"wrote {args.csv}")
    else:
        # Compact stdout table for a freeze run.
        print(
            f"{'name':<36} {'voice':5} {'decl':5} {'disc':5} "
            f"{'contract':8} {'exec'}"
        )
        for row in rows:
            print(
                f"{row.name:<36} "
                f"{'yes' if row.in_voice_catalog else '-':5} "
                f"{'yes' if row.declared_in_budget else '-':5} "
                f"{'yes' if row.discoverable else '-':5} "
                f"{len(row.contract_tests):<8} "
                f"{row.execution_status}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
