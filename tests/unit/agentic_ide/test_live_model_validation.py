"""Both Instant tools refuse ids outside the live list (2026-11-17)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from jarvis.agentic_ide.folders import probe_project
from jarvis.agentic_ide.model_lists import ModelListSources, ModelListStore, install_shared_store
from jarvis.agentic_ide.orchestration import WorkspaceOrchestrator
from jarvis.agentic_ide.session import Session, Terminal
from jarvis.live.state import LiveLedger
from jarvis.society.coding_threads import match_model, match_provider


def _install(tmp_path: Path) -> ModelListStore:
    grok = tmp_path / "grok.json"
    codex = tmp_path / "codex.json"
    claude = tmp_path / "claude-ccd.json"
    grok.write_text(
        json.dumps(
            {
                "fetched_at": "t",
                "models": {
                    "grok-4.7": {
                        "info": {
                            "id": "grok-4.7",
                            "name": "Grok 4.7",
                            "reasoning_efforts": [{"id": "high", "value": "high", "default": True}],
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    codex.write_text(
        json.dumps(
            {
                "fetched_at": "t",
                "models": [
                    {
                        "slug": "gpt-6.1-sol",
                        "display_name": "Sol",
                        "default_reasoning_level": "low",
                        "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}],
                        "visibility": "list",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    claude.write_text(
        json.dumps(
            {
                "version": 2,
                "fetchedAt": 1,
                "staleAt": 2,
                "catalog": {
                    "config": {
                        "models": [
                            {
                                "id": "claude-haiku-5-5",
                                "name": "Haiku 5.5",
                                "runtime": {
                                    "effort_levels": ["low", "medium", "high"],
                                    "default_effort": "medium",
                                },
                            },
                            {
                                "id": "claude-sonnet-5-5",
                                "name": "Sonnet 5.5",
                                "runtime": {
                                    "effort_levels": ["low", "medium", "high"],
                                    "default_effort": "medium",
                                },
                            },
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    def codex_lister():
        from jarvis.agentic_ide.model_lists import parse_codex_cache

        entries = parse_codex_cache(json.loads(codex.read_text(encoding="utf-8")), read_at=0.0)
        return [
            {
                "id": m.id,
                "label": m.display_name,
                "efforts": list(m.efforts),
                "default_effort": m.default_effort,
            }
            for m in entries
        ]

    store = ModelListStore(
        store_path=tmp_path / "store.json",
        sources=ModelListSources(
            grok_cache=grok,
            codex_cache=codex,
            claude_catalog_file=claude,
            codex_lister=codex_lister,
        ),
        now=lambda: 1_000.0,
    )
    install_shared_store(store)
    return store


class RecordingRegistry:
    def __init__(self, workspace: Session) -> None:
        self._workspace = workspace
        self.workspaces = [workspace]
        self.sessions = [workspace]
        self.active_id = workspace.id
        self.adds: list[dict] = []

    @property
    def session(self):
        return self._workspace

    def get(self, workspace_id: str | None = None):
        if workspace_id in (None, "", self._workspace.id):
            return self._workspace
        return None

    async def add_terminal(self, **kwargs):
        self.adds.append(dict(kwargs))
        term = Terminal(
            f"t{len(self._workspace.terminals) + 1}",
            str(kwargs.get("name") or "T"),
            str(kwargs.get("agent") or "claude"),
            "Claude",
            len(self._workspace.terminals),
            status="live",
            activity="idle",
            activity_at=time.time(),
            model=str(kwargs.get("model") or ""),
            effort=str(kwargs.get("effort") or ""),
        )
        self._workspace.terminals.append(term)
        return term

    async def add_terminals(self, count: int, **kwargs):
        created = []
        for _ in range(max(1, int(count))):
            created.append(await self.add_terminal(**kwargs))
        return created, False


class _Sessions:
    async def run(self, args):
        return {"delivery": "accepted", "submitted": True, "completed": False}


@pytest.fixture
def world(tmp_path: Path):
    from jarvis.agentic_ide import library

    store = _install(tmp_path)
    path = tmp_path / "Personal Jarvis"
    path.mkdir()
    project = library.ensure_project(path)
    workspace = Session(
        id=uuid4().hex,
        folder=str(path),
        name="Personal Jarvis",
        profile=probe_project(path),
        terminals=[],
        created_at=1,
        project_id=project.id,
    )
    registry = RecordingRegistry(workspace)
    ledger = LiveLedger(tmp_path / "receipts.db")
    orchestrator = WorkspaceOrchestrator(registry, _Sessions(), ledger)
    yield orchestrator, registry, workspace, store
    ledger.close()
    install_shared_store(None)


@pytest.mark.asyncio
async def test_workspace_orchestrate_accepts_live_id(world) -> None:
    orchestrator, registry, workspace, _store = world
    result = await orchestrator.run(
        {
            "action": "create",
            "workspace": "Personal Jarvis",
            "cli": "claude",
            "model": "claude-haiku-5-5",
            "effort": "low",
            "count": 1,
        }
    )
    assert result["status"] == "created", result
    assert registry.adds[-1]["model"] == "claude-haiku-5-5"


@pytest.mark.asyncio
async def test_workspace_orchestrate_refuses_id_not_in_live_list(world) -> None:
    orchestrator, registry, workspace, _store = world
    before = len(workspace.terminals)
    result = await orchestrator.run(
        {
            "action": "create",
            "workspace": "Personal Jarvis",
            "cli": "claude",
            "model": "claude-haiku-5-5-style-invented",
            "count": 1,
        }
    )
    assert result["status"] == "not_accepted", result
    reason = str(result.get("reason") or "")
    assert "claude-haiku-5-5-style-invented" in reason
    assert "live" in reason.lower() or "list" in reason.lower() or "age" in reason.lower()
    assert len(workspace.terminals) == before
    assert registry.adds == []


def test_coding_session_accepts_live_id(world) -> None:
    _orchestrator, _registry, _workspace, _store = world
    row = match_provider("Claude Code")
    assert row is not None
    assert match_model(row, "claude-haiku-5-5") == "claude-haiku-5-5"
    assert match_model(row, "Haiku 5.5") == "claude-haiku-5-5"


def test_coding_session_refuses_id_not_in_live_list(world) -> None:
    _orchestrator, _registry, _workspace, _store = world
    row = match_provider("Claude Code")
    assert row is not None
    assert match_model(row, "claude-haiku-5-5-style-invented") == ""
    assert match_model(row, "some-future-model") == ""


def test_workspace_tool_description_includes_live_ids(world) -> None:
    from jarvis.brain.workspace_tool import WorkspaceOrchestrationTool

    _orchestrator, _registry, _workspace, _store = world
    tool = WorkspaceOrchestrationTool(SimpleNamespace())
    text = tool.description
    assert "claude-haiku-5-5" in text
    assert "gpt-6.1-sol" in text
    assert "grok-4.7" in text
