"""Instant create can pass model/effort into coding panes (2026-11-17)."""

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
from jarvis.agentic_ide.session import Session, SessionError, Terminal
from jarvis.live.state import LiveLedger


def _install_create_store(tmp_path: Path) -> None:
    claude = tmp_path / "claude-ccd.json"
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
                                "id": "claude-haiku-4-5-20251001",
                                "name": "Haiku 4.5",
                                "runtime": {"effort_levels": [], "default_effort": ""},
                            },
                            {
                                "id": "claude-haiku-5-5",
                                "name": "Haiku 5.5",
                                "runtime": {
                                    "effort_levels": ["low", "medium", "high"],
                                    "default_effort": "medium",
                                },
                            },
                            {
                                "id": "claude-sonnet-5",
                                "name": "Sonnet 5",
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
    grok = tmp_path / "grok.json"
    grok.write_text(
        json.dumps(
            {
                "fetched_at": "t",
                "models": {
                    "grok-4.7": {
                        "info": {
                            "id": "grok-4.7",
                            "name": "Grok 4.7",
                            "reasoning_efforts": [
                                {"id": "high", "value": "high", "default": True}
                            ],
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    codex = tmp_path / "codex.json"
    codex.write_text(
        json.dumps(
            {
                "fetched_at": "t",
                "models": [
                    {
                        "slug": "gpt-6.1-sol",
                        "display_name": "Sol",
                        "default_reasoning_level": "low",
                        "supported_reasoning_levels": [{"effort": "low"}],
                        "visibility": "list",
                    }
                ],
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

    install_shared_store(
        ModelListStore(
            store_path=tmp_path / "store.json",
            sources=ModelListSources(
                grok_cache=grok,
                codex_cache=codex,
                claude_catalog_file=claude,
                codex_lister=codex_lister,
            ),
            now=lambda: 1_000.0,
        )
    )


class RecordingRegistry:
    """Records model/effort on add_terminal / add_terminals for Instant create."""

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
        model = str(kwargs.get("model") or "")
        effort = str(kwargs.get("effort") or "")
        agent = str(kwargs.get("agent") or "claude")
        name = str(kwargs.get("name") or f"T{len(self._workspace.terminals) + 1}")
        term = Terminal(
            f"t{len(self._workspace.terminals) + 1}",
            name,
            agent,
            agent.title(),
            len(self._workspace.terminals),
            status="live",
            activity="idle",
            activity_at=time.time(),
            model=model,
            effort=effort,
        )
        self._workspace.terminals.append(term)
        return term

    async def add_terminals(self, count: int, **kwargs):
        # Mirror production: forward model/effort into each add_terminal.
        created = []
        for _ in range(max(1, int(count))):
            created.append(await self.add_terminal(**kwargs))
        return created, False


class _Sessions:
    async def run(self, args):
        return {"delivery": "accepted", "submitted": True, "completed": False}


@pytest.fixture
def create_world(tmp_path):
    from jarvis.agentic_ide import library

    _install_create_store(tmp_path)
    path = tmp_path / "Personal Jarvis"
    path.mkdir()
    project = library.ensure_project(path)
    term = Terminal(
        "t1",
        "Alex",
        "claude",
        "Claude",
        0,
        status="live",
        activity="idle",
        activity_at=time.time(),
    )
    workspace = Session(
        id=uuid4().hex,
        folder=str(path),
        name="Personal Jarvis",
        profile=probe_project(path),
        terminals=[term],
        created_at=1,
        project_id=project.id,
    )
    registry = RecordingRegistry(workspace)
    ledger = LiveLedger(tmp_path / "receipts.db")
    orchestrator = WorkspaceOrchestrator(registry, _Sessions(), ledger)
    yield orchestrator, registry, workspace
    ledger.close()
    install_shared_store(None)


@pytest.mark.asyncio
async def test_create_with_model_reaches_terminal(create_world) -> None:
    orchestrator, registry, workspace = create_world
    before = len(workspace.terminals)
    result = await orchestrator.run(
        {
            "action": "create",
            "workspace": "Personal Jarvis",
            "cli": "claude",
            "model": "claude-haiku-4-5-20251001",
            "effort": "medium",
            "count": 1,
        }
    )
    assert result["status"] == "created", result
    assert len(workspace.terminals) == before + 1
    assert registry.adds, "create must call add_terminal/add_terminals"
    last = registry.adds[-1]
    assert last.get("model") == "claude-haiku-4-5-20251001"
    assert last.get("effort") == "medium"
    assert workspace.terminals[-1].model == "claude-haiku-4-5-20251001"
    assert workspace.terminals[-1].effort == "medium"


@pytest.mark.asyncio
async def test_create_rejects_unknown_model_honestly(create_world) -> None:
    orchestrator, registry, workspace = create_world
    before = len(workspace.terminals)
    result = await orchestrator.run(
        {
            "action": "create",
            "workspace": "Personal Jarvis",
            "cli": "claude",
            "model": "not-a-real-model-id",
            "count": 1,
        }
    )
    assert result["status"] == "not_accepted", result
    reason = str(result.get("reason") or "").lower()
    assert "not-a-real-model-id" in reason or "catalog" in reason or "model" in reason
    assert "not supported" in reason or "not in" in reason or "unknown" in reason or "catalog" in reason
    assert len(workspace.terminals) == before
    assert registry.adds == []


@pytest.mark.asyncio
async def test_agents_entry_model_reaches_add_terminals(create_world) -> None:
    orchestrator, registry, workspace = create_world
    result = await orchestrator.run(
        {
            "action": "create",
            "workspace": "Personal Jarvis",
            "agents": [
                {
                    "cli": "claude",
                    "count": 1,
                    "model": "claude-sonnet-5",
                    "effort": "medium",
                }
            ],
        }
    )
    assert result["status"] == "created", result
    assert registry.adds[-1].get("model") == "claude-sonnet-5"
    assert registry.adds[-1].get("effort") == "medium"


def test_workspace_tool_schema_declares_model_and_effort() -> None:
    from jarvis.brain.workspace_tool import WorkspaceOrchestrationTool

    tool = WorkspaceOrchestrationTool(SimpleNamespace())
    props = tool.schema["properties"]
    assert "model" in props
    assert "effort" in props
    agent_items = props["agents"]["items"]["properties"]
    assert "model" in agent_items
    assert "effort" in agent_items
