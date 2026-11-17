"""Codex reviewer pane uses a live list id, never a fixed stale name."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.agent_chat.codex_fallback import open_codex_pane
from jarvis.agentic_ide.model_lists import (
    ModelEntry,
    ModelListSources,
    ModelListStore,
    install_shared_store,
)


class _RecordingRegistry:
    def __init__(self) -> None:
        self.session = SimpleNamespace(id="ws1", terminals=[])
        self.sessions = [self.session]
        self.calls: list[dict] = []

    async def add_terminal(self, **kwargs):
        self.calls.append(dict(kwargs))
        term = SimpleNamespace(name="Codex", agent="codex", model=kwargs.get("model", ""))
        self.session.terminals.append(term)
        return term


@pytest.fixture
def live_codex(tmp_path: Path):
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
                        "supported_reasoning_levels": [
                            {"effort": "low"},
                            {"effort": "medium"},
                            {"effort": "high"},
                            {"effort": "xhigh"},
                        ],
                        "visibility": "list",
                    },
                    {
                        "slug": "gpt-5.6-terra",
                        "display_name": "Terra",
                        "default_reasoning_level": "medium",
                        "supported_reasoning_levels": [
                            {"effort": "low"},
                            {"effort": "medium"},
                            {"effort": "high"},
                        ],
                        "visibility": "list",
                    },
                ],
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
    claude = tmp_path / "claude.json"
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
                                "name": "Haiku",
                                "runtime": {
                                    "effort_levels": ["low"],
                                    "default_effort": "low",
                                },
                            }
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
    yield store
    install_shared_store(None)


@pytest.mark.asyncio
async def test_open_codex_pane_uses_live_model_not_gpt_5_2(live_codex: ModelListStore) -> None:
    reg = _RecordingRegistry()
    result = await open_codex_pane(reg)
    assert not isinstance(result, Exception)
    assert reg.calls, "must call add_terminal"
    model = str(reg.calls[0].get("model") or "")
    assert model
    assert model != "gpt-5.2"
    assert model != "gpt-5.5"
    assert model in live_codex.ensure("codex").ids()


def test_pick_codex_reviewer_never_returns_fixed_absent_name(live_codex: ModelListStore) -> None:
    picked = live_codex.pick_codex_reviewer()
    assert isinstance(picked, ModelEntry)
    assert picked.id != "gpt-5.2"
    assert picked.id in live_codex.ensure("codex").ids()
    assert live_codex.highest_effort(picked) in picked.efforts
