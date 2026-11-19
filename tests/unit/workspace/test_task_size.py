"""Instant size → live catalog model picks (2026-11-17)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.agentic_ide.model_lists import ModelListSources, ModelListStore, install_shared_store
from jarvis.workspace.task_size import TASK_SIZES, pick_for_size


def _write_caches(tmp_path: Path) -> dict[str, Path]:
    grok = tmp_path / "grok.json"
    codex = tmp_path / "codex.json"
    claude = tmp_path / "claude-ccd.json"
    grok.write_text(
        json.dumps(
            {
                "fetched_at": "2026-11-17T12:00:00Z",
                "models": {
                    "grok-4.7-build-fast": {
                        "info": {
                            "id": "grok-4.7-build-fast",
                            "name": "Fast",
                            "reasoning_efforts": [
                                {"id": "low", "value": "low", "default": False},
                                {"id": "high", "value": "high", "default": True},
                            ],
                        }
                    },
                    "grok-4.7": {
                        "info": {
                            "id": "grok-4.7",
                            "name": "Grok 4.7",
                            "reasoning_efforts": [
                                {"id": "low", "value": "low", "default": False},
                                {"id": "medium", "value": "medium", "default": False},
                                {"id": "high", "value": "high", "default": True},
                            ],
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    codex.write_text(
        json.dumps(
            {
                "fetched_at": "2026-11-17T12:00:00Z",
                "models": [
                    {
                        "slug": "gpt-6-luna",
                        "display_name": "Luna",
                        "default_reasoning_level": "medium",
                        "supported_reasoning_levels": [
                            {"effort": "low"},
                            {"effort": "medium"},
                            {"effort": "high"},
                        ],
                        "visibility": "list",
                    },
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
                            {
                                "id": "claude-opus-5-5",
                                "name": "Opus 5.5",
                                "runtime": {
                                    "effort_levels": ["low", "medium", "high", "xhigh"],
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
    return {"grok": grok, "codex": codex, "claude": claude}


@pytest.fixture
def live_store(tmp_path: Path):
    caches = _write_caches(tmp_path)
    def codex_lister():
        from jarvis.agentic_ide.model_lists import parse_codex_cache

        entries = parse_codex_cache(
            json.loads(caches["codex"].read_text(encoding="utf-8")), read_at=0.0
        )
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
            grok_cache=caches["grok"],
            codex_cache=caches["codex"],
            claude_catalog_file=caches["claude"],
            codex_lister=codex_lister,
        ),
        now=lambda: 1_000.0,
    )
    previous = install_shared_store(store)
    yield store
    install_shared_store(previous)


@pytest.mark.parametrize("size", list(TASK_SIZES))
@pytest.mark.parametrize("agent", ["claude", "codex", "grok-build"])
def test_each_size_gives_a_catalog_id(size: str, agent: str, live_store: ModelListStore) -> None:
    pick = pick_for_size(agent, size, store=live_store)
    assert pick.model, f"{agent}/{size} must name a live catalog model"
    result = live_store.ensure(
        {"claude": "claude", "codex": "codex", "grok-build": "grok"}[agent]
    )
    assert pick.model in result.ids()


def test_haiku_never_gets_strong_effort_on_simple(live_store: ModelListStore) -> None:
    pick = pick_for_size("claude", "simple", store=live_store)
    assert "haiku" in pick.model
    assert pick.effort in ("", "low")


def test_no_opus_for_simple(live_store: ModelListStore) -> None:
    pick = pick_for_size("claude", "simple", store=live_store)
    assert "opus" not in pick.model.lower()
    hard = pick_for_size("claude", "hard", store=live_store)
    assert "opus" in hard.model.lower()
    assert hard.effort in ("high", "xhigh", "max", "ultra")


def test_normal_and_hard_efforts(live_store: ModelListStore) -> None:
    assert pick_for_size("claude", "normal", store=live_store).effort == "medium"
    hard_codex = pick_for_size("codex", "hard", store=live_store)
    assert hard_codex.model in live_store.ensure("codex").ids()
    assert hard_codex.effort in ("high", "xhigh", "max", "ultra")
    assert "fast" in pick_for_size("grok-build", "simple", store=live_store).model


def test_unknown_size_or_cli_is_empty(live_store: ModelListStore) -> None:
    assert pick_for_size("claude", "tiny", store=live_store).model == ""
    assert pick_for_size("unknown-cli", "simple", store=live_store).model == ""


def test_never_returns_stale_gpt_5_2(live_store: ModelListStore) -> None:
    for size in TASK_SIZES:
        assert pick_for_size("codex", size, store=live_store).model != "gpt-5.2"
        assert pick_for_size("codex", size, store=live_store).model != "gpt-5.5"
