"""Self-refreshing CLI model lists (RED → GREEN, 2026-11-17).

Fake cache files only — never call a real Claude / Codex / Grok CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.agentic_ide import model_lists as ml


def _grok_payload(ids: list[tuple[str, str, list[str], str]]) -> dict:
    models = {}
    for mid, name, efforts, default in ids:
        models[mid] = {
            "info": {
                "id": mid,
                "name": name,
                "hidden": False,
                "reasoning_efforts": [
                    {
                        "id": e,
                        "value": e,
                        "label": e,
                        "default": e == default,
                    }
                    for e in efforts
                ],
            }
        }
    return {
        "fetched_at": "2026-11-17T12:00:00Z",
        "models": models,
    }


def _codex_payload(ids: list[tuple[str, str, list[str], str]]) -> dict:
    return {
        "fetched_at": "2026-11-17T12:00:00Z",
        "client_version": "0.169.0-test",
        "models": [
            {
                "slug": mid,
                "display_name": name,
                "default_reasoning_level": default,
                "supported_reasoning_levels": [{"effort": e, "description": e} for e in efforts],
                "visibility": "list",
            }
            for mid, name, efforts, default in ids
        ],
    }


def _claude_payload(ids: list[tuple[str, str, list[str], str]]) -> dict:
    return {
        "version": 2,
        "fetchedAt": 1_700_000_000_000,
        "staleAt": 1_700_003_600_000,
        "catalog": {
            "surface": "ccd",
            "config": {
                "id": "ccd",
                "models": [
                    {
                        "id": mid,
                        "name": name,
                        "section": "main",
                        "runtime": {
                            "effort_levels": efforts,
                            "default_effort": default,
                        },
                    }
                    for mid, name, efforts, default in ids
                ],
            },
            "state": {"id": "ccd"},
        },
    }


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return tmp_path / "cli_model_lists.json"


@pytest.fixture
def caches(tmp_path: Path) -> dict[str, Path]:
    grok = tmp_path / "grok_models_cache.json"
    codex = tmp_path / "codex_models_cache.json"
    claude_dir = tmp_path / "claude_model_catalog"
    claude_dir.mkdir()
    claude = claude_dir / "test-ccd.json"
    grok.write_text(
        json.dumps(
            _grok_payload(
                [
                    ("grok-4.7", "Grok 4.7", ["low", "medium", "high", "xhigh"], "high"),
                    ("grok-4.5", "Grok 4.5", ["low", "medium", "high"], "high"),
                ]
            )
        ),
        encoding="utf-8",
    )
    codex.write_text(
        json.dumps(
            _codex_payload(
                [
                    ("gpt-6.1-sol", "GPT-6.1-Sol", ["low", "medium", "high", "xhigh"], "low"),
                    ("gpt-5.6-terra", "GPT-5.6-Terra", ["low", "medium", "high"], "medium"),
                ]
            )
        ),
        encoding="utf-8",
    )
    claude.write_text(
        json.dumps(
            _claude_payload(
                [
                    ("claude-haiku-5-5", "Haiku 5.5", ["low", "medium", "high"], "medium"),
                    ("claude-opus-5-5", "Opus 5.5", ["low", "medium", "high", "xhigh", "max"], "medium"),
                ]
            )
        ),
        encoding="utf-8",
    )
    return {"grok": grok, "codex": codex, "claude": claude}


def _codex_lister(path: Path):
    """Fake account listing that mirrors the test cache file (confirmed)."""

    def load():
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = ml.parse_codex_cache(data, read_at=0.0)
        return [
            {
                "id": m.id,
                "label": m.display_name,
                "efforts": list(m.efforts),
                "default_effort": m.default_effort,
            }
            for m in entries
        ]

    return load


def _store(store: Path, caches: dict[str, Path], *, now: float = 1_000.0) -> ml.ModelListStore:
    return ml.ModelListStore(
        store_path=store,
        sources=ml.ModelListSources(
            grok_cache=caches["grok"],
            codex_cache=caches["codex"],
            claude_catalog_file=caches["claude"],
            codex_lister=_codex_lister(caches["codex"]),
        ),
        now=lambda: now,
        refresh_ttl_s=600.0,
    )


def test_removed_model_disappears_after_refresh(store: Path, caches: dict[str, Path]) -> None:
    clock = {"t": 1_000.0}
    lists = ml.ModelListStore(
        store_path=store,
        sources=ml.ModelListSources(
            grok_cache=caches["grok"],
            codex_cache=caches["codex"],
            claude_catalog_file=caches["claude"],
            codex_lister=_codex_lister(caches["codex"]),
        ),
        now=lambda: clock["t"],
        refresh_ttl_s=600.0,
    )
    first = lists.ensure("grok")
    assert "grok-4.5" in {m.id for m in first.models}

    caches["grok"].write_text(
        json.dumps(
            _grok_payload(
                [("grok-4.7", "Grok 4.7", ["low", "medium", "high", "xhigh"], "high")]
            )
        ),
        encoding="utf-8",
    )
    clock["t"] = 1_000.0 + 601.0
    second = lists.ensure("grok", force=True)
    ids = {m.id for m in second.models}
    assert "grok-4.7" in ids
    assert "grok-4.5" not in ids


def test_new_model_appears_after_refresh(store: Path, caches: dict[str, Path]) -> None:
    clock = {"t": 1_000.0}
    lists = ml.ModelListStore(
        store_path=store,
        sources=ml.ModelListSources(
            grok_cache=caches["grok"],
            codex_cache=caches["codex"],
            claude_catalog_file=caches["claude"],
            codex_lister=_codex_lister(caches["codex"]),
        ),
        now=lambda: clock["t"],
        refresh_ttl_s=600.0,
    )
    lists.ensure("codex")
    caches["codex"].write_text(
        json.dumps(
            _codex_payload(
                [
                    ("gpt-6.1-sol", "GPT-6.1-Sol", ["low", "medium", "high"], "low"),
                    ("gpt-6-luna", "GPT-6-Luna", ["low", "medium", "high"], "medium"),
                ]
            )
        ),
        encoding="utf-8",
    )
    clock["t"] = 1_700.0
    refreshed = lists.ensure("codex", force=True)
    assert refreshed.confirmed is True
    assert "gpt-6-luna" in {m.id for m in refreshed.models}


def test_broken_file_keeps_last_good_list_and_reports(store: Path, caches: dict[str, Path]) -> None:
    lists = _store(store, caches, now=1_000.0)
    good = lists.ensure("claude")
    assert good.ok
    assert {m.id for m in good.models} >= {"claude-haiku-5-5"}

    caches["claude"].write_text("{not-json", encoding="utf-8")
    kept = lists.ensure("claude", force=True)
    assert kept.ok is False
    assert kept.kept_last_good is True
    assert {m.id for m in kept.models} == {m.id for m in good.models}
    assert kept.error
    assert "unreadable" in kept.error.lower() or "json" in kept.error.lower() or "broken" in kept.error.lower()


def test_empty_list_is_an_error_not_no_models(store: Path, caches: dict[str, Path]) -> None:
    caches["grok"].write_text(
        json.dumps({"fetched_at": "2026-11-17T12:00:00Z", "models": {}}),
        encoding="utf-8",
    )
    lists = _store(store, caches)
    result = lists.ensure("grok", force=True)
    assert result.ok is False
    assert result.models == ()
    assert result.error
    assert "empty" in result.error.lower()


def test_normalised_fields_include_effort_and_read_at(store: Path, caches: dict[str, Path]) -> None:
    lists = _store(store, caches, now=2_500.0)
    result = lists.ensure("claude")
    haiku = next(m for m in result.models if m.id == "claude-haiku-5-5")
    assert haiku.display_name == "Haiku 5.5"
    assert "medium" in haiku.efforts
    assert haiku.default_effort == "medium"
    assert haiku.read_at == 2_500.0
    assert result.read_at == 2_500.0


def test_refusal_forces_refresh_even_inside_ttl(store: Path, caches: dict[str, Path]) -> None:
    clock = {"t": 1_000.0}
    lists = ml.ModelListStore(
        store_path=store,
        sources=ml.ModelListSources(
            grok_cache=caches["grok"],
            codex_cache=caches["codex"],
            claude_catalog_file=caches["claude"],
            codex_lister=_codex_lister(caches["codex"]),
        ),
        now=lambda: clock["t"],
        refresh_ttl_s=600.0,
    )
    lists.ensure("codex")
    caches["codex"].write_text(
        json.dumps(
            _codex_payload(
                [("gpt-6.1-sol", "GPT-6.1-Sol", ["low", "medium", "high"], "low")]
            )
        ),
        encoding="utf-8",
    )
    # Still inside TTL — ordinary ensure would keep the old snapshot.
    clock["t"] = 1_100.0
    stale = lists.ensure("codex")
    assert "gpt-5.6-terra" in {m.id for m in stale.models}

    after = lists.after_refusal("codex", "gpt-5.2")
    assert "gpt-5.6-terra" not in {m.id for m in after.models}
    assert "gpt-6.1-sol" in {m.id for m in after.models}
    assert after.confirmed is True
