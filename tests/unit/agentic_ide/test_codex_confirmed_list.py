"""RED: Codex picks use account-confirmed lists; cache alone is unconfirmed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.agentic_ide import model_lists as ml


def _cache_file(path: Path, *ids: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "fetched_at": "t",
                "models": [
                    {
                        "slug": mid,
                        "display_name": mid,
                        "default_reasoning_level": "low",
                        "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}],
                        "visibility": "list",
                    }
                    for mid in ids
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_unconfirmed_cache_id_is_refused(tmp_path: Path) -> None:
    """Cache-only Codex ids are not confirmed — Instant must not open them."""
    cache = _cache_file(tmp_path / "codex.json", "gpt-6.1-sol", "gpt-5.6-luna")

    def boom() -> None:
        raise RuntimeError("app-server listing down")

    store = ml.ModelListStore(
        store_path=tmp_path / "store.json",
        sources=ml.ModelListSources(codex_cache=cache, codex_lister=boom),
        now=lambda: 1_000.0,
    )
    result = store.ensure("codex", force=True)
    assert result.confirmed is False
    assert result.source == ml.SOURCE_CACHE_UNCONFIRMED
    checked = ml.validate_model_id("codex", "gpt-6.1-sol", store=store)
    assert isinstance(checked, dict)
    assert checked["status"] == "not_accepted"
    assert "unconfirmed" in checked["reason"].lower() or "review not run" in checked["reason"].lower()
    assert store.pick_codex_reviewer() is None


def test_confirmed_id_is_accepted(tmp_path: Path) -> None:
    cache = _cache_file(tmp_path / "codex.json", "cache-only-id")

    def lister():
        return [
            {
                "id": "gpt-6-sol",
                "label": "Sol",
                "efforts": ["low", "medium", "high"],
                "default_effort": "medium",
            }
        ]

    store = ml.ModelListStore(
        store_path=tmp_path / "store.json",
        sources=ml.ModelListSources(codex_cache=cache, codex_lister=lister),
        now=lambda: 1_000.0,
    )
    assert ml.validate_model_id("codex", "gpt-6-sol", store=store) == "gpt-6-sol"
    refused = ml.validate_model_id("codex", "cache-only-id", store=store)
    assert isinstance(refused, dict)
    assert refused["status"] == "not_accepted"
    picked = store.pick_codex_reviewer()
    assert picked is not None
    assert picked.id == "gpt-6-sol"


def test_refusal_refreshes_confirmed_list(tmp_path: Path) -> None:
    calls = {"n": 0}
    rows = [
        {
            "id": "gpt-6.1-sol",
            "label": "Sol",
            "efforts": ["low"],
            "default_effort": "low",
        }
    ]

    def lister():
        calls["n"] += 1
        if calls["n"] == 1:
            return list(rows)
        return [
            {
                "id": "gpt-5.6-terra",
                "label": "Terra",
                "efforts": ["medium"],
                "default_effort": "medium",
            }
        ]

    store = ml.ModelListStore(
        store_path=tmp_path / "store.json",
        sources=ml.ModelListSources(codex_lister=lister),
        now=lambda: 1_000.0,
        refresh_ttl_s=600.0,
    )
    assert "gpt-6.1-sol" in store.ensure("codex").ids()
    after = store.after_refusal("codex", "gpt-6.1-sol")
    assert "gpt-5.6-terra" in after.ids()
    assert "gpt-6.1-sol" not in after.ids()
    assert calls["n"] >= 2


def test_listing_failure_without_cache_names_failure(tmp_path: Path) -> None:
    def boom():
        return None

    store = ml.ModelListStore(
        store_path=tmp_path / "store.json",
        sources=ml.ModelListSources(codex_cache=None, codex_lister=boom),
        now=lambda: 1_000.0,
    )
    result = store.ensure("codex", force=True)
    assert result.ok is False
    assert result.confirmed is False
    assert result.models == ()
    assert result.error
    checked = ml.validate_model_id("codex", "gpt-6.1-sol", store=store)
    assert isinstance(checked, dict)
    assert "review not run" in checked["reason"].lower() or "unconfirmed" in checked["reason"].lower()


@pytest.mark.asyncio
async def test_open_codex_pane_refuses_when_listing_unconfirmed(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from jarvis.agent_chat.codex_fallback import CodexFallbackResult, open_codex_pane
    from jarvis.agentic_ide.model_lists import ModelListSources, ModelListStore, install_shared_store

    cache = _cache_file(tmp_path / "codex.json", "gpt-6.1-sol")

    def boom():
        raise RuntimeError("listing failed")

    store = ModelListStore(
        store_path=tmp_path / "store.json",
        sources=ModelListSources(codex_cache=cache, codex_lister=boom),
        now=lambda: 1_000.0,
    )
    install_shared_store(store)
    try:
        calls: list[dict] = []

        space = SimpleNamespace(id="ws1", terminals=[])

        class Reg:
            session = space
            sessions = [space]

            async def add_terminal(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(name="Codex", agent="codex")

        result = await open_codex_pane(Reg())
        assert isinstance(result, CodexFallbackResult)
        assert result.action in {"missing", "error"}
        assert "review not run" in result.message.lower()
        assert calls == []
    finally:
        install_shared_store(None)
