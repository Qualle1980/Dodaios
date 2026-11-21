"""RED: coding-session agents listing is memoised to cut Instant token churn."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_agents_second_call_returns_cached_compact_payload(monkeypatch) -> None:
    from jarvis.society import coding_threads as ct

    calls = {"n": 0}

    async def fake_installed():
        calls["n"] += 1
        return {"claude": True, "codex": True}

    async def fake_signed(_runners):
        return {}

    monkeypatch.setattr(ct, "_installed_clis", fake_installed)
    monkeypatch.setattr(ct, "_signed_in", fake_signed)
    monkeypatch.setattr(
        ct,
        "_provider_rows",
        lambda: [
            SimpleNamespace(
                id="openai-codex",
                agent="codex",
                runner="codex-cli",
                default_model="gpt-6-sol",
                curated_models=(),
            )
        ],
    )
    monkeypatch.setattr(
        ct,
        "_models",
        lambda _row: [
            {"id": "gpt-6-sol", "label": "Sol", "efforts": ["low", "high"]},
            {"id": "gpt-5.6-luna", "label": "Luna", "efforts": ["medium"]},
        ],
    )
    monkeypatch.setattr(ct, "_label_of", lambda _id: "Codex")

    tool = ct.CodingThreadTool(runtime=SimpleNamespace(), agent_id="instant", session_id="s1")
    first = await tool._do_agents(SimpleNamespace(), SimpleNamespace(), {})
    second = await tool._do_agents(SimpleNamespace(), SimpleNamespace(), {})

    assert calls["n"] == 1
    assert second.get("cached") is True
    first_size = len(json.dumps(first, ensure_ascii=False))
    second_size = len(json.dumps(second, ensure_ascii=False))
    # Memoised answer must not grow; models drop effort ladders in the payload.
    assert "efforts" not in json.dumps(first.get("coding_agents"), ensure_ascii=False)
    assert second_size <= first_size + 80
    assert first_size < 2500
