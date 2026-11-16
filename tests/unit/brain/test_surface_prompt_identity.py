"""A subscription CLI seat is told which model it is, not the last brain turn's.

Live 2026-11-15: a Claude seat and an Antigravity seat on the typed chat both
answered "I am Gemini". ``render_surface_prompt`` built the head through
``_build_system_prompt``, which named ``_active_turn_identity`` - the provider of
whatever brain turn ran last on the manager (Gemini, the configured primary).
"""
from __future__ import annotations

from pathlib import Path

import pytest

import jarvis.core.config as core_config
from jarvis.brain import identity_card as ic
from jarvis.brain.manager import BrainManager
from jarvis.core.config import load_config


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BrainManager:
    monkeypatch.setattr(core_config, "DATA_DIR", tmp_path)
    ic.reset_identity_card_cache()
    manager = BrainManager.__new__(BrainManager)
    manager._soul = None
    manager._user_profile = None
    manager._people = None
    manager._core_memory = None
    manager._system_prompt_extra = ""
    manager._wiki_context_suffix = ""
    manager._reply_language = "auto"
    manager._wiki_injector = None
    config = load_config()
    config.wiki_integration.vault_root = tmp_path / "vault"
    manager._config = config
    # The last brain turn ran on Gemini; the seat below is someone else.
    manager._active_turn_identity = ("gemini", "gemini-3.5-flash")
    yield manager
    ic.reset_identity_card_cache()


async def test_the_seat_is_named_not_the_last_brain_turn(manager: BrainManager) -> None:
    prompt, _turn = await manager.render_surface_prompt(
        user_text="Which model are you?", provider="claude-api", model="claude-opus-5"
    )
    assert "Anthropic Claude (model: claude-opus-5)" in prompt
    assert "Google Gemini" not in prompt


async def test_without_a_seat_no_stale_provider_is_named(manager: BrainManager) -> None:
    prompt, _turn = await manager.render_surface_prompt(user_text="Which model are you?")
    assert "ACTIVE BRAIN MODEL" not in prompt
    assert "Google Gemini" not in prompt


async def test_a_surface_build_leaves_the_brain_turn_identity_alone(
    manager: BrainManager,
) -> None:
    await manager.render_surface_prompt(user_text="hi", provider="antigravity", model=None)
    assert manager._active_turn_identity == ("gemini", "gemini-3.5-flash")
    assert "Google Gemini (model: gemini-3.5-flash)" in manager._build_system_prompt()
