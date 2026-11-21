"""The per-message recall must actually reach the prompt the provider sees.

Companion to tests/unit/memory/vault/test_turn_recall.py, which locks the
selection. This proves the wiring, through `render_surface_prompt` - the
function a surface actually calls. The previous PR taught the lesson the hard
way: a test that calls the helper directly stays green with the call unwired.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import jarvis.core.config as core_config
from jarvis.brain import identity_card as ic
from jarvis.brain.manager import BrainManager
from jarvis.core.config import load_config
from jarvis.memory.vault import MemoryVault, runtime
from jarvis.memory.vault.turn_recall import HEADER


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(core_config, "DATA_DIR", tmp_path)
    ic.reset_identity_card_cache()
    handle = MemoryVault(tmp_path / "vault.db")
    runtime.set_active(handle)
    try:
        yield handle
    finally:
        runtime.set_active(None)
        ic.reset_identity_card_cache()


def _manager(tmp_path: Path) -> BrainManager:
    """A BrainManager with __init__ bypassed — only the attrs the prompt needs."""
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
    return manager


async def test_a_memory_about_the_question_reaches_the_prompt(tmp_path, vault) -> None:
    vault.remember(
        "Ahmad met the notary in Hamburg on 2026-10-02.", "episode", entities=["hamburg"]
    )
    manager = _manager(tmp_path)

    prompt, _turn = await manager.render_surface_prompt(user_text="What happened in Hamburg?")

    assert HEADER.splitlines()[0] in prompt
    assert "notary" in prompt


async def test_an_unrelated_question_adds_nothing(tmp_path, vault) -> None:
    vault.remember(
        "Ahmad met the notary in Hamburg on 2026-10-02.", "episode", entities=["hamburg"]
    )
    manager = _manager(tmp_path)

    prompt, _turn = await manager.render_surface_prompt(
        user_text="How do I compile a kernel module?"
    )

    assert HEADER.splitlines()[0] not in prompt


async def test_an_empty_vault_adds_nothing(tmp_path, vault) -> None:
    manager = _manager(tmp_path)

    prompt, _turn = await manager.render_surface_prompt(user_text="What happened in Hamburg?")

    assert HEADER.splitlines()[0] not in prompt


async def test_the_standing_card_is_not_repeated_in_the_turn(tmp_path, vault) -> None:
    """One memory, one prompt: the card holds it, so the turn block must not."""
    vault.remember("Ahmad lives in Hamburg.", "profile", entities=["hamburg"])
    manager = _manager(tmp_path)

    prompt, _turn = await manager.render_surface_prompt(user_text="Tell me about Hamburg.")

    assert prompt.count("Ahmad lives in Hamburg.") == 1


async def test_a_broken_vault_still_yields_a_prompt(tmp_path, vault, monkeypatch) -> None:
    def explode(*_args, **_kwargs):
        raise RuntimeError("no recall today")

    monkeypatch.setattr(runtime, "turn_suffix", explode)
    manager = _manager(tmp_path)

    prompt, _turn = await manager.render_surface_prompt(user_text="What happened in Hamburg?")

    assert prompt, "a recall fault must cost the block, not the turn"
    assert HEADER.splitlines()[0] not in prompt


async def test_the_switched_off_vault_adds_nothing(tmp_path, vault) -> None:
    vault.remember(
        "Ahmad met the notary in Hamburg on 2026-10-02.", "episode", entities=["hamburg"]
    )
    manager = _manager(tmp_path)
    manager._config.memory.vault.enabled = False
    runtime.set_active(None)

    prompt, _turn = await manager.render_surface_prompt(user_text="What happened in Hamburg?")

    assert HEADER.splitlines()[0] not in prompt
