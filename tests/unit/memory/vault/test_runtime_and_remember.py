"""The off switch holds, and "remember that ..." reaches the vault as well.

Until ADR-0044 chunk 3 flips reads to the vault, the notebook stays the live
path for an explicit request: writing only to the vault would leave the
person's "remember that ..." with no visible effect for a step. So the tool
writes both - except when the vault refuses a credential, which must not land
in the notebook either.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from jarvis.core.config import JarvisConfig
from jarvis.memory.vault import MemoryVault, runtime
from jarvis.plugins.tool.remember import RememberTool


@pytest.fixture
def vault(tmp_path):
    """A temporary vault installed as the process-wide one, and taken out after."""
    handle = MemoryVault(tmp_path / "vault.db")
    runtime.set_active(handle)
    try:
        yield handle
    finally:
        runtime.set_active(None)


@dataclass
class FakeContext:
    """The part of an ExecutionContext this tool reads."""

    user_utterance: str = "Remember that I live in Berlin."


class FakeNotebook:
    """Stands in for `memory.learning.notebook.remember_explicitly`."""

    def __init__(self, *, missing: bool = False) -> None:
        self.saved: list[str] = []
        self.missing = missing

    def __call__(self, fact: str, *, evidence: str = "") -> Any:
        if self.missing:
            raise LookupError("no notebook")
        self.saved.append(fact)
        return object()


@pytest.fixture
def notebook(monkeypatch):
    """Keep the real notebook, and the real data directory, out of the test."""
    import jarvis.memory.learning.notebook as module

    fake = FakeNotebook()
    monkeypatch.setattr(module, "remember_explicitly", fake)
    return fake


# -- the off switch -----------------------------------------------------


def test_the_vault_is_on_by_default() -> None:
    assert runtime.vault_enabled(JarvisConfig()) is True


def test_switching_it_off_hands_back_no_vault(vault) -> None:
    """Even with a vault open, the switch decides."""
    config = JarvisConfig()
    config.memory.vault.enabled = False

    assert runtime.vault_enabled(config) is False
    assert runtime.active(config) is None


def test_a_config_without_a_memory_section_is_treated_as_on() -> None:
    """A partial object must not silently disable memory."""

    class Bare:
        pass

    assert runtime.vault_enabled(Bare()) is True


def test_the_path_follows_the_configured_data_dir(tmp_path) -> None:
    config = JarvisConfig()
    config.memory.data_dir = str(tmp_path)

    assert runtime.vault_path(config) == (tmp_path / "memory" / "vault.db").resolve()


def test_closing_releases_the_handle_and_the_next_use_reopens_it(tmp_path) -> None:
    config = JarvisConfig()
    # Never the real data dir: `active` opens the file, and on a developer's
    # machine that is their own vault.
    config.memory.data_dir = str(tmp_path)
    runtime.set_active(MemoryVault(tmp_path / "first.db"))

    runtime.close_active()
    reopened = runtime.active(config)

    try:
        assert reopened is not None
        assert reopened.path == runtime.vault_path(config)
    finally:
        runtime.close_active()


# -- the remember tool --------------------------------------------------


async def test_an_explicit_request_lands_in_both_stores(vault, notebook) -> None:
    result = await RememberTool().execute({"fact": "I live in Berlin."}, FakeContext())

    assert result.success
    assert notebook.saved == ["I live in Berlin."], "the live path must keep working"
    stored = vault.list()
    assert [m.text for m in stored] == ["I live in Berlin."]
    assert stored[0].origin == "user", "the user said it outright"
    assert stored[0].confidence == 1.0
    assert stored[0].source["via"] == "remember tool"


async def test_a_credential_reaches_neither_store(vault, notebook) -> None:
    secret = "sk-" + "proj-" + "A" * 40

    result = await RememberTool().execute({"fact": f"My key is {secret}"}, FakeContext())

    assert result.success is False
    assert "credential" in (result.error or "")
    assert notebook.saved == [], "a refused secret must not reach the notebook"
    assert vault.list() == []


async def test_a_broken_vault_does_not_break_the_live_path(notebook, monkeypatch) -> None:
    """The notebook is still what the person sees, so a vault fault is not fatal."""

    def explode(_config: Any) -> Any:
        raise RuntimeError("no vault today")

    monkeypatch.setattr(runtime, "active", explode)

    result = await RememberTool().execute({"fact": "I live in Berlin."}, FakeContext())

    assert result.success
    assert notebook.saved == ["I live in Berlin."]


async def test_an_empty_fact_is_rejected_before_any_store(vault, notebook) -> None:
    result = await RememberTool().execute({"fact": "   "}, FakeContext())

    assert result.success is False
    assert notebook.saved == []
    assert vault.list() == []
