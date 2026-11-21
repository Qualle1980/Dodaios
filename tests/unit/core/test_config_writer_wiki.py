"""Durable, platform-neutral persistence for the selected Obsidian vault."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from jarvis.core.config import JarvisConfig
from jarvis.core.config_writer import set_wiki_vault_root, set_wiki_verified_only


@pytest.mark.parametrize(
    "selected",
    [
        "D:\\Knowledge\\Main Vault\\Jarvis",
        "/srv/knowledge/main-vault/Jarvis",
    ],
)
def test_selected_vault_survives_config_reload(
    tmp_path: Path,
    selected: str,
) -> None:
    config_path = tmp_path / "jarvis.toml"
    config_path.write_text(
        "[wiki_integration]\n"
        "enabled = true\n"
        "subscribe_idle = false\n",
        encoding="utf-8",
    )

    set_wiki_vault_root(selected, path=config_path)

    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    # The file must keep the user's path VERBATIM (a Windows rewrite must not
    # mangle a POSIX vault path or vice versa).
    assert parsed["wiki_integration"]["vault_root"] == selected
    reloaded = JarvisConfig.model_validate(parsed)
    # In memory the value is a Path; compare Path-to-Path so the assertion does
    # not fail on the host platform's separator rendering of a FOREIGN-style
    # path (str(WindowsPath("/srv/x")) == r"\srv\x").
    assert reloaded.wiki_integration.vault_root == Path(selected)
    assert reloaded.wiki_integration.enabled is True
    assert reloaded.wiki_integration.subscribe_idle is False


@pytest.mark.parametrize("enabled", [True, False])
def test_verified_only_switch_survives_config_reload(
    tmp_path: Path,
    enabled: bool,
) -> None:
    """The gate from #157 is reachable through the writer, not by hand (AP-7)."""
    config_path = tmp_path / "jarvis.toml"
    config_path.write_text(
        "[wiki_integration]\nenabled = true\nsubscribe_idle = false\n",
        encoding="utf-8",
    )

    set_wiki_verified_only(enabled, path=config_path)

    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert parsed["wiki_integration"]["verified_only"] is enabled
    reloaded = JarvisConfig.model_validate(parsed)
    assert reloaded.wiki_integration.verified_only is enabled
    # Neighbouring keys in the same table survive the patch.
    assert reloaded.wiki_integration.enabled is True
    assert reloaded.wiki_integration.subscribe_idle is False


def test_verified_only_rejects_a_non_bool(tmp_path: Path) -> None:
    """A truthy string would land in the TOML and read back as a string."""
    config_path = tmp_path / "jarvis.toml"
    config_path.write_text("[wiki_integration]\nenabled = true\n", encoding="utf-8")

    with pytest.raises(TypeError):
        set_wiki_verified_only("true", path=config_path)  # type: ignore[arg-type]

    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert "verified_only" not in parsed["wiki_integration"]
