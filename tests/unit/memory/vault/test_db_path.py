"""The vault's file is resolved once, absolutely, and apart from the wiki's."""
from __future__ import annotations

import os
from pathlib import Path

from jarvis.memory.vault import resolve_vault_db_path
from jarvis.memory.wiki.db_path import resolve_wiki_db_path


def test_the_path_is_absolute_and_independent_of_the_process_cwd(tmp_path) -> None:
    """AP-8's neighbour: a relative data dir must anchor to the repo, not the CWD."""
    before = resolve_vault_db_path()
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert resolve_vault_db_path() == before
    finally:
        os.chdir(cwd)

    assert before.is_absolute()


def test_an_absolute_data_dir_is_taken_as_given(tmp_path) -> None:
    assert resolve_vault_db_path(tmp_path) == (tmp_path / "memory" / "vault.db").resolve()


def test_it_is_not_the_wikis_database() -> None:
    """Someone must be able to export or delete their memory without the wiki."""
    assert resolve_vault_db_path() != resolve_wiki_db_path()
    assert resolve_vault_db_path().name == "vault.db"
    assert resolve_vault_db_path().parent.name == "memory"


def test_the_directory_is_created_on_first_open(tmp_path) -> None:
    from jarvis.memory.vault import MemoryVault

    target = Path(tmp_path) / "fresh" / "memory" / "vault.db"
    with MemoryVault(target) as vault:
        assert vault.stats()["current"] == 0

    assert target.exists()
