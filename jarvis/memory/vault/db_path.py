"""Canonical SQLite path of the personal memory vault (ADR-0044).

Its own file, next to the wiki's ``jarvis.db`` rather than inside it: the vault
holds what Dodaios knows about the user, which someone must be able to export,
hand over or delete on its own, without taking the wiki's pages with it.
"""
from __future__ import annotations

from pathlib import Path

from jarvis.core.paths import repo_root


def resolve_vault_db_path(data_dir: str | Path | None = None) -> Path:
    """Return one absolute ``vault.db`` path, independent of the process CWD."""
    raw = Path(data_dir) if data_dir is not None else Path("data")
    directory = raw if raw.is_absolute() else repo_root() / raw
    return (directory / "memory" / "vault.db").resolve(strict=False)


__all__ = ["resolve_vault_db_path"]
