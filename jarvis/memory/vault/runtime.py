"""The one memory vault this process writes to (ADR-0044).

Opened on demand and kept, because a background filing run touches it once per
candidate and SQLite connections are not free. The handle is shared, which the
store allows: it serialises its own writes.

This holds only what the writing side needs. The always-on profile block and
the per-turn recall that read from here arrive with chunk 3; adding them now
would be code nothing calls.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from jarvis.memory.vault.db_path import resolve_vault_db_path
from jarvis.memory.vault.store import MemoryVault

log = logging.getLogger(__name__)

_lock = threading.Lock()
_active: MemoryVault | None = None


def vault_enabled(config: Any) -> bool:
    """Whether the vault may be written at all (``[memory.vault] enabled``)."""
    vault_cfg = getattr(getattr(config, "memory", None), "vault", None)
    return bool(getattr(vault_cfg, "enabled", True))


def vault_path(config: Any) -> Any:
    """Where the file lives: ``[memory] data_dir`` resolved, then ``memory/vault.db``."""
    data_dir = str(getattr(getattr(config, "memory", None), "data_dir", "") or "").strip()
    return resolve_vault_db_path(data_dir or None)


def active(config: Any) -> MemoryVault | None:
    """The open vault, opening it on first use. ``None`` while it is switched off."""
    global _active
    if not vault_enabled(config):
        return None
    with _lock:
        if _active is None:
            _active = MemoryVault(vault_path(config))
        return _active


def set_active(vault: MemoryVault | None) -> None:
    """Replace the handle. Used by tests and by a configuration change."""
    global _active
    with _lock:
        old, _active = _active, vault
    if old is not None and old is not vault:
        try:
            old.close()
        except Exception:  # noqa: BLE001 - closing a stale handle must not fail a swap
            log.debug("memory vault: could not close the previous handle", exc_info=True)


def close_active() -> None:
    set_active(None)
