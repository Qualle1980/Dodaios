"""The one memory vault this process writes to (ADR-0044).

Opened on demand and kept, because a background filing run touches it once per
candidate and SQLite connections are not free. The handle is shared, which the
store allows: it serialises its own writes.

The always-on profile block lives with the identity card
(``jarvis/brain/identity_card.py``) and the per-message recall in
``turn_recall``; what stays here is the handle itself, the per-message entry
point and the kind guess both the ``remember`` tool and the one-time import of
the older stores need.
"""
from __future__ import annotations

import logging
import re
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


# Words that hint at a kind, in the languages the product takes input in. Used
# wherever a kind has to be guessed without a model call: an explicit "remember
# that ..." and the one-time import of the older stores.
_KIND_WORDS = {
    "procedural": "always never don't stop immer nie niemals siempre nunca",  # i18n-allow
    "preference": (
        "like love hate prefer favorite favourite "
        "mag liebe hasse lieber bevorzuge prefiero gusta"  # i18n-allow
    ),
    "relationship": (
        "wife husband partner girlfriend boyfriend mother father mom dad sister brother "
        "son daughter friend colleague boss "
        "frau mann freundin freund mutter vater schwester bruder sohn tochter "  # i18n-allow
        "kollege kollegin chef chefin"  # i18n-allow
    ),
    "project": "project deadline goal plan launch projekt ziel abgabe frist",  # i18n-allow
}

_KIND_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (kind, re.compile(r"\b(" + "|".join(map(re.escape, words.split())) + r")\b", re.I))
    for kind, words in _KIND_WORDS.items()
)


def guess_kind(text: str) -> str:
    """A kind for a sentence, without a model call. ``profile`` when nothing hints."""
    for kind, pattern in _KIND_HINTS:
        if pattern.search(text):
            return kind
    return "profile"


def turn_suffix(config: Any, user_text: str) -> str:
    """Memories matching this message, as a prompt block; ``""`` when none fit.

    Excludes whatever the standing block already carries, so a memory is never
    paid for twice in one prompt, and counts what it shows as *surfaced* —
    being shown is not evidence of having helped (ADR-0044 principle 7).

    Never raises: a turn must not fail over ambient knowledge.
    """
    try:
        from jarvis.memory.vault.turn_recall import turn_block  # noqa: PLC0415

        vault = active(config)
        if vault is None or not user_text.strip():
            return ""
        standing = frozenset(
            memory.id
            for _section, memories in vault.profile_selection()
            for memory in memories
        )
        block, shown = turn_block(vault, user_text, standing_ids=standing)
        if shown:
            vault.mark_used(shown)
        return block
    except Exception:  # noqa: BLE001 — recall must never break a turn
        log.warning("memory vault: turn recall unavailable", exc_info=True)
        return ""
