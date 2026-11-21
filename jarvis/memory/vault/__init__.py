"""Personal memory vault: the main assistant's long-term memory (ADR-0044)."""

from jarvis.memory.vault.db_path import resolve_vault_db_path
from jarvis.memory.vault.models import KINDS, Memory, SearchHit, WriteResult
from jarvis.memory.vault.store import MemoryVault

__all__ = [
    "KINDS",
    "Memory",
    "MemoryVault",
    "SearchHit",
    "WriteResult",
    "resolve_vault_db_path",
]
