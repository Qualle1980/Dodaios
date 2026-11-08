"""remember tool: the user says "remember X" → X is saved to Jarvis' MEMORY.md.

Every later conversation (voice, chat) shows it in its prompt.

The notebook lives in the Society lead's folder of the vault
(``society/jarvis/MEMORY.md``, see :mod:`jarvis.memory.learning.notebook`),
so the person can read and edit it in the app or in Obsidian. Without a
running notebook (a headless test, a boot that disabled learning) the fact
goes to the legacy core-memory file instead, so the request is never lost.

Risk tier: safe — only writes local files.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from jarvis.core.config import DATA_DIR
from jarvis.core.protocols import ExecutionContext, ToolResult

log = logging.getLogger(__name__)


class RememberTool:
    name: str = "remember"
    risk_tier: str = "safe"
    description: str = (
        "Save something the user explicitly asked you to remember for future "
        "conversations into Dodaios' long-term MEMORY.md. Pass one self-contained "
        "sentence in the user's language that still makes sense later (resolve "
        "'this' or 'that'). Every later conversation sees it."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "What to remember, as one self-contained sentence.",
            },
            "category": {
                "type": "string",
                "description": "Optional label (e.g. 'preference', 'project'); "
                "used only by the legacy fallback store.",
                "default": "general",
            },
        },
        "required": ["fact"],
    }

    async def execute(self, args: dict[str, Any], ctx: ExecutionContext) -> ToolResult:
        from jarvis.memory.learning.notebook import remember_explicitly

        fact = " ".join(str(args.get("fact") or "").split())
        if not fact:
            return ToolResult(success=False, output=None, error="fact is missing")
        said = str(getattr(ctx, "user_utterance", "") or "")
        # The vault is written first and the notebook stays the live path: the
        # vault is not read by any prompt until ADR-0044 chunk 3, so switching
        # now would leave the person's "remember that ..." with no effect for a
        # step. A memory the vault refuses (a secret) must still not reach the
        # notebook, so a refusal ends the call here.
        refusal = await self._also_in_vault(fact, said)
        if refusal is not None:
            return refusal
        try:
            change = await asyncio.to_thread(remember_explicitly, fact, evidence=said)
        except LookupError:  # no MEMORY.md notebook: use the legacy store
            return await self._legacy(fact, str(args.get("category") or "general").strip())
        except (ValueError, OSError) as exc:  # reported to the model as a failed save
            return ToolResult(success=False, output=None, error=f"not saved: {exc}")
        if change is None:
            return ToolResult(success=True, output=f"Already in MEMORY.md: {fact}")
        return ToolResult(success=True, output=f"Saved to MEMORY.md: {fact}")

    @staticmethod
    async def _also_in_vault(fact: str, said: str) -> ToolResult | None:
        """Write the fact to the memory vault too (ADR-0044).

        Returns a failed :class:`ToolResult` when the vault REFUSED the fact -
        that only happens for something shaped like a credential (AP-2), and
        such a fact must not land in the notebook either. ``None`` means the
        caller carries on: stored, already known, or the vault is switched off
        or unavailable.
        """
        try:
            from jarvis.core import config as cfg  # noqa: PLC0415 - off the import floor
            from jarvis.memory.vault import runtime  # noqa: PLC0415
            from jarvis.memory.vault.candidates import vault_kind  # noqa: PLC0415

            vault = await asyncio.to_thread(runtime.active, cfg.load_config())
            if vault is None:
                return None
            result = await asyncio.to_thread(
                vault.remember,
                fact,
                vault_kind("other"),
                origin="user",
                source={"via": "remember tool", "evidence": said[:500]},
            )
        except Exception:  # noqa: BLE001 - the notebook is still the live path
            log.warning("remember: could not also write to the memory vault", exc_info=True)
            return None
        if result.action == "refused":
            return ToolResult(
                success=False,
                output=None,
                error="not saved: that looks like a credential, and credentials are never stored",
            )
        return None

    @staticmethod
    async def _legacy(fact: str, category: str) -> ToolResult:
        from jarvis.memory import CORE_MEMORY_FILENAME, CoreMemory
        from jarvis.memory.learning.notebook import OWNER_ID
        from jarvis.memory.write_feedback import announce_write

        def _write() -> None:
            with announce_write(OWNER_ID, CORE_MEMORY_FILENAME, operation="add"):
                mem = CoreMemory.load(DATA_DIR / CORE_MEMORY_FILENAME)
                mem.add_fact(fact, category=category or "general")

        try:
            await asyncio.to_thread(_write)
        except Exception as exc:  # noqa: BLE001 — reported to the caller as a failed save
            return ToolResult(success=False, output=None, error=str(exc))
        return ToolResult(success=True, output=f"Remembered: [{category}] {fact}")
