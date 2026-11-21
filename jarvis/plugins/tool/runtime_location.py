"""``runtime_location`` — where this Dodaios process is running.

Answers "are you on the server?", "which instance?", and whether the Control
Key path is active. Same facts as ``GET /api/health``; voice must call this
instead of guessing. Read-only, safe-tier, voice-catalogued like ``product_help``.
"""

from __future__ import annotations

from typing import Any

from jarvis.core.protocols import ExecutionContext, ToolResult
from jarvis.core.runtime_location import runtime_location_facts


class RuntimeLocationTool:
    read_only = True

    name = "runtime_location"
    risk_tier = "safe"
    description = (
        "Where this Dodaios process is running right now: instance (default/dev), "
        "bind host and whether that bind is loopback-only or reachable externally, "
        "listen port, and whether the Control Key path is active. Call this for "
        "any question about this deployment — 'are you on the server?', 'which "
        "machine?', 'is this reachable from outside?' — never guess."
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    async def execute(self, args: dict[str, Any], ctx: ExecutionContext) -> ToolResult:
        del args, ctx
        version: str | None
        try:
            from jarvis import __version__ as version
        except Exception:  # noqa: BLE001 — version is optional enrichment
            version = None
        try:
            facts = runtime_location_facts(version=version)
        except Exception as exc:  # noqa: BLE001
            # Not silence: the failure is handed to the caller as a failed
            # ToolResult carrying the exception type and message.
            return ToolResult(
                success=False,
                output=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        return ToolResult(success=True, output=facts)
