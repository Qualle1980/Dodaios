"""``agent-usage`` — read plan percentages for connected coding-CLI seats.

Router-tier, read-only, ``safe``. Calls :func:`jarvis.agent_usage.collect` over
the accounts from :mod:`jarvis.agent_accounts` and returns the same figures the
Accounts panel shows: used percent per window, remaining percent, and whether
the reading is ``live`` or ``cached``.

Offered only when the user explicitly asks (see
``jarvis.brain.agent_usage_gate``); never injects percentages into ordinary
answers. No estimation — unavailable seats are reported as unavailable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, ClassVar

from jarvis.core.protocols import ExecutionContext, ToolResult

log = logging.getLogger(__name__)

_PLATFORMS = ("claude", "codex", "grok-build")


def _remaining(percent_used: float) -> float:
    return max(0.0, min(100.0, 100.0 - float(percent_used)))


def format_usage_report(readings: dict[str, Any]) -> str:
    """Plain-text report the model can quote. No invented numbers."""
    if not readings:
        return "No connected coding-CLI seats reported usage."
    lines: list[str] = []
    for account_id in sorted(readings):
        usage = readings[account_id]
        to_dict = getattr(usage, "to_dict", None)
        data = to_dict() if callable(to_dict) else dict(usage)
        status = str(data.get("status") or "unavailable")
        platform = str(data.get("platform") or "")
        source = str(data.get("source") or "")
        header = f"{account_id} ({platform})"
        if status != "ok":
            message = str(data.get("message") or status)
            lines.append(f"{header}: {status} — {message}")
            continue
        source_label = source if source in {"live", "cached"} else "unknown"
        lines.append(f"{header}: source={source_label}")
        windows = data.get("windows") or ()
        if not windows:
            lines.append("  (no windows)")
            continue
        for window in windows:
            w = window if isinstance(window, dict) else getattr(window, "to_dict", lambda: {})()
            kind = str(w.get("kind") or "other")
            used = float(w.get("percent") or 0.0)
            remaining = _remaining(used)
            minutes = w.get("window_minutes")
            scope = w.get("scope_label")
            resets = w.get("resets_at")
            bits = [
                f"{kind}",
                f"{used:.0f}% used",
                f"{remaining:.0f}% remaining",
            ]
            if minutes:
                bits.append(f"window={minutes} min")
            if scope:
                bits.append(f"scope={scope}")
            if resets:
                bits.append(f"resets_at={resets}")
            lines.append("  - " + ", ".join(bits))
    return "\n".join(lines)


class AgentUsageTool:
    """Read-only plan usage for Claude / Codex / Grok Build seats."""

    read_only = True

    name: ClassVar[str] = "agent-usage"
    risk_tier: ClassVar[str] = "safe"
    description: ClassVar[str] = (
        "Read how much of each connected coding-CLI subscription is already used "
        "(Claude, Codex, Grok Build). Returns used and remaining percent per "
        "window (session/5h and weekly where the provider has them), and whether "
        "each reading is live or cached. Call ONLY when the user explicitly asks "
        "for usage, quota, weekly/session limit, or remaining percent. Never add "
        "these figures to an ordinary answer. No parameters required; optional "
        "platform filter and refresh=true to bypass the short cache."
    )
    schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "platform": {
                "type": "string",
                "enum": list(_PLATFORMS),
                "description": "Limit the reading to one CLI platform.",
            },
            "refresh": {
                "type": "boolean",
                "description": "Bypass the short TTL cache and re-read providers.",
            },
        },
        "required": [],
        "input_examples": [
            {},
            {"platform": "claude"},
            {"platform": "claude", "refresh": True},
        ],
    }

    def __init__(
        self,
        *,
        list_accounts_fn: Callable[[], list[Any]] | None = None,
        collect_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self._list_accounts = list_accounts_fn
        self._collect = collect_fn

    async def execute(
        self, args: dict[str, Any], ctx: ExecutionContext
    ) -> ToolResult:  # noqa: ARG002 — ctx required by the tool protocol
        if not isinstance(args, dict):
            return ToolResult(
                success=False,
                output=None,
                error="invalid_input: agent-usage expects an object",
            )
        platform = args.get("platform")
        if platform is not None and platform not in _PLATFORMS:
            return ToolResult(
                success=False,
                output=None,
                error=f"invalid_input: platform must be one of {_PLATFORMS}",
            )
        refresh = bool(args.get("refresh") or False)
        try:
            accounts = self._load_accounts(platform)
            collect = self._collect
            if collect is None:
                from jarvis import agent_usage

                collect = agent_usage.collect
            readings = collect(accounts, refresh=refresh)
        except Exception as exc:  # noqa: BLE001 — status read must never break the turn
            log.warning("agent-usage failed: %s", exc, exc_info=True)
            return ToolResult(
                success=False,
                output=None,
                error=f"could not read agent usage: {type(exc).__name__}: {exc}",
            )
        report = format_usage_report(readings)
        return ToolResult(success=True, output=report)

    def _load_accounts(self, platform: str | None) -> list[Any]:
        if self._list_accounts is not None:
            accounts = list(self._list_accounts())
        else:
            from jarvis import agent_accounts

            if platform:
                accounts = list(agent_accounts.list_accounts(platform))
            else:
                accounts = list(agent_accounts.all_accounts())
        if platform:
            accounts = [a for a in accounts if getattr(a, "platform", None) == platform]
        return accounts
