"""Central destructive-action class for the approval gate (decision 0002).

A destructive action is a class, not a risk tier. Bypass may not answer for it;
a person still can. Classification reuses the existing shell impact classifier
and an optional per-tool hook — it does not reimplement either.
"""
from __future__ import annotations

import logging
from typing import Any

from jarvis.core.protocols import Tool

from .command_impact import DESTRUCTIVE, classify_command

log = logging.getLogger(__name__)


def is_destructive(tool: Tool, args: dict[str, Any]) -> bool:
    """True when this call deletes or destroys something.

    Order:
    1. A tool that declares ``is_destructive_for_args`` wins (same optional-hook
       pattern as ``risk_tier_for_args`` / ``describe_args``). A broken hook
       fails closed.
    2. A run_shell-shaped call (string ``command`` argument) is classified by
       ``command_impact.classify_command``. Anything the classifier cannot
       settle — missing/non-string command, or an unexpected raise — is
       destructive.
    3. Every other tool is not in this class.
    """
    hook = getattr(tool, "is_destructive_for_args", None)
    if callable(hook):
        try:
            return bool(hook(args))
        except Exception as exc:  # noqa: BLE001 — fail closed on a broken hook
            log.warning(
                "is_destructive_for_args on %r raised %r — treating as destructive",
                getattr(tool, "name", tool),
                exc,
            )
            return True

    if not isinstance(args, dict) or "command" not in args:
        return False

    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return True

    try:
        return classify_command(command).level == DESTRUCTIVE
    except Exception as exc:  # noqa: BLE001 — fail closed
        log.warning(
            "classify_command raised %r for %r — treating as destructive",
            exc,
            getattr(tool, "name", tool),
        )
        return True
