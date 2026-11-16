"""Whether a vendor CLI seat is signed in to the login it actually spends.

One reader for every caller that has to know if a coding seat can run: the
IDE picker's health sweep (``jarvis/ui/web/agent_chat_routes.py``) and the
coding-session tool that tells an agent which coding agents it can hand work
to (``jarvis/society/coding_threads.py``). It lives here, below both, so the
society layer does not import a web route.

Never a live API-key call: each CLI's own login file or ``auth status`` answer.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def cli_auth_status(runner: str) -> Any | None:
    """The vendor CLI's own login snapshot. ``None`` if ``runner`` is not a CLI
    this app can read the login of (OpenCode, Kimi and the DeepSeek harness
    keep theirs where no reader has been verified — see the registry)."""
    if runner == "glm-cli":
        # GLM Coding Plan is Claude Code plus a Z.ai key: the key IS the login.
        from types import SimpleNamespace

        from jarvis.workspace.agents import glm_spawn_env

        configured = glm_spawn_env() is not None
        return SimpleNamespace(
            connected=configured,
            mode="subscription",
            message="GLM Coding Plan: Z.ai key saved"
            if configured
            else "GLM Coding Plan: no Z.ai key — add one under API Keys",
        )
    if runner == "claude-cli":
        from jarvis.claude_auth import ClaudeAuthService

        return ClaudeAuthService().status()
    if runner == "codex-cli":
        from jarvis.codex_auth import CodexAuthService

        return CodexAuthService().status()
    if runner == "agy-cli":
        from jarvis.google_cli.auth_service import GoogleCliAuthService

        return GoogleCliAuthService().status()
    if runner == "grok-cli":
        from jarvis.grok_build_auth import GrokBuildAuthService

        return GrokBuildAuthService().status()
    return None


def cli_subscription_connected(runner: str, st: Any) -> bool:
    """Whether ``st`` is a login this CLI seat can actually spend.

    Claude and Grok Build report ``connected=True`` for a stored API key as
    well; that key belongs to a different picker row. Antigravity's
    ``api_key`` mode is the Gemini key, same split. Codex's ``connected``
    is the CLI's own auth.json (ChatGPT login or a key the CLI itself
    holds), which is what ``codex exec`` uses.
    """
    connected = bool(getattr(st, "connected", False))
    mode = (getattr(st, "mode", None) or "").strip().lower()
    if runner == "claude-cli":
        return connected and mode == "subscription"
    if runner == "grok-cli":
        return connected and mode == "subscription"
    if runner == "agy-cli":
        return connected and mode == "oauth-personal"
    return connected


def cli_login_snapshot(runner: str) -> tuple[str, str, str]:
    """Login presence for a vendor CLI seat. Never a live API-key call.

    Returns ``(status, reason, detail)``. A signed-in subscription is
    ``ok``; anything else is ``needs_setup`` so the picker does not borrow
    API-key words ("Key rejected") for a seat that has no key.
    """
    try:
        st = cli_auth_status(runner)
    except Exception as exc:  # noqa: BLE001 — one bad CLI must not lose the sweep
        log.info("agent chat: CLI login check for %s failed: %s", runner, exc)
        return (
            "unknown",
            "check_failed",
            f"The check itself failed ({type(exc).__name__})",
        )
    if st is None:
        # The CLI keeps its own login and this app has no verified reader for
        # it: no dot rather than a guessed one (the turn itself says if the
        # CLI is signed out).
        return "unknown", "self_managed", f"{runner}: keeps its own login"
    connected = cli_subscription_connected(runner, st)
    detail = (getattr(st, "message", None) or "").strip() or (
        f"{runner}: signed in" if connected else f"{runner}: not signed in"
    )
    if connected:
        return "ok", "ok", detail
    return "needs_setup", "not_configured", detail


def seat_signed_in(runner: str) -> bool | None:
    """``True``/``False`` for a seat whose login this app can read, else ``None``."""
    status, _reason, _detail = cli_login_snapshot(runner)
    if status == "ok":
        return True
    if status == "needs_setup":
        return False
    return None


__all__ = [
    "cli_auth_status",
    "cli_login_snapshot",
    "cli_subscription_connected",
    "seat_signed_in",
]
