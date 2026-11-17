"""Account-confirmed Codex ``model/list`` for the shared model-list store.

Lives under ``agent_chat`` so ``jarvis.agentic_ide`` never imports
``runner_cli`` (chat/IDE separation). The IDE store injects this callable.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any, Mapping, Sequence

from jarvis.agent_chat.cli_catalog import discover_codex_models

#: Same budget as ``jarvis.agentic_ide.model_lists.CODEX_LIST_TIMEOUT_S``.
_DEFAULT_TIMEOUT_S = 15.0


def list_confirmed_codex_models(
    *, timeout_s: float = _DEFAULT_TIMEOUT_S
) -> Sequence[Mapping[str, Any]] | None:
    """Run ``model/list`` for the installed Codex CLI (no coding turn)."""
    from jarvis.agent_chat.runner_cli import _account_env, _catalog_cwd, codex_argv_prefix

    def _call() -> Sequence[Mapping[str, Any]] | None:
        return discover_codex_models(
            codex_argv_prefix(),
            _account_env("codex"),
            _catalog_cwd(),
            timeout_s=timeout_s,
        )

    try:
        import asyncio

        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop: probe only. Silence is right — RuntimeError here means
        # "safe to call sync", not a failure to report.
        return _call()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_call).result(timeout=timeout_s + 5.0)


__all__ = ["list_confirmed_codex_models"]
