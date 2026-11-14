"""Per-operation model selection; never mutate the shared BrainManager for a call."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSelection:
    provider: str
    model: str | None
    reasoning_effort: str = ""
    # An operation-owned client preserves the selected account and billing path.
    # It is intentionally process-local, never persisted or shown in repr/logs.
    brain_override: Any = field(default=None, repr=False, compare=False)


operation_model: ContextVar[ModelSelection | None] = ContextVar("operation_model", default=None)


def _normalize_worker_provider(provider: str) -> str:
    from jarvis.missions.worker_runtime.provider_map import CODEX_SUBAGENT_SLUGS

    if provider in CODEX_SUBAGENT_SLUGS:
        return "codex"
    if provider in {"claude-subscription", "claude-code"}:
        return "claude-cli"
    if provider == "openai-api":
        return "openai"
    return provider


def worker_selection(config: Any) -> ModelSelection | None:
    """The configured ``[brain.worker]`` pick, regardless of credential state.

    Display and persistence read this. Runtime dispatch that needs a brain that
    can actually answer must use :func:`usable_worker_selection` instead — a
    configured ``claude-api`` with no Anthropic key is still returned here.
    """
    worker = getattr(getattr(config, "brain", None), "worker", None)
    if worker is None or not getattr(worker, "provider", ""):
        return None
    provider = _normalize_worker_provider(str(worker.provider))
    return ModelSelection(
        provider, worker.model, str(getattr(worker, "reasoning_effort", ""))
    )


def usable_worker_selection(config: Any) -> ModelSelection | None:
    """First Agents/worker-tier provider that can run on this machine.

    Prefer the configured ``[brain.worker]`` pick when it is tool-capable and
    credentialed. Otherwise walk the Agents-tier chain and its capability-gated
    rescue tail (keyless local, free-tier Gemini, other reachable keyed
    families). Returns ``None`` when nothing is usable so callers refuse
    honestly (AP-21/22/30) instead of hard-pinning a boot-disabled provider.
    """
    from jarvis.local_models.assistant_session import (
        chain_candidates,
        is_default_usable,
        is_tool_capable,
    )

    configured = worker_selection(config)
    if configured is not None:
        try:
            if is_tool_capable(configured.provider) and is_default_usable(configured.provider):
                return configured
        except Exception:  # noqa: BLE001 — a probe glitch falls through to the chain
            log.debug(
                "usable_worker_selection: configured probe failed for %s",
                configured.provider,
                exc_info=True,
            )

    pairs = chain_candidates(config)
    if not pairs:
        return None
    provider, model = pairs[0]
    effort = (
        configured.reasoning_effort
        if configured is not None
        and configured.provider == _normalize_worker_provider(provider)
        else ""
    )
    return ModelSelection(
        _normalize_worker_provider(provider),
        model or None,
        effort,
    )


@contextmanager
def use_operation_model(selection: ModelSelection | None):
    token = operation_model.set(selection)
    try:
        yield
    finally:
        operation_model.reset(token)
