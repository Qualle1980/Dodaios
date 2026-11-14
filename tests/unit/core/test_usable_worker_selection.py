"""Capability-gated worker-tier resolution (AP-21/22/30).

Defect shape (2026-11-10): ``[brain.worker] provider = claude-api`` while no
Anthropic key exists — the provider is disabled at boot, but
``worker_selection`` still returned it and scheduled tasks / seats hard-pinned
onto a dead brain. Mission workers already crossed families via
``_cross_family_last_resort_worker``; the Agents/task/seat path did not.
"""
from __future__ import annotations

import pytest

from jarvis.core.config import BrainTierConfig, JarvisConfig
from jarvis.core.model_selection import usable_worker_selection, worker_selection
from jarvis.local_models import assistant_session as policy


def _cfg(provider: str = "claude-api", model: str | None = None, **fallback: str) -> JarvisConfig:
    cfg = JarvisConfig()
    cfg.brain.primary = "gemini"
    cfg.brain.worker = BrainTierConfig(provider=provider, model=model, **fallback)
    return cfg


def test_worker_selection_still_reports_the_configured_dead_provider() -> None:
    """Display/persistence keep the configured pick; runtime uses usable_*."""
    cfg = _cfg("claude-api")
    selected = worker_selection(cfg)
    assert selected is not None
    assert selected.provider == "claude-api"


def test_usable_worker_resolves_to_a_keyed_tool_capable_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured claude-api is disabled; gemini has a key and can call tools."""
    cfg = _cfg("claude-api")
    monkeypatch.setattr(policy, "is_default_usable", lambda p: p == "gemini")
    monkeypatch.setattr(policy, "is_tool_capable", lambda p: p != "antigravity")
    monkeypatch.setattr(policy, "_known", lambda p: p in {"claude-api", "gemini", "openrouter"})

    selected = usable_worker_selection(cfg)
    assert selected is not None
    assert selected.provider == "gemini"


def test_usable_worker_refuses_honestly_when_nothing_is_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg("claude-api")
    monkeypatch.setattr(policy, "is_default_usable", lambda _p: False)
    monkeypatch.setattr(policy, "is_tool_capable", lambda _p: True)
    monkeypatch.setattr(policy, "_known", lambda _p: True)

    assert usable_worker_selection(cfg) is None


def test_usable_worker_never_selects_a_provider_that_cannot_call_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Antigravity drops tool schemas — it must never win the worker tier."""
    cfg = _cfg("antigravity", fallback_provider="gemini", fallback_model="g-flash")
    monkeypatch.setattr(policy, "is_default_usable", lambda _p: True)
    monkeypatch.setattr(policy, "is_tool_capable", lambda p: p != "antigravity")
    monkeypatch.setattr(policy, "_known", lambda p: p in {"antigravity", "gemini"})

    selected = usable_worker_selection(cfg)
    assert selected is not None
    assert selected.provider == "gemini"


def test_usable_worker_never_selects_a_provider_with_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg("claude-api", fallback_provider="openai", fallback_model="o1")
    # Only openrouter (rescue tail) has a key — configured openai does not.
    monkeypatch.setattr(policy, "is_default_usable", lambda p: p == "openrouter")
    monkeypatch.setattr(policy, "is_tool_capable", lambda _p: True)
    monkeypatch.setattr(
        policy,
        "_known",
        lambda p: p in {"claude-api", "openai", "openrouter", "gemini"},
    )

    selected = usable_worker_selection(cfg)
    assert selected is not None
    assert selected.provider == "openrouter"


def test_agents_tier_rescues_a_disabled_configured_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg("claude-api")
    monkeypatch.setattr(policy, "_known", lambda p: p in {"claude-api", "gemini"})
    tier = policy.agents_tier(
        cfg,
        usable=lambda p: p == "gemini",
        tool_capable=lambda p: p != "antigravity",
    )
    assert (tier.provider, tier.ready, tier.reason) == ("gemini", True, "")


def test_task_provider_chain_does_not_hard_pin_a_dead_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jarvis.brain.manager import BrainManager
    from jarvis.core.bus import EventBus

    cfg = _cfg("claude-api")
    mgr = BrainManager(config=cfg, bus=EventBus(), tools={}, tool_executor=None)  # type: ignore[arg-type]
    mgr._active_name = "gemini"
    mgr._dead_providers.add("claude-api")
    monkeypatch.setattr(policy, "is_default_usable", lambda p: p == "gemini")
    monkeypatch.setattr(policy, "is_tool_capable", lambda p: p != "antigravity")
    monkeypatch.setattr(policy, "_known", lambda p: p in {"claude-api", "gemini"})
    monkeypatch.setattr(mgr, "_fast_model", lambda name: f"{name}-fast")
    monkeypatch.setattr(mgr, "_deep_model", lambda name: f"{name}-deep")

    chain = mgr._task_provider_chain("fast")
    assert chain[0][0] == "gemini"
    assert "claude-api" not in [name for name, _ in chain]


def test_jarvis_seat_keeps_the_configured_worker_even_when_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seat is an identity: reachability never moves it to another family.

    An agent's own memory updates run on that agent's seat (AGENTS.md), and
    ``jarvis_seat``'s only runtime caller is the learning reviewer. Routing it
    through ``usable_worker_selection`` made an unusable ``codex`` seat review
    on ``gemini`` — see
    ``tests/unit/memory/learning/test_learning_seat.py``. A seat that cannot
    answer is reported by the reviewer, not swapped here.
    """
    from jarvis.society import seat_brain

    cfg = _cfg("claude-api")
    monkeypatch.setattr(policy, "is_default_usable", lambda p: p == "gemini")
    monkeypatch.setattr(policy, "is_tool_capable", lambda _p: True)
    monkeypatch.setattr(policy, "_known", lambda p: p in {"claude-api", "gemini"})
    # subscription_seat is imported inside jarvis_seat from task_agent.
    monkeypatch.setattr(
        "jarvis.core.task_agent.subscription_seat",
        lambda provider: None,
    )

    seat = seat_brain.jarvis_seat(cfg)
    assert seat.provider == "claude-api"
