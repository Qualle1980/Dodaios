"""The spawn precheck and the worker factory must answer about the SAME worker.

``BrainManager._heavy_worker_provider_viable`` decides whether a deterministic
heavy dispatch may start. ``jarvis.missions.init._worker_factory`` then decides
which worker actually runs. They are two readers of one routing decision, and
they read it from different places: the precheck sees the loaded config, the
factory re-resolves the provider live and reads the per-step model.

The precheck once asked ``_select_subagent_worker_kind(None, step_model)`` while
the factory asks it with ``_live_subagent_provider(...)`` — which, with no
``[brain.worker].provider`` configured, resolves to the ACTIVE BRAIN rather than
to nothing. The two therefore reasoned about different workers. These tests pin
the resolution rule itself against the real resolver, so a change to either side
fails here instead of silently drifting.

Deterministic — no provider call, no network, no model, no mission run.
"""
from __future__ import annotations

import pytest

from jarvis.brain.manager import BrainManager
from jarvis.core.bus import EventBus
from jarvis.core.config import BrainTierConfig, JarvisConfig
from jarvis.missions.init import _live_subagent_provider, _select_subagent_worker_kind

_TALKERS = ("chatgpt-web", "chatgpt-plan", "claude-cli", "ollama", "openai", "gemini")


def _manager(primary: str, worker: BrainTierConfig | None = None) -> BrainManager:
    config = JarvisConfig()
    config.brain.primary = primary
    config.brain.worker = worker
    return BrainManager(config=config, bus=EventBus(), tools={})


@pytest.mark.parametrize("primary", _TALKERS)
def test_the_factory_resolves_an_unconfigured_worker_to_the_active_brain(
    primary: str, monkeypatch
) -> None:
    """The rule the precheck has to mirror, asserted against the real resolver."""
    config = JarvisConfig()
    config.brain.primary = primary
    config.brain.worker = None
    monkeypatch.setattr("jarvis.core.config.load_config", lambda *a, **k: config)

    assert _live_subagent_provider(None) == primary


@pytest.mark.parametrize("primary", _TALKERS)
def test_the_precheck_allows_every_worker_the_factory_can_actually_build(
    primary: str,
) -> None:
    """Whatever family the active brain resolves to, the dispatch may start."""
    kind = _select_subagent_worker_kind(primary, "")

    assert kind != "gemini" or primary == "gemini"
    assert _manager(primary)._heavy_worker_provider_viable() is True


def test_a_configured_worker_provider_wins_over_the_active_brain() -> None:
    """The install's own shape: a plan-seat talker with a codex worker."""
    config = JarvisConfig()
    config.brain.primary = "chatgpt-web"
    config.brain.worker = BrainTierConfig(provider="openai-codex", model="")

    assert _select_subagent_worker_kind("openai-codex", "") == "codex_direct"
    mgr = BrainManager(config=config, bus=EventBus(), tools={})
    assert mgr._heavy_worker_provider_viable() is True


@pytest.mark.parametrize("step_model", ["gemini-3-flash", "gpt-5.5", ""])
def test_a_step_model_cannot_divert_a_resolved_provider(step_model: str) -> None:
    """A resolved provider is a hard lock, so the model never flips the gate.

    This is the config-drift case: the precheck reads brain.worker.model and
    the factory reads the per-step model. As long as a provider resolves, the
    two cannot disagree, because neither of them looks at the model at all.
    """
    assert _select_subagent_worker_kind("chatgpt-web", step_model) == "subjarvis"

    mgr = _manager("chatgpt-web", BrainTierConfig(provider="", model=step_model))

    assert mgr._heavy_worker_provider_viable() is True


def test_a_blind_gemini_spawn_with_nothing_configured_is_still_refused() -> None:
    """BUG-017: that worker 403s on an account without Gemini access."""
    config = JarvisConfig()
    config.brain.worker = BrainTierConfig(provider="", model="gemini-3-flash")
    mgr = BrainManager(config=config, bus=EventBus(), tools={})
    object.__setattr__(config.brain, "primary", "")

    assert _select_subagent_worker_kind(None, "gemini-3-flash") == "gemini"
    assert mgr._heavy_worker_provider_viable() is False


def test_switching_the_active_brain_moves_the_precheck_with_the_factory() -> None:
    """A config change must not leave the two sides on different answers."""
    config = JarvisConfig()
    config.brain.worker = None
    mgr = BrainManager(config=config, bus=EventBus(), tools={})

    for primary in ("gemini", "chatgpt-web", "ollama"):
        object.__setattr__(config.brain, "primary", primary)
        kind = _select_subagent_worker_kind(primary, "")

        assert mgr._heavy_worker_provider_viable() is (
            kind != "gemini" or bool(primary)
        ), primary
