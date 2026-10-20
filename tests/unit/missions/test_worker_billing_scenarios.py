"""Which worker runs, and whose money it spends, in the six real situations.

The heavy-worker fallback is where a wrong answer costs the user money: a
subscription install must never slide onto a stored API key on its own, and a
plan that is out of capacity must park rather than quietly bill something else.
These tests drive the actual resolvers with the probes stubbed, so the ordering
and the billing gates are asserted rather than read.

Deterministic — no network, no provider call, no model, no mission run.
"""
from __future__ import annotations

from typing import Any

import pytest

from jarvis.missions import init as mission_init
from jarvis.missions.workers.api_agent_worker import ApiAgentWorker
from jarvis.missions.workers.claude_direct_worker import ClaudeDirectWorker
from jarvis.missions.workers.codex_direct_worker import CodexDirectWorker

_TASK = "add a health endpoint to the gateway"


@pytest.fixture
def probes(monkeypatch) -> dict[str, Any]:
    """Every credential probe the resolvers consult, all denied by default."""
    state: dict[str, Any] = {
        "claude_binary": None,
        "claude_auth": False,
        "claude_cooldown": False,
        "codex_oauth": False,
        "codex_reauth": False,
        "codex_cooldown": False,
        "viable_keys": set(),
    }
    monkeypatch.setattr(
        "jarvis.missions.workers.claude_direct_worker._resolve_claude_binary",
        lambda: state["claude_binary"],
    )
    monkeypatch.setattr(
        mission_init, "_claude_cli_auth_viable", lambda: state["claude_auth"]
    )
    monkeypatch.setattr(
        "jarvis.claude_quota_state.claude_in_quota_cooldown",
        lambda: state["claude_cooldown"],
    )
    monkeypatch.setattr(
        "jarvis.missions.workers.codex_direct_worker._codex_oauth_available",
        lambda: state["codex_oauth"],
    )
    monkeypatch.setattr(
        "jarvis.codex_auth_state.codex_needs_reauth", lambda: state["codex_reauth"]
    )
    monkeypatch.setattr(
        "jarvis.codex_quota_state.codex_in_quota_cooldown",
        lambda: state["codex_cooldown"],
    )
    monkeypatch.setattr(
        mission_init,
        "_api_key_family_viable",
        lambda provider: provider in state["viable_keys"],
    )
    return state


def _cross(pinned: bool) -> Any:
    return mission_init._cross_family_last_resort_worker(
        _TASK, None, allow_metered=not pinned
    )


# --------------------------------------------------------------------------
# 1-5: the subscription ladder.
# --------------------------------------------------------------------------

def test_1_codex_available_and_authenticated_runs_on_codex(probes) -> None:
    probes["codex_oauth"] = True

    assert isinstance(_cross(pinned=True), CodexDirectWorker)


def test_2_an_exhausted_codex_quota_is_skipped_not_retried(probes) -> None:
    """A capped plan keeps its login connected, so presence is not enough."""
    probes["codex_oauth"] = True
    probes["codex_cooldown"] = True
    probes["viable_keys"] = {"openai", "gemini"}

    assert _cross(pinned=True) is None, "a pinned install must park, not bill a key"


def test_3_an_invalid_codex_login_is_skipped(probes) -> None:
    probes["codex_oauth"] = True
    probes["codex_reauth"] = True
    probes["viable_keys"] = {"openai"}

    assert _cross(pinned=True) is None


def test_4_the_claude_subscription_is_taken_before_codex(probes) -> None:
    """Claude is probed first, so a host with Claude never diverts."""
    probes["claude_binary"] = "/usr/bin/claude"
    probes["claude_auth"] = True
    probes["codex_oauth"] = True

    assert isinstance(_cross(pinned=True), ClaudeDirectWorker)


def test_4b_an_installed_but_dead_claude_hands_over_to_the_codex_plan(probes) -> None:
    probes["claude_binary"] = "/usr/bin/claude"
    probes["claude_auth"] = False
    probes["codex_oauth"] = True

    assert isinstance(_cross(pinned=True), CodexDirectWorker)


def test_5_with_no_subscription_a_pinned_install_never_reaches_a_key(probes) -> None:
    probes["viable_keys"] = {"claude-api", "gemini", "openrouter", "openai"}

    assert _cross(pinned=True) is None, (
        "every metered family was available and all of them must be refused"
    )


def test_5b_an_unpinned_install_may_cross_to_its_own_key(probes) -> None:
    """No subscription was ever connected: the user's single key is all there is."""
    probes["viable_keys"] = {"openrouter"}

    worker = _cross(pinned=False)

    assert isinstance(worker, ApiAgentWorker)


def test_5c_without_credentials_at_all_nothing_is_invented(probes) -> None:
    assert _cross(pinned=False) is None


# --------------------------------------------------------------------------
# 6: no worker configured, the active brain is an API-key provider.
# --------------------------------------------------------------------------

def test_6_the_selected_provider_runs_on_its_own_key(probes) -> None:
    """Not a fallback: openai is the family the user chose and stored a key for."""
    probes["viable_keys"] = {"openai"}

    worker = mission_init._resolve_api_agent_worker(
        "openai", _TASK, None, pinned=True
    )

    assert isinstance(worker, ApiAgentWorker)


def test_6b_a_pinned_install_does_not_cross_to_a_foreign_key(probes) -> None:
    """The selected family has no usable key: a subscription or parking, never
    somebody else's metered family."""
    probes["viable_keys"] = {"gemini", "openrouter"}

    with pytest.raises(mission_init.WorkerCapacityUnavailable):
        mission_init._resolve_api_agent_worker("openai", _TASK, None, pinned=True)
