"""The owner may allow their own API key to do background work.

Background work runs on subscriptions and free local models only, and waits
rather than touch a per-token key — the rule that stopped ~5 EUR vanishing in
three hours with nobody at the machine.

That rule had a cost its author did not intend. On an install that HAS connected
a subscription but whose usable credentials are free-tier API keys, every
candidate is refused and nothing is left, so background work defers for ever.
The memory learning loop IS background work, so the assistant stops learning
about its user. Measured 2026-11-10: a full day of use, zero new memories, while
the owner's question about his own subscription could not be answered.

``[brain] background_may_use_api_key`` lets the owner say an API key of theirs
may do that work. It widens WHO may be billed, not WHAT may be chosen: it names
no provider and no model (AP-21), and it is not a capability probe, because
whether an account sits on a free tier is not observable from this process.
Off by default, so an arbitrary downloader keeps the protection (AP-23).
"""

from __future__ import annotations

import pytest

from jarvis.brain import background_policy as policy


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    policy.reset_for_tests()
    yield
    policy.reset_for_tests()


def _probe(states: dict[str, bool | None]) -> None:
    policy._probe_override = lambda name: states.get(name)


def _allow_api(monkeypatch: pytest.MonkeyPatch, value: bool) -> None:
    """Stand in for the owner's setting without touching a real config file."""
    monkeypatch.setattr(policy, "api_keys_allowed", lambda: value)


# --- the default protects an arbitrary downloader -----------------------------


def test_the_switch_is_off_unless_the_owner_turns_it_on(monkeypatch) -> None:
    from jarvis.core.config import BrainConfig

    assert BrainConfig().background_may_use_api_key is False


def test_an_unreadable_config_keeps_the_purse_shut(monkeypatch) -> None:
    # The safe direction is to wait, never to bill, so a broken config must not
    # read as permission.
    def _boom() -> None:
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("jarvis.core.config.load_config", _boom)

    assert policy.api_keys_allowed() is False


def test_off_still_refuses_every_key_under_a_subscription(monkeypatch) -> None:
    _allow_api(monkeypatch, False)
    _probe({"claude-cli": True})

    # The subscription has to BE in the chain for the policy to see it; that is
    # how the real fallback chain is built.
    result = policy.background_providers(["gemini", "openrouter", "claude-cli"])

    assert result.subscription_mode
    assert "gemini" not in result.allowed
    assert "openrouter" not in result.allowed
    # The signed-in subscription is still allowed to do the work, so the reason
    # is the ordinary one; "waiting" appears only when nothing at all is left.
    assert result.allowed == ("claude-cli",)
    assert result.reason == "subscriptions and local models only"


# --- the measured failure, and that the switch fixes it -----------------------


def test_a_remembered_subscription_with_only_free_keys_left_nothing(monkeypatch) -> None:
    # This is 2026-11-10 exactly: chatgpt-plan still signed in, so subscription
    # mode holds, while the brain runs on gemini. Every candidate is refused and
    # the learning loop defers for ever.
    _allow_api(monkeypatch, False)
    _probe({"chatgpt-plan": True})

    result = policy.background_providers(
        ["gemini", "openrouter", "chatgpt-plan"], primary="gemini"
    )

    assert result.subscription_mode
    # chatgpt-plan is signed in but is no longer what the brain runs on, and the
    # two usable free-tier keys are refused, so nothing is left to do the work.
    assert "gemini" not in result.allowed
    assert "openrouter" not in result.allowed


def test_turning_it_on_lets_the_configured_api_provider_work(monkeypatch) -> None:
    _allow_api(monkeypatch, True)
    _probe({"chatgpt-plan": True})

    result = policy.background_providers(
        ["gemini", "openrouter", "chatgpt-plan"], primary="gemini"
    )

    assert result.subscription_mode
    assert "gemini" in result.allowed
    assert "you allowed" in result.reason


def test_order_is_preserved_so_the_cheapest_candidate_still_leads(monkeypatch) -> None:
    _allow_api(monkeypatch, True)
    _probe({"claude-cli": True})

    result = policy.background_providers(["ollama", "gemini", "openrouter"])

    assert result.allowed == ("ollama", "gemini", "openrouter")


# --- it widens who may bill, not what may be chosen ---------------------------


def test_local_and_subscription_are_unaffected_by_the_switch(monkeypatch) -> None:
    for value in (False, True):
        policy.reset_for_tests()
        monkeypatch.setenv("LOCALAPPDATA", "x")
        _allow_api(monkeypatch, value)
        _probe({"claude-cli": True})

        result = policy.background_providers(["ollama", "claude-cli"])

        assert "ollama" in result.allowed
        assert "claude-cli" in result.allowed


def test_a_signed_out_subscription_is_still_refused_when_keys_are_allowed(
    monkeypatch,
) -> None:
    # The switch must not resurrect a provider that cannot answer. A
    # subscription known to be signed out stays out either way.
    _allow_api(monkeypatch, True)
    _probe({"claude-cli": False, "codex": True})

    result = policy.background_providers(["claude-cli", "gemini", "codex"])

    assert "claude-cli" not in result.allowed
    assert "gemini" in result.allowed


def test_the_switch_names_no_provider(monkeypatch) -> None:
    # AP-21: the module's own docstring promises membership comes from the
    # provider cards, never a name list. This test fails if someone "fixes" the
    # feature by hardcoding a provider.
    import inspect

    source = inspect.getsource(policy.background_providers)
    for name in ("gemini", "openrouter", "groq", "anthropic", "openai"):
        assert name not in source, f"{name} is hardcoded in background_providers"


def test_reading_the_real_config_does_not_raise(monkeypatch) -> None:
    # api_keys_allowed loads the live configuration; it must answer a bool on
    # this machine rather than propagate whatever it finds.
    assert isinstance(policy.api_keys_allowed(), bool)
