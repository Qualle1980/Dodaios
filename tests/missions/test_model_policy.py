"""The model-tier policy: no worker escalates to a dearer model on its own.

Live on 2026-10-28 a mission was dispatched with ``step.model = "haiku"`` and
ran on ``claude-opus-4-8``, because the worker accepted the argument and never
read it while a module constant named the premium tier. These tests pin the
decision itself, with no worker and no CLI in the way.
"""
from __future__ import annotations

import pytest

from jarvis.missions.workers.model_policy import (
    DEFAULT_WORKER_MODEL,
    BillingSource,
    ModelTier,
    classify_billing_source,
    cross_family_model,
    explicitly_allowed_models,
    is_claude_model,
    actual_model_is_allowed,
    is_escalation,
    paid_api_usage_authorized,
    resolve_worker_model,
    retry_model_after_rejection,
    tier_of,
)


# --- the ordering ----------------------------------------------------------


def test_tiers_are_ordered_cheapest_first() -> None:
    assert ModelTier.ECONOMY < ModelTier.STANDARD < ModelTier.PREMIUM


@pytest.mark.parametrize(
    ("model", "tier"),
    [
        ("haiku", ModelTier.ECONOMY),
        ("HAIKU", ModelTier.ECONOMY),
        ("claude-haiku-4-5", ModelTier.ECONOMY),
        ("claude-3-5-haiku-20241022", ModelTier.ECONOMY),
        ("sonnet", ModelTier.STANDARD),
        ("claude-sonnet-5-5", ModelTier.STANDARD),
        ("claude-3-5-sonnet-latest", ModelTier.STANDARD),
        ("opus", ModelTier.PREMIUM),
        ("claude-opus-4-8", ModelTier.PREMIUM),
        ("claude-fable-5", ModelTier.PREMIUM),
    ],
)
def test_aliases_and_ids_resolve_to_their_tier(model: str, tier: ModelTier) -> None:
    assert tier_of(model) is tier


@pytest.mark.parametrize("model", ["", None, "grok-4.3", "gpt-5", "gemini-3-pro", "banana"])
def test_a_non_claude_model_has_no_tier(model: str | None) -> None:
    """Unplaceable is not cheap. A foreign id must never look like a tier."""
    assert tier_of(model) is None
    assert is_claude_model(model) is False


# --- the escalation rule ---------------------------------------------------


def test_cheaper_or_equal_is_not_an_escalation() -> None:
    assert is_escalation(requested="sonnet", candidate="sonnet") is False
    assert is_escalation(requested="sonnet", candidate="haiku") is False
    assert is_escalation(requested="opus", candidate="claude-sonnet-5-5") is False


def test_a_dearer_tier_is_an_escalation() -> None:
    assert is_escalation(requested="haiku", candidate="sonnet") is True
    assert is_escalation(requested="haiku", candidate="claude-opus-4-8") is True
    assert is_escalation(requested="sonnet", candidate="opus") is True


def test_an_unplaceable_candidate_counts_as_an_escalation() -> None:
    """The burden of proof sits on the expensive side."""
    assert is_escalation(requested="haiku", candidate="grok-4.3") is True


# --- resolution ------------------------------------------------------------


def test_an_explicit_request_wins_over_configuration() -> None:
    """The precedence ``ApiAgentWorker._resolve_worker_model`` documents."""
    choice = resolve_worker_model(requested="haiku", configured="claude-opus-4-8")
    assert choice.model == "haiku"
    assert choice.source == "step"


def test_a_configured_model_is_an_approval_of_its_tier() -> None:
    choice = resolve_worker_model(requested="", configured="claude-opus-4-8")
    assert choice.model == "claude-opus-4-8"
    assert choice.source == "config"


def test_nothing_requested_and_nothing_configured_is_never_premium() -> None:
    """The defect: an absent configuration used to select Opus."""
    choice = resolve_worker_model(requested="", configured="")
    assert choice.model == DEFAULT_WORKER_MODEL
    assert choice.tier is ModelTier.STANDARD
    assert choice.tier < ModelTier.PREMIUM


def test_an_unknown_configured_model_falls_to_the_default_not_to_premium() -> None:
    choice = resolve_worker_model(requested="", configured="grok-4.3")
    assert choice.model == DEFAULT_WORKER_MODEL
    assert choice.blocked == "grok-4.3"


def test_a_foreign_requested_model_is_refused_not_forwarded() -> None:
    choice = resolve_worker_model(requested="gpt-5", configured="")
    assert choice.model == DEFAULT_WORKER_MODEL
    assert choice.blocked == "gpt-5"


# --- cross-family fallback -------------------------------------------------


def test_a_family_fallback_does_not_approve_a_tier() -> None:
    """Codex login dead -> Claude. The family moved; the tier may not rise."""
    choice = cross_family_model("gpt-5-codex")
    assert choice.model == DEFAULT_WORKER_MODEL
    assert choice.tier < ModelTier.PREMIUM


def test_a_family_fallback_still_honours_a_claude_request() -> None:
    assert cross_family_model("haiku").model == "haiku"


# --- the retry ladder ------------------------------------------------------


def test_a_rejected_model_retries_downwards_only() -> None:
    assert retry_model_after_rejection("claude-opus-4-8") == "sonnet"
    assert retry_model_after_rejection("sonnet") == "haiku"


def test_the_ladder_ends_instead_of_upgrading() -> None:
    """Exhausted means stop. Dropping --model would hand the CLI the choice."""
    assert retry_model_after_rejection("haiku") is None
    assert (
        retry_model_after_rejection("claude-opus-4-8", already_tried=("sonnet", "haiku"))
        is None
    )


def test_a_rejected_unknown_model_gets_one_safe_attempt() -> None:
    assert retry_model_after_rejection("claude-fable-5") == "sonnet"
    assert retry_model_after_rejection("banana") == DEFAULT_WORKER_MODEL
    assert retry_model_after_rejection("banana", already_tried=("sonnet",)) is None


# --- billing ---------------------------------------------------------------


def test_an_oauth_bearer_is_a_subscription_seat() -> None:
    assert (
        classify_billing_source({"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-x"})
        is BillingSource.SUBSCRIPTION_OAUTH
    )


def test_a_classic_api_key_is_metered() -> None:
    assert (
        classify_billing_source({"ANTHROPIC_API_KEY": "sk-ant-api03-x"})
        is BillingSource.API_KEY
    )


def test_an_oauth_token_in_the_api_key_slot_is_still_a_seat() -> None:
    assert (
        classify_billing_source({"ANTHROPIC_API_KEY": "sk-ant-oat01-x"})
        is BillingSource.SUBSCRIPTION_OAUTH
    )


def test_oauth_wins_when_both_are_present() -> None:
    """The CLI reads the OAuth slot first, so that is what gets billed."""
    assert (
        classify_billing_source(
            {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-x", "ANTHROPIC_API_KEY": "sk-ant-api03-y"}
        )
        is BillingSource.SUBSCRIPTION_OAUTH
    )


def test_no_credential_in_the_environment_is_unknown_never_free() -> None:
    """The CLI may authenticate from the user's own store; this layer cannot
    see it, and a guess would be the kind of claim that hid an Opus run."""
    assert classify_billing_source({}) is BillingSource.UNKNOWN


# --- strict id validation (review of PR #67) -------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "gpt-opus-4",
        "grok-sonnet-1",
        "my-haiku-model",
        "sonnet-pro",
        "opus-x",
        "gemini-sonnet",
        "llama-3-opus",
        "haiku.gpt",
    ],
)
def test_a_foreign_id_containing_a_family_word_is_not_a_claude_model(
    model: str,
) -> None:
    """The first matcher accepted any string containing the family word.

    ``gpt-opus-4`` and ``grok-sonnet-1`` would have been priced as Claude
    tiers and then handed to ``claude --model``, which cannot run them. The id
    must be Anthropic's own.
    """
    assert tier_of(model) is None
    assert is_claude_model(model) is False


@pytest.mark.parametrize(
    ("model", "tier"),
    [
        ("claude-3-opus-20240229", ModelTier.PREMIUM),
        ("claude-sonnet-4-5", ModelTier.STANDARD),
        ("anthropic/claude-sonnet-4-5", ModelTier.STANDARD),
        ("us.anthropic.claude-opus-4-1-v1:0", ModelTier.PREMIUM),
        ("CLAUDE-HAIKU-4-5", ModelTier.ECONOMY),
    ],
)
def test_real_claude_ids_still_resolve(model: str, tier: ModelTier) -> None:
    """Including the vendor-prefixed spellings a config may legitimately hold."""
    assert tier_of(model) is tier


def test_a_foreign_id_is_refused_rather_than_forwarded_to_the_cli() -> None:
    choice = resolve_worker_model(requested="gpt-opus-4", configured="")
    assert choice.model == DEFAULT_WORKER_MODEL
    assert choice.blocked == "gpt-opus-4"


# --- paid API authorization (review of PR #67) -----------------------------


def test_the_user_switch_authorizes_paid_api_use(monkeypatch) -> None:
    """``[missions] paid_api_fallback`` is the explicit yes, and it is enough."""
    monkeypatch.setattr(
        "jarvis.missions.capacity.paid_api_fallback_enabled", lambda: True
    )
    monkeypatch.setattr("jarvis.missions.init.install_is_pinned", lambda: True)
    assert paid_api_usage_authorized() is True


def test_a_subscription_install_with_the_switch_off_is_not_authorized(
    monkeypatch,
) -> None:
    """AGENTS.md: a failing subscription makes work WAIT, never bill a key."""
    monkeypatch.setattr(
        "jarvis.missions.capacity.paid_api_fallback_enabled", lambda: False
    )
    monkeypatch.setattr("jarvis.missions.init.install_is_pinned", lambda: True)
    assert paid_api_usage_authorized() is False


def test_an_install_with_no_subscription_may_use_its_own_key(monkeypatch) -> None:
    """A single-key install must keep working — its key IS the primary path."""
    monkeypatch.setattr(
        "jarvis.missions.capacity.paid_api_fallback_enabled", lambda: False
    )
    monkeypatch.setattr("jarvis.missions.init.install_is_pinned", lambda: False)
    assert paid_api_usage_authorized() is True


def test_an_unreadable_pinning_signal_does_not_authorize_spending(
    monkeypatch,
) -> None:
    def _boom() -> bool:
        raise RuntimeError("signal unreadable")

    monkeypatch.setattr(
        "jarvis.missions.capacity.paid_api_fallback_enabled", lambda: False
    )
    monkeypatch.setattr("jarvis.missions.init.install_is_pinned", _boom)
    assert paid_api_usage_authorized() is False


# --- accepting what the CLI reports (review of PR #67) ---------------------


@pytest.mark.parametrize(
    ("approved", "reported"),
    [
        ("sonnet", "claude-sonnet-5-5"),  # alias vs full id, same tier
        ("claude-sonnet-5-5", "sonnet"),  # and the other way round
        ("sonnet", "sonnet"),
        ("sonnet", "claude-haiku-4-5"),  # cheaper
        ("opus", "claude-sonnet-4-5"),
        ("haiku", ""),  # nothing reported: nothing to object to
    ],
)
def test_an_equal_or_cheaper_reported_model_is_allowed(
    approved: str, reported: str
) -> None:
    assert actual_model_is_allowed(approved=approved, reported=reported) is True


@pytest.mark.parametrize(
    ("approved", "reported"),
    [
        ("haiku", "claude-opus-4-8"),  # the live 2026-10-28 shape
        ("sonnet", "opus"),
        ("haiku", "sonnet"),
        ("sonnet", "gpt-opus-4"),  # foreign id: unplaceable, so refused
        ("sonnet", "some-new-model-9"),  # unknown id: refused as well
        ("sonnet", "claude-instant-1.2"),  # a Claude id outside the table
    ],
)
def test_a_dearer_or_unverifiable_reported_model_is_not_allowed(
    approved: str, reported: str
) -> None:
    assert actual_model_is_allowed(approved=approved, reported=reported) is False


def test_the_acceptance_check_is_the_escalation_rule() -> None:
    """One rule, not two: the check after the fact and the decision before it
    must never disagree, so the former is defined by the latter."""
    for approved, reported in (
        ("haiku", "claude-opus-4-8"),
        ("sonnet", "claude-haiku-4-5"),
        ("sonnet", "some-new-model-9"),
        ("opus", "claude-sonnet-5-5"),
    ):
        assert actual_model_is_allowed(approved=approved, reported=reported) is (
            not is_escalation(requested=approved, candidate=reported)
        )


def test_an_explicitly_allowed_model_passes(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_ALLOWED_WORKER_MODELS", " some-new-model-9 , x-9 ")
    assert (
        actual_model_is_allowed(approved="sonnet", reported="SOME-NEW-MODEL-9") is True
    )
    assert actual_model_is_allowed(approved="sonnet", reported="x-9") is True
    # Only what was named: a dearer KNOWN model is still refused.
    assert actual_model_is_allowed(approved="haiku", reported="opus") is False


def test_nothing_is_allowed_by_default(monkeypatch) -> None:
    monkeypatch.delenv("JARVIS_ALLOWED_WORKER_MODELS", raising=False)
    assert explicitly_allowed_models() == frozenset()
