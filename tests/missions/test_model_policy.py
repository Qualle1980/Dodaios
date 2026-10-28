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
    is_claude_model,
    is_escalation,
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
