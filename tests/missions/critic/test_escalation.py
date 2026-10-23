"""The critic model never switches up on its own (mandate 2026-10-23).

It used to escalate to ``claude-opus-4-8`` on the third iteration, for a
security tag, and after a low-confidence round. Every round now runs on the
one critic tier.
"""
from __future__ import annotations

import pytest

from jarvis.missions.critic.escalation import (
    CRITIC_MODEL,
    DEFAULT_FALLBACK_MODEL,
    MODEL_TIER_BY_ITERATION,
    choose_critic_model,
)


@pytest.mark.parametrize("iteration", [0, 1, 2, 3])
@pytest.mark.parametrize("security_tag", [False, True])
@pytest.mark.parametrize("prior_confidence", [None, 0.1, 0.9])
def test_every_round_runs_on_the_one_tier(
    iteration: int, security_tag: bool, prior_confidence: float | None
) -> None:
    model = choose_critic_model(
        iteration, security_tag=security_tag, prior_confidence=prior_confidence
    )
    assert model == CRITIC_MODEL == "sonnet"


def test_no_table_entry_or_fallback_names_a_stronger_model() -> None:
    assert set(MODEL_TIER_BY_ITERATION.values()) == {CRITIC_MODEL}
    assert DEFAULT_FALLBACK_MODEL == CRITIC_MODEL
    assert "opus" not in CRITIC_MODEL
