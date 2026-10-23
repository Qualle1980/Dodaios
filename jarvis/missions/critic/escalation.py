"""Critic model choice: one tier, never an automatic upgrade.

The critic used to escalate on its own to the frontier model
(``claude-opus-4-8``) on the third iteration, for a security tag, or after a
low-confidence round. That is an automatic switch to a stronger and more
expensive model nobody approved for that call, which the maintainer ruled out
on 2026-10-23 ("no automatic switch to a more capable or more expensive
model"). Every round now runs on :data:`CRITIC_MODEL`. A security tag or a
low confidence still shapes the VERDICT (``requires_escalation`` keeps
escalating the mission to the user); it no longer changes the model.

Source: Research-Doc §F.6 + ADR-0009 §2 (escalation principle), amended by the
2026-10-23 mandate above.
"""
from __future__ import annotations

from typing import Final

#: The one critic tier. The CLI alias resolves to the subscription's current
#: Sonnet; it is never upgraded automatically.
CRITIC_MODEL: Final[str] = "sonnet"

#: Kept for callers that label the next round's model; it now names the same
#: tier as every other round.
FRONTIER_MODEL: Final[str] = CRITIC_MODEL

MODEL_TIER_BY_ITERATION: Final[dict[int, str]] = {
    0: CRITIC_MODEL,
    1: CRITIC_MODEL,
    2: CRITIC_MODEL,
}

DEFAULT_FALLBACK_MODEL: Final[str] = CRITIC_MODEL


def choose_critic_model(
    iteration: int,
    *,
    security_tag: bool = False,
    prior_confidence: float | None = None,
) -> str:
    """The model for this critic round: always :data:`CRITIC_MODEL`.

    ``security_tag`` and ``prior_confidence`` are accepted so callers keep
    their signature; neither may raise the tier (mandate 2026-10-23).
    """
    del iteration, security_tag, prior_confidence
    return CRITIC_MODEL


__all__ = [
    "CRITIC_MODEL",
    "DEFAULT_FALLBACK_MODEL",
    "FRONTIER_MODEL",
    "MODEL_TIER_BY_ITERATION",
    "choose_critic_model",
]
