"""One place that decides which Claude model a worker may run, and on whose bill.

Why this exists. On 2026-10-28 a live mission was dispatched with
``step.model = "haiku"`` and ran on ``claude-opus-4-8``. Nothing was broken in
the routing: ``ClaudeDirectWorker.spawn()`` accepted the ``model`` argument and
never read it, and the model came from a module constant whose value was the
most expensive tier. The mission's own telemetry then reported ``haiku``,
because the event was built from the request rather than from what ran. So the
escalation was invisible from both ends.

Three rules follow from that, and this module is where they live so they cannot
drift apart across the workers (BUG-008 class):

1. **An explicit pick wins.** The step's model is the caller's decision, the
   same precedence ``ApiAgentWorker._resolve_worker_model`` already documents.
2. **Nobody escalates on their own.** A tier above what was asked for needs an
   explicit approval — a step pin or an operator's config entry. An absent or
   unrecognised configuration resolves DOWN to
   :data:`DEFAULT_WORKER_MODEL`, never up to the premium tier. This is the
   maintainer mandate of 2026-10-23 ("no automatic switch to a more capable or
   more expensive model"), which ``critic/escalation.py`` already applies to
   the critic and which the worker path never received.
3. **A family approval is not a tier approval.** Crossing from a configured
   provider to Claude (a dead codex login, an exhausted window) says nothing
   about which Claude tier may run, and a foreign model id — ``grok-4.3``,
   ``gpt-5`` — is not a Claude model at all and must never reach
   ``claude --model``.

Billing is the other half. The claude CLI bills either a subscription seat (an
OAuth bearer) or a metered API key, and which one it used is not something to
assume: :func:`classify_billing_source` answers from the worker environment
and says :data:`BillingSource.UNKNOWN` when the environment does not prove it.
An unproven run is never reported as a free one.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Final, Mapping

logger = logging.getLogger(__name__)


class ModelTier(int, Enum):
    """Claude tiers, ordered by capability and price.

    ``int`` so a comparison is the ordering itself. ``PREMIUM`` holds Opus and
    the approved-access Fable family: both are the expensive end, and treating
    an unknown premium id as cheap is the mistake this module exists to stop.
    """

    ECONOMY = 1  # haiku
    STANDARD = 2  # sonnet
    PREMIUM = 3  # opus, fable


#: Tier for a bare CLI alias. The aliases are what the CLI itself accepts, so
#: they are also the names this module hands back.
_ALIAS_TIERS: Final[dict[str, ModelTier]] = {
    "haiku": ModelTier.ECONOMY,
    "sonnet": ModelTier.STANDARD,
    "opus": ModelTier.PREMIUM,
    "fable": ModelTier.PREMIUM,
}

#: Full model ids: ``claude-<family>-<version>`` in either order
#: (``claude-3-5-sonnet-20241022``, ``claude-sonnet-5-5``, ``claude-opus-4-8``).
#: Matching on the family word rather than a version list is deliberate: a
#: version table goes stale and a stale table would silently answer "unknown"
#: for a next-generation Opus, which is the one answer that must never be wrong
#: in the cheap direction.
_FAMILY_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(haiku|sonnet|opus|fable)\b"
)

#: The model a Claude worker runs when nothing was asked for and nothing is
#: configured. The middle tier on purpose: reachable on a subscription (unlike
#: the approved-access Fable family, live mission 019ec615), capable enough for
#: mission work, and not the premium tier that an absent config used to select.
DEFAULT_WORKER_MODEL: Final[str] = "sonnet"

#: What a rejected model may be retried with: equal tier or cheaper, never up.
#: Used when the CLI refuses the requested model (an approved-access id on a
#: subscription that cannot reach it). Dropping ``--model`` instead would hand
#: the choice to the CLI's own default, which is not a decision anyone here
#: made and can be more expensive than what was asked for.
_DOWNGRADE_LADDER: Final[dict[ModelTier, tuple[str, ...]]] = {
    ModelTier.PREMIUM: ("sonnet", "haiku"),
    ModelTier.STANDARD: ("haiku",),
    ModelTier.ECONOMY: (),
}


class BillingSource(str, Enum):
    """Who pays for a worker run.

    ``UNKNOWN`` is a real answer, not a placeholder: the CLI can authenticate
    from the user's own credential store, which the worker environment does not
    expose. Reporting that run as a subscription run would be a claim this code
    cannot support.
    """

    SUBSCRIPTION_OAUTH = "subscription_oauth"
    API_KEY = "api_key"
    UNKNOWN = "unknown"


#: Reported as the actual model until the CLI itself has named one.
UNKNOWN_MODEL: Final[str] = "unknown"


def normalise(model: str | None) -> str:
    """Lower-cased, trimmed model string. ``None`` becomes ``""``."""
    return (model or "").strip().lower()


def tier_of(model: str | None) -> ModelTier | None:
    """Tier of a Claude model id or alias, or ``None`` when it is not one.

    ``None`` means "this code cannot place it", which callers must treat as
    unsafe rather than cheap — a foreign provider's id (``grok-4.3``) and a
    typo both land here.
    """
    name = normalise(model)
    if not name:
        return None
    alias = _ALIAS_TIERS.get(name)
    if alias is not None:
        return alias
    match = _FAMILY_RE.search(name)
    if match is None:
        return None
    return _ALIAS_TIERS[match.group(1)]


def is_claude_model(model: str | None) -> bool:
    """True when ``model`` is a Claude id/alias this module recognises."""
    return tier_of(model) is not None


def is_escalation(*, requested: str | None, candidate: str | None) -> bool:
    """True when running ``candidate`` instead of ``requested`` costs more.

    An unplaceable candidate counts as an escalation whenever something
    placeable was requested: it cannot be shown to be equal or cheaper, and the
    burden of proof belongs on the more expensive side.
    """
    requested_tier = tier_of(requested)
    if requested_tier is None:
        return False  # nothing to compare against
    candidate_tier = tier_of(candidate)
    if candidate_tier is None:
        return True
    return candidate_tier > requested_tier


@dataclass(frozen=True)
class ModelChoice:
    """The resolved model, and the evidence for why it is that one."""

    model: str
    tier: ModelTier
    #: ``"step"`` (an explicit per-task pin), ``"config"`` (an operator's
    #: configured model) or ``"default"`` (:data:`DEFAULT_WORKER_MODEL`).
    source: str
    #: Set when a candidate was refused for being a tier above the request, or
    #: for not being a Claude model at all. Carries the refused value so the
    #: log and the tests can name it.
    blocked: str | None = None


def resolve_worker_model(
    *,
    requested: str | None,
    configured: str | None,
) -> ModelChoice:
    """Decide which model a Claude worker runs.

    Inputs:
        requested: the step's explicit pin — the caller's own decision.
        configured: the operator's configured model for this worker (an
            explicit config entry, so an explicit approval of that tier).

    Output:
        A :class:`ModelChoice`. Precedence is request, then configuration, then
        :data:`DEFAULT_WORKER_MODEL`. A configured value that is not a Claude
        model is refused rather than passed to ``claude --model``, and a
        configured value above an explicit request never overrides it.
    """
    requested_name = normalise(requested)
    configured_name = normalise(configured)

    if requested_name:
        if is_claude_model(requested_name):
            tier = tier_of(requested_name)
            assert tier is not None  # noqa: S101 — is_claude_model proved it
            return ModelChoice(requested_name, tier, "step")
        # A step pin this module cannot place is not usable as a claude
        # --model: a foreign provider's id reaches the CLI as an instant
        # failure (the 2026-06-08 "primary provider is grok" incident). Fall
        # through to the configured/default path and record the refusal.
        logger.warning(
            "model policy: requested model %r is not a Claude model — "
            "refusing to pass it to the claude CLI.",
            requested_name,
        )
        blocked = requested_name
    else:
        blocked = None

    if configured_name and is_claude_model(configured_name):
        tier = tier_of(configured_name)
        assert tier is not None  # noqa: S101 — is_claude_model proved it
        return ModelChoice(configured_name, tier, "config", blocked)

    if configured_name:
        logger.warning(
            "model policy: configured worker model %r is not a Claude model — "
            "using %s instead of guessing a tier.",
            configured_name,
            DEFAULT_WORKER_MODEL,
        )
        blocked = blocked or configured_name

    tier = tier_of(DEFAULT_WORKER_MODEL)
    assert tier is not None  # noqa: S101 — a constant of this module
    return ModelChoice(DEFAULT_WORKER_MODEL, tier, "default", blocked)


def cross_family_model(requested: str | None) -> ModelChoice:
    """Decide the model when a worker runs as another family's fallback.

    A dead codex login or an exhausted window may move the work to Claude —
    that is a family decision, already logged by the worker factory. It carries
    no approval for a tier, and the requested model belongs to the other
    family, so neither value may select a premium Claude model here. The answer
    is the request when it happens to be a Claude model, otherwise
    :data:`DEFAULT_WORKER_MODEL`.
    """
    name = normalise(requested)
    if name and is_claude_model(name):
        tier = tier_of(name)
        assert tier is not None  # noqa: S101 — is_claude_model proved it
        return ModelChoice(name, tier, "step")
    tier = tier_of(DEFAULT_WORKER_MODEL)
    assert tier is not None  # noqa: S101 — a constant of this module
    return ModelChoice(
        DEFAULT_WORKER_MODEL, tier, "default", blocked=name or None,
    )


def retry_model_after_rejection(
    rejected: str | None, *, already_tried: tuple[str, ...] = ()
) -> str | None:
    """The next model to try after the CLI refused ``rejected``, or ``None``.

    ``None`` means stop: there is no equal-or-cheaper Claude model left, and
    the honest outcome is the failure the CLI reported. Removing ``--model`` to
    let the CLI pick is not an option — its default is not a decision this code
    made and can be the premium tier.
    """
    tier = tier_of(rejected)
    if tier is None:
        # An unplaceable model was refused. The one safe retry is the
        # documented default, and only if it has not been tried.
        return (
            DEFAULT_WORKER_MODEL
            if DEFAULT_WORKER_MODEL not in already_tried
            and normalise(rejected) != DEFAULT_WORKER_MODEL
            else None
        )
    for candidate in _DOWNGRADE_LADDER[tier]:
        if candidate not in already_tried:
            return candidate
    return None


def classify_billing_source(env: Mapping[str, str]) -> BillingSource:
    """Which bill a claude-CLI run lands on, judged from its environment.

    * An OAuth bearer (``CLAUDE_CODE_OAUTH_TOKEN`` / ``ANTHROPIC_OAUTH_TOKEN``,
      or an ``sk-ant-oat`` token in the API-key slot) is a subscription seat.
      The CLI reads the OAuth slot first, so it wins when both are present.
    * A classic ``ANTHROPIC_API_KEY`` on its own is metered.
    * Neither proves nothing: the CLI then authenticates from the user's own
      credential store, which this environment does not expose, so the answer
      is :data:`BillingSource.UNKNOWN` — never "free".
    """
    oauth = (
        env.get("CLAUDE_CODE_OAUTH_TOKEN")
        or env.get("ANTHROPIC_OAUTH_TOKEN")
        or ""
    ).strip()
    api_key = (env.get("ANTHROPIC_API_KEY") or "").strip()
    if oauth:
        return BillingSource.SUBSCRIPTION_OAUTH
    if api_key.startswith("sk-ant-oat"):
        # Routed into the wrong slot by a caller, but still a seat bearer.
        return BillingSource.SUBSCRIPTION_OAUTH
    if api_key:
        return BillingSource.API_KEY
    return BillingSource.UNKNOWN


__all__ = [
    "DEFAULT_WORKER_MODEL",
    "UNKNOWN_MODEL",
    "BillingSource",
    "ModelChoice",
    "ModelTier",
    "classify_billing_source",
    "cross_family_model",
    "is_claude_model",
    "is_escalation",
    "normalise",
    "resolve_worker_model",
    "retry_model_after_rejection",
    "tier_of",
]
