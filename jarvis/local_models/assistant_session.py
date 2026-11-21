"""Session policy of the setup assistant: one chat per install, on the Agents tier.

The assistant is a ``local-models`` agent-chat session. It runs on the
Jarvis-Agents tier (``[brain.worker]`` provider / model, the fallback pair
when the primary is unknown to the chat) — never on the voice brain, so a
person whose voice runs on Ollama still gets a capable cloud model to set
Ollama up. There is ONE session per install: the routes reuse the newest
``local-models`` session and re-create it only when the worker tier moved.

No usable Agents-tier credential → an honest refusal (:data:`NOT_READY`),
never a silent fallback onto another tier.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Final

log = logging.getLogger(__name__)

__all__ = [
    "NOT_READY",
    "SURFACE",
    "AgentsTier",
    "agents_tier",
    "chain_candidates",
    "ensure_session",
    "is_default_usable",
    "is_tool_capable",
    "session_state",
]

SURFACE: Final[str] = "local-models"
NOT_READY: Final[str] = (
    "Pick a Tool Model with an API key first (API Keys → Tool Model) — the setup "
    "assistant runs on it, billed through that key."
)
#: The tier is connected but every provider in its chain drives a vendor CLI
#: on a flat prompt (Antigravity, for one) and cannot call the assistant's
#: tools; a fallback with an API key fixes it.
NO_TOOLS: Final[str] = (
    "The picked Tool Model cannot call tools here (a CLI-only provider such as "
    "Antigravity). Pick an API provider — Gemini, OpenAI, Vertex or OpenRouter — "
    "under API Keys → Tool Model and try again."
)


@dataclass(frozen=True, slots=True)
class AgentsTier:
    provider: str
    model: str
    ready: bool
    reason: str


def _known(provider: str) -> bool:
    from jarvis.agent_chat.catalog import provider_row
    from jarvis.agent_chat.runner_api import supports_api_runner

    return provider_row(provider) is not None or supports_api_runner(provider)


def is_tool_capable(provider: str) -> bool:
    """Whether ``provider``'s brain plugin can call tools inside the brain loop.

    Read from the plugin itself (``can_call_tools`` / ``supports_tools``) via
    the ``jarvis.brain`` entry point, never from the name (AP-21). A plugin
    that cannot be loaded here answers True: the turn then fails honestly
    instead of a healthy provider being skipped by a probe glitch.
    """
    from jarvis.core.entry_points import entry_points

    try:
        eps = [ep for ep in entry_points(group="jarvis.brain") if ep.name == provider]
        if not eps:
            return True
        cls = eps[0].load()
        try:
            brain = cls()
        except TypeError:
            brain = cls(model=None)
        probe = getattr(brain, "can_call_tools", None)
        if callable(probe):
            return bool(probe())
        return bool(getattr(brain, "supports_tools", True))
    except Exception:  # noqa: BLE001 — a probe glitch must not hide a working provider
        log.debug("agents tier: tool-capability probe for %s failed", provider, exc_info=True)
        return True


def is_default_usable(provider: str) -> bool:
    """A credential exists — or, for a keyless local runtime, the server answers.

    A subscription login, an Agents-tier secret, or a provider-spec credential
    still count. ``auth_mode == "none"`` alone does not: that only means "needs
    no API key", and treating it as usable let a stopped Ollama win the worker
    chain while memory learning wrote nothing (2026-11-10). Local billing cards
    must prove reachability through
    :func:`jarvis.local_models.runtime_reachable.is_local_runtime_reachable`
    (plugin ``runtime_reachable`` or a card ``base_url`` GET) — capability, not
    keylessness (AP-21).
    """
    from jarvis.brain.resolver import _subscription_candidates, _subscription_connected
    from jarvis.core.config import get_jarvis_agent_secret, load_config

    try:
        if provider in _subscription_candidates(load_config()) and _subscription_connected(
            provider
        ):
            return True
    except Exception:  # noqa: BLE001 — an unreadable login is "not connected", logged
        log.debug("agents tier: subscription probe for %s failed", provider, exc_info=True)

    try:
        if get_jarvis_agent_secret(provider):
            return True
    except Exception:  # noqa: BLE001 — no credential readable counts as none
        log.debug("agents tier: credential lookup for %s failed", provider, exc_info=True)
    try:
        from jarvis.brain.app_control import is_credential_present
        from jarvis.ui.web.provider_spec import get_spec, provider_billing

        spec = get_spec(provider)
        if spec is None or not is_credential_present(spec):
            return False
        # Keyless local: credential presence is vacuously true — prove the
        # runtime answers before the worker/agents chain selects it.
        if provider_billing(spec) == "local":
            from jarvis.local_models.runtime_reachable import is_local_runtime_reachable

            return is_local_runtime_reachable(provider)
        return True
    except Exception:  # noqa: BLE001 — same: unreadable is "none"
        log.debug("agents tier: main credential lookup for %s failed", provider, exc_info=True)
        return False


def _tool_model_pair(cfg: Any) -> tuple[str, str] | None:
    """The user's Tool Model pick, ``None`` when it is ``auto``.

    The same reader the Tool Model tab and the IDE's prompt writer use
    (``resolver._tool_model_selection``), so the assistant runs on exactly the
    model the user pointed that tab at — and is billed through that key.
    """
    from jarvis.brain.resolver import _tool_model_selection

    try:
        provider, model = _tool_model_selection(cfg)
    except Exception:  # noqa: BLE001 — a malformed section reads as unset, logged
        log.debug("agents tier: tool-model selection unreadable", exc_info=True)
        return None
    if not provider or provider == "auto":
        return None
    return provider, str(model or "").strip()


def _chain(cfg: Any) -> list[tuple[str, str]]:
    """The Tool Model first; the Agents-tier chain only as a fallback."""
    out: list[tuple[str, str]] = []
    pinned = _tool_model_pair(cfg)
    if pinned is not None:
        out.append(pinned)
    brain = getattr(cfg, "brain", None)
    worker = getattr(brain, "worker", None)
    pairs = [
        ("provider", "model"),
        ("fallback_provider", "fallback_model"),
        ("fallback_provider_2", "fallback_model_2"),
    ] if worker is not None else []
    for p_key, m_key in pairs:
        provider = str(getattr(worker, p_key, "") or "").strip()
        model = str(getattr(worker, m_key, "") or "").strip()
        if provider and (provider, model) not in out:
            out.append((provider, model))
    # Last: the chat the user talks to, when it is a SUBSCRIPTION seat. Still
    # gated like every other pair (known, tool-capable, signed in). A keyed
    # primary is never added: the agent chain bills a key only where the user
    # pointed the Tool Model or the Agents tier at it.
    primary = str(getattr(brain, "primary", "") or "").strip()
    if primary and all(p != primary for p, _ in out) and _is_subscription(cfg, primary):
        out.append((primary, ""))
    return out


def _provider_model(cfg: Any, provider: str) -> str:
    """The configured model for ``provider``, or empty for the provider default."""
    providers = getattr(getattr(cfg, "brain", None), "providers", None) or {}
    return str(getattr(providers.get(provider), "model", "") or "").strip()


def _rescue_candidates(cfg: Any) -> list[tuple[str, str]]:
    """Capability-gated tail when the configured Agents/worker pick cannot run.

    Used only after the configured Tool Model / ``[brain.worker]`` chain is
    present but yields no usable provider — typically ``claude-api`` with no
    Anthropic key (AP-22). Order prefers keyless local brains, then free-tier
    Gemini when a key exists, then every other keyed brain family from the
    shared secrets registry. Credential and tool-capability filters apply in
    :func:`chain_candidates` / :func:`agents_tier`; this list is candidates
    only.
    """
    from jarvis.core.config import PROVIDER_SECRET_CANDIDATES
    from jarvis.ui.web.provider_spec import PROVIDERS

    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(provider: str) -> None:
        name = (provider or "").strip()
        if not name or name in seen:
            return
        seen.add(name)
        out.append((name, _provider_model(cfg, name)))

    for spec in PROVIDERS:
        if spec.tier != "brain" or spec.auth_mode != "none" or spec.account_login:
            continue
        _add(spec.id)

    # Free-tier Gemini before other metered API families when a key exists.
    if "gemini" in PROVIDER_SECRET_CANDIDATES:
        _add("gemini")
    for provider in PROVIDER_SECRET_CANDIDATES:
        if provider == "gemini":
            continue
        _add(provider)
    return out


def _filter_pairs(
    pairs: list[tuple[str, str]],
    *,
    usable: Callable[[str], bool],
    tool_capable: Callable[[str], bool],
) -> list[tuple[str, str]]:
    """Keep known, tool-capable, credentialed pairs — never a silent dead brain."""
    out: list[tuple[str, str]] = []
    for provider, model in pairs:
        if not _known(provider) or not tool_capable(provider):
            continue
        try:
            if usable(provider):
                out.append((provider, model))
        except Exception:  # noqa: BLE001 — a failing probe is "not ready", with the reason logged
            log.info("agents tier: usability probe for %s failed", provider, exc_info=True)
    return out


def _is_subscription(cfg: Any, provider: str) -> bool:
    from jarvis.brain.resolver import _subscription_candidates

    try:
        return provider in _subscription_candidates(cfg)
    except Exception:  # noqa: BLE001 — unknown billing counts as "not a subscription", logged
        log.debug("agents tier: billing lookup for %s failed", provider, exc_info=True)
        return False


def chain_candidates(
    cfg: Any,
    *,
    usable: Callable[[str], bool] | None = None,
    tool_capable: Callable[[str], bool] | None = None,
) -> list[tuple[str, str]]:
    """The worker pairs worth a live probe: known, tool-capable, credentialed.

    Walks the configured Tool Model / Agents chain first. Only when that chain
    is present but yields no usable provider does the capability rescue tail
    run (local / free Gemini / other keyed families). An unset worker tier
    stays empty — picking a Tool Model is still required.
    """
    can_tools = tool_capable or is_tool_capable
    probe = usable or is_default_usable
    configured = _chain(cfg)
    out = _filter_pairs(configured, usable=probe, tool_capable=can_tools)
    if out or not configured:
        return out
    seen = {provider for provider, _model in out}
    rescue = [
        (provider, model)
        for provider, model in _rescue_candidates(cfg)
        if provider not in seen
    ]
    return _filter_pairs(rescue, usable=probe, tool_capable=can_tools)


#: (provider, model) -> (ok, reason, monotonic) — one real call per pair per
#: ten minutes; a run and the session read share it.
_LIVE_CACHE: dict[tuple[str, str], tuple[bool, str, float]] = {}
_LIVE_TTL_S: Final[float] = 600.0


def _reset_for_tests() -> None:
    _LIVE_CACHE.clear()
    try:
        from jarvis.local_models.runtime_reachable import reset_for_tests as _reset_reach

        _reset_reach()
    except Exception:  # noqa: BLE001 — test helper; absence must not break imports
        log.debug("agents tier: reachability reset skipped", exc_info=True)


async def probe_live(
    cfg: Any,
    pairs: list[tuple[str, str]],
    *,
    timeout_s: float = 20.0,
    tester: Callable[..., Awaitable[Any]] | None = None,
) -> dict[tuple[str, str], tuple[bool, str]]:
    """One real one-token generation per pair, classified like the API-keys
    page's "Test" (``provider_test``): a key that authenticates but is refused
    for quota or billing is NOT usable here, and the sentence says so."""
    import time

    from jarvis.brain import provider_test
    from jarvis.ui.web.provider_spec import get_spec

    out: dict[tuple[str, str], tuple[bool, str]] = {}
    now = time.monotonic()
    for provider, model in pairs:
        hit = _LIVE_CACHE.get((provider, model))
        if hit is not None and now - hit[2] < _LIVE_TTL_S:
            out[(provider, model)] = (hit[0], hit[1])
            continue
        spec = get_spec(provider)
        if spec is None:
            out[(provider, model)] = (False, f"{provider} is not a provider this build knows.")
            continue
        run = tester or provider_test.run_provider_test
        try:
            result = await run(spec, cfg, model=model or None, timeout_s=timeout_s)
            ok = result.status == provider_test.OK
            reason = "" if ok else _plain_reason(provider, model, result)
        except Exception as exc:  # noqa: BLE001 — a probe crash is "not usable", logged
            log.warning("agents tier: live probe of %s/%s crashed: %s", provider, model, exc)
            ok, reason = False, f"{provider} ({model}): probe failed — {exc}"
        _LIVE_CACHE[(provider, model)] = (ok, reason, time.monotonic())
        out[(provider, model)] = (ok, reason)
    return out


_PLAIN_BY_STATUS: Final[dict[str, str]] = {
    "no_credits": "the account has no credit or quota left for this model",
    "rate_limited": "the provider is rate-limiting this key right now",
    "bad_key": "the key was refused",
    "model_unavailable": "the provider no longer offers this model",
    "unreachable": "the provider could not be reached",
    "not_configured": "no key is stored for it",
}


def _plain_reason(provider: str, model: str, result: Any) -> str:
    """One readable sentence, never a dumped error body.

    The status vocabulary is the API-keys page's; the provider's own text is
    kept only when it is short and free of JSON braces (a 404 body is not
    something a person should have to read in a status line).
    """
    status = str(getattr(result, "status", "") or "error")
    head = f"{provider} ({model or 'default model'}): "
    plain = _PLAIN_BY_STATUS.get(status)
    detail = str(getattr(result, "detail", "") or "").strip()
    if plain:
        return head + plain + "."
    if detail and "{" not in detail and len(detail) <= 140:
        return head + detail
    return head + status.replace("_", " ") + "."


def agents_tier(
    cfg: Any,
    *,
    usable: Callable[[str], bool] | None = None,
    tool_capable: Callable[[str], bool] | None = None,
    live: dict[tuple[str, str], tuple[bool, str]] | None = None,
) -> AgentsTier:
    """The pair the assistant runs on and whether it can run right now.

    Walks the worker chain (primary, fallback, fallback 2) and takes the first
    provider that is known, can call tools, has a credential and — when a
    ``live`` map from :func:`probe_live` is given — answered a real call.
    When the configured chain is present but nothing in it can run, a
    capability-gated rescue tail (local / free Gemini / other keyed families)
    is tried once. ``usable`` is the credential / login probe (the routes pass
    the provider-agnostic worker check the API-keys page uses; the default
    reads the Agents-tier secret); ``tool_capable`` defaults to asking the
    plugin.
    """
    configured = _chain(cfg)
    if not configured:
        return AgentsTier("", "", False, NOT_READY)
    known = [(p, m) for p, m in configured if _known(p)]
    if not known:
        return AgentsTier(configured[0][0], "", False, NOT_READY)
    can_tools = tool_capable or is_tool_capable
    handy = [(p, m) for p, m in known if can_tools(p)]
    if not handy:
        return AgentsTier(known[0][0], known[0][1], False, NO_TOOLS)
    probe = usable or is_default_usable
    last_reason = NOT_READY

    def _first_ready(pairs: list[tuple[str, str]]) -> AgentsTier | None:
        nonlocal last_reason
        for provider, model in pairs:
            try:
                ok = bool(probe(provider))
            except Exception:  # noqa: BLE001 — a failing probe is "not ready", logged
                log.info("agents tier: usability probe for %s failed", provider, exc_info=True)
                ok = False
            if not ok:
                continue
            if live is not None and (provider, model) in live:
                alive, reason = live[(provider, model)]
                if not alive:
                    last_reason = reason or NOT_READY
                    continue
            return AgentsTier(provider, model, True, "")
        return None

    ready = _first_ready(handy)
    if ready is not None:
        return ready

    # Configured chain exhausted — rescue only when a worker/Tool Model was set.
    rescue_known = [
        (p, m)
        for p, m in _rescue_candidates(cfg)
        if _known(p) and can_tools(p) and all(p != cp for cp, _ in configured)
    ]
    # A live map only covers the pairs that were probed; do not auto-accept an
    # unprobed rescue provider as ready when the caller asked for live proof.
    if live is None:
        rescued = _first_ready(rescue_known)
        if rescued is not None:
            return rescued
    return AgentsTier(handy[0][0], handy[0][1], False, last_reason)


def _current(svc: Any) -> Any | None:
    sessions = svc.store.list_sessions(limit=1, surface=SURFACE)
    return sessions[0] if sessions else None


def session_state(
    svc: Any,
    cfg: Any,
    *,
    usable: Callable[[str], bool] | None = None,
    live: dict[tuple[str, str], tuple[bool, str]] | None = None,
) -> dict:
    """``GET /session``: the session id (or null), the pair, readiness, reason."""
    tier = agents_tier(cfg, usable=usable, live=live)
    current = _current(svc)
    matches = (
        current is not None
        and current.provider == tier.provider
        and (not tier.model or current.model == tier.model)
    )
    return {
        "session_id": current.session_id if matches and current is not None else None,
        "surface": SURFACE,
        "provider": tier.provider,
        "model": tier.model,
        "ready": tier.ready,
        "reason": tier.reason,
    }


def ensure_session(
    svc: Any,
    cfg: Any,
    *,
    usable: Callable[[str], bool] | None = None,
    live: dict[tuple[str, str], tuple[bool, str]] | None = None,
) -> Any:
    """The one ``local-models`` session, created or re-created for the current pair.

    Raises ``PermissionError(reason)`` when the Agents tier cannot run.
    """
    tier = agents_tier(cfg, usable=usable, live=live)
    if not tier.ready:
        raise PermissionError(tier.reason or NOT_READY)
    current = _current(svc)
    if (
        current is not None
        and current.provider == tier.provider
        and (not tier.model or current.model == tier.model)
    ):
        return current
    return svc.create_session(
        provider=tier.provider,
        model=tier.model,
        title="Local models setup",
        surface=SURFACE,
    )
