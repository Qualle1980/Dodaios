"""Send a coding-handback review to a Codex pane when Instant has no evidence.

Instant (ChatGPT plan) always reviews first. After its correction budget is
spent, :func:`jarvis.agent_chat.review_routing.choose_reviewer` decides whether
Instant already carried the evidence. When it did not, this module delivers
one review prompt to a Codex coding pane through the same Agentic-IDE registry
``workspace-orchestrate`` already uses (:func:`jarvis.agentic_ide.session.get_registry`).
If no Codex pane is open, it opens one with ``add_terminal(agent=\"codex\")`` —
the same create path used for other coding CLIs — then sends the brief.
``Registry.send_prompt`` registers the wait via ``track_submission``. An open
failure returns an honest ``review not run`` line; nothing claims a review ran.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from jarvis.agent_chat.delegation_acceptance import Receipt
from jarvis.agent_chat.review_routing import ReviewEvidence, choose_reviewer
from jarvis.agentic_ide.session import AgentBusyError, SessionError

log = logging.getLogger(__name__)

CodexFallbackAction = Literal["skipped", "sent", "busy", "missing", "error"]


@dataclass(frozen=True)
class CodexFallbackResult:
    """What happened when the wait tried to hand the review to Codex."""

    action: CodexFallbackAction
    message: str = ""
    pane: str = ""


def default_registry() -> Any:
    """The process-wide Agentic-IDE registry — same accessor as workspace-orchestrate."""
    from jarvis.agentic_ide.session import get_registry

    return get_registry()


def evidence_from_payload(payload: dict[str, Any] | None) -> ReviewEvidence | None:
    """Rebuild :class:`ReviewEvidence` from a correction / unverified notice."""
    if not payload:
        return None
    raw = payload.get("receipts")
    receipts: list[Receipt] = []
    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                continue
            try:
                receipts.append(
                    Receipt(
                        tool=str(row.get("tool") or ""),
                        args=dict(row.get("args") or {})
                        if isinstance(row.get("args"), dict)
                        else {},
                        success=bool(row.get("success")),
                        output=str(row.get("output") or ""),
                        cwd=str(row.get("cwd") or ""),
                    )
                )
            except Exception:  # noqa: BLE001 - a bad row must not kill the fallback
                log.warning("codex fallback: could not rebuild a tool receipt", exc_info=True)
    return ReviewEvidence(
        receipts=tuple(receipts),
        verdict_text=str(payload.get("verdict_text") or ""),
        tool_error=bool(payload.get("tool_error")),
    )


def serialize_receipts(receipts: Sequence[Receipt]) -> list[dict[str, Any]]:
    """JSON-safe receipt rows for a notice payload."""
    out: list[dict[str, Any]] = []
    for receipt in receipts:
        out.append(
            {
                "tool": receipt.tool,
                "args": dict(receipt.args) if isinstance(receipt.args, dict) else {},
                "success": bool(receipt.success),
                "output": str(receipt.output or ""),
                "cwd": str(receipt.cwd or ""),
            }
        )
    return out


def review_prompt_for_codex(
    *,
    original: str,
    gaps: Sequence[str],
    work: Sequence[str],
) -> str:
    """Brief Codex to review the handback Instant could not evidence."""
    missing = "\n".join(f"- {gap}" for gap in gaps) or "- no evidence was recorded at all"
    return (
        "Review this delegated coding handback. Instant (ChatGPT plan) already "
        "tried and could not produce the required evidence.\n"
        f"Jobs: {list(work)}\nMissing evidence:\n{missing}\n"
        "Read the change in that job's own checkout and run its tests there. "
        "Report plainly what you verified and what is still unproven. "
        "Do not claim the work is good without that evidence.\n"
        "Original user request:\n" + original
    )


def find_codex_pane(registry: Any) -> tuple[Any, Any] | None:
    """First open pane whose agent is Codex, front workspace first when possible."""
    sessions = list(getattr(registry, "sessions", None) or [])
    active = getattr(registry, "session", None)
    ordered = []
    if active is not None:
        ordered.append(active)
    for space in sessions:
        if active is not None and getattr(space, "id", None) == getattr(active, "id", None):
            continue
        ordered.append(space)
    for space in ordered:
        for term in list(getattr(space, "terminals", None) or []):
            if str(getattr(term, "agent", "") or "").lower() == "codex":
                return space, term
    return None


def _workspace_for_open(registry: Any) -> Any | None:
    """Active workspace first, else the first open session — same order as find."""
    active = getattr(registry, "session", None)
    if active is not None:
        return active
    sessions = list(getattr(registry, "sessions", None) or [])
    return sessions[0] if sessions else None


async def open_codex_pane(registry: Any) -> tuple[Any, Any] | CodexFallbackResult:
    """Open one Codex pane the way workspace-orchestrate create opens a CLI.

    Uses ``registry.add_terminal(agent=\"codex\")`` on an existing workspace.
    Returns ``(workspace, terminal)`` on success, or a :class:`CodexFallbackResult`
    with ``action=\"missing\"`` / ``\"error\"`` and an honest not-run message.
    """
    space = _workspace_for_open(registry)
    if space is None:
        message = (
            "review not run: could not open a Codex pane because "
            "no Agentic-IDE workspace is open."
        )
        log.info("codex fallback: %s", message)
        return CodexFallbackResult(action="missing", message=message)

    add = getattr(registry, "add_terminal", None)
    if add is None or not callable(add):
        message = (
            "review not run: could not open a Codex pane because "
            "the workspace registry cannot open panes."
        )
        log.info("codex fallback: %s", message)
        return CodexFallbackResult(action="missing", message=message)

    workspace_id = str(getattr(space, "id", "") or "") or None
    model = ""
    effort = ""
    try:
        from jarvis.agentic_ide.hand_off_retry import review_key, shared_gate
        from jarvis.agentic_ide.model_lists import shared_store

        store = shared_store()
        picked = store.pick_codex_reviewer()
        if picked is None:
            listing = store.ensure("codex", force=True)
            note = listing.error or "no confirmed Codex model for this account"
            message = f"review not run: {note}"
            log.info("codex fallback: %s", message)
            return CodexFallbackResult(action="missing", message=message)
        model = picked.id
        effort = store.highest_effort(picked)
        gate_key = review_key(owner_session=workspace_id or "codex-fallback", title="codex-review")
        blocked = shared_gate().allow_open(gate_key, model)
        if blocked:
            log.info("codex fallback: %s", blocked)
            return CodexFallbackResult(action="missing", message=blocked)
        shared_gate().note_open(gate_key, model)
    except Exception:  # noqa: BLE001 — opening without a model still lets the CLI default
        log.warning("codex fallback: live reviewer model unavailable", exc_info=True)
    try:
        term = await add(
            agent="codex",
            workspace_id=workspace_id,
            **({"model": model} if model else {}),
            **({"effort": effort} if effort else {}),
        )
    except SessionError as exc:
        message = f"review not run: could not open a Codex pane because {exc}"
        log.warning("codex fallback: %s", message)
        return CodexFallbackResult(action="missing", message=message)
    except Exception as exc:  # noqa: BLE001 - must surface honesty, never swallow
        log.exception("codex fallback: opening a Codex pane failed")
        return CodexFallbackResult(
            action="error",
            message=(
                "review not run: could not open a Codex pane because "
                f"{type(exc).__name__}: {exc}"
            ),
        )

    if term is None:
        message = (
            "review not run: could not open a Codex pane because "
            "add_terminal returned no pane."
        )
        log.warning("codex fallback: %s", message)
        return CodexFallbackResult(action="missing", message=message)

    # Prefer the workspace that owns the new pane when the registry returns one.
    found = find_codex_pane(registry)
    if found is not None:
        return found
    return space, term


async def attempt_codex_fallback(
    *,
    evidence: ReviewEvidence | None,
    prompt: str,
    registry: Any | None = None,
) -> CodexFallbackResult:
    """Send ``prompt`` to a Codex pane after every coding run.

    Instant may already have complete evidence; Codex still reviews
    (``choose_reviewer`` always returns Codex). When no Codex pane is open,
    opens one via ``add_terminal(agent=\"codex\")`` (the same create path
    workspace-orchestrate uses for Grok and Claude panes), then sends the
    brief. On ``sent``, ``Registry.send_prompt`` (or a test double that
    mirrors it) registers delegated work through ``track_submission`` so the
    chat wait can hold for the report. Busy panes and open failures return an
    honest ``review not run`` message — never ``passed``. The caller must not
    retry in a loop.
    """
    # Evidence completeness is recorded for status; it never skips Codex.
    _ = choose_reviewer(evidence)

    reg = registry if registry is not None else default_registry()
    found = find_codex_pane(reg)
    if found is None:
        opened = await open_codex_pane(reg)
        if isinstance(opened, CodexFallbackResult):
            return opened
        found = opened

    owner, term = found
    name = str(getattr(term, "name", "") or "Codex")
    workspace_id = str(getattr(owner, "id", "") or "") or None
    try:
        await reg.send_prompt(
            name,
            prompt,
            workspace_id=workspace_id,
            when_busy="refuse",
        )
    except AgentBusyError as exc:
        log.warning(
            "codex fallback: Codex pane %s is busy; review not sent: %s",
            name,
            exc,
        )
        return CodexFallbackResult(action="busy", message=str(exc), pane=name)
    except SessionError as exc:
        log.warning(
            "codex fallback: Codex pane unavailable (%s): %s",
            name,
            exc,
        )
        return CodexFallbackResult(
            action="missing",
            message=f"review not run: could not send to a Codex pane because {exc}",
            pane=name,
        )
    except Exception as exc:  # noqa: BLE001 - must surface honesty, never swallow
        log.exception("codex fallback: send to %s failed", name)
        return CodexFallbackResult(
            action="error",
            message=f"review not run: could not send to a Codex pane because {type(exc).__name__}: {exc}",
            pane=name,
        )
    log.info("codex fallback: sent review to Codex pane %s", name)
    return CodexFallbackResult(action="sent", pane=name)


__all__ = [
    "CodexFallbackResult",
    "attempt_codex_fallback",
    "default_registry",
    "evidence_from_payload",
    "find_codex_pane",
    "open_codex_pane",
    "review_prompt_for_codex",
    "serialize_receipts",
]
