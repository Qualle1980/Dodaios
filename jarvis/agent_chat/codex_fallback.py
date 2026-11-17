"""Send a coding-handback review to a Codex pane when Instant has no evidence.

Instant (ChatGPT plan) always reviews first. After its correction budget is
spent, :func:`jarvis.agent_chat.review_routing.choose_reviewer` decides whether
Instant already carried the evidence. When it did not, this module delivers
one review prompt to a Codex coding pane through the same Agentic-IDE registry
``workspace-orchestrate`` already uses (:func:`jarvis.agentic_ide.session.get_registry`),
and ``Registry.send_prompt`` registers the wait via ``track_submission``.
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


async def attempt_codex_fallback(
    *,
    evidence: ReviewEvidence | None,
    prompt: str,
    registry: Any | None = None,
) -> CodexFallbackResult:
    """Send ``prompt`` to a Codex pane when Instant's evidence is incomplete.

    On ``sent``, ``Registry.send_prompt`` (or a test double that mirrors it)
    registers delegated work through ``track_submission`` so the chat wait can
    hold for the report. Busy and missing panes return an honest message; the
    caller must not retry in a loop.
    """
    if choose_reviewer(evidence) == "instant":
        return CodexFallbackResult(action="skipped")

    reg = registry if registry is not None else default_registry()
    found = find_codex_pane(reg)
    if found is None:
        message = (
            "No Codex coding pane is open to take the review after Instant "
            "could not produce the evidence."
        )
        log.info("codex fallback: %s", message)
        return CodexFallbackResult(action="missing", message=message)

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
        return CodexFallbackResult(action="missing", message=str(exc), pane=name)
    except Exception as exc:  # noqa: BLE001 - must surface honesty, never swallow
        log.exception("codex fallback: send to %s failed", name)
        return CodexFallbackResult(
            action="error",
            message=f"{type(exc).__name__}: {exc}",
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
    "review_prompt_for_codex",
    "serialize_receipts",
]
