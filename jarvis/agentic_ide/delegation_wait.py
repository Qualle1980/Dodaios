"""Return coding jobs to the calling chat using recorded lifecycle evidence.

An idle terminal or a delivery acknowledgement is not task completion. Only a
fresh CLI end marker plus the answer to this submission can finish a wait.
Unsupported/remote transcript formats produce an honest unverified result.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from jarvis.core.delegated_work import DelegatedWork, register_delegated_work

from . import agent_transcript

log = logging.getLogger(__name__)


#: Claude tiers a chat-addressed pane may serve work on. A launch flag is a
#: request; this is the list the CLI's own record is held against afterwards.
ALLOWED_CLAUDE_TIERS = ("sonnet", "haiku")


def _timeline(term: Any) -> Any | None:
    """The pane's own record: its events AND what model its last reply carried.

    One seam, so the served model and the conversation are read from the same
    file at the same moment -- reading them separately could attest a model to
    a completion it did not produce.
    """
    from .session import account_home

    try:
        return agent_transcript.read_timeline(
            term.agent,
            term.resume.id,
            home=account_home(term.agent, term.account),
            live=True,
        )
    except Exception:  # noqa: BLE001 - an unreadable record is "no record"
        log.warning("delegation handback: the pane's transcript could not be read", exc_info=True)
        return None


def served_model_verdict(term: Any, served: str) -> tuple[str, str]:
    """What the CLI's OWN record says it ran, judged against what we allow.

    Returns ``(state, detail)`` with state ``ok``, ``unattested`` or
    ``refused``. The launch flag cannot settle this: the vendor decides which
    model answers, and the only authoritative statement arrives with the first
    reply. So this runs AFTER the work, and a wrong tier is reported rather
    than retried -- the request was already spent, and sending a second one
    would spend another.
    """
    if str(getattr(term, "agent", "") or "") != "claude":
        return ("ok", "")
    name = str(served or "").strip().lower()
    if not name:
        return ("unattested", "the pane published no model for this reply")
    if any(tier in name for tier in ALLOWED_CLAUDE_TIERS):
        return ("ok", name)
    return ("refused", name)


def _identity(term: Any) -> tuple:
    return term.process_generation, term.last_submit_at


def _read(term: Any, identity: tuple) -> dict[str, Any] | None:
    if getattr(term, "computer_id", "") or not agent_transcript.can_read(term.agent):
        return {
            "status": "unverified",
            "report": (
                "This coding session has no supported local completion record. "
                "Inspect its context; delivery alone does not prove completion."
            ),
        }
    if term.resume is None:
        return None
    timeline = _timeline(term)
    events = (
        timeline.events if timeline is not None else []
    )
    expected = " ".join(
        agent_transcript._clip(agent_transcript._spoken(term.last_prompt)).split()
    )
    since = int(float(identity[1] or 0) * 1000)
    # The newest record of THIS submission. A note typed into the running turn
    # after it is part of the same work, not a reason to stop waiting.
    start = next(
        (
            i
            for i in range(len(events) - 1, -1, -1)
            if events[i]["kind"] == "user_message"
            and int(events[i].get("ts_ms") or 0) >= since
            and " ".join(events[i]["payload"].get("text", "").split()) == expected
        ),
        None,
    )
    if start is None:
        return None
    own = events[start + 1 :]
    calls = {e["payload"].get("call_id"): e["payload"] for e in own if e["kind"] == "tool_call"}
    for event in own:
        if event["kind"] == "tool_result":
            calls.pop(event["payload"].get("call_id"), None)
    for call in calls.values():
        if str(call.get("name", "")).rsplit(".", 1)[-1] in {
            "request_user_input",
            "request_user_input_async",
            "AskUserQuestion",
        }:
            return {"status": "needs_input", "report": str(call.get("input") or "")}
    if not own or own[-1]["kind"] != "turn_finished":
        return (
            {
                "status": "running",
                "progress": str(own[-1].get("ts_ms")) + ":" + str(own[-1].get("seq")),
            }
            if own
            else None
        )
    finish = own[-1]["payload"]
    status = finish.get("status")
    report = next(
        (
            str(e["payload"].get("text") or "")
            for e in reversed(own)
            if e["kind"] == "assistant_text"
            and e["payload"].get("turn_id") == finish.get("turn_id")
        ),
        "",
    )
    if status in {"cancelled", "error"}:
        return {"status": status, "report": report or str(finish.get("error") or status)}
    if status != "done" or not report.strip():
        return None
    served = str(getattr(timeline, "model", "") or "")
    state, detail = served_model_verdict(term, served)
    if state == "refused":
        # The work may well be done; what is refused is calling it an
        # acceptable delivery. Nothing is resent: the request is already spent.
        return {
            "status": "model_unverified",
            "report": report,
            "model": served,
            "evidence": (
                f"The pane's own record says this reply came from {detail!r}, which is "
                "not a tier a chat-addressed Claude pane may run on. Nothing was resent."
            ),
        }
    return {
        "status": "completed",
        "report": report,
        "model": served,
        "evidence": (
            "Recorded CLI completion and matching report; not independent verification"
            + ("" if state == "ok" and served else f"; {detail}")
        ),
    }


def _target(owner: Any, term: Any) -> dict[str, Any]:
    """Who did this work, where, and on which revision.

    The chat's acceptance gate (:mod:`jarvis.agent_chat.delegation_acceptance`)
    verifies a handback against the checkout it actually came from, so the
    identity has to travel WITH the result rather than be inferred later from
    whichever workspace happens to be visible. The revision is read from the
    workspace's own git directory -- a plain local read, like every other
    probe here; it stays empty when the folder is not a checkout.
    """
    root = str(getattr(owner, "folder", "") or "")
    head = ""
    if root:
        try:
            from pathlib import Path

            from .git_local import git_out

            head = (git_out(["rev-parse", "HEAD"], Path(root)) or "").strip()
        except Exception:  # noqa: BLE001 - no revision is a missing binding, not a failure
            log.debug("delegation handback: could not read the workspace revision", exc_info=True)
    return {
        "project_id": str(getattr(owner, "project_id", "") or ""),
        "workspace_id": str(getattr(owner, "id", "") or ""),
        "terminal_id": "pane:" + str(getattr(term, "history_id", "") or ""),
        "root": root,
        "head_sha": head,
        "at": time.time(),
    }


def track_submission(registry: Any, owner: Any, term: Any) -> None:
    if term.submitted is False or term.last_submit_at is None:
        return
    identity = _identity(term)

    async def probe() -> dict[str, Any] | None:
        def current() -> bool:
            return registry.find_terminal("pane:" + term.history_id, owner.id) == (owner, term)

        if not current():
            return {"status": "stopped", "report": "The delegated coding pane was closed."}
        if identity != _identity(term):
            return {
                "status": "superseded",
                "report": (
                    "The pane received another submission or restarted. "
                    "Completion of the original task is unverified."
                ),
            }
        if term.status in {"failed", "exited"}:
            return {"status": term.status, "report": f"Coding process ended: {term.status}."}
        result = await asyncio.to_thread(_read, term, identity)
        if not current() or identity != _identity(term):
            return {
                "status": "superseded",
                "report": "The coding session changed while reading its result.",
            }
        if result is not None:
            # The parent verifies against THIS workspace and revision.
            result = {**result, "target": await asyncio.to_thread(_target, owner, term)}
        return result

    register_delegated_work(
        DelegatedWork(
            key=f"coding:{term.history_id}:{identity[0]}:{identity[1]}",
            name=str(term.name),
            probe=probe,
        )
    )
