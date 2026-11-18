"""Park risky Instant intents until the user answers ja/nein.

Pending action lives in the project state file (not in model memory).
Inspired by LangGraph interrupt resume and Temporal approval signals —
ideas only; no third-party runtime.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from jarvis.host_controller.intent import ParsedIntent

log = logging.getLogger(__name__)

_RISKY_RE = re.compile(
    r"("
    r"\bmerge\b.*\bmain\b|\bmain\b.*\bmerge\b|"
    r"\bdelete\b|\brm\s+-rf\b|\bremove-item\b|\bdrop\s+table\b|"
    r"\bspend\b|\bbill\b|\bpurchase\b|\bpaid\s+fallback\b|"
    r"\brepo\s+settings\b|\bbranch\s+protection\b|\bwebhook\b|"
    r"\bforce[- ]push\b|\boutside\s+the\s+project\b|\bother\s+repo\b"
    r")",
    re.IGNORECASE,
)

_YES = frozenset({"ja", "yes", "y", "ok", "okay", "approve", "approved"})
_NO = frozenset({"nein", "no", "n", "cancel", "abort", "reject", "denied"})


def is_risky_intent(intent: ParsedIntent) -> bool:
    """True when the intent text asks for a risky host action."""
    data = intent.data
    blobs: list[str] = []
    for key in ("task", "reason", "title", "findings", "project"):
        value = data.get(key)
        if isinstance(value, str):
            blobs.append(value)
        elif isinstance(value, list):
            blobs.extend(str(item) for item in value)
    for task in data.get("tasks") or []:
        if isinstance(task, dict):
            blobs.append(str(task.get("title") or ""))
            blobs.append(str(task.get("acceptance") or ""))
    text = " ".join(blobs)
    return bool(_RISKY_RE.search(text))


def approval_question(intent: ParsedIntent) -> str:
    kind = intent.kind
    summary = ""
    if kind == "ASSIGN_TASK":
        summary = str(intent.data.get("task") or "")[:200]
    elif kind == "ACCEPT_TASK":
        summary = f"accept #{intent.data.get('issue_number')} @ {intent.data.get('commit_sha')}"
    else:
        summary = kind
    # i18n-allow: runtime chat question to the user
    return (
        f"Risikante Aktion ({kind}): {summary}. "  # i18n-allow
        f"Soll ich das ausführen? Antworte mit ja oder nein."  # i18n-allow
    )


def parse_approval_answer(text: str) -> bool | None:
    raw = (text or "").strip().lower()
    if not raw:
        return None
    # Accept a single-token answer, optionally with punctuation.
    token = re.split(r"[\s,.!?]+", raw, maxsplit=1)[0]
    if token in _YES or raw in _YES:
        return True
    if token in _NO or raw in _NO:
        return False
    return None


def load_state(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("host approval state unreadable at %s: %s", path, exc)
        return {}
    return raw if isinstance(raw, dict) else {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def store_pending(path: Path, *, intent_text: str, question: str) -> dict[str, Any]:
    state = load_state(path)
    state["pending_action"] = {
        "intent_text": intent_text,
        "question": question,
    }
    save_state(path, state)
    return state


def clear_pending(path: Path) -> dict[str, Any]:
    state = load_state(path)
    state.pop("pending_action", None)
    save_state(path, state)
    return state


def get_pending(path: Path) -> dict[str, Any] | None:
    state = load_state(path)
    pending = state.get("pending_action")
    return pending if isinstance(pending, dict) else None
