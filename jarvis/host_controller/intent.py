"""Strict Instant intent parser.

ChatGPT Instant emits text only. Its JSON sometimes contains Markdown-style
``\\_`` inside string values (``ASSIGN\\_TASK``, ``test\\_x.py``), which is
invalid JSON and used to drop a delegation silently via ``json.loads``.
This parser normalises that one escape inside strings, logs it, and rejects
every other unsupported escape (including ``\\x``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from jarvis.host_controller.types import INTENT_KEYS, INTENT_KINDS

log = logging.getLogger(__name__)

_JSON_ESCAPES = frozenset({'"', "\\", "/", "b", "f", "n", "r", "t", "u"})


class IntentParseError(ValueError):
    """The Instant text is not a host intent the controller may run."""


@dataclass(frozen=True)
class ParsedIntent:
    intent_id: str
    kind: str
    data: dict[str, Any]
    normalised_underscore: bool = False


def _extract_candidate(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        raise IntentParseError("empty intent text")
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    raise IntentParseError("intent must be a JSON object; raw text is never executed")


def _normalise_string_escapes(candidate: str) -> tuple[str, bool]:
    """Rewrite ``\\_`` → ``_`` only inside JSON string values; reject ``\\x`` etc."""
    out: list[str] = []
    i = 0
    in_string = False
    normalised = False
    length = len(candidate)
    while i < length:
        ch = candidate[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue
        if ch == "\\":
            if i + 1 >= length:
                raise IntentParseError("unterminated escape in JSON string")
            nxt = candidate[i + 1]
            if nxt == "_":
                out.append("_")
                normalised = True
                i += 2
                continue
            if nxt == "x":
                raise IntentParseError("unsupported escape \\x in JSON string")
            if nxt == "u":
                hex_part = candidate[i + 2 : i + 6]
                if len(hex_part) != 4 or any(
                    c not in "0123456789abcdefABCDEF" for c in hex_part
                ):
                    raise IntentParseError("bad unicode escape in JSON string")
                out.append(candidate[i : i + 6])
                i += 6
                continue
            if nxt not in _JSON_ESCAPES:
                raise IntentParseError(f"unsupported escape \\{nxt} in JSON string")
            out.append(ch)
            out.append(nxt)
            i += 2
            continue
        out.append(ch)
        if ch == '"':
            in_string = False
        i += 1
    if in_string:
        raise IntentParseError("unterminated JSON string")
    return "".join(out), normalised


def parse_intent(text: str) -> ParsedIntent:
    """Parse Instant output into a validated host intent.

    Never treats free text as a shell command. Unknown keys and unknown kinds
    raise :class:`IntentParseError`.
    """
    candidate = _extract_candidate(text)
    cleaned, normalised = _normalise_string_escapes(candidate)
    if normalised:
        log.info(
            "host intent normalised Markdown underscore escapes (\\_) inside JSON strings"
        )
    try:
        data = json.loads(cleaned)
    except ValueError as exc:
        raise IntentParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise IntentParseError("intent root must be an object")

    intent_id = data.get("intent_id")
    kind = data.get("kind")
    if not isinstance(intent_id, str) or not intent_id.strip():
        raise IntentParseError("intent_id is required")
    if not isinstance(kind, str) or kind not in INTENT_KINDS:
        raise IntentParseError(f"unknown intent kind: {kind!r}")

    allowed = INTENT_KEYS[kind]
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise IntentParseError(f"unknown key(s): {', '.join(unknown)}")

    project = data.get("project")
    if kind != "PLAN" and (not isinstance(project, str) or not project.strip()):
        # PLAN also requires project; checked below with title/tasks.
        if "project" in allowed and (not isinstance(project, str) or not project.strip()):
            raise IntentParseError("project is required")
    if not isinstance(project, str) or not project.strip():
        raise IntentParseError("project is required")

    if kind == "PLAN":
        title = data.get("title")
        tasks = data.get("tasks")
        if not isinstance(title, str) or not title.strip():
            raise IntentParseError("PLAN requires title")
        if not isinstance(tasks, list) or not tasks:
            raise IntentParseError("PLAN requires a non-empty tasks list")
        for item in tasks:
            if not isinstance(item, dict):
                raise IntentParseError("each PLAN task must be an object")
            if set(item) - {"title", "acceptance"}:
                raise IntentParseError("unknown key in PLAN task")
            if not isinstance(item.get("title"), str) or not item["title"].strip():
                raise IntentParseError("PLAN task title is required")
            if "acceptance" in item and not isinstance(item["acceptance"], str):
                raise IntentParseError("PLAN task acceptance must be a string")

    if kind == "ASSIGN_TASK":
        if not isinstance(data.get("issue_number"), int):
            raise IntentParseError("ASSIGN_TASK requires integer issue_number")
        worker = data.get("worker")
        if worker not in {"claude", "grok"}:
            raise IntentParseError("ASSIGN_TASK worker must be claude or grok")
        if "allow_subagents" in data and not isinstance(data["allow_subagents"], bool):
            raise IntentParseError("allow_subagents must be a boolean")
        if "subagent_kind" in data and data["subagent_kind"] is not None:
            if not isinstance(data["subagent_kind"], str):
                raise IntentParseError("subagent_kind must be a string or null")

    if kind == "REQUEST_REVIEW":
        if not isinstance(data.get("issue_number"), int):
            raise IntentParseError("REQUEST_REVIEW requires integer issue_number")
        sha = data.get("commit_sha")
        if not isinstance(sha, str) or not sha.strip():
            raise IntentParseError("REQUEST_REVIEW requires a non-empty commit_sha")

    if kind == "REQUEST_FIX":
        if not isinstance(data.get("issue_number"), int):
            raise IntentParseError("REQUEST_FIX requires integer issue_number")
        findings = data.get("findings")
        if not isinstance(findings, list) or not all(isinstance(x, str) for x in findings):
            raise IntentParseError("REQUEST_FIX requires findings: list[str]")

    if kind == "ACCEPT_TASK":
        if not isinstance(data.get("issue_number"), int):
            raise IntentParseError("ACCEPT_TASK requires integer issue_number")
        sha = data.get("commit_sha")
        if not isinstance(sha, str) or not sha.strip():
            raise IntentParseError("ACCEPT_TASK requires a non-empty commit_sha")
        if not isinstance(data.get("tests_green"), bool):
            raise IntentParseError("ACCEPT_TASK requires boolean tests_green")

    if kind == "BLOCKED":
        if not isinstance(data.get("issue_number"), int):
            raise IntentParseError("BLOCKED requires integer issue_number")
        if "reason" in data and not isinstance(data["reason"], str):
            raise IntentParseError("BLOCKED reason must be a string")

    return ParsedIntent(
        intent_id=intent_id.strip(),
        kind=kind,
        data=data,
        normalised_underscore=normalised,
    )
