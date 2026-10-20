"""Redact credentials from text that leaves this process.

A key the person typed or dictated is in the conversation like any other
sentence, and the conversation is copied into a delegated worker's mission
prompt as context. From there it reaches a third-party CLI's argv, its log,
and the mission event store — three places a credential must never be.
AGENTS.md §3/AP-2: voice and chat never accept a secret, and nothing commits
one; this is the same rule applied to text on its way OUT of the app.

The patterns were already written for the CLI usage log
(:mod:`jarvis.clis.usage_log`, which now imports them from here) and are kept
in ONE place so the two callers cannot drift apart.

Deliberately conservative: these match credential SHAPES with enough entropy
and a known prefix to be unambiguous. A redaction that fires on ordinary prose
would silently remove task context, which is its own failure — so a token this
does not recognise is not redacted, and this is a safety net rather than a
guarantee. The guarantee stays where it belongs: credentials live in
``get_secret``, never in a prompt.
"""
from __future__ import annotations

import re
from typing import Final

#: What a redacted credential is replaced with. Readable on purpose: a worker
#: that sees it knows a value was withheld rather than mistyped.
REDACTED: Final[str] = "<REDACTED>"

SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"gho_[A-Za-z0-9]{36}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{80,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    re.compile(r"ya29\.[0-9A-Za-z_-]{50,}"),
    re.compile(r"xox[bpoa]-[A-Za-z0-9-]{20,}"),
    re.compile(r"sk_(live|test)_[A-Za-z0-9]{20,}"),
    re.compile(r"rk_(live|test)_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(?:authorization|api[_-]?key|bearer)[\s:=]+[A-Za-z0-9._-]{16,}"),
)


def scrub_secrets(text: str) -> str:
    """Replace every recognised credential in ``text`` with :data:`REDACTED`."""
    if not text:
        return ""
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


__all__ = ["REDACTED", "SECRET_PATTERNS", "scrub_secrets"]
