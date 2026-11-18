"""Read/write project memory via Obsidian / Ahmad-Context tooling.

Host adds only the top short snippets to the prompt. Writes use the existing
``obsidian_memory_sync.py remember`` path — never raw transcripts, never secrets.
"""

from __future__ import annotations

import logging
import re
import subprocess
from collections.abc import Callable
from typing import Any

from jarvis.core.process_utils import NO_WINDOW_CREATIONFLAGS

log = logging.getLogger(__name__)

Searcher = Callable[[str], list[dict[str, Any]]]
RememberFn = Callable[[str], None]

DEFAULT_SEARCH_SCRIPT = r"D:\Projects\pc-control\worker\obsidian-agent-tools.mjs"
DEFAULT_REMEMBER_SCRIPT = r"D:\Projects\AK-References\tools\obsidian_memory_sync.py"
_SECRET_RE = re.compile(
    r"(api[_-]?key|secret|token|password|bearer\s+[a-z0-9._\-]+)",
    re.IGNORECASE,
)


def _scrub(text: str) -> str:
    return _SECRET_RE.sub("[redacted]", text or "")


def _default_search(query: str) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["node", DEFAULT_SEARCH_SCRIPT, "search", query],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
            creationflags=NO_WINDOW_CREATIONFLAGS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("memory bridge search failed: %s", exc)
        return []
    if result.returncode != 0:
        return []
    import json

    try:
        payload = json.loads(result.stdout or "{}")
    except ValueError as exc:
        log.debug("host controller: fallback taken (%s)", exc)
        return []
        return []
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "title": str(row.get("title") or row.get("file") or "note"),
                "preview": str(row.get("preview") or "")[:400],
            }
        )
    return out


def _default_remember(text: str) -> None:
    scrubbed = _scrub(text).strip()
    if not scrubbed:
        return
    try:
        subprocess.run(
            [
                "python",
                DEFAULT_REMEMBER_SCRIPT,
                "remember",
                "--source",
                "grok",
                "--project",
                "Dodaios",
                "--text",
                scrubbed,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            creationflags=NO_WINDOW_CREATIONFLAGS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("memory bridge remember failed: %s", exc)


class MemoryBridge:
    def __init__(
        self,
        *,
        searcher: Searcher | None = None,
        remember_fn: RememberFn | None = None,
        max_snippets: int = 3,
        max_chars: int = 2500,
    ) -> None:
        self._searcher = searcher or _default_search
        self._remember = remember_fn or _default_remember
        self.max_snippets = max_snippets
        self.max_chars = max_chars

    def prompt_snippets(self, query: str) -> str:
        hits = list(self._searcher(query) or [])[: self.max_snippets]
        if not hits:
            return ""
        lines = ["# Project memory (short snippets; ignore if irrelevant)"]
        used = 0
        for hit in hits:
            title = _scrub(str(hit.get("title") or "note"))
            preview = _scrub(str(hit.get("preview") or "")).strip()
            if len(preview) > 280:
                preview = preview[:277] + "…"
            block = f"- {title}: {preview}"
            if used + len(block) > self.max_chars:
                break
            lines.append(block)
            used += len(block)
        return "\n".join(lines)

    def remember_decision(self, text: str) -> None:
        scrubbed = _scrub(text).strip()
        if not scrubbed:
            return
        # Never store raw transcripts.
        if "user:" in scrubbed.lower() and "assistant:" in scrubbed.lower():
            scrubbed = scrubbed.split("\n")[0][:500]
        self._remember(scrubbed)
