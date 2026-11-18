"""Research-before-planning: GitHub code search + open PRs for PLAN words."""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable
from typing import Any

from jarvis.core.process_utils import NO_WINDOW_CREATIONFLAGS

log = logging.getLogger(__name__)

CodeSearch = Callable[[str], list[dict[str, Any]]]
PrSearch = Callable[[str], list[dict[str, Any]]]


def _gh_json(args: list[str]) -> Any:
    result = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
        check=False,
        creationflags=NO_WINDOW_CREATIONFLAGS,
    )
    if result.returncode != 0:
        log.warning("gh research failed: %s", (result.stderr or "")[:200])
        return None
    text = (result.stdout or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError as exc:
        log.warning("gh research returned non-JSON: %s", exc)
        return None


def default_code_search(query: str, *, repo: str = "") -> list[dict[str, Any]]:
    args = ["search", "code", query, "--limit", "5", "--json", "path,repository,url"]
    if repo:
        args.extend(["--repo", repo])
    payload = _gh_json(args)
    if not isinstance(payload, list):
        return []
    out: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "path": str(row.get("path") or ""),
                "text": str(row.get("url") or row.get("path") or ""),
            }
        )
    return out


def default_pr_search(query: str, *, repo: str = "") -> list[dict[str, Any]]:
    args = [
        "search",
        "prs",
        query,
        "--state",
        "open",
        "--limit",
        "5",
        "--json",
        "number,title,url",
    ]
    if repo:
        args.extend(["--repo", repo])
    payload = _gh_json(args)
    if not isinstance(payload, list):
        return []
    out: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "number": row.get("number"),
                "title": str(row.get("title") or ""),
            }
        )
    return out


def collect_research(
    *,
    title: str,
    tasks: list[dict[str, Any]],
    code_search: CodeSearch | None = None,
    pr_search: PrSearch | None = None,
) -> dict[str, Any]:
    words = [title]
    for task in tasks:
        words.append(str(task.get("title") or ""))
    query = " ".join(w for w in words if w).strip() or title
    # No injected searcher → empty result (unit tests stay offline). Production
    # wires ``default_code_search`` / ``default_pr_search`` at the host boundary.
    code_fn = code_search if code_search is not None else (lambda _q: [])
    pr_fn = pr_search if pr_search is not None else (lambda _q: [])
    code_hits = list(code_fn(query) or [])
    pr_hits = list(pr_fn(query) or [])
    return {"query": query, "code": code_hits, "prs": pr_hits}


def format_research_block(research: dict[str, Any]) -> str:
    lines = ["## Host research (before plan)", f"Query: {research.get('query') or ''}"]
    code = list(research.get("code") or [])
    if code:
        lines.append("Code hits:")
        for hit in code[:5]:
            path = hit.get("path") or hit.get("text") or ""
            lines.append(f"- {path}")
    else:
        lines.append("Code hits: none")
    prs = list(research.get("prs") or [])
    if prs:
        lines.append("Open PRs:")
        for pr in prs[:5]:
            lines.append(f"- #{pr.get('number')}: {pr.get('title')}")
    else:
        lines.append("Open PRs: none")
    return "\n".join(lines)
