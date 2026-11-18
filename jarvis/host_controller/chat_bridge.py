"""Connects the chat to the host controller in project mode.

Project mode is on while a state file names an active project. It is set and
cleared by the operator through the functions here. While it is on:

* the brain's system prompt carries ``instant_skill.md`` (the intent format);
* a reply that is one host intent JSON object is executed by the host and
  answered in plain text.

Replies that are not host intents return ``None`` and take the normal path.
Nothing here starts work on its own: the host runs only what Instant asked
for in a reply.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jarvis.host_controller.instant import handle_instant_text

STATE_NAME = "host_active_project.json"
_SKILL_PATH = Path(__file__).with_name("instant_skill.md")
_FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def _default_data_dir() -> Path:
    from jarvis.core import config as cfg

    return Path(cfg.DATA_DIR)


def _state_path(data_dir: Path | None) -> Path:
    return (data_dir or _default_data_dir()) / STATE_NAME


def set_active_project(project: str, repo: str, *, data_dir: Path | None = None) -> dict[str, str]:
    """Turn project mode on for ``project`` in GitHub repo ``repo``."""
    project = project.strip()
    repo = repo.strip()
    if not project or not repo or "/" not in repo:
        raise ValueError("project and repo (owner/name) are required")
    state = {"project": project, "repo": repo}
    path = _state_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return state


def clear_active_project(*, data_dir: Path | None = None) -> bool:
    """Turn project mode off. Returns whether it was on."""
    path = _state_path(data_dir)
    if not path.exists():
        return False
    path.unlink()
    return True


def active_project(*, data_dir: Path | None = None) -> dict[str, str] | None:
    """The active project state, or ``None`` when project mode is off or unreadable."""
    path = _state_path(data_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    project = raw.get("project")
    repo = raw.get("repo")
    if not isinstance(project, str) or not isinstance(repo, str) or not project or not repo:
        return None
    return {"project": project, "repo": repo}


def instant_directive() -> str:
    """The text the brain's system prompt gets while project mode is on."""
    return _SKILL_PATH.read_text(encoding="utf-8")


def parse_host_intent(reply: str) -> str | None:
    """The intent text in ``reply``, or ``None`` when it is not an intent at all.

    A reply counts as an intent when it is one JSON object carrying a ``kind``
    and no ``route``. The text is NOT validated here: the host's own parser
    does that, so a broken intent is reported as rejected instead of being
    silently answered as prose.
    """
    text = (reply or "").strip()
    fenced = _FENCE.fullmatch(text)
    candidate = fenced.group(1) if fenced else text
    if not (candidate.startswith("{") and candidate.endswith("}")):
        return None
    if '"route"' in candidate or '"kind"' not in candidate:
        return None
    return candidate


def _summary(result: Any) -> str:
    if not result.ok:
        return (
            f"Nicht ausgeführt: {result.error}"  # i18n-allow # noqa: E501
        )
    data = result.data or {}
    if result.kind == "PLAN":
        issues = data.get("issues", [])
        return f"Plan angelegt: {len(issues)} Aufgabe(n) im Projekt {data.get('project', '')}."  # i18n-allow # noqa: E501
    if result.kind == "REPORT_PROGRESS":
        return (
            f"Stand: {data.get('percent', 0)} % abgenommen "  # i18n-allow # noqa: E501
            f"({data.get('accepted', 0)} von {data.get('planned', 0)} Aufgaben), "
            f"{data.get('in_progress', 0)} in Arbeit."
        )
    return f"Ausgeführt: {result.kind}."  # i18n-allow # noqa: E501


def route_reply(
    reply: str,
    *,
    data_dir: Path | None = None,
    store: Any = None,
    workers: Any = None,
    run_dir: Path | None = None,
) -> str | None:
    """Execute a host intent from ``reply`` when project mode is on.

    Returns the text to show, or ``None`` when the reply should take the
    normal path. ``store`` and ``workers`` are injectable for tests; in
    production they are built from the active project.
    """
    state = active_project(data_dir=data_dir)
    if state is None:
        return None
    intent_text = parse_host_intent(reply)
    if intent_text is None:
        return None
    base = data_dir or _default_data_dir()
    if store is None:
        from jarvis.host_controller.gh_store import GhProjectStore

        store = GhProjectStore(repo=state["repo"])
    if workers is None:
        from jarvis.host_controller.workers import HostWorkerBundle

        workers = HostWorkerBundle(work_root=base / "host_work", repo_root=Path.cwd())
    result = handle_instant_text(
        intent_text,
        state["project"],
        store=store,
        workers=workers,
        run_dir=run_dir or base / "host" / state["project"],
    )
    return _summary(result)
