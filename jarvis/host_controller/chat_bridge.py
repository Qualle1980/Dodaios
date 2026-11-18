"""Connects the chat to the host controller in project mode.

Project mode is on while a state file names an active project. It is set and
cleared by the operator through the functions here. While it is on:

* the brain's system prompt carries ``instant_skill.md`` plus a bounded state
  package and short memory snippets;
* a reply that is one host intent JSON object is executed by the host and
  answered in plain text;
* risky intents are parked until the user answers ja/nein;
* a plain STATUS request is answered from GitHub by the host.

Replies that are not host intents return ``None`` and take the normal path.
Nothing here starts work on its own: the host runs only what Instant asked
for in a reply (or what the user approved).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jarvis.host_controller.approval import (
    approval_question,
    clear_pending,
    get_pending,
    is_risky_intent,
    parse_approval_answer,
    store_pending,
)
from jarvis.host_controller.instant import handle_instant_text
from jarvis.host_controller.intent import IntentParseError, parse_intent
from jarvis.host_controller.state_package import (
    build_state_package,
    format_status_reply,
    looks_like_status_request,
)
from jarvis.host_controller.workers import build_live_host_workers

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
    path = _state_path(data_dir)
    existing = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = raw
        except (OSError, ValueError):
            existing = {}
    state = {**existing, "project": project, "repo": repo}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return {"project": project, "repo": repo}


def clear_active_project(*, data_dir: Path | None = None) -> bool:
    """Turn project mode off. Returns whether it was on."""
    path = _state_path(data_dir)
    if not path.exists():
        return False
    path.unlink()
    return True


def active_project(*, data_dir: Path | None = None) -> dict[str, Any] | None:
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
    return dict(raw)


def instant_directive() -> str:
    """The text the brain's system prompt gets while project mode is on."""
    return _SKILL_PATH.read_text(encoding="utf-8")


def project_prompt_addon(
    *,
    data_dir: Path | None = None,
    store: Any = None,
    repo_root: Path | None = None,
    progress_path: Path | None = None,
    memory_bridge: Any = None,
    open_prs: list[dict[str, Any]] | None = None,
) -> str:
    """Extra system-prompt text for project mode. Empty when mode is off."""
    state = active_project(data_dir=data_dir)
    if state is None:
        return ""
    parts = [instant_directive()]
    if store is not None:
        run_dir = (data_dir or _default_data_dir()) / "host" / state["project"]
        package = build_state_package(
            project=state["project"],
            store=store,
            repo_root=repo_root or Path.cwd(),
            progress_path=progress_path or (run_dir / "progress.md"),
            open_prs=open_prs,
        )
        if package.strip():
            parts.append(package)
    if memory_bridge is not None:
        snippets = memory_bridge.prompt_snippets(state["project"])
        if snippets.strip():
            parts.append(snippets)
    return "\n\n".join(parts)


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
        research = data.get("research") or {}
        base = f"Plan angelegt: {len(issues)} Aufgabe(n) im Projekt {data.get('project', '')}."  # i18n-allow # noqa: E501
        hits = list(research.get("code") or []) + list(research.get("prs") or [])
        summary = data.get("research_summary") or ""
        if hits and summary:
            return f"{base}\n{summary}"
        return base
    if result.kind == "REPORT_PROGRESS":
        ready = data.get("ready") or []
        ready_txt = f" Bereit: {ready}." if ready else ""  # i18n-allow
        return (
            f"Stand: {data.get('percent', 0)} % abgenommen "  # i18n-allow # noqa: E501
            f"({data.get('accepted', 0)} von {data.get('planned', 0)} Aufgaben), "
            f"{data.get('in_progress', 0)} in Arbeit.{ready_txt}"
        )
    if result.kind == "ACCEPT_TASK" and data.get("final_report"):
        return f"Ausgeführt: {result.kind}. Abschlussbericht geschrieben; warte auf dich."  # i18n-allow # noqa: E501
    return f"Ausgeführt: {result.kind}."  # i18n-allow # noqa: E501


def _default_workers(*, data_dir: Path, repo_root: Path | None = None) -> Any:
    return build_live_host_workers(
        repo_root=repo_root or Path.cwd(),
        work_root=data_dir / "host_work",
    )


def _production_searchers(repo: str) -> tuple[Any, Any]:
    from jarvis.host_controller.research import default_code_search, default_pr_search

    return (
        (lambda q, _repo=repo: default_code_search(q, repo=_repo)),
        (lambda q, _repo=repo: default_pr_search(q, repo=_repo)),
    )


def _run_intent_text(
    intent_text: str,
    *,
    state: dict[str, Any],
    data_dir: Path,
    store: Any,
    workers: Any,
    run_dir: Path,
    wire_live_research: bool = False,
) -> str:
    code_search = None
    pr_search = None
    if wire_live_research:
        code_search, pr_search = _production_searchers(str(state.get("repo") or ""))
    result = handle_instant_text(
        intent_text,
        state["project"],
        store=store,
        workers=workers,
        run_dir=run_dir,
        code_search=code_search,
        pr_search=pr_search,
    )
    return _summary(result)


def handle_project_user_turn(
    text: str,
    *,
    data_dir: Path | None = None,
    store: Any = None,
    workers: Any = None,
    run_dir: Path | None = None,
    repo_root: Path | None = None,
) -> str | None:
    """Handle STATUS and ja/nein approval before Instant sees the turn.

    Returns the text to show, or ``None`` when the turn should continue to the
    model.
    """
    state = active_project(data_dir=data_dir)
    if state is None:
        return None
    base = data_dir or _default_data_dir()
    path = _state_path(data_dir)
    run_path = Path(run_dir) if run_dir is not None else base / "host" / state["project"]

    answer = parse_approval_answer(text)
    pending = get_pending(path)
    if pending is not None and answer is not None:
        intent_text = str(pending.get("intent_text") or "")
        clear_pending(path)
        if answer is False:
            return "Abgelehnt. Die Aktion wurde nicht ausgeführt."  # i18n-allow
        live = store is None or workers is None
        if store is None:
            from jarvis.host_controller.gh_store import GhProjectStore

            store = GhProjectStore(repo=state["repo"])
        if workers is None:
            workers = _default_workers(data_dir=base, repo_root=repo_root)
        return _run_intent_text(
            intent_text,
            state=state,
            data_dir=base,
            store=store,
            workers=workers,
            run_dir=run_path,
            wire_live_research=live,
        )

    if looks_like_status_request(text):
        if store is None:
            from jarvis.host_controller.gh_store import GhProjectStore

            store = GhProjectStore(repo=state["repo"])
        return format_status_reply(project=state["project"], store=store)

    return None


def route_reply(
    reply: str,
    *,
    data_dir: Path | None = None,
    store: Any = None,
    workers: Any = None,
    run_dir: Path | None = None,
    repo_root: Path | None = None,
) -> str | None:
    """Execute a host intent from ``reply`` when project mode is on.

    Returns the text to show, or ``None`` when the reply should take the
    normal path. ``store`` and ``workers`` are injectable for tests; in
    production they are built from the active project (live worker drivers).
    """
    state = active_project(data_dir=data_dir)
    if state is None:
        return None
    intent_text = parse_host_intent(reply)
    if intent_text is None:
        return None
    base = data_dir or _default_data_dir()
    path = _state_path(data_dir)
    run_path = Path(run_dir) if run_dir is not None else base / "host" / state["project"]

    try:
        parsed = parse_intent(intent_text)
    except IntentParseError as exc:
        return f"Nicht ausgeführt: {exc}"  # i18n-allow

    if is_risky_intent(parsed):
        question = approval_question(parsed)
        store_pending(path, intent_text=intent_text, question=question)
        return question

    live = store is None or workers is None
    if store is None:
        from jarvis.host_controller.gh_store import GhProjectStore

        store = GhProjectStore(repo=state["repo"])
    if workers is None:
        workers = _default_workers(data_dir=base, repo_root=repo_root)
    return _run_intent_text(
        intent_text,
        state=state,
        data_dir=base,
        store=store,
        workers=workers,
        run_dir=run_path,
        wire_live_research=live,
    )
