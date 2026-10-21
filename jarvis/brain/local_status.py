"""Answer a status question from real local data, with no model at all.

The sibling of :mod:`jarvis.brain.one_shot_reply`. That module answers the
closed set whose answer is a fixed line or the clock; this one answers the
questions whose answer is a FACT about this machine — which project is
checked out, whether a coding agent is still running, whether anything is
waiting for an approval, whether the assistant is ready.

Measured 2026-10-21 on the live install: "Wie ist der Projektstand?",  # i18n-allow
"Läuft Claude noch?", "Gibt es offene Freigaben?" and "Ist Jarvis bereit?"  # i18n-allow
all classified as ``capable-default`` and went to a full model turn over the
browser bridge — 7,1-7,2 s warm, plus a model call on the user's plan, to
report facts this process already holds.

Two rules decide everything here:

* **Never invent a fact.** Every answer is built from a value this module
  actually read. When a source is not wired up or fails, the answer SAYS the
  value is unavailable; it never substitutes a plausible default. A confident
  wrong status reading is worse than a slow correct one.
* **Never block the turn.** Each source is given a short deadline and runs off
  the event loop. A source that is slow or hung degrades to "unavailable" for
  that one answer instead of making a status question cost more than the model
  turn it replaces.

Sources are INJECTED (``StatusSources``) rather than imported here, so this
module has no dependency on the mission or agent layers and a test pins every
value. Whatever the caller cannot supply is reported as unavailable.
"""
from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .intent_router import one_shot_kind

log = logging.getLogger(__name__)

#: Per-source deadline. A status answer that takes longer than this has lost
#: its only advantage over asking the model, so the source degrades instead.
SOURCE_TIMEOUT_S = 1.5

#: Questions this module answers, per kind. Anchored to the WHOLE utterance
#: like the one-shot matcher: a status question followed by a real request is
#: not one of these and still reaches the model.
_PATTERNS: dict[str, tuple[str, ...]] = {
    "project_status": (
        r"wie\s+ist\s+der\s+(?:aktuelle\s+)?projekt(?:stand|status)",  # i18n-allow
        r"was\s+ist\s+der\s+(?:projekt)?stand",  # i18n-allow
        r"wie\s+(?:ist|steht)\s+(?:es\s+)?(?:um\s+)?das\s+projekt",  # i18n-allow
        r"(?:what|how)\s+is\s+the\s+project\s+(?:status|state)",
        r"project\s+status",
    ),
    "agents_running": (
        r"l[äa]uft\s+(?:noch\s+)?(?:claude|codex|der\s+agent)(?:\s+noch)?",  # i18n-allow
        r"laufen\s+(?:noch\s+)?(?:agenten|die\s+agenten)",  # i18n-allow
        r"(?:is|are)\s+(?:claude|codex|the\s+agents?)\s+still\s+running",
    ),
    "pending_approvals": (
        r"gibt\s+es\s+(?:noch\s+)?offene\s+freigaben",  # i18n-allow
        r"(?:sind|gibt)\s+(?:es\s+)?freigaben\s+offen",  # i18n-allow
        r"offene\s+freigaben",  # i18n-allow
        r"(?:any|are\s+there)\s+pending\s+approvals",
    ),
    "readiness": (
        r"ist\s+(?:alles|jarvis)\s+bereit",  # i18n-allow
        r"bist\s+du\s+bereit",  # i18n-allow
        r"(?:is\s+)?(?:everything|jarvis)\s+ready",
        r"are\s+you\s+ready",
    ),
}


def status_kind(user_text: str) -> str | None:
    """Which status question ``user_text`` is, or None for everything else.

    Checked AFTER :func:`jarvis.brain.intent_router.one_shot_kind`, which owns
    the greeting/clock set; a one-shot is never also a status question.
    """
    text = (user_text or "").strip().strip("?!.,").lower()
    if not text or one_shot_kind(user_text) is not None:
        return None
    for kind, patterns in _PATTERNS.items():
        for pattern in patterns:
            if re.fullmatch(rf"(?:jarvis[,\s]+)?{pattern}", text, re.IGNORECASE):
                return kind
    return None


@dataclass(frozen=True, slots=True)
class StatusSources:
    """Where the facts come from. Anything left None reports as unavailable.

    ``repo`` is the checkout to describe. The three callables are supplied by
    the runtime that owns the data: the caller knows whether a mission
    coordinator or an agent registry exists, this module does not.
    """

    repo: Path | None = None
    pending_approvals: Callable[[], Awaitable[list[Any]]] | None = None
    running_agents: Callable[[], Awaitable[list[str]]] | None = None
    readiness: Callable[[], Awaitable[dict[str, bool]]] | None = None


# --- phrasing ---------------------------------------------------------------
# Short on purpose: these are spoken as well as written, and a status answer
# that runs to a paragraph is a worse answer than a slow one.
_UNAVAILABLE = {
    "de": "Das kann ich gerade nicht auslesen.",  # i18n-allow: runtime reply
    "en": "I cannot read that right now.",
    "es": "No puedo consultarlo ahora mismo.",  # i18n-allow
}
_PROJECT = {
    "de": "Projekt {name}, Branch {branch}, letzter Commit {commit}. {dirty}",  # i18n-allow
    "en": "Project {name}, branch {branch}, last commit {commit}. {dirty}",
    "es": "Proyecto {name}, rama {branch}, último commit {commit}. {dirty}",  # i18n-allow
}
_DIRTY = {
    "de": ("Keine offenen Änderungen.", "{n} offene Änderungen."),  # i18n-allow
    "en": ("No uncommitted changes.", "{n} uncommitted changes."),
    "es": ("Sin cambios pendientes.", "{n} cambios pendientes."),  # i18n-allow
}
_AGENTS = {
    "de": ("Es läuft kein Agent.", "Aktiv: {names}."),  # i18n-allow
    "en": ("No agent is running.", "Running: {names}."),
    "es": ("Ningún agente activo.", "Activos: {names}."),  # i18n-allow
}
_APPROVALS = {
    "de": ("Keine offenen Freigaben.", "{n} offene Freigaben."),  # i18n-allow
    "en": ("No pending approvals.", "{n} pending approvals."),
    "es": ("Sin aprobaciones pendientes.", "{n} aprobaciones pendientes."),  # i18n-allow
}
_READY = {
    "de": ("Alles bereit.", "Noch nicht bereit: {missing}."),  # i18n-allow
    "en": ("Everything is ready.", "Not ready yet: {missing}."),
    "es": ("Todo listo.", "Aún no listo: {missing}."),  # i18n-allow
}


def _lang(lang: str) -> str | None:
    code = (lang or "").split("-")[0].lower()
    return code if code in _UNAVAILABLE else None


async def _guarded(awaitable: Awaitable[Any], what: str) -> Any | None:
    """Run a source under the deadline. A failure is None, never a guess."""
    try:
        return await asyncio.wait_for(awaitable, timeout=SOURCE_TIMEOUT_S)
    except TimeoutError:
        log.info("local status: %s did not answer within %.1fs", what, SOURCE_TIMEOUT_S)
    except Exception:
        log.info("local status: %s failed", what, exc_info=True)
    return None


def _git(repo: Path, *args: str) -> str | None:
    """One read-only git call. Returns None rather than raising or guessing."""
    from jarvis.core.process_utils import NO_WINDOW_CREATIONFLAGS

    try:
        done = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            encoding="utf-8",  # AP-1: Windows would otherwise decode as cp1252
            errors="replace",
            timeout=SOURCE_TIMEOUT_S,
            creationflags=NO_WINDOW_CREATIONFLAGS,  # AP-1
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        log.info("local status: git %s failed", args[0] if args else "", exc_info=True)
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()


#: A spoken commit subject is a headline, not a changelog entry.
_COMMIT_CHARS = 60


async def _project_status(repo: Path | None, lang: str) -> str | None:
    if repo is None:
        return None
    # ``Path(".")`` has an empty ``.name``, which produced "Projekt ,".
    # Resolve first so the answer names the directory the user would name.
    # Off the loop: resolve() touches the filesystem.
    root = await asyncio.to_thread(repo.resolve)
    # The three reads are independent, so they run together: serially they
    # cost ~420 ms here, which is a large share of a status answer's budget.
    branch, commit, porcelain = await asyncio.gather(
        asyncio.to_thread(_git, root, "rev-parse", "--abbrev-ref", "HEAD"),
        asyncio.to_thread(_git, root, "log", "-1", "--format=%s"),
        asyncio.to_thread(_git, root, "status", "--porcelain"),
    )
    if branch is None or porcelain is None:
        return None
    changed = len([ln for ln in porcelain.splitlines() if ln.strip()])
    clean, dirty = _DIRTY[lang]
    subject = (commit or "").strip() or "-"
    if len(subject) > _COMMIT_CHARS:
        subject = subject[: _COMMIT_CHARS - 1].rstrip() + "…"
    return _PROJECT[lang].format(
        name=root.name or str(root),
        branch=branch,
        commit=subject,
        dirty=clean if changed == 0 else dirty.format(n=changed),
    )


async def local_status_answer(
    user_text: str, lang: str, *, sources: StatusSources | None = None
) -> str | None:
    """The real local answer for a status question, or None to use a model.

    None means "not a status question" or "no language line": the caller then
    proceeds normally. A recognised question whose data could not be read
    returns an honest "cannot read that", NOT a model fallback — the user asked
    for a local fact, and silently spending a model call on it is the behaviour
    this module exists to remove.
    """
    kind = status_kind(user_text)
    if kind is None:
        return None
    language = _lang(lang)
    if language is None:
        return None
    src = sources or StatusSources()
    unavailable = _UNAVAILABLE[language]

    if kind == "project_status":
        return await _guarded(
            _project_status(src.repo, language), "project status"
        ) or unavailable

    if kind == "agents_running":
        if src.running_agents is None:
            return unavailable
        names = await _guarded(src.running_agents(), "running agents")
        if names is None:
            return unavailable
        none_line, some_line = _AGENTS[language]
        return none_line if not names else some_line.format(names=", ".join(names))

    if kind == "pending_approvals":
        if src.pending_approvals is None:
            return unavailable
        items = await _guarded(src.pending_approvals(), "pending approvals")
        if items is None:
            return unavailable
        none_line, some_line = _APPROVALS[language]
        return none_line if not items else some_line.format(n=len(items))

    if kind == "readiness":
        if src.readiness is None:
            return unavailable
        checks = await _guarded(src.readiness(), "readiness")
        if checks is None:
            return unavailable
        missing = sorted(name for name, ok in checks.items() if not ok)
        ready_line, not_ready = _READY[language]
        return ready_line if not missing else not_ready.format(missing=", ".join(missing))

    return None


__all__ = [
    "SOURCE_TIMEOUT_S",
    "StatusSources",
    "local_status_answer",
    "status_kind",
]
