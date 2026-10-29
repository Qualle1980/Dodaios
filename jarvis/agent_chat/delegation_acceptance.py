"""A coding agent's "done" is delivery, not acceptance.

:mod:`jarvis.agentic_ide.delegation_wait` already says so in the only place
that can know: a completion it reads from a CLI transcript carries
``evidence="Recorded CLI completion and matching report; not independent
verification"``. What was missing is the consequence. The report travels back
into the original chat turn, where the chief is *asked* to check the diff and
run the tests — and an instruction is not a guarantee. A model that answers
"PASS, all tests pass" without having looked is indistinguishable, to the
person reading the chat, from one that did.

So this module turns the handback's own honesty into a condition. When a wave
hands back a coding completion that nobody has independently verified, the
next wave runs with an :class:`AcceptanceGate`. The gate watches what the turn
actually executes, through the one authorized path (``ToolExecutor``'s
``ActionProposed`` / ``ActionExecuted`` records, AP-3), and asks for two
separate facts before an acceptance claim may stand:

* version-control state for the work itself (``workspace-git``, or a read-only
  ``git`` command), and
* a test run (``pytest``, ``npm test``, ``cargo test`` …).

Either one alone is not acceptance: a diff proves something was written, a
green suite with no diff read proves nothing about *this* work.

Nothing here decides whether work was good. It decides whether the chat may
TELL the person it was good. Without the evidence the answer is rewritten to
an honest unverified one; the work, the pane and the report are untouched, and
no retry is started.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

log = logging.getLogger(__name__)

#: The status a coding handback reports when only the CLI said it was done.
COMPLETED = "completed"
#: The marker ``jarvis/agentic_ide/delegation_wait.py`` stamps on such a row.
NOT_INDEPENDENT = "not independent verification"


# ------------------------------------------------------------------ the claim

# An acceptance claim is a statement that the delegated work is good: a
# verdict word, or a sentence that the tests passed. Deliberately narrow — a
# report that stays descriptive ("T1 says it added subtract()") is not a
# verdict and must pass through untouched.
_ACCEPTANCE_RE = re.compile(
    r"(?:"
    r"\bpass(?:ed|es)?\b|\bgreen\b|\bverified\b|\bconfirmed\b|"
    r"\ball\s+tests?\s+(?:pass|passed|are\s+green)\b|"
    r"\b(?:works|working)\s+(?:now|correctly)\b|\bdone\s+and\s+(?:tested|verified)\b|"
    r"\bbestanden\b|\bgepr(?:u|ue)ft\b|\bverifiziert\b|"  # i18n-allow: German output matcher
    r"\balles\s+gr(?:u|ue)n\b|\berfolgreich\s+getestet\b|"  # i18n-allow: German output matcher
    r"\btests?\s+(?:laufen\s+)?(?:gr(?:u|ue)n|durch)\b|"  # i18n-allow: German output matcher
    r"\bfunktioniert\s+(?:jetzt|korrekt)\b|"  # i18n-allow: German output matcher
    r"\baprobado\b|\bverificado\b|\bcomprobado\b|"  # i18n-allow: Spanish output matcher
    r"\btodas\s+las\s+pruebas\s+pasan\b|"  # i18n-allow: Spanish output matcher
    r"\bverificado\s+e\s+aprovado\b|\btestes?\s+passaram\b"  # i18n-allow: Portuguese output
    r")",
    re.IGNORECASE,
)

#: A clause that explicitly withholds acceptance; it beats the words above.
_DISCLAIMED_RE = re.compile(
    r"(?:"
    r"\bunverified\b|\bnot\s+verified\b|\bnot\s+yet\s+verified\b|\bcannot\s+confirm\b|"
    r"\bno\s+independent\b|\bblocked\b|"
    r"\bnicht\s+(?:gepr(?:u|ue)ft|verifiziert|best(?:a|ae)tigt)\b|"  # i18n-allow: DE output
    r"\bungepr(?:u|ue)ft\b|\bnoch\s+offen\b|"  # i18n-allow: DE output
    r"\bsin\s+verificar\b|\bno\s+verificado\b|"  # i18n-allow: ES output
    r"\bn(?:a|ã)o\s+verificado\b"  # i18n-allow: PT output
    r")",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Casefolded and accent-stripped, so one pattern matches both spellings.

    The same fold :mod:`jarvis.brain.action_honesty` applies, for the same
    reason: a model writes the accents, a regex should not have to spell every
    variant of them.
    """
    folded = unicodedata.normalize("NFKD", str(text or "").casefold())
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def claims_acceptance(text: str) -> bool:
    """Whether ``text`` tells the person the delegated work is good."""
    body = _normalize(text)
    if not body.strip():
        return False
    if _DISCLAIMED_RE.search(body):
        # The answer names its own limit; that is the honest shape already.
        return False
    return bool(_ACCEPTANCE_RE.search(body))


# --------------------------------------------------------------- the evidence

_GIT_READ = ("diff", "status", "show", "log", "rev-parse", "describe", "blame")
_TEST_RUNNERS = (
    "pytest",
    "unittest",
    "tox",
    "nox",
    "npm test",
    "npm run test",
    "yarn test",
    "pnpm test",
    "vitest",
    "jest",
    "cargo test",
    "go test",
    "dotnet test",
    "mvn test",
    "gradle test",
    "rspec",
    "phpunit",
    "ctest",
)
#: Tools that read version-control state as their whole purpose.
_GIT_TOOLS = frozenset({"workspace-git"})


def _command_of(args: Any) -> str:
    if not isinstance(args, dict):
        return ""
    for key in ("command", "cmd", "args", "arguments"):
        value = args.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return " ".join(str(part) for part in value)
    return ""


def reads_version_control(tool_name: str, args: Any) -> bool:
    """A call that shows what the delegated work actually did to the checkout."""
    if tool_name in _GIT_TOOLS:
        return True
    command = _command_of(args).strip().lower()
    if not command:
        return False
    # Only a git invocation, and only one of its reading subcommands: a commit
    # or a checkout is a change, not a look at one.
    if not re.match(r"^(?:[\"']?[^\s\"']*\bgit\b[\"']?)(?:\s|$)", command):
        return False
    tail = command.split("git", 1)[1]
    return any(re.search(rf"\b{sub}\b", tail) for sub in _GIT_READ)


def runs_tests(tool_name: str, args: Any) -> bool:
    """A call that actually executes a test suite."""
    command = _command_of(args).strip().lower()
    if not command:
        return False
    return any(runner in command for runner in _TEST_RUNNERS)


@dataclass
class Evidence:
    """What the turn has really been observed to do, by tool record only."""

    version_control: bool = False
    tests: bool = False
    #: Tool names seen, for the log line that explains a refusal.
    seen: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.version_control and self.tests

    def missing(self) -> tuple[str, ...]:
        out: list[str] = []
        if not self.version_control:
            out.append("git")
        if not self.tests:
            out.append("tests")
        return tuple(out)

    def note(self, tool_name: str, args: Any) -> None:
        """Record one SUCCESSFUL tool call."""
        name = str(tool_name or "")
        self.seen.append(name)
        if reads_version_control(name, args):
            self.version_control = True
        if runs_tests(name, args):
            self.tests = True


class EvidenceWatch:
    """Collects the turn's evidence from the executor's own bus records.

    Paired like the timeline's step rows: ``ActionProposed`` carries the call's
    arguments, ``ActionExecuted`` says whether it worked, and only a successful
    pair counts. Defensive throughout — a malformed event costs the gate
    nothing and must never reach the brain (AP-18).
    """

    __slots__ = ("_bus", "_open", "_trace_id", "evidence")

    def __init__(self, bus: Any | None, trace_id: UUID) -> None:
        self._bus = bus
        self._trace_id = trace_id
        self._open: list[tuple[str, dict[str, Any]]] = []
        self.evidence = Evidence()

    def start(self) -> None:
        if self._bus is not None and hasattr(self._bus, "subscribe_all"):
            self._bus.subscribe_all(self._on_event)

    def stop(self) -> None:
        if self._bus is not None and hasattr(self._bus, "unsubscribe_all"):
            try:
                self._bus.unsubscribe_all(self._on_event)
            except Exception:  # noqa: BLE001 — detaching must never raise
                log.debug("acceptance gate: could not detach the evidence watch", exc_info=True)

    async def _on_event(self, event: Any) -> None:
        try:
            self._translate(event)
        except Exception:  # noqa: BLE001 — AP-18: never leave a subscriber
            log.debug("acceptance gate: evidence watch skipped an event", exc_info=True)

    def _translate(self, event: Any) -> None:
        name = type(event).__name__
        if getattr(event, "trace_id", None) != self._trace_id:
            return
        if name == "ActionProposed":
            args = getattr(event, "args", None)
            self._open.append(
                (str(getattr(event, "tool_name", "") or ""), args if isinstance(args, dict) else {})
            )
        elif name == "ActionExecuted":
            tool_name = str(getattr(event, "tool_name", "") or "")
            for index, (open_name, args) in enumerate(self._open):
                if open_name == tool_name:
                    self._open.pop(index)
                    if bool(getattr(event, "success", False)):
                        self.evidence.note(tool_name, args)
                    return


# ------------------------------------------------------------------- the gate


@dataclass(frozen=True)
class AcceptanceGate:
    """Armed for the wave that answers an unverified coding completion.

    ``work`` names the handback rows that need acceptance, for the log and for
    a caller that wants to say which assignment is still open.
    """

    work: tuple[str, ...]

    def verdict(self, text: str, evidence: Evidence, *, language: str = "en") -> str | None:
        """The rewritten answer, or ``None`` when the answer may stand."""
        if evidence.complete or not claims_acceptance(text):
            return None
        return unverified_answer(text, missing=evidence.missing(), language=language)


def gate_for(rows: list[dict[str, Any]]) -> AcceptanceGate | None:
    """The gate a wave of delegation results needs, or ``None``.

    Only a coding completion that says of itself that it was not independently
    verified arms one. A failure, a question or a stop needs no gate: those
    rows carry no acceptance to guard.
    """
    work = tuple(
        str(row.get("task_id") or "")
        for row in rows
        if str(row.get("status") or "") == COMPLETED
        and NOT_INDEPENDENT in str(row.get("evidence") or "")
    )
    return AcceptanceGate(work=work) if work else None


# Each line names its own limit with the words ``claims_acceptance`` reads as
# a disclaimer, so the honest answer can never be mistaken for a verdict.
_UNVERIFIED: dict[str, str] = {
    "en": (
        "The coding agent reported its work as finished, so this is unverified: I have "
        "not read the diff{tests}. Say the word and I will look at the change and run "
        "the tests."
    ),
    # i18n-allow: runtime chat output
    "de": (
        "Der Coding-Agent meldet seine Arbeit als fertig, "  # i18n-allow: DE chat output
        "das ist aber nicht geprüft: "  # i18n-allow: DE chat output
        "ich habe den Diff nicht gelesen{tests}. "  # i18n-allow: DE chat output
        "Sag ein Wort, dann sehe ich mir die "  # i18n-allow: DE chat output
        "Änderung an und lasse die Tests laufen."  # i18n-allow: DE chat output
    ),
    # i18n-allow: runtime chat output
    "es": (
        "El agente de código dice que terminó, pero esto queda sin verificar: no he "
        "leído el diff{tests}. Dímelo y reviso el cambio y ejecuto las pruebas."
    ),
    # i18n-allow: runtime chat output
    "pt": (
        "O agente de código diz que terminou, mas isto fica não verificado: não li o "
        "diff{tests}. Diz uma palavra e eu vejo a alteração e corro os testes."
    ),
}
_AND_TESTS: dict[str, str] = {
    "en": " and have not run the tests",
    "de": " und die Tests nicht ausgeführt",  # i18n-allow: runtime chat output
    "es": " ni he ejecutado las pruebas",  # i18n-allow: runtime chat output
    "pt": " nem corri os testes",  # i18n-allow: runtime chat output
}


def unverified_answer(
    text: str, *, missing: tuple[str, ...] = (), language: str = "en"
) -> str:
    """The honest answer that replaces an unearned acceptance claim."""
    lang = language if language in _UNVERIFIED else "en"
    tests = _AND_TESTS[lang] if "tests" in missing or not missing else ""
    return _UNVERIFIED[lang].format(tests=tests)


__all__ = [
    "COMPLETED",
    "NOT_INDEPENDENT",
    "AcceptanceGate",
    "Evidence",
    "EvidenceWatch",
    "claims_acceptance",
    "gate_for",
    "reads_version_control",
    "runs_tests",
    "unverified_answer",
]
