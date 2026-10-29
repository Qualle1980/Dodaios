"""A coding agent's "done" is delivery. Acceptance has to be earned, per job.

:mod:`jarvis.agentic_ide.delegation_wait` already says what a CLI completion is
worth, in the only place that can know: it stamps every one of them
``evidence="Recorded CLI completion and matching report; not independent
verification"``. That row travels back into the original chat turn, where
:func:`jarvis.agent_chat.delegation_wait.continuation_prompt` *asks* the chief
to read the diff and run the tests. An instruction is not a guarantee: an
answer of "PASS, all tests pass" from a model that never looked is, to the
person reading the chat, indistinguishable from a verified one.

So the handback's own honesty becomes a condition. A wave that answers an
unverified coding job runs with an :class:`AcceptanceGate`, and the gate asks
for evidence **bound to that job**, not for two true booleans:

* a **change read** — ``workspace-git`` naming that job's workspace, or a
  ``git diff`` / ``git show`` run inside its checkout — that actually returned
  something;
* a **test run** — a real runner as the command's own program, not a word
  inside it, without ``--collect-only`` and friends, inside that checkout, and
  reporting a pass;
* both **after** the job was handed back, so a suite that ran before the
  change cannot certify it;
* and, where the command names a revision, that revision must be the one the
  job finished on.

An acceptance claim is read clause by clause. A disclaimer in one sentence
must not license a verdict in the next: ``"Unverified: not yet verified. PASS,
all tests pass."`` is a claim, and it used to pass this gate untouched.

What the gate cannot do is grant anything. It approves nothing, runs nothing,
sends nothing and retries nothing. Where the evidence is missing — including
when its own recording broke — the answer becomes an honest one that NAMES the
gap, and the delegated work, the pane and the report are left exactly as they
are.
"""

from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: The marker ``jarvis/agentic_ide/delegation_wait.py`` stamps on a completion
#: whose only evidence is the CLI's own end-of-turn record.
NOT_INDEPENDENT = "not independent verification"
#: The work-key prefix that module registers.
CODING_PREFIX = "coding:"

#: Why a claim was refused. Named, because "unverified" alone tells the person
#: nothing about what is still missing.
DIFF_MISSING = "diff_missing"
TEST_MISSING = "test_missing"
WRONG_WORKSPACE = "wrong_workspace"
STALE_EVIDENCE = "stale_evidence"
OBSERVER_ERROR = "observer_error"


# ------------------------------------------------------------------ the claim


def _normalize(text: str) -> str:
    """Casefolded and accent-stripped, so one pattern matches both spellings.

    The same fold :mod:`jarvis.brain.action_honesty` applies, for the same
    reason: a model writes the accents, a regex should not have to spell every
    variant of them.
    """
    folded = unicodedata.normalize("NFKD", str(text or "").casefold())
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


# A verdict word, or a sentence that the work is finished and sound. Kept to
# statements ABOUT the outcome: a report that stays descriptive ("T1 says it
# added subtract()") is not a verdict and must pass through untouched.
_ACCEPTANCE_RE = re.compile(
    r"(?:"
    r"\bpass(?:ed|es)?\b|\bgreen\b|\bverified\b|\bconfirmed\b|"
    r"\ball\s+tests?\s+(?:pass|passed|are\s+green)\b|"
    r"\b(?:works|working)\s+(?:now|correctly|fine)\b|\beverything\s+works\b|"
    r"\bdone\s+and\s+(?:tested|verified)\b|"
    r"\b(?:complete|finished)\s+and\s+(?:correct|working|tested)\b|"
    r"\bno\s+(?:errors|issues)\b|"
    r"\bbestanden\b|\bgepr(?:u|ue)ft\b|\bverifiziert\b|"  # i18n-allow: DE matcher
    r"\balles\s+gr(?:u|ue)n\b|\berfolgreich\s+getestet\b|"  # i18n-allow: DE matcher
    r"\btests?\s+(?:laufen\s+)?(?:gr(?:u|ue)n|durch)\b|"  # i18n-allow: DE matcher
    r"\bfunktioniert\s+(?:jetzt|korrekt|einwandfrei)\b|"  # i18n-allow: DE matcher
    r"\bl(?:a|ae)uft\s+(?:jetzt\s+)?(?:einwandfrei|fehlerfrei|durch)\b|"  # i18n-allow: DE
    r"\bfehlerfrei\b|\bohne\s+fehler\b|"  # i18n-allow: DE matcher
    r"\b(?:fertig|erledigt)\s+und\s+\w+|"  # i18n-allow: DE matcher
    r"\baprobado\b|\bverificado\b|\bcomprobado\b|"  # i18n-allow: ES matcher
    r"\btodas\s+las\s+pruebas\s+pasan\b|"  # i18n-allow: ES matcher
    r"\bfunciona\s+(?:ahora|correctamente)\b|"  # i18n-allow: ES matcher
    r"\btestes?\s+passaram\b|\bfunciona\s+agora\b"  # i18n-allow: PT matcher
    r")"
)

#: A clause that withholds acceptance. It only ever disclaims ITS OWN clause.
_DISCLAIMED_RE = re.compile(
    r"(?:"
    r"\bunverified\b|\bnot\s+verified\b|\bnot\s+yet\s+verified\b|\bcannot\s+confirm\b|"
    r"\bno\s+independent\b|\bblocked\b|\bnot\s+confirmed\b|"
    r"\bnothing\s+(?:is\s+|was\s+|has\s+been\s+)?"
    r"(?:checked|verified|confirmed|tested|proven)\b|"
    r"\bno\s+(?:test|tests|diff|evidence|proof)\b|"
    r"\bnichts\s+(?:ist\s+|wurde\s+)?"  # i18n-allow: DE matcher
    r"(?:gepr(?:u|ue)ft|verifiziert|best(?:a|ae)tigt|getestet|belegt)\b|"  # i18n-allow: DE
    r"\bnada\s+(?:esta\s+)?(?:verificado|comprobado)\b|"  # i18n-allow: ES matcher
    r"\bhave\s+not\s+(?:read|run|checked|verified)\b|"
    r"\bhaven'?t\s+(?:read|run|checked|verified)\b|"
    r"\bnicht\s+(?:gepr(?:u|ue)ft|verifiziert|best(?:a|ae)tigt|gelesen|"  # i18n-allow: DE
    r"ausgef(?:u|ue)hrt)\b|"  # i18n-allow: DE matcher
    r"\bungepr(?:u|ue)ft\b|\bnoch\s+offen\b|\bkeine\s+pr(?:u|ue)fung\b|"  # i18n-allow: DE
    r"\bsin\s+verificar\b|\bno\s+verificado\b|\bno\s+he\s+leido\b|"  # i18n-allow: ES
    r"\bno\s+lo\s+he\s+verificado\b|"  # i18n-allow: ES matcher
    r"\bnao\s+verificado\b|\bnao\s+o\s+verifiquei\b|\bnao\s+li\b"  # i18n-allow: PT
    r")"
)

#: Clause boundaries: sentence punctuation, a colon, a semicolon, a newline,
#: or a spaced dash. A decimal point or a clock colon between digits is not one.
_CLAUSE_SPLIT_RE = re.compile(
    r"(?:[!?…]|(?<!\d)[.:](?!\d)|[;\n\r]|\s[-–—]\s)+"
)


def claims_acceptance(text: str) -> bool:
    """Whether ``text`` tells the person the delegated work is good.

    Judged per clause. A model that opens with "this is unverified" and then
    writes "PASS, all tests pass" has made the claim; reading the disclaimer
    as cover for the whole answer was exactly the hole the review found.
    """
    body = _normalize(text)
    if not body.strip():
        return False
    for clause in _CLAUSE_SPLIT_RE.split(body):
        if not clause.strip():
            continue
        if _DISCLAIMED_RE.search(clause):
            continue
        if _ACCEPTANCE_RE.search(clause):
            return True
    return False


# --------------------------------------------------------------- the receipts


@dataclass(frozen=True)
class Receipt:
    """One tool call that really ran, as the gate needs to judge it."""

    tool: str
    args: Mapping[str, Any]
    success: bool
    output: str = ""
    #: Where the call ran. For the folder tools this is the chat's own folder.
    cwd: str = ""
    at: float = field(default_factory=time.time)


@dataclass
class EvidenceLedger:
    """Every receipt of one turn, plus whether the recording itself held.

    ``observer_error`` is load-bearing: a gate that cannot see what happened
    must refuse the claim rather than wave it through. Since the recorder never
    changes a tool's own behaviour, this can only ever withhold an acceptance.
    """

    receipts: list[Receipt] = field(default_factory=list)
    observer_error: bool = False

    def note(
        self,
        tool: str,
        args: Any,
        *,
        success: bool,
        output: Any = "",
        cwd: str = "",
    ) -> None:
        try:
            self.receipts.append(
                Receipt(
                    tool=str(tool or ""),
                    args=dict(args) if isinstance(args, Mapping) else {},
                    success=bool(success),
                    output=str(output or ""),
                    cwd=str(cwd or ""),
                )
            )
        except Exception:  # noqa: BLE001 - a lost receipt has to fail closed
            self.observer_error = True
            log.warning("acceptance gate: a tool receipt could not be recorded", exc_info=True)


class RecordingTool:
    """A tool that reports what it did into a ledger, and changes nothing else.

    Wrapping happens through the turn's own ``tool_filter``, which is applied
    to the merged tool surface, so the folder tools and the workspace tools are
    recorded the same way. Everything but ``execute`` is forwarded untouched --
    the risk tier, the read-only capability and the approval summary stay the
    wrapped tool's own, because the executor has to make exactly the decision
    it would have made without this.
    """

    def __init__(self, tool: Any, ledger: EvidenceLedger, *, cwd: str = "") -> None:
        self._tool = tool
        self._ledger = ledger
        self._cwd = cwd

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RecordingTool({self._tool!r})"

    async def execute(self, args: Any, ctx: Any) -> Any:
        result = await self._tool.execute(args, ctx)
        try:
            self._ledger.note(
                getattr(self._tool, "name", ""),
                args,
                success=bool(getattr(result, "success", False)),
                output=getattr(result, "output", "") or getattr(result, "error", "") or "",
                cwd=str(getattr(self._tool, "cwd", "") or self._cwd),
            )
        except Exception:  # noqa: BLE001 - the tool's own result is never affected
            self._ledger.observer_error = True
            log.warning("acceptance gate: recording a tool call failed", exc_info=True)
        return result


def recording_filter(ledger: EvidenceLedger, *, cwd: str = ""):
    """A ``TurnOverride.tool_filter`` that records every call of the turn."""

    def _wrap(tools: dict[str, Any]) -> dict[str, Any]:
        try:
            return {name: RecordingTool(tool, ledger, cwd=cwd) for name, tool in tools.items()}
        except Exception:  # noqa: BLE001 - never lose the turn's hands over this
            ledger.observer_error = True
            log.warning("acceptance gate: the turn's tools could not be recorded", exc_info=True)
            return tools

    return _wrap


# ------------------------------------------------------------- the predicates

_GIT_READS = ("diff", "show")
#: ``workspace-git`` operations that report a workspace's own changes.
_GIT_TOOL_READS = frozenset({"state"})
_GIT_TOOL = "workspace-git"

_RUNNERS = frozenset(
    {
        "pytest", "py.test", "tox", "nox", "vitest", "jest", "rspec", "phpunit",
        "ctest", "unittest", "behave", "mocha",
    }
)
#: Programs that only run a suite with the right subcommand after them.
_RUNNER_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "cargo": frozenset({"test", "nextest"}),
    "go": frozenset({"test"}),
    "dotnet": frozenset({"test"}),
    "mvn": frozenset({"test", "verify"}),
    "gradle": frozenset({"test", "check"}),
    "gradlew": frozenset({"test", "check"}),
    "npm": frozenset({"test", "t"}),
    "yarn": frozenset({"test"}),
    "pnpm": frozenset({"test"}),
    "npx": frozenset({"vitest", "jest", "mocha"}),
    "bun": frozenset({"test"}),
}
#: A run that deliberately executes nothing.
_NO_RUN_FLAGS = (
    "--collect-only", "--co", "--help", "-h", "--version", "--dry-run",
    "--fixtures", "--markers", "--list", "--list-tests", "--setup-only",
)
#: Something in the output has to say a suite actually passed.
_PASSED_RE = re.compile(
    r"(?:\b\d+\s+passed\b|\ball\s+tests?\s+passed\b|\btests?\s+passed\b|"
    r"\btest\s+run\s+successful\b|\bbuild\s+success\b|\b0\s+failed\b|"
    r"^ok\b|\bok\s*$|^PASS\b)",
    re.IGNORECASE | re.MULTILINE,
)
#: The shell marker the chat's RunCommand appends; not output of its own.
_EXIT_MARKER_RE = re.compile(r"\[exit\s+-?\d+\b[^\]]*\]")
_SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b")


def _command_of(args: Any) -> str:
    if not isinstance(args, Mapping):
        return ""
    for key in ("command", "cmd"):
        value = args.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return " ".join(str(part) for part in value)
    return ""


def _tokens(command: str) -> list[str]:
    """The command's words, with quotes stripped -- enough to read its argv0."""
    return [token.strip("'\"") for token in str(command or "").split() if token.strip("'\"")]


def _program(token: str) -> str:
    """The bare program name of a token, without its path or extension."""
    tail = re.split(r"[\\/]", token)[-1].casefold()
    return tail[:-4] if tail.endswith(".exe") else tail


def reads_change(tool_name: str, args: Any) -> bool:
    """A call that shows what the delegated work did to the checkout.

    ``git status`` and ``git rev-parse`` are deliberately NOT enough: they say
    a checkout exists and where it stands, never what this work changed.
    """
    name = str(tool_name or "")
    if name == _GIT_TOOL:
        op = str(args.get("op") or "") if isinstance(args, Mapping) else ""
        return op in _GIT_TOOL_READS
    tokens = _tokens(_command_of(args))
    if not tokens or _program(tokens[0]) != "git":
        return False
    rest = tokens[1:]
    while rest and rest[0].startswith("-"):
        # Skip git's own options, including "-C <path>" and "--no-pager".
        flag = rest.pop(0)
        if flag in {"-C", "-c", "--git-dir", "--work-tree"} and rest:
            rest.pop(0)
    return bool(rest) and rest[0] in _GIT_READS


def runs_tests(tool_name: str, args: Any) -> bool:
    """A call that actually executes a test suite.

    The program has to BE a runner, not merely be mentioned: ``echo pytest``
    prints a word and exits 0, and it used to satisfy this.
    """
    tokens = _tokens(_command_of(args))
    if not tokens:
        return False
    if any(flag in tokens for flag in _NO_RUN_FLAGS):
        return False
    program = _program(tokens[0])
    rest = tokens[1:]
    if program in {"python", "python3", "py", "uv", "poetry", "pdm", "hatch", "pipenv"}:
        # "python -m pytest", "poetry run pytest", "uv run pytest ..."
        while rest and (rest[0].startswith("-") or rest[0] in {"run", "exec"}):
            rest.pop(0)
        return bool(rest) and _program(rest[0]) in _RUNNERS
    if program in _RUNNER_SUBCOMMANDS:
        while rest and rest[0].startswith("-"):
            rest.pop(0)
        if rest and rest[0] == "run":
            rest.pop(0)
        return bool(rest) and rest[0].casefold() in _RUNNER_SUBCOMMANDS[program]
    return program in _RUNNERS


def _said_something(output: str) -> bool:
    """Output that is more than the shell's own exit marker."""
    return bool(_EXIT_MARKER_RE.sub("", str(output or "")).strip())


def reports_a_pass(output: str) -> bool:
    """Whether a test run's own output says a suite passed."""
    return bool(_PASSED_RE.search(str(output or "")))


def _inside(path: str, root: str) -> bool:
    """Whether ``path`` is ``root`` or below it, case-folded as the OS does."""
    if not path or not root:
        return False
    try:
        here = os.path.normcase(os.path.abspath(str(path)))
        there = os.path.normcase(os.path.abspath(str(root)))
    except Exception:  # noqa: BLE001 - an unusable path is simply not inside
        return False
    return here == there or here.startswith(there + os.sep)


# -------------------------------------------------------------------- the job


@dataclass(frozen=True)
class DelegatedJob:
    """One coding handback that still needs the chief's own verification."""

    task_id: str
    status: str = ""
    project_id: str = ""
    workspace_id: str = ""
    #: The workspace's folder on disk; evidence has to come from inside it.
    root: str = ""
    #: The revision the job finished on, when the handback could read one.
    head_sha: str = ""
    #: When it was handed back. Evidence older than this proves nothing.
    at: float = 0.0

    def addresses_me(self, receipt: Receipt) -> bool:
        """Whether this receipt was taken in this job's own workspace."""
        if receipt.tool == _GIT_TOOL:
            named = str(receipt.args.get("workspace_id") or "")
            # An omitted workspace means "the visible one", which is not
            # provably this job's -- which is the whole point of the check.
            return bool(named) and named == self.workspace_id
        return _inside(receipt.cwd, self.root)

    def right_revision(self, receipt: Receipt) -> bool:
        """A command naming a revision must name the one this job ended on."""
        if not self.head_sha:
            return True
        named = _SHA_RE.findall(_command_of(receipt.args).casefold())
        return all(self.head_sha.casefold().startswith(token) for token in named)

    def evidence_gaps(self, ledger: EvidenceLedger) -> tuple[str, ...]:
        """What is still missing before this job may be called good."""
        if ledger.observer_error:
            return (OBSERVER_ERROR,)
        change = test = False
        off_workspace = stale = False
        for receipt in ledger.receipts:
            is_change = reads_change(receipt.tool, receipt.args)
            is_test = runs_tests(receipt.tool, receipt.args)
            if not (is_change or is_test) or not receipt.success:
                continue
            if not self.addresses_me(receipt) or not self.right_revision(receipt):
                off_workspace = True
                continue
            if self.at and receipt.at < self.at:
                stale = True
                continue
            if is_change and (receipt.tool == _GIT_TOOL or _said_something(receipt.output)):
                change = True
            if is_test and reports_a_pass(receipt.output):
                test = True
        gaps: list[str] = []
        if not change:
            gaps.append(DIFF_MISSING)
        if not test:
            gaps.append(TEST_MISSING)
        if gaps and off_workspace:
            gaps.append(WRONG_WORKSPACE)
        if gaps and stale:
            gaps.append(STALE_EVIDENCE)
        return tuple(gaps)


# ------------------------------------------------------------------- the gate


@dataclass(frozen=True)
class AcceptanceGate:
    """Armed for the wave that answers coding work nobody has verified."""

    jobs: tuple[DelegatedJob, ...]

    @property
    def work(self) -> tuple[str, ...]:
        return tuple(job.task_id for job in self.jobs)

    def gaps(self, ledger: EvidenceLedger) -> tuple[str, ...]:
        """Every job's missing evidence, in a stable order."""
        seen: list[str] = []
        for job in self.jobs:
            for gap in job.evidence_gaps(ledger):
                if gap not in seen:
                    seen.append(gap)
        return tuple(seen)

    def verdict(self, text: str, ledger: EvidenceLedger, *, language: str = "en") -> str | None:
        """The rewritten answer, or ``None`` when the answer may stand."""
        if not claims_acceptance(text):
            return None
        gaps = self.gaps(ledger)
        if not gaps:
            return None
        log.warning(
            "Delegated coding work %s was called good with missing evidence (%s); "
            "answering honestly instead.",
            list(self.work), ", ".join(gaps),
        )
        return unverified_answer(gaps, language=language)


def gate_for(rows: list[dict[str, Any]]) -> AcceptanceGate | None:
    """The gate a wave of delegation results needs, or ``None``.

    EVERY coding handback arms a job, whatever it says about itself. Not only
    the ones that say ``completed`` -- a crashed, blocked or superseded agent
    is further from acceptance, not closer to it. And not only the ones that
    carry the ``not independent verification`` marker: deciding on the absence
    of a prose string would mean a row that says nothing at all counts as
    verified, which is exactly the wrong default. The one way past this gate
    is a handback that explicitly reports ``verified`` -- a claim no CLI
    completion makes today, and which has to be earned where it is set.
    """
    jobs: list[DelegatedJob] = []
    for row in rows or ():
        task_id = str(row.get("task_id") or "")
        if not task_id.startswith(CODING_PREFIX):
            continue
        if row.get("verified") is True:
            continue
        target = row.get("target")
        target = target if isinstance(target, Mapping) else {}
        jobs.append(
            DelegatedJob(
                task_id=task_id,
                status=str(row.get("status") or ""),
                project_id=str(target.get("project_id") or ""),
                workspace_id=str(target.get("workspace_id") or ""),
                root=str(target.get("root") or ""),
                head_sha=str(target.get("head_sha") or ""),
                at=float(target.get("at") or 0.0),
            )
        )
    return AcceptanceGate(jobs=tuple(jobs)) if jobs else None


# ------------------------------------------------------------ the honest line

_OPENING: dict[str, str] = {
    "en": "The coding agent reported its work as finished; I have not verified it: ",
    # i18n-allow: runtime chat output
    "de": (
        "Der Coding-Agent meldet seine Arbeit als fertig, "  # i18n-allow: DE chat output
        "ich habe sie aber nicht geprüft: "  # i18n-allow: DE chat output
    ),
    # i18n-allow: runtime chat output
    "es": "El agente de código dice que terminó, pero no lo he verificado: ",
    # i18n-allow: runtime chat output
    "pt": "O agente de código diz que terminou, mas não o verifiquei: ",
}
_GAP_TEXT: dict[str, dict[str, str]] = {
    "en": {
        DIFF_MISSING: "the change itself is unread",
        TEST_MISSING: "no test run of mine has succeeded",
        WRONG_WORKSPACE: "what I did read was not this project's checkout",
        STALE_EVIDENCE: "the evidence I have predates the change",
        OBSERVER_ERROR: "I could not record what ran, so nothing is established",
    },
    # i18n-allow: runtime chat output
    "de": {
        DIFF_MISSING: "die Änderung selbst ist ungelesen",  # i18n-allow: DE chat output
        TEST_MISSING: "kein Testlauf von mir war erfolgreich",  # i18n-allow: DE output
        WRONG_WORKSPACE: "nicht das Checkout dieses Projekts",  # i18n-allow: DE output
        STALE_EVIDENCE: "meine Belege sind älter als die Änderung",  # i18n-allow: DE chat output
        OBSERVER_ERROR: "ich konnte nicht mitschreiben, was lief",  # i18n-allow: DE output
    },
    # i18n-allow: runtime chat output
    "es": {
        DIFF_MISSING: "el cambio en sí sigue sin leer",
        TEST_MISSING: "ninguna ejecución mía de pruebas indica éxito",
        WRONG_WORKSPACE: "lo que leí no era el checkout de este proyecto",
        STALE_EVIDENCE: "mis pruebas son anteriores al cambio",
        OBSERVER_ERROR: "no pude registrar qué se ejecutó, así que nada queda probado",
    },
    # i18n-allow: runtime chat output
    "pt": {
        DIFF_MISSING: "a alteração em si continua por ler",
        TEST_MISSING: "nenhuma execução minha de testes indica sucesso",
        WRONG_WORKSPACE: "o que li não era o checkout deste projeto",
        STALE_EVIDENCE: "as minhas provas são anteriores à alteração",
        OBSERVER_ERROR: "não consegui registar o que correu, logo nada está provado",
    },
}
_CLOSING: dict[str, str] = {
    "en": " I am carrying on with that check.",
    "de": " Ich prüfe das weiter.",  # i18n-allow: runtime chat output
    "es": " Sigo con esa comprobación.",  # i18n-allow: runtime chat output
    "pt": " Vou continuar essa verificação.",  # i18n-allow: runtime chat output
}
_JOIN: dict[str, str] = {
    "en": " and ",
    "de": " und ",  # i18n-allow: runtime chat output
    "es": " y ",  # i18n-allow: runtime chat output
    "pt": " e ",  # i18n-allow: runtime chat output
}


def unverified_answer(gaps: tuple[str, ...] = (), *, language: str = "en") -> str:
    """The honest answer that replaces an unearned acceptance claim.

    It names what is actually missing, and it does NOT ask the person to
    authorize the check again -- they already asked for the work, and the
    check is part of it.
    """
    lang = language if language in _OPENING else "en"
    table = _GAP_TEXT[lang]
    parts = [table[gap] for gap in gaps if gap in table] or [table[DIFF_MISSING]]
    return _OPENING[lang] + _JOIN[lang].join(parts) + "." + _CLOSING[lang]


__all__ = [
    "CODING_PREFIX",
    "DIFF_MISSING",
    "NOT_INDEPENDENT",
    "OBSERVER_ERROR",
    "STALE_EVIDENCE",
    "TEST_MISSING",
    "WRONG_WORKSPACE",
    "AcceptanceGate",
    "DelegatedJob",
    "EvidenceLedger",
    "Receipt",
    "RecordingTool",
    "claims_acceptance",
    "gate_for",
    "reads_change",
    "recording_filter",
    "reports_a_pass",
    "runs_tests",
    "unverified_answer",
]
