"""Worker-prompt composition + the artifact-language directive.

This is the single chokepoint that guarantees every dispatched worker is told
to produce **English code artifacts by default**, regardless of the language the
user spoke in.

Why a dedicated module
----------------------
The mission-dispatch chain is mostly German-flavoured: the user speaks German,
``spawn_worker._build_mission_prompt`` wraps the request with German labels
("Aufgabe", "Wortlaut des Nutzers"), and for short missions the verbatim German
utterance is handed to the worker unchanged (the decomposer's heuristic
single-step path never rewrites it). With no instruction to the contrary, the
worker mirrors the request language and writes German identifiers, comments and
docstrings — a direct violation of the repo Output Language Policy ("every
artifact an agent produces is English").

The fix is one directive, prepended to the worker prompt at the one place every
worker prompt is built (``orchestrator._run_iterations``), so it covers all
decomposition paths (heuristic single-step, LLM multi-step), all worker CLIs
(Claude / Codex / Gemini), every critic iteration, and any future spawn entry
point — provider-agnostically.

The directive is deliberately a *default*, not an absolute: it governs the
*medium* (code), never the *subject matter*. A deliverable whose content is
meant to be German (a German news page, quoted text, localized copy) stays
German; only the surrounding code artifacts default to English.
"""
from __future__ import annotations

from typing import Final

from jarvis.core.secret_scrub import scrub_secrets

#: First line of a worker prompt whose request is a read order. The codex
#: worker turns it into ``--sandbox read-only``; the model reads it as the order.
READ_ONLY_MARKER: Final[str] = "READ-ONLY TASK:"

# Prepended to every worker prompt. Kept provider-agnostic and language-neutral:
# it must hold no matter what language the request arrived in (the leak was
# "German request -> German code"), so it can never be a de/en toggle.
ARTIFACT_LANGUAGE_DIRECTIVE: Final[str] = (
    "ARTIFACT LANGUAGE - DEFAULT ENGLISH (this is an international repository). "
    "Write every CODE ARTIFACT in English: identifiers, function/class/variable "
    "names, comments, docstrings, log and error messages, commit messages, test "
    "names, and technical documentation. This holds even when this task "
    "description, the user's words, or the surrounding conversation are in "
    "another language - do NOT mirror the request language into the code. "
    "EXCEPTION: keep another language only where that language is the actual "
    "SUBJECT MATTER rather than the medium - user-facing copy / UI text or "
    "document content in the language the user asked for, quoted or localized "
    "text, and test fixtures that assert on non-English input. When the user "
    "EXPLICITLY requests a specific language for the artifact itself, honor that "
    "request."
)


# Prepended to every worker prompt, right after the language directive. Root
# cause (2026-06-22, mission 019ef052): the user asked for a SINGLE HTML file;
# the worker shipped four (index.html + app.js + styles.css + assets/). The
# standing _QUALITY_DIRECTIVE ("never downgrade to a minimal version, treat a
# skeleton hint as a floor not a ceiling") reads a single self-contained file as
# a forbidden minimal version, so the worker built the "production-quality" norm
# of a split multi-file project — overriding the user's explicit form constraint.
# Nothing in the chain told the worker that honoring a requested SHAPE is part of
# the job. This directive is that missing layer. Kept ASCII / provider- and
# language-agnostic: a form constraint must be honored no matter the request
# language or which worker CLI runs.
OUTPUT_SHAPE_DIRECTIVE: Final[str] = (
    "OUTPUT SHAPE - HONOR EXPLICIT FORM CONSTRAINTS. When the request fixes the "
    "SHAPE or packaging of the deliverable - a single self-contained file, one "
    "HTML file with its CSS and JS inlined, exactly one script, a specific file "
    "count, or a named output format - treat that as a HARD requirement and "
    "deliver in EXACTLY that form, even when it means inlining or consolidating "
    "what you would otherwise split across several files. Honoring a constraint "
    "the user set is part of satisfying the request; it is NEVER a downgrade, a "
    "stub, or a minimal version. When the request fixes no shape, use your best "
    "professional judgment for how to package the result."
)


# Prepended to every worker prompt. AGENTS.md section 0 ("Look before you
# build") is the repo's first rule for every coding agent, and until now no code
# path put it in front of a worker: it lived in a file the worker may never
# open. The cost is measured twice. On 2026-11-07 an agent began rebuilding a
# memory-vault subsystem by hand that already sat complete in the upstream tree
# (ADR-0044). On 2026-11-09 three tools were declared to a live call with an
# empty schema; the model sent empty arguments to one of them 177 times in a
# single turn and the app was unusable. That exact failure shape had been
# recorded two days earlier and the note was never opened. Both were search
# failures, not reasoning failures. Kept ASCII / provider- and
# language-agnostic: the habit must hold whichever worker CLI runs.
LOOK_BEFORE_YOU_BUILD_DIRECTIVE: Final[str] = (
    "LOOK BEFORE YOU BUILD - SEARCH FIRST, EVERY TASK. The first step of your "
    "task is not the task: it is finding out whether it already exists. Do this "
    "for every fix and every feature, not only when you are stuck. Search in "
    "this order, cheapest first: (1) the diagnostics already emitted - logs, "
    "events, recorded output; a theory formed before reading our own output is "
    "a guess. (2) This repository - search for the CAPABILITY, not for the file "
    "name you expect. (3) Recorded lessons and earlier results available "
    "through your tools. When you find it, read it, keep the reasoning, and "
    "adapt it to our names and paths; a file pasted in unchanged is worse than "
    "none. SEARCH THE FAILURE, NOT ONLY THE FEATURE: before changing something, "
    "ask who has hit this bug in this code before and how it was solved. AN "
    "EMPTY SEARCH IS A RESULT - state in one line what you took, what you "
    "adapted, or why nothing fit. Reporting that nothing existed is required; "
    "silence about it is not."
)


SUPERVISOR_BOUNDARY_DIRECTIVE: Final[str] = (
    "SUPERVISOR BOUNDARY - USE ONLY EXPLICITLY GRANTED DODAIOS TOOLS. You may "
    "use the worker's own sandboxed shell and file tools inside the assigned "
    "worktree. Never invoke jarvis, jarvisctl, or jctl to control the running "
    "app; never operate the live desktop or Computer Use; never read control "
    "credentials; and never change live configuration. For knowledge and "
    "research work, use only the supervisor-granted Dodaios tools that appear "
    "in your tool list: the wiki tools (wiki-list, wiki-recall, "
    "wiki-page-read, wiki-ingest), web search "
    "(search_web), and contact lookup (contact-lookup). A tool absent from "
    "your tool list is not available for this mission - do not call it. A "
    "denied or failed supervisor tool call is a normal, recoverable event: "
    "adapt your approach, use an alternative, or deliver the remainder of "
    "the task and state the limitation in your summary."
)


# The worker's closing words are what the user hears about the mission. Left
# alone they are a file list or nothing at all, so the readback could not say
# whether the work is finished or what the user still has to decide.
FINAL_REPORT_DIRECTIVE: Final[str] = (
    "FINAL REPORT - CLOSE WITH A SHORT SUMMARY. End your work with a few plain "
    "sentences: what you built or found, where things stand (done, partly done, "
    "or blocked and why), and any decision the user still has to make. It is "
    "read back to the user, so leave out file listings, commands and logs "
    "unless they are the point. When you stop early or are blocked, say so "
    "there instead of ending silently."
)


def compose_worker_prompt(prior_block: str, step_prompt: str) -> str:
    """Build the full worker prompt: framing directives first.

    Leads with the artifact-language directive, the output-shape directive,
    the supervisor boundary and the final-report directive, so they frame the
    whole task before any prior feedback or instruction is read.
    ``prior_block`` is the (possibly empty) rendered critic-reflection block
    that carries correction context across iterations. ``step_prompt`` is the
    task instruction for this step (verbatim user utterance or a decomposed
    sub-task).
    """
    from ..conductor.lessons import lessons_block
    from ..stream_evidence import clean_request_body
    from ..task_kind import is_read_task

    head = (
        f"{ARTIFACT_LANGUAGE_DIRECTIVE}\n\n{LOOK_BEFORE_YOU_BUILD_DIRECTIVE}\n\n"
        f"{OUTPUT_SHAPE_DIRECTIVE}\n\n"
        f"{SUPERVISOR_BOUNDARY_DIRECTIVE}\n\n{FINAL_REPORT_DIRECTIVE}"
    )
    # Classified HERE, on the request alone: the directives above would read
    # as a read-only order themselves (see ``codex_direct_worker._sandbox_for``).
    if step_prompt.lstrip().startswith(READ_ONLY_MARKER) or is_read_task(
        clean_request_body(step_prompt)
    ):
        head = f"{READ_ONLY_MARKER} do not create, modify or delete any file.\n\n{head}"
    # Rules the app has already LEARNED from earlier failures of this kind.
    # ``lessons_block`` existed with a single caller - the delegation retry path
    # - so a worker and every fan-out child started each mission without them:
    # the app learned, stored, and then never handed the lesson to the agent
    # doing the work. It reads a process-wide store and returns "" when no store
    # is active or nothing was learned, so this adds nothing when there is
    # nothing to add.
    learned = lessons_block(step_prompt)
    if learned.strip():
        head = f"{head}\n\n{learned}"
    body = (
        f"{head}\n\n{prior_block}\n\n{step_prompt}"
        if prior_block and prior_block.strip()
        else f"{head}\n\n{step_prompt}"
    )
    # Last stop before the text becomes a worker CLI's input. The step prompt
    # is already redacted where it is built, but the reflection block is critic
    # output about a diff and a log, so it is not under that guarantee — and
    # this is the one place every decomposition path, every CLI and every
    # iteration passes through. Redacting twice costs nothing.
    return scrub_secrets(body)


__all__ = [
    "ARTIFACT_LANGUAGE_DIRECTIVE",
    "LOOK_BEFORE_YOU_BUILD_DIRECTIVE",
    "READ_ONLY_MARKER",
    "OUTPUT_SHAPE_DIRECTIVE",
    "FINAL_REPORT_DIRECTIVE",
    "SUPERVISOR_BOUNDARY_DIRECTIVE",
    "compose_worker_prompt",
]
