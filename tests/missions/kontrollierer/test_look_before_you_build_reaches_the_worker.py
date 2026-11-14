"""Every worker and every fan-out child is told to search before it builds.

Root cause: ``AGENTS.md`` section 0 ("Look before you build") is the repo's
first rule for every coding agent, and no code path ever put it in front of a
worker -- it lived in a file the worker may never open. The cost is measured
twice. On 2026-11-07 an agent started rebuilding a memory-vault subsystem that
already sat complete upstream (ADR-0044). On 2026-11-09 three tools reached a
live call with an empty schema; the model sent empty arguments to one of them
177 times in one turn and the app was unusable, and the note recording that
exact failure shape two days earlier was never opened.

The second gap is the same shape one layer down: ``lessons_block`` renders the
rules the app LEARNED from earlier failures, and it had a single caller (the
delegation retry path). A mission worker and every fan-out child therefore
started without them -- the app learned, stored, and never handed the lesson to
the agent doing the work.

``compose_worker_prompt`` is the one chokepoint every decomposition path, every
worker CLI and every iteration passes through, so these tests pin both there.
"""
from __future__ import annotations

import jarvis.missions.conductor.lessons as lessons_module
from jarvis.missions.kontrollierer.worker_prompt import (
    LOOK_BEFORE_YOU_BUILD_DIRECTIVE,
    READ_ONLY_MARKER,
    compose_worker_prompt,
)

# --- The directive itself -----------------------------------------------------


def test_directive_states_the_search_order_not_only_the_slogan() -> None:
    text = LOOK_BEFORE_YOU_BUILD_DIRECTIVE.lower()
    # A worker that is only told "search first" does not know where to look.
    assert "diagnostics" in text or "logs" in text
    assert "repository" in text
    assert "lesson" in text


def test_directive_requires_reporting_an_empty_search() -> None:
    # "Upstream has the same gap, nothing to adopt" is a finding that justifies
    # building. Silence about it is what let the same work be redone by hand.
    text = LOOK_BEFORE_YOU_BUILD_DIRECTIVE.lower()
    assert "empty search" in text
    assert "result" in text


def test_directive_names_searching_the_failure_not_only_the_feature() -> None:
    # The 177-call loop was found by searching the feature and missing the
    # recorded failure of the same shape.
    text = LOOK_BEFORE_YOU_BUILD_DIRECTIVE.lower()
    assert "failure" in text


def test_directive_applies_to_every_task_not_only_when_stuck() -> None:
    text = LOOK_BEFORE_YOU_BUILD_DIRECTIVE.lower()
    assert "every" in text
    assert "stuck" in text


def test_directive_is_ascii_so_every_worker_cli_reads_it_intact() -> None:
    # Worker CLIs are spawned as subprocesses on Windows, where the default
    # encoding is cp1252 (AGENTS.md AP-1). A typographic character here would
    # arrive mangled at some worker and not at others.
    LOOK_BEFORE_YOU_BUILD_DIRECTIVE.encode("ascii")


# --- It reaches the worker ----------------------------------------------------


def test_the_directive_is_in_every_composed_worker_prompt() -> None:
    prompt = compose_worker_prompt("", "Add a button to the costs view")
    assert LOOK_BEFORE_YOU_BUILD_DIRECTIVE in prompt


def test_the_directive_survives_a_prior_reflection_block() -> None:
    # The iteration path passes a critic-reflection block; the framing
    # directives must still be there on the second and later attempts.
    prompt = compose_worker_prompt("The previous diff broke the boot path.", "Fix it")
    assert LOOK_BEFORE_YOU_BUILD_DIRECTIVE in prompt


def test_the_directive_is_in_a_read_only_task_too() -> None:
    # A read-only investigation is exactly the task that should search first.
    prompt = compose_worker_prompt("", f"{READ_ONLY_MARKER} find out why boot is slow")
    assert READ_ONLY_MARKER in prompt
    assert LOOK_BEFORE_YOU_BUILD_DIRECTIVE in prompt


def test_the_directive_comes_before_the_task_instruction() -> None:
    # It has to frame the task, not trail it: a worker that has already read its
    # instruction has already started planning the build.
    step = "Rebuild the memory vault from scratch"
    prompt = compose_worker_prompt("", step)
    assert prompt.index(LOOK_BEFORE_YOU_BUILD_DIRECTIVE) < prompt.index(step)


# --- Learned lessons reach the worker ----------------------------------------


class _FakeLessonStore:
    """Stands in for ``LessonStore``; ``lessons_block`` only calls ``render``."""

    def __init__(self, rendered: str) -> None:
        self._rendered = rendered
        self.calls: list[int] = []

    def render(self, lane: str, limit: int = 5) -> str:  # noqa: ARG002 - lane unused
        self.calls.append(limit)
        return self._rendered


def test_learned_rules_reach_the_worker(monkeypatch) -> None:
    store = _FakeLessonStore("- Read the flight recorder before theorising.")
    monkeypatch.setattr(lessons_module, "_active", store)

    prompt = compose_worker_prompt("", "Fix the live tool loop")

    assert "Read the flight recorder before theorising." in prompt
    assert store.calls, "the lesson store was never consulted"


def test_learned_rules_come_before_the_task_instruction(monkeypatch) -> None:
    rule = "- Check tool.schema before declaring a tool."
    monkeypatch.setattr(lessons_module, "_active", _FakeLessonStore(rule))
    step = "Declare the memory tools to the live call"

    prompt = compose_worker_prompt("", step)

    assert prompt.index(rule) < prompt.index(step)


def test_no_store_adds_nothing_and_does_not_raise(monkeypatch) -> None:
    # The common case: no mission learning active. The prompt must be unchanged
    # in substance and must not grow a stray empty rule block.
    monkeypatch.setattr(lessons_module, "_active", None)

    prompt = compose_worker_prompt("", "Add a button")

    assert LOOK_BEFORE_YOU_BUILD_DIRECTIVE in prompt
    assert "lessons learned from earlier orders" not in prompt


def test_an_empty_render_adds_no_rule_block(monkeypatch) -> None:
    # ``LessonStore.render`` answers the literal "(none)" when it knows nothing;
    # that must not reach a worker as a heading with no rules under it.
    monkeypatch.setattr(lessons_module, "_active", _FakeLessonStore("(none)"))

    prompt = compose_worker_prompt("", "Add a button")

    assert "lessons learned from earlier orders" not in prompt
