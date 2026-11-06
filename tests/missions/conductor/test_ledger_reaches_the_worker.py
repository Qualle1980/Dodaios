"""The order's ledger travels to the isolated worker with its instruction.

Magentic-One's executors share one group chat, so the task record and every
earlier result reach them for free. Ours run in their own worktree (AP-10)
and read nothing but the handed-over text. A live order on 2026-11-06 showed
what that costs: its third step re-ran the first step's search, because its
instruction was everything it knew.
"""
from __future__ import annotations

import json
from collections.abc import Sequence

from jarvis.missions.conductor.ledger import (
    WORKER_BRIEF_CHARS,
    WORKER_BRIEF_STEPS,
    worker_brief,
)
from jarvis.missions.conductor.loop import Conductor, Message

TEAM = {"codex": "reads and edits files", "claude-code": "large refactors"}


def _ledger(*, done: bool = False, instruction: str = "Step") -> str:
    return json.dumps({
        "is_request_satisfied": {"reason": "r", "answer": done},
        "is_in_loop": {"reason": "r", "answer": False},
        "is_progress_being_made": {"reason": "r", "answer": True},
        "next_executor": {"reason": "r", "answer": "codex"},
        "instruction": {"reason": "r", "answer": instruction},
    })


class _Seat:
    def __init__(self, ledgers: list[str]) -> None:
        self.ledgers = list(ledgers)

    async def __call__(self, messages: Sequence[Message]) -> str:
        prompt = messages[-1][1]
        if "pre-survey" in prompt:
            return "GIVEN OR VERIFIED FACTS: a.py exists"
        if "Devise a short bullet-point plan" in prompt:
            return "- codex: look"
        if "We worked on this request" in prompt:
            return "Done."
        return self.ledgers.pop(0) if self.ledgers else "garbage"


class _Team:
    """Records what each step actually received."""

    def __init__(self) -> None:
        self.handed: list[str] = []

    async def __call__(self, executor: str, instruction: str) -> str:
        self.handed.append(instruction)
        return f"result {len(self.handed)}"


# ── the builder ───────────────────────────────────────────────────────────


def test_an_order_with_nothing_established_hands_over_the_bare_instruction() -> None:
    assert worker_brief("Do it", "", []) == "Do it"


def test_the_fact_sheet_travels_with_the_instruction() -> None:
    brief = worker_brief("Do it", "GIVEN: a.py has 24 types", [])

    assert "GIVEN: a.py has 24 types" in brief
    assert brief.endswith("Do it")


def test_an_earlier_result_travels_so_it_is_not_found_again() -> None:
    brief = worker_brief("Now the ADRs", "facts", [("coding", "all 24 types mirrored")])

    assert "all 24 types mirrored" in brief
    assert "[coding]" in brief


def test_only_the_last_few_results_travel_so_a_long_order_stays_bounded() -> None:
    earlier = [("coding", f"finding {n}") for n in range(WORKER_BRIEF_STEPS + 3)]

    brief = worker_brief("Next", "facts", earlier)

    assert "finding 0" not in brief
    assert f"finding {WORKER_BRIEF_STEPS + 2}" in brief
    assert brief.count("[coding]") == WORKER_BRIEF_STEPS


def test_one_enormous_result_is_cut_instead_of_filling_the_brief() -> None:
    brief = worker_brief("Next", "facts", [("coding", "x" * (WORKER_BRIEF_CHARS * 3))])

    assert "x" * WORKER_BRIEF_CHARS in brief
    assert "x" * (WORKER_BRIEF_CHARS + 1) not in brief


def test_a_step_that_returned_nothing_is_left_out_of_the_brief() -> None:
    brief = worker_brief("Next", "facts", [("coding", "   "), ("quality", "approved")])

    assert "[coding]" not in brief
    assert "approved" in brief


# ── the loop ──────────────────────────────────────────────────────────────


async def test_the_first_step_carries_the_facts_but_no_earlier_result() -> None:
    seat, team = _Seat([_ledger(instruction="Look"), _ledger(done=True)]), _Team()

    await Conductor(seat, TEAM).run("Where is it?", team)

    assert len(team.handed) == 1
    assert "a.py exists" in team.handed[0]
    assert "Results of earlier steps" not in team.handed[0]


async def test_the_second_step_is_told_what_the_first_one_found() -> None:
    seat = _Seat([_ledger(instruction="Look"), _ledger(instruction="Now judge"),
                  _ledger(done=True)])
    team = _Team()

    await Conductor(seat, TEAM).run("Where is it?", team)

    assert len(team.handed) == 2
    assert "result 1" in team.handed[1], "the second worker never learned the first result"
    assert team.handed[1].rstrip().endswith("Now judge")


async def test_the_record_keeps_the_bare_instruction_not_the_brief() -> None:
    seat = _Seat([_ledger(instruction="Look"), _ledger(instruction="Now judge"),
                  _ledger(done=True)])
    team = _Team()

    out = await Conductor(seat, TEAM).run("Where is it?", team)

    assert [step.instruction for step in out.steps] == ["Look", "Now judge"]
