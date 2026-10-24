"""The conductor loop over a scripted seat: no model, no worker."""
from __future__ import annotations

import json
from collections.abc import Sequence

from jarvis.missions.conductor.loop import Conductor, Message

TEAM = {"codex": "reads and edits files", "claude-code": "large refactors"}


def _ledger(*, done: bool = False, progress: bool = True, loop: bool = False,
            nxt: str = "codex", instruction: str = "List the TODOs in a.py") -> str:
    return "Here you go:\n" + json.dumps({
        "is_request_satisfied": {"reason": "r", "answer": done},
        "is_in_loop": {"reason": "same reply twice", "answer": loop},
        "is_progress_being_made": {"reason": "blocked", "answer": progress},
        "next_executor": {"reason": "r", "answer": nxt},
        "instruction": {"reason": "r", "answer": instruction},
    })


class _Seat:
    """Answers by prompt kind; ledgers are served in order."""

    def __init__(self, ledgers: list[str]) -> None:
        self.ledgers = list(ledgers)
        self.calls: list[str] = []

    async def __call__(self, messages: Sequence[Message]) -> str:
        prompt = messages[-1][1]
        if "pre-survey" in prompt:
            self.calls.append("facts")
            return "1. GIVEN OR VERIFIED FACTS: a.py exists"
        if "We are not making the progress" in prompt:
            self.calls.append("facts_update")
            return "updated facts"
        if "Devise a short bullet-point plan" in prompt:
            self.calls.append("plan")
            return "- codex: list the TODOs"
        if "what went wrong" in prompt:
            self.calls.append("plan_update")
            return "- codex: search with rg instead"
        if "We worked on this request" in prompt:
            self.calls.append("final")
            return "a.py holds 2 TODOs."
        self.calls.append("ledger")
        return self.ledgers.pop(0) if self.ledgers else "garbage"


class _Team:
    def __init__(self) -> None:
        self.steps: list[tuple[str, str]] = []

    async def __call__(self, executor: str, instruction: str) -> str:
        self.steps.append((executor, instruction))
        return "a.py:3 TODO x\na.py:9 TODO y"


async def test_one_step_then_done_writes_the_final_answer() -> None:
    seat, team = _Seat([_ledger(), _ledger(done=True)]), _Team()
    conductor = Conductor(seat, TEAM, lessons="- search before answering")
    out = await conductor.run("TODOs in a.py?", team)

    assert out.status == "done" and out.answer == "a.py holds 2 TODOs."
    assert team.steps == [("codex", "List the TODOs in a.py")]
    assert out.steps[0].result.startswith("a.py:3")
    assert seat.calls == ["facts", "plan", "ledger", "ledger", "final"]


async def test_lessons_reach_the_plan_prompt() -> None:
    seen: list[str] = []

    async def _seat(messages: Sequence[Message]) -> str:
        seen.append(messages[-1][1])
        return _ledger(done=True) if "Recall the request" in messages[-1][1] else "x"

    await Conductor(_seat, TEAM, lessons="- verify the file on disk").run("t", _Team())
    assert any("verify the file on disk" in p for p in seen if "bullet-point plan" in p)


async def test_stalls_replan_once_per_budget_then_stop_honestly() -> None:
    stuck = _ledger(progress=False)
    seat, team = _Seat([stuck] * 10), _Team()
    out = await Conductor(seat, TEAM, max_stalls=2, max_replans=1, max_rounds=10).run("t", team)

    assert out.status == "stopped" and out.replans == 1
    assert seat.calls.count("plan_update") == 1 and seat.calls[-1] == "final"
    # A stalled refresh below the bound still hands the step out (Magentic-One).
    assert len(team.steps) == 2


async def test_an_unknown_executor_or_garbage_is_retried_then_stops() -> None:
    seat, team = _Seat([_ledger(nxt="gemini"), "no json", "{}"]), _Team()
    out = await Conductor(seat, TEAM, max_json_retries=3).run("t", team)

    assert out.status == "no_ledger" and team.steps == []
    assert seat.calls.count("ledger") == 3 and seat.calls[-1] == "final"


async def test_the_round_bound_still_ends_in_an_answer() -> None:
    seat, team = _Seat([_ledger()] * 5), _Team()
    out = await Conductor(seat, TEAM, max_rounds=3).run("t", team)

    assert out.status == "max_rounds" and out.rounds == 3 and len(team.steps) == 3
    assert out.answer == "a.py holds 2 TODOs."


async def test_preview_plans_without_executing() -> None:
    seat = _Seat([_ledger(nxt="claude-code", instruction="Refactor the router")])
    preview = await Conductor(seat, TEAM).preview("Refactor the router")

    assert preview.first is not None and preview.first.next_executor.answer == "claude-code"
    assert preview.plan.startswith("- codex")
    assert seat.calls == ["facts", "plan", "ledger"]


def test_a_conversation_reaches_the_seat_as_one_prompt_with_the_ask_last() -> None:
    from jarvis.missions.conductor.loop import as_single_prompt

    assert as_single_prompt([("user", "only ask")]) == "only ask"
    prompt = as_single_prompt([("user", "facts?"), ("assistant", "a.py exists"),
                               ("user", "Devise a plan")])
    assert prompt.index("facts?") < prompt.index("a.py exists") < prompt.index("Devise a plan")
    assert prompt.rstrip().endswith("Devise a plan")
    assert "Answer ONLY this now" in prompt


async def test_a_halted_step_ends_the_loop_without_another_seat_call() -> None:
    from jarvis.missions.conductor.loop import OrderHalted

    seat = _Seat([_ledger()] * 5)

    async def _halt(executor: str, instruction: str) -> str:
        raise OrderHalted("waiting_capacity", "outcome: waiting_capacity")

    out = await Conductor(seat, TEAM).run("t", _halt)
    assert out.status == "halted" and out.answer == ""
    assert seat.calls == ["facts", "plan", "ledger"]  # no final answer, no second ledger
    assert out.steps[0].result == "outcome: waiting_capacity"
