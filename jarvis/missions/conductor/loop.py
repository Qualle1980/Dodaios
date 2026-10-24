"""The conductor loop: plan an order, hand out steps, watch progress, answer.

Two loops around the records of :mod:`.ledger`:

* **Outer loop** — facts, then a plan (both on the conductor seat), then the
  task record opens a fresh thread.
* **Inner loop** — after every executor result the progress record is
  refreshed (strict JSON, bounded retries). Satisfied: the conductor writes
  the final answer. Stalled ``max_stalls`` times: facts and plan are rewritten
  and the outer loop starts again, at most ``max_replans`` times, then the
  order stops with an honest answer. Otherwise the named executor gets the
  instruction and its result joins the thread.

The model never picks a tool outside the team: :func:`.ledger.parse_progress`
rejects an unknown executor, and ``execute`` only ever sees a team name.

Control flow adapted from Magentic-One's orchestrator in microsoft/autogen
(``_magentic_one_orchestrator.py``: ``handle_start``, ``_reenter_outer_loop``,
``_orchestrate_step``, ``_update_task_ledger``, ``_prepare_final_answer``);
MIT License, Copyright (c) Microsoft Corporation — the full notice is in
:mod:`.ledger`. Changes: no agent runtime or group-chat topics, executors are
Dodaios lanes behind one ``execute`` callable, results come back as text,
learned rules feed the plan, and a round bound always ends in a final answer
instead of an exception.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

from .ledger import (
    DEFAULT_MAX_REPLANS,
    DEFAULT_MAX_STALLS,
    FACTS_PROMPT,
    FACTS_UPDATE_PROMPT,
    FINAL_ANSWER_PROMPT,
    LEDGER_PROMPT,
    PLAN_PROMPT,
    PLAN_UPDATE_PROMPT,
    PROGRESS_PROMPT,
    ProgressEntry,
    StallCounter,
    parse_progress,
)

logger = logging.getLogger(__name__)

#: Upper bound on inner-loop rounds; each costs one seat call plus one step.
DEFAULT_MAX_ROUNDS: Final[int] = 8
#: Attempts to get a usable progress record before the order stops.
DEFAULT_MAX_JSON_RETRIES: Final[int] = 3

Message = tuple[Literal["user", "assistant"], str]
#: One completion on the conductor seat: the whole conversation in, text out.
Ask = Callable[[Sequence[Message]], Awaitable[str]]
#: Run one step: ``(executor, instruction)`` in, the executor's result as text out.
Execute = Callable[[str, str], Awaitable[str]]

Status = Literal["done", "stopped", "max_rounds", "no_ledger", "halted"]


class OrderHalted(Exception):  # noqa: N818 - a control signal, not an error
    """Raised by ``execute`` when no further step may run (capacity, budget).

    The loop ends at once with status ``halted`` and no further seat call:
    the caller parks or fails the order itself.
    """

    def __init__(self, reason: str, result: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.result = result


def as_single_prompt(messages: Sequence[Message]) -> str:
    """The conversation as ONE prompt: the transcript first, the current ask last.

    Every seat answers this the same way, whatever roles its transport keeps.
    """
    if not messages:
        return ""
    *earlier, (_, current) = messages
    if not earlier:
        return current
    transcript = "\n\n".join(f"[{role}]\n{text}" for role, text in earlier)
    return (
        "Conversation so far (context only, already answered):\n\n"
        f"{transcript}\n\n---\nAnswer ONLY this now:\n\n{current}"
    )


@dataclass
class Step:
    """One handed-out step and what came back."""

    executor: str
    instruction: str
    result: str


@dataclass
class Outcome:
    """How an order ended."""

    status: Status
    answer: str
    facts: str
    plan: str
    rounds: int
    replans: int
    steps: list[Step] = field(default_factory=list)
    ledgers: list[ProgressEntry] = field(default_factory=list)


@dataclass
class Preview:
    """The conductor's first decision without running anything (shadow mode)."""

    facts: str
    plan: str
    first: ProgressEntry | None


class Conductor:
    """Conduct one order over a fixed team of executors."""

    def __init__(
        self,
        ask: Ask,
        team: dict[str, str],
        *,
        lessons: str = "(none)",
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        max_stalls: int = DEFAULT_MAX_STALLS,
        max_replans: int = DEFAULT_MAX_REPLANS,
        max_json_retries: int = DEFAULT_MAX_JSON_RETRIES,
    ) -> None:
        if not team:
            raise ValueError("a conductor needs at least one executor")
        self._ask = ask
        self._team = dict(team)
        self._names = list(team)
        self._lessons = lessons or "(none)"
        self._max_rounds = max_rounds
        self._max_stalls = max_stalls
        self._max_replans = max_replans
        self._max_json_retries = max(1, max_json_retries)

    @property
    def team_text(self) -> str:
        return "\n".join(f"{name}: {role}" for name, role in self._team.items())

    # -- outer loop ------------------------------------------------------

    async def _first_ledger(self, task: str) -> tuple[str, str]:
        planning: list[Message] = [("user", FACTS_PROMPT.format(task=task))]
        facts = await self._ask(planning)
        planning += [
            ("assistant", facts),
            ("user", PLAN_PROMPT.format(team=self.team_text, lessons=self._lessons)),
        ]
        plan = await self._ask(planning)
        return facts, plan

    async def _update_ledger(
        self, task: str, thread: list[Message], facts: str
    ) -> tuple[str, str]:
        context = [*thread, ("user", FACTS_UPDATE_PROMPT.format(task=task, facts=facts))]
        facts = await self._ask(context)
        context += [("assistant", facts), ("user", PLAN_UPDATE_PROMPT.format(team=self.team_text))]
        plan = await self._ask(context)
        return facts, plan

    def _open_thread(self, task: str, facts: str, plan: str) -> list[Message]:
        return [("assistant", LEDGER_PROMPT.format(
            task=task, team=self.team_text, facts=facts, plan=plan))]

    # -- inner loop ------------------------------------------------------

    async def _progress(self, task: str, thread: list[Message]) -> ProgressEntry | None:
        prompt = PROGRESS_PROMPT.format(
            task=task, team=self.team_text, names=", ".join(self._names))
        for attempt in range(1, self._max_json_retries + 1):
            raw = await self._ask([*thread, ("user", prompt)])
            entry = parse_progress(raw, self._names)
            if entry is not None:
                return entry
            logger.info("conductor: unusable progress record (attempt %d/%d)",
                        attempt, self._max_json_retries)
        return None

    async def _final_answer(self, task: str, thread: list[Message]) -> str:
        return await self._ask([*thread, ("user", FINAL_ANSWER_PROMPT.format(task=task))])

    async def preview(self, task: str) -> Preview:
        """Facts, plan and the first progress record; nothing is executed."""
        facts, plan = await self._first_ledger(task)
        first = await self._progress(task, self._open_thread(task, facts, plan))
        return Preview(facts=facts, plan=plan, first=first)

    async def run(self, task: str, execute: Execute) -> Outcome:
        """Conduct ``task`` to a final answer. Never raises on a model's reply shape."""
        facts, plan = await self._first_ledger(task)
        thread = self._open_thread(task, facts, plan)
        stalls = StallCounter(max_stalls=self._max_stalls, max_replans=self._max_replans)
        steps: list[Step] = []
        ledgers: list[ProgressEntry] = []

        def _outcome(status: Status, answer: str, rounds: int) -> Outcome:
            return Outcome(status=status, answer=answer, facts=facts, plan=plan,
                           rounds=rounds, replans=stalls.replans, steps=steps, ledgers=ledgers)

        for rounds in range(1, self._max_rounds + 1):
            entry = await self._progress(task, thread)
            if entry is None:
                logger.warning("conductor: no usable progress record, stopping the order")
                return _outcome("no_ledger", await self._final_answer(task, thread), rounds)
            ledgers.append(entry)
            decision = stalls.decide(entry)
            if decision == "done":
                return _outcome("done", await self._final_answer(task, thread), rounds)
            if decision == "stop":
                logger.info("conductor: stalled after %d replans, stopping", stalls.replans)
                return _outcome("stopped", await self._final_answer(task, thread), rounds)
            if decision == "replan":
                logger.info("conductor: stalled, replanning (%d/%d)",
                            stalls.replans, self._max_replans)
                facts, plan = await self._update_ledger(task, thread, facts)
                thread = self._open_thread(task, facts, plan)
                continue
            executor = entry.next_executor.answer
            instruction = entry.instruction.answer
            thread.append(("assistant", f"[conductor -> {executor}] {instruction}"))
            try:
                result = await execute(executor, instruction)
            except OrderHalted as halt:
                steps.append(Step(executor=executor, instruction=instruction, result=halt.result))
                logger.info("conductor: order halted (%s), no further seat call", halt.reason)
                return _outcome("halted", "", rounds)
            steps.append(Step(executor=executor, instruction=instruction, result=result))
            thread.append(("user", f"[{executor}] {result}"))
        logger.info("conductor: round bound %d reached", self._max_rounds)
        return _outcome("max_rounds", await self._final_answer(task, thread), self._max_rounds)


__all__ = [
    "DEFAULT_MAX_JSON_RETRIES", "DEFAULT_MAX_ROUNDS", "Ask", "Conductor", "Execute", "OrderHalted",
    "as_single_prompt",
    "Message", "Outcome", "Preview", "Step",
]
