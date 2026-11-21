"""Whether a worker may have the helpers it asked for, and why.

A mission worker that wants to fan out now REQUESTS helpers instead of simply
taking them: the request and the decision are both recorded on the mission
record (``SubagentsRequested`` / ``SubagentsDecided``), so a fan-out is
visible where every other mission hop already is, and a denial comes back as
an instruction to carry on alone.

The decision is deterministic code, not a model call. Two reasons. Asking a
model "do you really need ten helpers?" spends money on every fan-out and can
be talked into yes by the same worker that wants them; and a fan-out that
multiplies is a control-flow problem, which is exactly where a bound belongs
(AP-19 counts per unit of work). The rules only ever REFUSE work that cannot
pay for itself:

* one helper is not parallel work — the worker can do that part itself;
* the same task twice is one task, however it is worded;
* a batch already run for this mission is a loop, not progress;
* more helpers than the mission has left is refused with the count.

Everything else is approved. A denial always names what to do instead,
because a worker that reads "denied" and nothing else simply asks again.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class HelperRequest:
    """One worker's ask: this many helpers, these tasks, this much left."""

    mission_id: str
    worker_id: str
    prompts: list[str] = field(default_factory=list)
    budget_left: int = 0

    @property
    def tasks(self) -> list[str]:
        """The non-empty tasks, in order, each stripped."""
        return [p.strip() for p in self.prompts if isinstance(p, str) and p.strip()]


#: Every answer the decision can give. The worker reads ``reason``; the
#: mission board translates ``code``, so no English model text reaches a
#: German screen.
DecisionCode = Literal[
    "ok",
    "no_task",
    "empty_task",
    "single_helper",
    "duplicate_tasks",
    "budget_spent",
    "over_budget",
    "repeat_batch",
]


@dataclass(frozen=True)
class Decision:
    """The answer: the reason the worker is told, and a code the UI can read."""

    approved: bool
    reason: str
    code: DecisionCode = "ok"


_CONTINUE: str = "Continue without a helper and do this part yourself."


def decide(
    request: HelperRequest, *, earlier: Sequence[frozenset[str]] = ()
) -> Decision:
    """Whether ``request`` is granted. ``earlier`` are this mission's granted batches."""
    tasks = request.tasks
    if len(tasks) != len([p for p in request.prompts if isinstance(p, str)]):
        return Decision(
            False, f"An empty task is not a helper. {_CONTINUE}", "empty_task"
        )
    if not tasks:
        return Decision(False, f"No task was given. {_CONTINUE}", "no_task")
    if len(tasks) == 1:
        return Decision(
            False,
            "One helper is not parallel work: there is nothing for it to run "
            f"beside. {_CONTINUE}",
            "single_helper",
        )
    unique = {t.strip().casefold() for t in tasks}
    if len(unique) != len(tasks):
        return Decision(
            False,
            "Two helpers were given the same task, so one of them would "
            f"repeat the other's work. {_CONTINUE} Or split the work "
            "differently and ask once more.",
            "duplicate_tasks",
        )
    if request.budget_left <= 0:
        return Decision(
            False,
            "This mission has no helpers left. Continue on your own and say "
            "plainly what you could not finish.",
            "budget_spent",
        )
    if len(tasks) > request.budget_left:
        return Decision(
            False,
            f"{len(tasks)} helpers were asked for and {request.budget_left} "
            f"are left. {_CONTINUE} Or ask again for at most "
            f"{request.budget_left}.",
            "over_budget",
        )
    batch = frozenset(t.strip().casefold() for t in tasks)
    if batch in set(earlier):
        return Decision(
            False,
            "These helpers already ran for this mission with the same tasks, "
            "so running them again repeats the work. Use what came back, or "
            "change what you are asking for.",
            "repeat_batch",
        )
    return Decision(
        True,
        f"{len(tasks)} independent tasks, {request.budget_left} helpers left.",
        "ok",
    )


__all__ = ["Decision", "DecisionCode", "HelperRequest", "decide"]
