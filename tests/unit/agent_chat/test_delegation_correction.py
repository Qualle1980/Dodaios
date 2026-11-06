"""One bounded correction attempt after a rejected handback (Phase C2).

The acceptance gate rejects an answer whose evidence is missing. Slice 1 wrote
that down; this is the return path. The whole risk of it lives in one
distinction: a job that DELIVERED and cannot prove it may be corrected, while a
job that is blocked, crashed or superseded may not be re-dispatched on
suspicion -- ``test_a_blocked_or_failed_agent_is_never_accepted_and_never_retried``
is that contract, and these tests pin the boundary it rests on.
"""

from __future__ import annotations

from dataclasses import dataclass

from jarvis.agent_chat.delegation_correction import (
    MAX_CORRECTIONS,
    correctable,
    correction_prompt,
)

ORDER = "Add subtract() to the calculator and run the tests."
GAPS = ("no test run inside the job's checkout", "no change read")


@dataclass(frozen=True)
class FakeJob:
    task_id: str
    status: str
    workspace_id: str = "ws-1"


@dataclass(frozen=True)
class FakeGate:
    jobs: tuple[FakeJob, ...]

    @property
    def work(self) -> tuple[str, ...]:
        return tuple(job.task_id for job in self.jobs)


def test_only_a_job_that_claims_delivery_may_be_corrected() -> None:
    """``completed`` is the one status that claims the work was handed back.
    Everything else is further from acceptance, not closer to it, so it is not
    re-dispatched: the person is told honestly instead."""
    assert correctable(FakeGate((FakeJob("coding-1", "completed"),))) == ("coding-1",)
    for status in ("needs_input", "error", "cancelled", "superseded", "running", ""):
        gate = FakeGate((FakeJob("coding-1", status),))
        assert correctable(gate) == (), status


def test_a_mixed_wave_corrects_only_the_jobs_that_delivered() -> None:
    gate = FakeGate((
        FakeJob("coding-1", "completed"),
        FakeJob("coding-2", "needs_input"),
        FakeJob("coding-3", "completed"),
    ))
    assert correctable(gate) == ("coding-1", "coding-3")


def test_a_gate_without_jobs_is_not_correctable() -> None:
    assert correctable(FakeGate(())) == ()
    assert correctable(None) == ()


def test_the_correction_names_the_gaps_and_forbids_a_claim() -> None:
    prompt = correction_prompt(
        gaps=GAPS, work=("coding-1",), original=ORDER, rules="", attempt=1
    )
    for gap in GAPS:
        assert gap in prompt
    assert "coding-1" in prompt
    assert ORDER in prompt
    # The correction is an instruction to get evidence, never a licence to
    # claim the work is good.
    lowered = prompt.lower()
    assert "do not claim" in lowered or "never claim" in lowered
    assert "attempt 1" in lowered and str(MAX_CORRECTIONS) in prompt


def test_the_learned_rules_travel_with_the_correction() -> None:
    """The conductor's seat may have written a rule from the first failure
    (slice 1). The next attempt has to see it, or the review bought nothing."""
    rules = "- Run the tests inside the job's own checkout before claiming a pass."
    prompt = correction_prompt(
        gaps=GAPS, work=("coding-1",), original=ORDER, rules=rules, attempt=2
    )
    assert rules in prompt
    assert "attempt 2" in prompt.lower()


def test_the_correction_is_not_read_as_the_persons_own_words() -> None:
    """A composed prompt that quotes the order would otherwise be routed as if
    the person had typed it, which is how a pane got re-ordered once
    (``is_continuation_prompt``)."""
    from jarvis.agent_chat.delegation_wait import is_continuation_prompt

    prompt = correction_prompt(
        gaps=GAPS, work=("coding-1",), original=ORDER, rules="", attempt=1
    )
    assert is_continuation_prompt(prompt)


def test_two_attempts_is_the_bound() -> None:
    """A real coding run costs money per round, so the bound is small and
    explicit rather than left to the wave budget."""
    assert MAX_CORRECTIONS == 2
