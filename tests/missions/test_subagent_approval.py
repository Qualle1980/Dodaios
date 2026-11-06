"""A helper is requested, decided and recorded — never just taken."""
from __future__ import annotations

from jarvis.missions.subagent_approval import HelperRequest, decide


def _request(*prompts: str, budget_left: int = 10) -> HelperRequest:
    return HelperRequest(
        mission_id="m1", worker_id="w1", prompts=list(prompts), budget_left=budget_left,
    )


def test_real_parallel_work_is_approved() -> None:
    decision = decide(_request("audit the router", "audit the brain"))
    assert decision.approved
    assert decision.reason


def test_one_helper_is_not_parallel_work() -> None:
    decision = decide(_request("audit the router"))
    assert not decision.approved
    assert "yourself" in decision.reason


def test_the_same_task_twice_is_not_two_helpers() -> None:
    decision = decide(_request("audit the router", "audit the router "))
    assert not decision.approved
    assert "same" in decision.reason.lower()


def test_a_batch_already_run_is_a_loop_and_is_denied() -> None:
    earlier = [frozenset({"audit the router", "audit the brain"})]
    decision = decide(_request("audit the brain", "audit the router"), earlier=earlier)
    assert not decision.approved
    assert "already" in decision.reason.lower()


def test_a_batch_beyond_what_is_left_is_denied_with_the_count() -> None:
    decision = decide(_request("a", "b", "c", budget_left=2))
    assert not decision.approved
    assert "2" in decision.reason


def test_nothing_left_denies_before_anything_is_spawned() -> None:
    decision = decide(_request("a", "b", budget_left=0))
    assert not decision.approved


def test_a_denial_tells_the_worker_what_to_do_instead() -> None:
    # The denial is an instruction to carry on, not an error to retry: a
    # worker that reads "denied" and nothing else tries again in a loop.
    for request in (_request("only one"),
                    _request("x", "x"),
                    _request("a", "b", budget_left=1)):
        decision = decide(request)
        assert not decision.approved
        assert "continue" in decision.reason.lower()


def test_blank_prompts_do_not_count_as_helpers() -> None:
    assert not decide(_request("real work", "   ")).approved
