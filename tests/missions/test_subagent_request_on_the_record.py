"""The fan-out tool asks, is answered, and both hops reach the mission record."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jarvis.missions.events import SubagentsDecided, SubagentsRequested
from jarvis.missions.subagent_approval import Decision, HelperRequest
from jarvis.plugins.tool.spawn_subagents import SpawnSubagentsTool


@dataclass
class _Ctx:
    """The execution context a broker-brokered worker call carries."""

    config: dict[str, Any]
    user_utterance: str = "split this up"


def _ctx(**extra: Any) -> _Ctx:
    return _Ctx(config={
        "tool_origin": "mission_worker",
        "mission_id": "m1",
        "worker_id": "w1",
        **extra,
    })


class _Runner:
    """Records what was asked for and what the children were told to do."""

    def __init__(self) -> None:
        self.recorded: list[tuple[HelperRequest, Decision]] = []
        self.runs: list[list[str]] = []

    async def record_request(self, request: HelperRequest, decision: Decision) -> None:
        self.recorded.append((request, decision))

    async def run(self, *, mission_id: str, prompts: list[str], timeout_s: int) -> list[Any]:
        del mission_id, timeout_s
        self.runs.append(list(prompts))
        return [
            type("R", (), {"index": i, "ok": True, "answer": f"done {i}", "workspace": "w",
                           "error": "", "duration_s": 1.0})()
            for i, _ in enumerate(prompts)
        ]


async def test_an_approved_fan_out_is_recorded_and_runs() -> None:
    runner = _Runner()
    tool = SpawnSubagentsTool(runner=runner)

    result = await tool.execute({"prompts": ["audit the router", "audit the brain"]}, _ctx())

    assert result.success
    assert runner.runs == [["audit the router", "audit the brain"]]
    request, decision = runner.recorded[0]
    assert decision.approved and request.worker_id == "w1"


async def test_a_refused_fan_out_spawns_nothing_and_is_still_recorded() -> None:
    runner = _Runner()
    tool = SpawnSubagentsTool(runner=runner)

    result = await tool.execute({"prompts": ["just the one thing"]}, _ctx())

    assert not result.success
    assert runner.runs == []            # nothing was spawned
    assert len(runner.recorded) == 1    # but the ask is on the record
    _, decision = runner.recorded[0]
    assert not decision.approved
    assert result.error == decision.reason


async def test_the_refusal_the_worker_reads_says_what_to_do_instead() -> None:
    tool = SpawnSubagentsTool(runner=_Runner())
    result = await tool.execute({"prompts": ["one thing"]}, _ctx())
    assert "yourself" in (result.error or "")


async def test_the_same_batch_twice_is_refused_as_a_loop() -> None:
    runner = _Runner()
    tool = SpawnSubagentsTool(runner=runner)
    batch = {"prompts": ["audit the router", "audit the brain"]}

    assert (await tool.execute(batch, _ctx())).success
    again = await tool.execute({"prompts": ["audit the brain", "audit the router"]}, _ctx())

    assert not again.success
    assert "already ran" in (again.error or "")
    assert len(runner.runs) == 1


async def test_a_refusal_costs_no_budget() -> None:
    runner = _Runner()
    tool = SpawnSubagentsTool(runner=runner)

    for _ in range(12):
        await tool.execute({"prompts": ["one thing"]}, _ctx())
    ok = await tool.execute({"prompts": ["a", "b"]}, _ctx())

    assert ok.success, "refused asks must not eat the mission's budget"


async def test_asking_for_more_than_is_left_names_the_number() -> None:
    runner = _Runner()
    tool = SpawnSubagentsTool(runner=runner)

    await tool.execute({"prompts": [f"task {i}" for i in range(9)]}, _ctx())
    over = await tool.execute({"prompts": ["x", "y", "z"]}, _ctx())

    assert not over.success
    assert "1" in (over.error or "")
    assert len(runner.runs) == 1


def test_both_events_carry_the_fields_the_board_reads() -> None:
    requested = SubagentsRequested(worker_id="w1", count=2, budget_left=8, tasks=["a", "b"])
    decided = SubagentsDecided(worker_id="w1", approved=False, count=2, reason="no")
    assert requested.event_type == "SubagentsRequested"
    assert decided.event_type == "SubagentsDecided"
