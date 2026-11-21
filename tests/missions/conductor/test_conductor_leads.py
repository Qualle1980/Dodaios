"""The four domain leads are roles the conductor hands work to, nothing more."""
from __future__ import annotations

import pytest

from jarvis.missions.conductor.lanes import FAMILY, LANES
from jarvis.missions.conductor.leads import (
    LEADS,
    lane_for,
    may_write,
    resolve_executor,
    roster,
)


def test_the_roster_is_the_four_leads_and_nothing_else() -> None:
    assert set(roster("add a flag to the parser")) == {
        "planning", "coding", "quality", "memory",
    }


def test_every_lead_describes_itself_for_the_plan() -> None:
    for name, role in roster("anything").items():
        assert role.strip(), name
        assert len(role) > 20, name


def test_a_lead_is_never_an_executor_name() -> None:
    # The roster names roles; the executors stay an implementation detail the
    # model cannot address directly.
    assert not set(LEADS) & set(FAMILY)


def test_only_the_coding_lead_may_change_files() -> None:
    assert may_write("coding")
    assert not may_write("planning")
    assert not may_write("quality")
    assert not may_write("memory")


def test_planning_answers_on_the_seat_without_a_worker() -> None:
    assert resolve_executor("planning", "what is a rebase?") == "conductor"


def test_the_coding_lead_follows_the_order_s_own_lane() -> None:
    # The deterministic first pass still decides how big the work is; the lead
    # does not re-route it with a second judgement.
    assert lane_for("coding", "refactor the whole router") == "deep"
    assert lane_for("coding", "fix the typo in a.py") == "change"


def test_a_read_only_lead_never_gets_a_writing_sandbox() -> None:
    for lead in ("planning", "quality", "memory"):
        assert LANES[lane_for(lead, "look at a.py")].sandbox != "workspace-write"


def test_the_quality_lead_is_of_another_family_than_the_builder() -> None:
    reviewer = resolve_executor("quality", "check it", builder="codex",
                               connected=("codex", "claude-code"))
    assert reviewer is not None
    assert FAMILY[reviewer] != FAMILY["codex"]


def test_quality_without_a_second_family_is_no_reviewer_at_all() -> None:
    # No fake independence: one family connected means nobody can review it.
    assert resolve_executor("quality", "check it", builder="codex",
                            connected=("codex",)) is None


def test_a_lead_whose_chain_has_nobody_connected_is_unavailable() -> None:
    assert resolve_executor("coding", "fix a.py", connected=()) is None


def test_an_executor_the_user_named_resolves_to_itself() -> None:
    # `requested_executor` pins one worker for the whole order; that name
    # travels through the same resolution instead of a second code path.
    assert resolve_executor("codex", "let codex count the lines",
                            connected=("codex",)) == "codex"


def test_an_unknown_name_resolves_to_nothing() -> None:
    assert resolve_executor("sergeant-major", "do it", connected=("codex",)) is None


@pytest.mark.parametrize("lead", sorted(LEADS))
def test_every_lead_resolves_with_both_workers_connected(lead: str) -> None:
    assert resolve_executor(lead, "change a.py", builder="codex",
                            connected=("codex", "claude-code")) is not None
