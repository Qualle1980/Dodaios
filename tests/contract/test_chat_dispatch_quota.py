"""No new coding agent on a seat that is nearly out of plan.

The decision is made BEFORE anything starts, on the seat the work would
actually run on, from :mod:`jarvis.agent_usage` — which answers per account
without spawning a pane and says whether its numbers are live or cached. The
review of 2026-10-29 found the rule existed as a function and reached no
dispatch, and that judging a *set* of seats was wrong in both directions.

Nothing here calls a provider: the usage reader is replaced by a scripted one.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from jarvis.agent_usage import AccountUsage, UsageWindow
from jarvis.agentic_ide import orchestration
from tests.contract import test_workspace_orchestration as support

rig = support.rig
runnable = support.runnable


def _usage(account_id: str, used: float, *, status: str = "ok") -> AccountUsage:
    return AccountUsage(
        account_id=account_id,
        platform="claude",
        status=status,
        windows=(UsageWindow(kind="session", percent=used, severity="normal"),),
        source="live",
        as_of=__import__("time").time(),
    )


@pytest.fixture
def seats(monkeypatch):
    """A scripted usage reader for named seats, and the active one."""
    state: dict[str, Any] = {"readings": {}, "active": "seat-a", "supported": True}

    class _Account:
        def __init__(self, account_id: str) -> None:
            self.id = account_id
            self.platform = "claude"

    monkeypatch.setattr(
        "jarvis.agent_usage.supports_usage", lambda platform: state["supported"]
    )
    monkeypatch.setattr(
        "jarvis.agent_accounts.list_accounts",
        lambda platform: [_Account(k) for k in state["readings"]] or [_Account(state["active"])],
    )
    monkeypatch.setattr(
        "jarvis.agent_accounts.active_account", lambda platform: _Account(state["active"])
    )
    monkeypatch.setattr(
        "jarvis.agent_usage.collect", lambda accounts, **kw: dict(state["readings"])
    )
    return state


async def _send(rig, **extra):
    orchestrator = rig[0]
    resolved = await support.target(rig)
    return await orchestrator.run(
        {
            "action": "send",
            **resolved,
            "request_id": uuid4().hex,
            "prompt": "Add subtract() to calc.py.",
            **extra,
        }
    )


async def test_a_seat_with_ten_percent_left_starts_nothing(rig, runnable, seats):
    """95 % used leaves 5 %: refused, and the pane is never written to."""
    _orchestrator, registry, sessions = rig
    pane = registry.sessions[0].terminals[0]
    pane.agent, pane.account = "claude", "seat-a"
    pane.model = "sonnet"
    seats["readings"] = {"seat-a": _usage("seat-a", 95.0)}

    result = await _send(rig)

    assert result["status"] == "not_accepted", result
    assert result["quota"] == "low"
    assert result["input_written"] is False
    assert "Nothing was sent" in result["reason"]
    # The low-level gateway was never reached: nothing was spent at all.
    assert not sessions.calls


async def test_a_healthy_seat_is_not_blocked_by_another_seat_running_low(
    rig, runnable, seats
):
    """Judging a set of seats refused a good target because a neighbour was low."""
    _orchestrator, registry, sessions = rig
    pane = registry.sessions[0].terminals[0]
    pane.agent, pane.account = "claude", "seat-a"
    pane.model = "sonnet"
    seats["readings"] = {
        "seat-a": _usage("seat-a", 20.0),
        "seat-b": _usage("seat-b", 99.0),
    }

    result = await _send(rig)

    assert result["status"] == "accepted", result
    assert len(sessions.calls) == 1


async def test_an_unknown_target_is_not_cleared_by_a_healthy_neighbour(
    rig, runnable, seats
):
    """A seat that publishes nothing stays unknown, whatever the others say."""
    _orchestrator, registry, _sessions = rig
    pane = registry.sessions[0].terminals[0]
    pane.agent, pane.account = "claude", "seat-a"
    pane.model = "sonnet"
    seats["readings"] = {
        "seat-a": _usage("seat-a", 0.0, status="unsupported"),
        "seat-b": _usage("seat-b", 1.0),
    }

    state, reason = await orchestration._chat_seat_quota("claude", "seat-a")
    assert state == "unknown", reason
    # Unknown is not a refusal: several CLIs publish no usage at all, and
    # refusing on absence would make delegation impossible rather than safe.
    # It is reported instead -- see the create receipt's ``quota`` field.
    result = await _send(rig)
    assert result["status"] == "accepted", result


async def test_a_cli_without_usage_support_is_unknown_not_blocked(rig, runnable, seats):
    seats["supported"] = False
    state, reason = await orchestration._chat_seat_quota("codex")
    assert state == "unknown" and "no plan usage" in reason


async def test_an_unreadable_budget_is_unknown_not_empty(rig, runnable, monkeypatch):
    """A reader that raises must not look like an exhausted plan."""
    monkeypatch.setattr("jarvis.agent_usage.supports_usage", lambda platform: True)

    def _boom(*_a, **_k):
        raise RuntimeError("no credentials")

    monkeypatch.setattr("jarvis.agent_accounts.list_accounts", _boom)
    state, reason = await orchestration._chat_seat_quota("claude", "seat-a")
    assert state == "unknown" and "could not be read" in reason


async def test_creating_a_pane_on_an_exhausted_seat_is_refused(rig, runnable, seats):
    """The same decision on the create path: no pane, no prompt, no spend."""
    orchestrator, registry, sessions = rig
    seats["readings"] = {"seat-a": _usage("seat-a", 96.0)}
    owner = registry.sessions[0]
    before = [t.history_id for t in owner.terminals]

    result = await orchestrator.run(
        {
            "action": "create",
            "workspace_id": owner.id,
            "project_id": owner.project_id,
            "cli": "Claude Code",
            "count": 1,
            "prompt": "Add subtract().",
            "request_id": uuid4().hex,
        }
    )

    assert result["status"] == "not_accepted", result
    assert result["quota"] == "low"
    assert [t.history_id for t in owner.terminals] == before
    assert not sessions.calls


async def test_a_created_pane_says_so_when_the_budget_could_not_be_read(
    rig, runnable, seats
):
    """Unknown is carried out loud, on the receipt the chief has to report."""
    orchestrator, registry, _sessions = rig
    seats["supported"] = False
    owner = registry.sessions[0]

    result = await orchestrator.run(
        {
            "action": "create",
            "workspace_id": owner.id,
            "project_id": owner.project_id,
            "cli": "Claude Code",
            "count": 1,
            "request_id": uuid4().hex,
        }
    )

    assert result["status"] == "created", result
    assert result["quota"] == "unknown"
    assert "no plan usage" in result["quota_detail"]
