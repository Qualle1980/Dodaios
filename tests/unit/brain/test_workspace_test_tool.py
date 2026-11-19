"""``workspace-test``: run a workspace's configured suite, report pass or fail.

It is the General's only way to run tests. It has no shell behind it: the
schema has no field for a program, and whatever the model sends, the command
that runs is the one the user configured.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import jsonschema
import pytest

from jarvis.brain.tool_gateway import BrainSupervisorToolGateway
from jarvis.brain.workspace_git_tool import WorkspaceGitTool
from jarvis.brain.workspace_test_tool import WorkspaceTestTool
from jarvis.brain.workspace_tool import WorkspaceOrchestrationTool
from jarvis.core.config import AgenticIdeConfig
from jarvis.core.protocols import ExecutionContext
from jarvis.core.tool_read_only import allows_read
from tests.fakes.fake_workspace_checkout import FakeCheckoutOrchestrator


def _ctx() -> ExecutionContext:
    return ExecutionContext(trace_id=uuid4(), user_utterance="", config={}, memory_read=None)


class _RecordingGateway:
    def __init__(self, reply: dict[str, Any] | Exception) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def run(self, args: dict[str, Any], *, trace_id: str = "") -> dict[str, Any]:
        self.calls.append(dict(args))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_the_schema_offers_a_workspace_and_no_program() -> None:
    props = WorkspaceTestTool.schema["properties"]
    assert WorkspaceTestTool.name == "workspace-test"
    assert {"workspace", "workspace_id"} <= set(props)
    assert not {"command", "cmd", "args", "argv", "program", "shell"} & set(props)
    assert WorkspaceTestTool.schema["additionalProperties"] is False
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"command": "rm -rf /"}, WorkspaceTestTool.schema)


def test_it_is_a_read_tool_for_every_argument() -> None:
    tool = WorkspaceTestTool(_RecordingGateway({"status": "ok"}))
    assert allows_read(tool, {"workspace_id": "w1"})
    assert tool.risk_tier == "safe"


async def test_it_asks_the_gateway_for_a_test_run_and_strips_any_command() -> None:
    gateway = _RecordingGateway({"status": "ok", "passed": True, "exit_code": 0})
    tool = WorkspaceTestTool(gateway)
    result = await tool.execute({"workspace_id": "w1", "command": "evil"}, _ctx())
    assert result.success is True
    assert gateway.calls == [{"workspace_id": "w1", "action": "test"}]


async def test_a_failing_suite_is_a_successful_read_that_says_failed() -> None:
    """The run happened and its answer is "failed": that is a result, not a tool error."""
    gateway = _RecordingGateway({"status": "ok", "passed": False, "exit_code": 1})
    result = await WorkspaceTestTool(gateway).execute({"workspace_id": "w1"}, _ctx())
    assert result.success is True and result.output["passed"] is False


async def test_a_refusal_is_not_a_success() -> None:
    gateway = _RecordingGateway(
        {"status": "not_accepted", "success": False, "code": "no_test_command", "reason": "x"}
    )
    result = await WorkspaceTestTool(gateway).execute({"workspace_id": "w1"}, _ctx())
    assert result.success is False and result.output["code"] == "no_test_command"


async def test_invalid_arguments_come_back_as_the_tool_error() -> None:
    result = await WorkspaceTestTool(_RecordingGateway(ValueError("bad selector"))).execute(
        {}, _ctx()
    )
    assert result.success is False and result.error == "bad selector"


def _checkout(root: Path) -> None:
    root.mkdir()
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    git = ("git", "-C", str(root))
    subprocess.run([*git, "init", "-q"], check=True, capture_output=True)
    subprocess.run([*git, "config", "user.email", "t@example.invalid"], check=True)
    subprocess.run([*git, "config", "user.name", "Test"], check=True)
    subprocess.run([*git, "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        [*git, "commit", "-q", "-m", "init", "--no-gpg-sign"], check=True, capture_output=True
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
async def test_end_to_end_through_the_tool_on_a_real_checkout(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _checkout(root)
    config = AgenticIdeConfig(
        test_commands={"*": [sys.executable, "-c", "print('1 passed')"]}
    )
    tool = WorkspaceTestTool(FakeCheckoutOrchestrator(str(root), config=config))
    result = await tool.execute({"workspace_id": "w1"}, _ctx())
    assert result.success is True
    assert result.output["passed"] is True and len(result.output["commit"]) == 40


# ------------------------------------------------------------- registration


def test_it_is_offered_wherever_workspace_git_is() -> None:
    manager = SimpleNamespace(_tools={})
    gateway = BrainSupervisorToolGateway(
        manager, workspace_tool=WorkspaceOrchestrationTool(_RecordingGateway({}))
    )
    offered = gateway.with_addressed_workspace_tool({})
    assert WorkspaceGitTool.name in offered
    assert isinstance(offered.get(WorkspaceTestTool.name), WorkspaceTestTool)
    # Both address the same orchestrator, so both resolve the same workspace.
    assert offered[WorkspaceTestTool.name].gateway is offered[WorkspaceGitTool.name].gateway
