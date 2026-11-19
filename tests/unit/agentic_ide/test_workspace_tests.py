"""A workspace's test suite is run by naming the workspace; the command is the user's.

The model picks a workspace, never a program: what runs is the argv the user
configured under ``[agentic_ide.test_commands]``. These tests run real
processes (the test interpreter itself) in a real git checkout.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from jarvis.agentic_ide import workspace_tests
from jarvis.core.config import AgenticIdeConfig
from tests.fakes.fake_workspace_checkout import FakeCheckoutOrchestrator

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _checkout(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    git = ("git", "-C", str(root))
    subprocess.run([*git, "init", "-q"], check=True, capture_output=True)
    subprocess.run([*git, "config", "user.email", "t@example.invalid"], check=True)
    subprocess.run([*git, "config", "user.name", "Test"], check=True)
    subprocess.run([*git, "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        [*git, "commit", "-q", "-m", "init", "--no-gpg-sign"], check=True, capture_output=True
    )
    return subprocess.run(
        [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _config(*argv: str, key: str = "*") -> AgenticIdeConfig:
    return AgenticIdeConfig(test_commands={key: list(argv)})


def _same_folder(printed: str, folder: Path) -> bool:
    return Path(printed).resolve() == folder.resolve()


def _python(code: str) -> tuple[str, ...]:
    return (sys.executable, "-c", code)


async def test_a_passing_suite_reports_pass_exit_code_and_the_commit(tmp_path) -> None:
    head = _checkout(tmp_path / "repo")
    orchestrator = FakeCheckoutOrchestrator(
        str(tmp_path / "repo"), config=_config(*_python("print('3 passed in 0.1s')"))
    )
    result = await orchestrator.run({"action": "test", "workspace_id": "w1"})
    assert result["status"] == "ok", result
    assert result["passed"] is True and result["exit_code"] == 0
    assert result["commit"] == head
    assert result["workspace_id"] == "w1" and result["project_id"] == "p1"
    assert "3 passed" in result["summary"]


async def test_a_failing_suite_reports_fail_with_its_exit_code(tmp_path) -> None:
    _checkout(tmp_path / "repo")
    orchestrator = FakeCheckoutOrchestrator(
        str(tmp_path / "repo"),
        config=_config(*_python("import sys; print('1 failed'); sys.exit(1)")),
    )
    result = await orchestrator.run({"action": "test", "workspace_id": "w1"})
    assert result["status"] == "ok", result
    assert result["passed"] is False and result["exit_code"] == 1
    assert "1 failed" in result["summary"]


async def test_the_suite_runs_inside_the_workspace_checkout(tmp_path) -> None:
    _checkout(tmp_path / "repo")
    orchestrator = FakeCheckoutOrchestrator(
        str(tmp_path / "repo"), config=_config(*_python("import os; print(os.getcwd())"))
    )
    result = await orchestrator.run({"action": "test", "workspace_id": "w1"})
    assert _same_folder(result["summary"].strip().splitlines()[-1], tmp_path / "repo")


async def test_a_command_in_the_arguments_is_never_run(tmp_path) -> None:
    """The model names a workspace; a program it supplies is ignored."""
    _checkout(tmp_path / "repo")
    marker = tmp_path / "repo" / "pwned.txt"
    orchestrator = FakeCheckoutOrchestrator(
        str(tmp_path / "repo"), config=_config(*_python("print('configured')"))
    )
    result = await orchestrator.run(
        {
            "action": "test",
            "workspace_id": "w1",
            "command": f'"{sys.executable}" -c "open(r\'{marker}\', \'w\')"',
            "cmd": ["rm", "-rf", "."],
        }
    )
    assert "configured" in result["summary"]
    assert not marker.exists()


async def test_no_configured_command_runs_nothing_and_says_where_to_set_it(tmp_path) -> None:
    _checkout(tmp_path / "repo")
    orchestrator = FakeCheckoutOrchestrator(str(tmp_path / "repo"), config=AgenticIdeConfig())
    result = await orchestrator.run({"action": "test", "workspace_id": "w1"})
    assert result["status"] == "not_accepted" and result["success"] is False
    assert result["code"] == "no_test_command"
    assert "agentic_ide.test_commands" in result["reason"]
    assert "passed" not in result


async def test_an_unknown_workspace_is_refused_before_anything_runs(tmp_path) -> None:
    _checkout(tmp_path / "repo")
    marker = tmp_path / "ran.txt"
    orchestrator = FakeCheckoutOrchestrator(
        str(tmp_path / "repo"),
        config=_config(*_python(f"open(r'{marker}', 'w').write('x')")),
    )
    result = await orchestrator.run({"action": "test", "workspace_id": "nope"})
    assert result["status"] != "ok" and "passed" not in result
    assert not marker.exists()


async def test_a_suite_that_outlives_its_timeout_is_a_refusal_not_a_pass(tmp_path) -> None:
    _checkout(tmp_path / "repo")
    orchestrator = FakeCheckoutOrchestrator(
        str(tmp_path / "repo"),
        config=_config(*_python("import time; time.sleep(30)")),
        timeout_s=0.5,
    )
    result = await orchestrator.run({"action": "test", "workspace_id": "w1"})
    assert result["status"] == "not_accepted" and result["code"] == "timeout"
    assert "passed" not in result


async def test_a_missing_program_is_a_refusal_with_the_reason(tmp_path) -> None:
    _checkout(tmp_path / "repo")
    orchestrator = FakeCheckoutOrchestrator(
        str(tmp_path / "repo"), config=_config("no-such-test-runner-xyz", "-q")
    )
    result = await orchestrator.run({"action": "test", "workspace_id": "w1"})
    assert result["status"] == "not_accepted" and result["code"] == "missing"
    assert "no-such-test-runner-xyz" in result["reason"]


async def test_the_summary_keeps_the_tail_of_a_long_output(tmp_path) -> None:
    _checkout(tmp_path / "repo")
    code = "for i in range(500): print('line', i)\nprint('5 passed')"
    orchestrator = FakeCheckoutOrchestrator(
        str(tmp_path / "repo"), config=_config(*_python(code))
    )
    result = await orchestrator.run({"action": "test", "workspace_id": "w1"})
    assert result["summary"].rstrip().endswith("5 passed")
    assert "line 0\n" not in result["summary"]
    assert len(result["summary"]) <= workspace_tests.SUMMARY_CHARS


# ------------------------------------------------------- which command applies


def test_a_project_s_own_entry_wins_over_the_default() -> None:
    config = AgenticIdeConfig(
        test_commands={"*": ["pytest"], "p1": ["npm", "test"], "Other": ["go", "test"]}
    )
    project = {"id": "p1", "name": "Bubble", "path": "/x"}
    assert workspace_tests.configured_command(config, project) == ["npm", "test"]


def test_a_project_is_also_found_by_its_name() -> None:
    config = AgenticIdeConfig(test_commands={"*": ["pytest"], "bubble": ["cargo", "test"]})
    project = {"id": "p1", "name": "Bubble", "path": "/x"}
    assert workspace_tests.configured_command(config, project) == ["cargo", "test"]


def test_the_default_applies_when_the_project_has_no_entry() -> None:
    config = AgenticIdeConfig(test_commands={"*": ["pytest", "-q"]})
    assert workspace_tests.configured_command(config, {"id": "p9", "name": "Z"}) == [
        "pytest",
        "-q",
    ]


def test_nothing_configured_means_no_command() -> None:
    assert workspace_tests.configured_command(AgenticIdeConfig(), {"id": "p1"}) == []


def test_a_malformed_entry_is_dropped_instead_of_failing_the_boot() -> None:
    """AP-16 spirit: a bad value degrades to "not configured", never a dead app."""
    config = AgenticIdeConfig(
        test_commands={"*": "pytest -q", "p1": [], "p2": ["pytest", 3], "p3": ["ok"]}
    )
    assert config.test_commands == {"p3": ["ok"]}
