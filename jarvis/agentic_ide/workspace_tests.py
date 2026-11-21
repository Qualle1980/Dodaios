"""Run a coding workspace's own test suite, addressed like every other workspace action.

A request names a workspace (or leaves it out for the visible one), never a
path and never a program. The folder comes from the workspace and the argv
from ``[agentic_ide.test_commands]``, which only the user writes. So a model
that can ask for a test run still cannot run anything else: there is no field
a command could arrive in, and any it sends is never read.

The answer is a verdict, not a transcript: pass or fail, the exit code, the
commit the checkout stood on and the tail of the output. The acceptance gate
(:mod:`jarvis.agent_chat.delegation_acceptance`) reads ``passed``,
``workspace_id`` and ``commit`` from it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from jarvis.core.process_utils import NO_WINDOW_CREATIONFLAGS, resolve_executable

log = logging.getLogger(__name__)

#: A suite that runs longer is reported as unfinished, never as a pass.
TIMEOUT_S = 600.0
#: How much of the output's tail comes back. The verdict is ``passed``; this is
#: what a person needs to see why.
SUMMARY_CHARS = 2000
_SUMMARY_LINES = 25
#: The key that applies to every project without an entry of its own.
_DEFAULT_KEY = "*"
_CONFIG_KEY = "agentic_ide.test_commands"


def configured_command(config: Any, project: dict[str, Any]) -> list[str]:
    """The argv configured for ``project``: its id, then its name, then ``*``."""
    table = dict(getattr(config, "test_commands", None) or {})
    if not table:
        return []
    project_id = str(project.get("id") or "")
    if project_id and project_id in table:
        return list(table[project_id])
    name = str(project.get("name") or "").casefold()
    if name:
        for key, argv in table.items():
            if key != _DEFAULT_KEY and key.casefold() == name:
                return list(argv)
    return list(table.get(_DEFAULT_KEY) or [])


def _load_config() -> Any:
    from jarvis.core.config import load_config

    return load_config().agentic_ide


def _summary(output: str) -> str:
    lines = [line.rstrip() for line in str(output or "").splitlines() if line.strip()]
    tail = "\n".join(lines[-_SUMMARY_LINES:])
    return tail[-SUMMARY_CHARS:]


def _commit(folder: str) -> str:
    from .git_local import git_out

    return (git_out(["rev-parse", "HEAD"], Path(folder)) or "").strip()


def _execute(argv: list[str], folder: str, timeout_s: float) -> dict[str, Any]:
    """Run the suite on the worker thread. Never through a shell."""
    env = dict(os.environ)
    # A Python suite on Windows would otherwise print in the console code page.
    env.setdefault("PYTHONIOENCODING", "utf-8")
    commit = _commit(folder)
    try:
        done = subprocess.run(
            [resolve_executable(argv[0]), *argv[1:]],
            cwd=folder,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
            env=env,
            creationflags=NO_WINDOW_CREATIONFLAGS,
        )
    except FileNotFoundError:
        log.info("workspace-test: configured program %r not found; nothing ran", argv[0])
        return {
            "status": "not_accepted", "success": False, "code": "missing",
            "reason": f"The configured test program {argv[0]!r} is not installed or not on PATH.",
        }
    except subprocess.TimeoutExpired:
        log.info("workspace-test: suite in %s exceeded %gs; reported as unfinished", folder,
                 timeout_s)
        return {
            "status": "not_accepted", "success": False, "code": "timeout",
            "reason": f"The test suite did not finish within {timeout_s:g} seconds; "
                      "nothing is known about whether it passes.",
        }
    return {
        "status": "ok",
        "passed": done.returncode == 0,
        "exit_code": done.returncode,
        "commit": commit,
        "summary": _summary(done.stdout),
    }


async def run(
    orchestrator: Any,
    args: dict[str, Any],
    *,
    config: Any = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Resolve the workspace a request names and run its configured suite there."""
    graph = await asyncio.to_thread(orchestrator.graph)
    picked = orchestrator._workspace(args, graph, "")
    if isinstance(picked, dict):
        return picked
    project, workspace = picked
    target = {
        "project_id": project["id"],
        "workspace_id": workspace["id"],
        "workspace": workspace["name"],
    }
    settings = config if config is not None else await asyncio.to_thread(_load_config)
    argv = configured_command(settings, project)
    if not argv:
        return {
            "status": "not_accepted", "success": False, "code": "no_test_command",
            "reason": (
                f"No test command is configured for {project.get('name') or project['id']}. "
                f"The user sets one under {_CONFIG_KEY} (by project id, name or '*'); "
                "nothing was run."
            ),
            **target,
        }
    result = await asyncio.to_thread(
        _execute, argv, str(workspace["folder"]), float(timeout_s or TIMEOUT_S)
    )
    return {**result, "op": "test", "command": " ".join(argv), **target}


__all__ = ["SUMMARY_CHARS", "TIMEOUT_S", "configured_command", "run"]
