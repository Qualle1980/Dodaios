"""The typed chat starts where the person's files are, like the spoken turn does.

Live forensic (2026-10-30, the maintainer's own install): asked about their own
files, the Jarvis chat answered that it could not reach them, while the same
question spoken was answered correctly. Both surfaces run the same brain.

The cause was never permission. ``agent_chat.tools.resolve_path`` passes an
absolute path through untouched, so the chat could always have read the file;
the spoken turn reaches the disk through ``run-shell``, which inherits the
app's own working directory. What differed is where each surface *starts*:
the chat's folder tools were rooted at ``chat_workspace_dir()``, a folder
under ``AppData\\Local`` that holds nothing the person put there. ``Ls`` with
no argument, a relative path, and every glob therefore described an empty
scratch directory -- so the model concluded, reasonably, that it had no access.

A chat that picked its own folder keeps it. Only the default moves.
"""
from __future__ import annotations

from pathlib import Path

from jarvis.agent_chat.surface_kits import _jarvis_tools


class _Session:
    """Just the attributes the kit reads."""

    def __init__(self, cwd: str = "", permission_mode: str = "") -> None:
        self.cwd = cwd
        self.permission_mode = permission_mode
        self.session_id = "s1"


def _root_of(tools: dict) -> Path:
    return tools["Read"].cwd


def test_a_chat_that_chose_a_folder_keeps_it() -> None:
    chosen = Path.home() / "some-project"
    tools = _jarvis_tools(None, None, _Session(cwd=str(chosen)))
    assert _root_of(tools) == chosen


def test_the_default_is_the_persons_own_home_not_a_hidden_scratch_folder() -> None:
    from jarvis.core.paths import chat_workspace_dir

    tools = _jarvis_tools(None, None, _Session())
    root = _root_of(tools)

    assert root == Path.home()
    assert root != chat_workspace_dir()
    # The guarantee in one sentence: listing the folder with no argument shows
    # the person their own files.
    assert root.is_dir()


def test_plan_mode_still_gets_no_writing_hands() -> None:
    """The default moving must not quietly widen what plan mode may do."""
    tools = _jarvis_tools(None, None, _Session(permission_mode="plan"))
    assert "Read" in tools
    assert "Write" not in tools
    assert "Edit" not in tools
