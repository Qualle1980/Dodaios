"""workspace-orchestrate description stays under the Instant token budget."""

from __future__ import annotations

from types import SimpleNamespace

from jarvis.brain.workspace_tool import WorkspaceOrchestrationTool


def test_workspace_orchestrate_base_description_under_800_tokens() -> None:
    """Measured 2026-11-17: base was ~1398 tokens; keep the trimmed copy under 800."""
    tool = WorkspaceOrchestrationTool(SimpleNamespace())
    tokens = len(tool._BASE_DESCRIPTION) // 4
    assert tokens < 800, f"base description ~{tokens} tokens; Instant tools budget needs <800"
    assert "live" in tool._BASE_DESCRIPTION.lower() or "Live" in tool.description
    assert "create" in tool._BASE_DESCRIPTION.lower()
    assert "while_busy" in tool._BASE_DESCRIPTION or "busy" in tool._BASE_DESCRIPTION.lower()


def test_workspace_orchestrate_full_description_under_1100_tokens() -> None:
    tool = WorkspaceOrchestrationTool(SimpleNamespace())
    # Live blurb adds a few hundred tokens at most; hard cap for the whole field.
    tokens = len(tool.description) // 4
    assert tokens < 1100, f"full description ~{tokens} tokens; expected under 1100 after trim"
