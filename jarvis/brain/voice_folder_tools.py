"""Read-only project file access for Live. No guessed cwd or drive-wide scan."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.core.protocols import ExecutionContext, ToolResult

_READS = frozenset({"Read", "Ls", "Glob", "Grep"})


class ScopedVoiceReadTool:
    """Validate every requested path before handing off to the existing folder tool."""

    risk_tier = "safe"
    read_only = True

    def __init__(self, inner: Any, root: Path) -> None:
        self._inner = inner
        self._root = root.resolve()
        self.name = inner.name
        self.description = (
            f"{inner.description} Only inside the selected project; "
            "relative paths start at the project root."
        )
        self.schema = inner.schema

    async def execute(self, args: dict[str, Any], ctx: ExecutionContext) -> ToolResult:
        args = dict(args or {})
        key = "file_path" if self.name == "Read" else "path"
        raw = args.get(key) or "."
        if not isinstance(raw, str) or not raw.strip():
            return ToolResult(success=False, output=None, error="A project-relative path is required.")
        try:
            candidate = (self._root / raw).resolve()
            candidate.relative_to(self._root)
        except (ValueError, OSError, RuntimeError):
            return ToolResult(
                success=False, output=None,
                error="Path is outside the selected project.",
            )
        args[key] = str(candidate)
        return await self._inner.execute(args, ctx)


def scoped_voice_read_tools(tools: dict[str, Any], root: Path) -> dict[str, ScopedVoiceReadTool]:
    """Never surface Write, Edit or RunCommand to Live through this bridge."""
    return {
        name: ScopedVoiceReadTool(tool, root)
        for name, tool in tools.items()
        if name in _READS
    }
