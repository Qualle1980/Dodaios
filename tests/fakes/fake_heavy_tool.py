"""A tool of a chosen name whose schema weighs a chosen number of bytes.

Some surfaces fit their tool declarations into a BYTE budget, not a count
(``jarvis.live.tools._CATALOG_BYTE_BUDGET``). A test that only varies how
MANY tools exist cannot reach such a cut: one real tool with a large schema
ends the budget where twenty small ones do not. Live 2026-11-09 the live
catalog declared 13 of 146 tools, and the app-steering group alone carried
about 17 KB of the 24 KB -- a weight no fixture reproduced.

So this fake takes the one property that matters there: its wire size.
"""

from typing import Any

from jarvis.core.protocols import ExecutionContext, ToolResult


class FakeHeavyTool:
    """A side-effect-free tool that costs ``weight_bytes`` on the wire."""

    def __init__(self, name: str, weight_bytes: int, risk_tier: str = "safe") -> None:
        self.name = name
        self.risk_tier = risk_tier
        self.description = f"{name}: a tool whose declaration is deliberately large."
        # The padding lives in a property description because that is where a
        # real tool's bulk sits too -- workspace-orchestrate's own schema is
        # 8.6 KB of documented enum values and per-action prose.
        padding = {"type": "string", "description": "x" * max(0, weight_bytes)}
        self.schema: dict[str, Any] = {
            "type": "object",
            "properties": {"padding": padding},
            "required": [],
        }
        self.calls: list[ExecutionContext] = []

    async def execute(self, args: dict[str, Any], ctx: ExecutionContext) -> ToolResult:
        self.calls.append(ctx)
        return ToolResult(success=True, output={"calls": len(self.calls)})
