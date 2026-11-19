"""Run a coding workspace's configured test suite: the General's only way to test.

The General holds no shell. Verifying delegated coding work still needs a test
run, so this tool runs exactly one thing: the suite the user configured for the
project (``[agentic_ide.test_commands]``), inside the workspace's own checkout.
Its schema has no field for a program, and :meth:`execute` forwards only the
workspace selector, so nothing a model writes can become a command.
"""

from __future__ import annotations

from jarvis.core.protocols import ExecutionContext, ToolResult, WorkspaceOrchestrationGateway

#: The only arguments that reach the gateway. Everything else is dropped here,
#: not merely rejected by the schema, because not every caller validates it.
_SELECTORS = ("project", "workspace", "project_id", "workspace_id")


class WorkspaceTestTool:
    name = "workspace-test"
    # Running the user's own configured suite changes no repository state.
    risk_tier = "safe"
    read_only = True
    is_action_tool = False
    description = (
        "Run a coding workspace's test suite, the command the user configured for its "
        "project, and report passed or failed, the exit code, the commit it ran on and "
        "the last lines of output. You cannot choose the command. Name the workspace as "
        "in workspace-orchestrate (workspace or workspace_id); use the delegated job's "
        "workspace_id to verify that job. A refusal says why and ran nothing: pass it on."
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {key: {"type": "string"} for key in _SELECTORS},
    }

    def __init__(self, gateway: WorkspaceOrchestrationGateway) -> None:
        self.gateway = gateway

    def read_only_for_args(self, args: dict) -> bool:
        return True

    def describe_args(self, args: dict) -> dict:
        return {
            "level": "read",
            "project": str(args.get("project_id") or args.get("project") or ""),
            "workspace": str(args.get("workspace_id") or args.get("workspace") or ""),
            "agent": "",
            "task": "run the configured tests",
        }

    async def execute(self, args: dict, ctx: ExecutionContext) -> ToolResult:
        selector = {key: args[key] for key in _SELECTORS if args.get(key)}
        try:
            result = await self.gateway.run(
                {**selector, "action": "test"}, trace_id=str(ctx.trace_id)
            )
        except ValueError as exc:
            # Invalid arguments go back to the model as the tool error.
            return ToolResult(success=False, output=None, error=str(exc))
        # A suite that ran and failed is a successful read whose answer is
        # "failed"; only a run that did not happen is a tool failure.
        return ToolResult(
            success=result.get("success") is not False and result.get("status") == "ok",
            output=result,
        )
