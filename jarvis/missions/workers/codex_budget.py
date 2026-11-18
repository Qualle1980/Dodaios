"""Per-run budget for one ``codex exec`` run.

Why: a single long Codex run can make hundreds of model calls, and each call
re-sends the whole conversation. Measured on 2026-11-13: one session with 321
model calls used about 2.9 million tokens, most of the account total for that
week. Codex itself has no per-run cap, so the worker enforces one.

Two limits, both read from the event stream the worker already parses:

* tokens: input plus output, summed over ``turn.completed`` usage records;
* tools: ``command_execution`` items, i.e. shell calls the model made.

Crossing either limit stops the run. The worker kills the process and reports
the reason; nothing is retried automatically, so the cap cannot hide a loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: Input plus output tokens one run may spend. Chosen above the largest normal
#: run measured on this machine (about 0.1 million) and far below the 2.9
#: million of the 321-call session.
CODEX_RUN_TOKEN_CAP = 1_500_000

#: Shell calls one run may make before it is stopped.
CODEX_RUN_TOOL_CAP = 120


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


@dataclass
class CodexRunBudget:
    token_cap: int = CODEX_RUN_TOKEN_CAP
    tool_cap: int = CODEX_RUN_TOOL_CAP
    tokens: int = 0
    tools: int = 0
    turns: int = 0

    def observe_turn_usage(self, usage: Mapping[str, Any] | None) -> None:
        """Add one ``turn.completed`` usage record. Missing fields count as zero."""
        self.turns += 1
        if not isinstance(usage, Mapping):
            return
        self.tokens += _int(usage.get("input_tokens")) + _int(usage.get("output_tokens"))

    def observe_tool_item(self, item: Mapping[str, Any] | None) -> None:
        """Count one ``command_execution`` item (a shell call)."""
        if isinstance(item, Mapping) and item.get("type") == "command_execution":
            self.tools += 1

    def stop_reason(self) -> str | None:
        """The first limit crossed, as a message for the operator, else ``None``."""
        if self.tokens > self.token_cap:
            return (
                f"codex run stopped: {self.tokens} tokens over {self.turns} turn(s) "
                f"exceeds the per-run cap of {self.token_cap}"
            )
        if self.tools > self.tool_cap:
            return (
                f"codex run stopped: {self.tools} shell calls exceed the per-run cap "
                f"of {self.tool_cap}"
            )
        return None
