"""Per-project run caps and quota wait. Never silent paid-key fallback.

Mirrors the honesty of ``jarvis.agent_chat.delegation_budget``: when a
subscription window is spent, new work waits; it does not fall onto an API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectBudget:
    max_runs: int
    runs_used: dict[str, int] = field(default_factory=dict)

    def remaining(self, project: str) -> int:
        used = self.runs_used.get(project, 0)
        return max(0, self.max_runs - used)

    def consume(self, project: str) -> bool:
        if self.remaining(project) <= 0:
            return False
        self.runs_used[project] = self.runs_used.get(project, 0) + 1
        return True
