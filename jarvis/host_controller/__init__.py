"""Instant host controller: validated intents → auditable GitHub actions.

ChatGPT Instant is text-only. The host parses a strict JSON intent, keeps
project state in GitHub (milestone/issues), dispatches workers, requires an
independent Codex review on an exact commit SHA, and recovers from GitHub on
restart. Ideas adapted from Temporal event-history recovery, LangGraph
interrupts/gates, and Prefect named wait states — no third-party runtime.
"""

from __future__ import annotations

from jarvis.host_controller.controller import HostController, HostResult
from jarvis.host_controller.gh_store import GhProjectStore, HostIssue, HostMilestone
from jarvis.host_controller.instant import handle_instant_text
from jarvis.host_controller.intent import IntentParseError, ParsedIntent, parse_intent
from jarvis.host_controller.workers import (
    HostWorkerBundle,
    ReviewResult,
    WorkerResult,
    build_live_host_workers,
)

__all__ = [
    "GhProjectStore",
    "HostController",
    "HostIssue",
    "HostMilestone",
    "HostResult",
    "HostWorkerBundle",
    "IntentParseError",
    "ParsedIntent",
    "ReviewResult",
    "WorkerResult",
    "build_live_host_workers",
    "handle_instant_text",
    "parse_intent",
]
