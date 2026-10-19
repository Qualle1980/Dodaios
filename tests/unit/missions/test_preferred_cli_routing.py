"""An explicitly requested worker family reaches the worker.

``Step.preferred_cli`` carries the family a routed task asked for via the
prompt's ``Zielort/Kontext:`` line. It may swap between the two subscription
CLIs and nothing else: the 2026-05-29 mandate (a deliberately configured
provider is the single source of truth, and Gemini is never a silent fallback)
must still hold for every other provider.
"""

from __future__ import annotations

import pytest

from jarvis.missions.init import _select_subagent_worker_kind
from jarvis.missions.kontrollierer.decomposer import Step
from jarvis.missions.stream_evidence import TARGET_CLI_MARKER, parse_target_cli


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (f"Aufgabe: x\n{TARGET_CLI_MARKER} codex.\nmehr", "codex"),
        (f"{TARGET_CLI_MARKER} claude", "claude"),
        (f"{TARGET_CLI_MARKER} CODEX.", "codex"),
        # A mention inside the task text must never redirect a worker.
        ("Bitte nutze codex fuer diese Aufgabe.", None),  # i18n-allow: quotes a German chat turn
        (f"{TARGET_CLI_MARKER} gemini.", None),
        ("", None),
    ],
)
def test_only_an_exact_marker_line_counts(prompt: str, expected: str | None) -> None:
    assert parse_target_cli(prompt) == expected


def test_step_has_no_preference_by_default() -> None:
    assert Step(slug="s", prompt="p").preferred_cli is None


@pytest.mark.parametrize(
    ("provider", "preferred", "expected"),
    [
        # The new capability: swap between the two subscription CLIs.
        ("openai-codex", "claude", "claude_direct"),
        ("claude-api", "codex", "codex_direct"),
        # Same family as configured: unchanged.
        ("openai-codex", "codex", "codex_direct"),
        ("claude-api", "claude", "claude_direct"),
        # No preference: every decision stays exactly as before.
        ("openai-codex", None, "codex_direct"),
        ("claude-api", None, "claude_direct"),
        # The mandate holds: a deliberately configured provider is never
        # diverted, and the Gemini API key is never reached by a request.
        ("gemini", "claude", "gemini"),
        ("openai", "claude", "api_agent"),
        ("openrouter", "codex", "api_agent"),
        (None, "claude", "subjarvis"),
        # An unknown family is ignored rather than guessed at.
        ("openai-codex", "gemini", "codex_direct"),
        ("openai-codex", "", "codex_direct"),
    ],
)
def test_worker_kind_honours_only_a_valid_cli_swap(
    provider: str | None, preferred: str | None, expected: str
) -> None:
    assert _select_subagent_worker_kind(provider, "", preferred) == expected


def test_the_old_two_argument_call_is_unchanged() -> None:
    """Existing callers must keep their behaviour without passing the new arg."""
    assert _select_subagent_worker_kind("openai-codex", "") == "codex_direct"
    assert _select_subagent_worker_kind("claude-api", "sonnet") == "claude_direct"
