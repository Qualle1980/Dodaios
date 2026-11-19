"""Pane input-line readiness must see markers inside box-drawing borders.

Grok draws ``│ >   │``; Claude Code draws ``❯`` + NBSP between ``─`` rules.
Matching only when the stripped line *starts* with the marker never sees
those rows, so the pane never becomes ready and the brief is never typed.
"""

from __future__ import annotations

from typing import Any

from jarvis.agentic_ide.fleet_actions import input_line_visible
from jarvis.agentic_ide.session import _input_line_holds


class _Pane:
    """Minimal live pane whose screen is the given rows."""

    def __init__(self, lines: tuple[str, ...], *, agent: str = "grok-build") -> None:
        self.agent = agent
        self.status = "live"
        self.pty_id = "pty-1"
        self._lines = lines

    @property
    def transcript(self) -> Any:
        lines = self._lines

        class _T:
            class _Screen:
                @staticmethod
                def display() -> tuple[str, ...]:
                    return lines

            screen = _Screen()

            @staticmethod
            def tail(_n: int) -> tuple[str, ...]:
                return lines

        return _T()


def test_grok_bordered_input_row_is_ready() -> None:
    """The grok-build composer row sits inside a box: the marker is not at column 0."""
    pane = _Pane(("some banner", "│ >   │"), agent="grok-build")
    assert input_line_visible(pane) is True


def test_claude_nbsp_row_between_rules_is_ready() -> None:
    """Claude Code's input row is ``❯`` + NBSP between two horizontal rules."""
    pane = _Pane(
        ("────────", "❯\u00a0", "────────"),
        agent="claude",
    )
    assert input_line_visible(pane) is True


def test_old_prompt_above_plain_bottom_is_not_ready() -> None:
    """A past ``>`` in the transcript is not the live input line."""
    pane = _Pane(("> old prompt", "status ok"), agent="grok-build")
    assert input_line_visible(pane) is False


def test_grok_consent_banner_is_not_ready() -> None:
    """The Opt-in banner has no composer; readiness must stay false."""
    pane = _Pane(
        ("Help improve Grok [Opt out] [Opt in]",),
        agent="grok-build",
    )
    assert input_line_visible(pane) is False


def test_plain_empty_screen_is_not_ready() -> None:
    pane = _Pane(("",), agent="grok-build")
    assert input_line_visible(pane) is False


def test_input_line_holds_sees_through_grok_borders() -> None:
    """Submit confirmation must read text inside ``│ > … │``."""
    assert _input_line_holds(["│ > analyse the routing │"], "analyse the routing") is True


def test_input_line_holds_sees_claude_nbsp_marker() -> None:
    assert _input_line_holds(["────────", "❯\u00a0analyse the routing", "────────"], "analyse the routing") is True
