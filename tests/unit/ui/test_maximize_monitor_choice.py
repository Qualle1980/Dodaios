"""Which monitor a maximize belongs on, with two screens at different scales.

Real setup that produced the bug: a 125%-scaled laptop at (0,0,1920,1200) and a
100% monitor at (-474,-1080,1446,0). Minimizing a window that was maximized on
the laptop and reopening it left the bare grey frame -- Windows parks a
minimized window off-screen, ``MonitorFromWindow`` then answered for that
parking spot, the maximize was capped to the OTHER monitor's work area, and the
WebView child landed on that monitor's rectangle, measured as exactly
(-474,-1080,1446,-48). Only on the secondary-scale screen, only maximized.
"""

from __future__ import annotations

from jarvis.ui.window_frame import maximize_reference_rect

LAPTOP = (0, 0, 1920, 1200)
EXTERNAL = (-474, -1080, 1446, 0)
PARKED = (-32000, -32000, -31840, -31972)  # where Windows puts a minimized window


def test_a_minimized_window_is_asked_about_through_its_restore_rect() -> None:
    """The bug: the parked rectangle names the wrong monitor."""
    assert maximize_reference_rect(
        minimized=True, window_rect=PARKED, restore_rect=LAPTOP
    ) == LAPTOP


def test_a_visible_window_is_asked_about_through_its_own_rect() -> None:
    """Dragged to the other screen and maximized there: that screen wins, not
    the one the restore rectangle remembers."""
    assert maximize_reference_rect(
        minimized=False, window_rect=EXTERNAL, restore_rect=LAPTOP
    ) == EXTERNAL


def test_a_missing_window_rect_falls_back_to_the_restore_rect() -> None:
    assert maximize_reference_rect(
        minimized=False, window_rect=None, restore_rect=LAPTOP
    ) == LAPTOP


def test_a_minimized_window_without_a_restore_rect_uses_what_it_has() -> None:
    assert maximize_reference_rect(
        minimized=True, window_rect=PARKED, restore_rect=None
    ) == PARKED


def test_nothing_to_go_on_is_reported_rather_than_guessed() -> None:
    """The caller then leaves the maximize uncapped instead of picking a
    monitor at random."""
    assert maximize_reference_rect(
        minimized=True, window_rect=None, restore_rect=None
    ) is None
