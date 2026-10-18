"""JARVIS HUD clip frames for the desktop overlay and the Jarvis Bar."""
from __future__ import annotations

from jarvis.ui import jarvis_clip
from jarvis.ui.jarvis_clip import JarvisClipRenderer

_KEY = (255, 0, 255)


def test_sheet_ships_the_whole_clip() -> None:
    assert jarvis_clip.frame_count() == 35


def test_frames_cycle_and_busy_modes_play_faster() -> None:
    assert jarvis_clip.frame_index(0.0) == 0
    assert jarvis_clip.frame_index(1 / 24) == 1
    assert jarvis_clip.frame_index(35 / 24) == 0
    assert jarvis_clip.frame_index(1 / 24, "think") == 2


def test_renderer_keeps_corners_on_the_colour_key() -> None:
    frame = JarvisClipRenderer(size=96, color_key=_KEY).render(0.5, "idle", None)
    assert frame.size == (96, 96)
    assert frame.getpixel((0, 0)) == _KEY
    assert frame.getpixel((95, 95)) == _KEY
    assert frame.getpixel((48, 48)) != _KEY


def test_circle_mask_is_binary() -> None:
    assert set(jarvis_clip.circle_mask(40).getdata()) == {0, 255}
