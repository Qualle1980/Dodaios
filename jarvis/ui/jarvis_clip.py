"""The JARVIS HUD clip as frames for the desktop overlay and the Jarvis Bar.

The sprite sheet ``assets/jarvis_clip.png`` holds the boot clip's HUD ring as
square frames side by side. Both desktop surfaces composite over an exact
colour key, so every frame is cut to a hard-edged circle (no feathered alpha).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw

log = logging.getLogger(__name__)

CLIP_PATH = Path(__file__).resolve().parent / "assets" / "jarvis_clip.png"

#: Native clip rate; busy modes play it faster so the ring visibly works.
_FPS = 24.0
_MODE_SPEED = {"think": 2.0, "dictate_transcribing": 2.0, "speak": 1.5, "connect": 1.5}


@lru_cache(maxsize=1)
def _sheet() -> tuple[Image.Image, ...]:
    sheet = Image.open(CLIP_PATH).convert("RGB")
    side = sheet.height
    count = sheet.width // side
    return tuple(sheet.crop((i * side, 0, (i + 1) * side, side)) for i in range(count))


def frame_count() -> int:
    return len(_sheet())


def frame_index(t: float, mode: str = "idle") -> int:
    """Which frame plays at time *t* (seconds) in *mode*."""
    return int(t * _FPS * _MODE_SPEED.get(mode, 1.0)) % frame_count()


@lru_cache(maxsize=8)
def circle_mask(diameter: int) -> Image.Image:
    """Binary disc mask: the colour key needs hard edges."""
    mask = Image.new("L", (diameter, diameter), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, diameter - 1, diameter - 1), fill=255)
    return mask


@lru_cache(maxsize=128)
def _scaled(index: int, diameter: int) -> Image.Image:
    return _sheet()[index].resize((diameter, diameter), Image.Resampling.LANCZOS)


def clip_disc(t: float, mode: str, diameter: int) -> Image.Image:
    """The frame for *t* as an RGB square; paste it with ``circle_mask``."""
    return _scaled(frame_index(t, mode), max(1, int(diameter)))


class JarvisClipRenderer:
    """Overlay renderer that plays the HUD clip in a circle over the colour key."""

    def __init__(self, size: int, color_key: tuple[int, int, int]) -> None:
        self._size = int(size)
        self._color_key = color_key
        _sheet()  # fail at build time so the caller can fall back

    def render(self, t: float, mode: str, ext_level: float | None) -> Image.Image:
        frame = Image.new("RGB", (self._size, self._size), self._color_key)
        frame.paste(clip_disc(t, mode, self._size), (0, 0), circle_mask(self._size))
        return frame

    def start_mouth_anim(self, duration_s: float, t_now: float) -> None:
        """The clip has no mouth; speaking only speeds the ring up."""

    def stop_mouth_anim(self) -> None:
        """See ``start_mouth_anim``."""
