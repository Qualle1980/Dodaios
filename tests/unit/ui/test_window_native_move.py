"""Handing a caption drag to the system's own window-move loop.

The page used to drag through pywebview's JavaScript reposition, which Windows
never learns about, so Snap did nothing. ``begin_native_move`` releases the
mouse capture and posts ``WM_NCLBUTTONDOWN``/``HTCAPTION`` instead.

The Win32 call itself cannot be exercised off Windows (and on Windows it blocks
until the user drops the window), so these tests pin the contract around it:
the platform gate, the missing-window answer, and the exact message it sends.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from jarvis.ui.window_command import begin_native_move


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_non_windows_hosts_say_so_instead_of_doing_nothing(platform: str) -> None:
    assert begin_native_move(1234, platform=platform) == {
        "ok": False,
        "reason": "unsupported_platform",
    }


@pytest.mark.parametrize("hwnd", [None, 0])
def test_without_a_window_handle_nothing_is_sent(hwnd: int | None) -> None:
    assert begin_native_move(hwnd, platform="win32") == {"ok": False, "reason": "no_window"}


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 message plumbing")
def test_it_releases_the_capture_then_posts_a_caption_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Order matters: while the WebView still holds the capture, the system's
    move loop exits at once and the drag looks broken."""
    import ctypes

    calls: list[tuple[str, Any]] = []

    class _User32:
        def ReleaseCapture(self) -> int:  # noqa: N802 — Win32 name
            calls.append(("release", None))
            return 1

        def SendMessageW(self, hwnd: Any, msg: int, wparam: int, lparam: int) -> int:  # noqa: N802
            calls.append(("send", (getattr(hwnd, "value", hwnd), msg, wparam, lparam)))
            return 0

    monkeypatch.setattr(ctypes, "windll", type("_W", (), {"user32": _User32()})())

    assert begin_native_move(4242, platform="win32") == {"ok": True}
    assert [name for name, _ in calls] == ["release", "send"]
    # WM_NCLBUTTONDOWN with HTCAPTION: "the user grabbed your title bar".
    assert calls[1][1] == (4242, 0x00A1, 2, 0)
