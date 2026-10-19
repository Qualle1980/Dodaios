"""Handing a caption drag to the system's own window-move loop.

The page otherwise drags through pywebview's JavaScript reposition, which
Windows never learns about, so Snap does nothing. ``begin_native_move``
releases the mouse capture and posts ``WM_NCLBUTTONDOWN``/``HTCAPTION``.

THREAD AFFINITY is what these tests mostly guard. The first version ran both
Win32 calls on the request's worker thread; ``ReleaseCapture`` only frees the
calling thread's capture, so the WebView kept the mouse, the move loop exited
at once, and the window became completely unmovable (live 2026-10-19). The
calls must therefore reach the window's own thread, and the page must be told
when that is impossible so it can keep a drag that works.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from jarvis.ui.window_command import begin_native_move, native_drag_available


class _Form:
    """Stands in for the WinForms form: records what gets marshalled."""

    def __init__(self) -> None:
        self.invoked: list[Any] = []

    def BeginInvoke(self, action: Any) -> None:  # noqa: N802 — WinForms name
        self.invoked.append(action)


class _Window:
    def __init__(self, native: Any) -> None:
        self.native = native


@pytest.fixture
def pythonnet(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stand-in for pythonnet's ``System.Action``.

    ``System`` only becomes importable once pywebview's WinForms backend has
    loaded the .NET runtime. In the live app the form exists precisely because
    that happened; a bare test process has no runtime, so the real code
    correctly answers ``no_ui_thread`` there. These tests want the other
    branch, so they supply the delegate factory.
    """
    module = types.ModuleType("System")
    module.Action = lambda fn: fn  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "System", module)


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_non_windows_hosts_say_so_instead_of_doing_nothing(
    platform: str, pythonnet: None
) -> None:
    window = _Window(_Form())
    assert begin_native_move(window, 1234, platform=platform) == {
        "ok": False,
        "reason": "unsupported_platform",
    }
    assert not window.native.invoked
    assert native_drag_available(window, platform=platform) is False


@pytest.mark.parametrize("hwnd", [None, 0])
def test_without_a_window_handle_nothing_is_sent(hwnd: int | None) -> None:
    window = _Window(_Form())
    assert begin_native_move(window, hwnd, platform="win32") == {
        "ok": False,
        "reason": "no_window",
    }
    assert not window.native.invoked


def test_a_window_that_cannot_marshal_is_reported_not_guessed(pythonnet: None) -> None:
    """The page keeps pywebview's drag region on this answer. Reporting ok here
    is what left the window unmovable."""
    for native in (None, object()):
        window = _Window(native)
        assert begin_native_move(window, 42, platform="win32") == {
            "ok": False,
            "reason": "no_ui_thread",
        }
        assert native_drag_available(window, platform="win32") is False


def test_the_grab_is_marshalled_onto_the_window_thread(pythonnet: None) -> None:
    """Both Win32 calls must leave the caller's thread, or the capture stays
    with the WebView and the move loop exits immediately."""
    window = _Window(_Form())
    assert begin_native_move(window, 4242, platform="win32") == {"ok": True}
    assert len(window.native.invoked) == 1
    assert callable(window.native.invoked[0])


def test_the_marshalled_work_releases_then_posts_a_caption_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Order matters: the capture must be gone before the caption click."""
    import ctypes

    from jarvis.ui import window_command

    calls: list[tuple[str, Any]] = []

    class _User32:
        def ReleaseCapture(self) -> int:  # noqa: N802 — Win32 name
            calls.append(("release", None))
            return 1

        def SendMessageW(self, hwnd: Any, msg: int, wparam: int, lparam: int) -> int:  # noqa: N802
            calls.append(("send", (getattr(hwnd, "value", hwnd), msg, wparam, lparam)))
            return 0

    monkeypatch.setattr(ctypes, "windll", type("_W", (), {"user32": _User32()})())

    window_command._grab_caption(4242)

    assert [name for name, _ in calls] == ["release", "send"]
    # WM_NCLBUTTONDOWN with HTCAPTION: "the user grabbed your title bar".
    assert calls[1][1] == (4242, 0x00A1, 2, 0)


def test_without_pythonnet_the_page_keeps_its_own_drag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No .NET runtime, no delegate — and the page must learn that."""
    monkeypatch.setitem(sys.modules, "System", None)
    window = _Window(_Form())
    assert begin_native_move(window, 42, platform="win32") == {
        "ok": False,
        "reason": "no_ui_thread",
    }
    assert native_drag_available(window, platform="win32") is False
