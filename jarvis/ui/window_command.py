"""Minimize, maximize, close and drag for the desktop window, from the page.

The page draws those buttons itself once the native title bar is gone
(``frameless``). The methods are the same ones the operating system would
call, so closing still runs the window's normal close path.

Dragging is different: it must be handed to Windows rather than carried out by
us. See :func:`begin_native_move`.

Headless hosts have no window. Callers get ``ok: false`` and do not pretend
a button was pressed.
"""

from __future__ import annotations

import sys
from typing import Any

_ACTIONS = frozenset({"minimize", "maximize", "close"})

# Win32 caption drag. Handing the gesture to the window manager is what makes
# Snap, Aero shake and the edge/corner previews work: those are features of the
# system's own move loop, and nothing we do from the page can imitate them.
_WM_NCLBUTTONDOWN = 0x00A1
_HTCAPTION = 2


def begin_native_move(hwnd: int | None, *, platform: str | None = None) -> dict[str, Any]:
    """Hand an in-progress caption drag to Windows' own window-move loop.

    The page used to drag through pywebview's drag region, which repositions
    the window from JavaScript. Windows never learns that a move is happening,
    so Snap does nothing: no edge preview, no half-screen drop, no shake.
    Releasing the mouse capture and posting ``WM_NCLBUTTONDOWN`` with
    ``HTCAPTION`` instead tells the window manager "the user just grabbed your
    title bar", and it runs its native modal move loop with every snap gesture
    included.

    ``SendMessageW`` does not return until that loop ends (the user drops the
    window), which is why the caller must be on a worker thread.

    The window frame returns ``HTCLIENT`` for the caption strip on purpose
    (``window_frame``), so the page keeps its own caption buttons; this is the
    one place that asks for caption behaviour, and only for the strip the page
    nominates.

    Non-Windows hosts report ``unsupported_platform`` rather than silently
    doing nothing: macOS drags its native title bar itself, and the page never
    draws a drag strip there.
    """
    plat = platform if platform is not None else sys.platform
    if plat != "win32":
        return {"ok": False, "reason": "unsupported_platform"}
    if not hwnd:
        return {"ok": False, "reason": "no_window"}
    import ctypes  # noqa: PLC0415 — Win32 only, never imported on other hosts

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    # Without ReleaseCapture the WebView still owns the mouse and the move loop
    # exits immediately, which looks exactly like a drag that does nothing.
    user32.ReleaseCapture()
    user32.SendMessageW(ctypes.c_void_p(int(hwnd)), _WM_NCLBUTTONDOWN, _HTCAPTION, 0)
    return {"ok": True}


def window_chrome(platform: str | None = None) -> dict[str, Any]:
    """Where the page should put the window buttons.

    macOS uses Cocoa's native traffic lights, so the page draws no controls.
    Windows and Linux keep their web controls on the right. ``frameless`` is
    filled in by the desktop shell from the live window.
    """
    plat = platform if platform is not None else sys.platform
    if plat == "darwin":
        name, controls = "darwin", "none"
    elif plat == "win32":
        name, controls = "windows", "trailing"
    else:
        name, controls = "linux", "trailing"
    return {"ok": True, "frameless": False, "platform": name, "controls": controls}


def run_window_command(
    window: Any,
    action: str,
    *,
    maximized: bool | None = None,
) -> dict[str, Any]:
    """Run one caption action on ``window``.

    ``maximized`` is the operating system's answer when the caller has one
    (Windows ``IsZoomed``). Without it, the last action this process took is
    the fallback answer. macOS uses native caption controls instead of this
    web-caption toggle: Cocoa's restore API only deminiaturizes a window.
    """
    if window is None:
        return {"ok": False, "reason": "no_window"}
    if action not in _ACTIONS:
        return {"ok": False, "reason": "unknown_action"}
    if action == "minimize":
        window.minimize()
        window._jarvis_maximized = False
        return {"ok": True, "maximized": False}
    if action == "close":
        window.destroy()
        return {"ok": True, "maximized": False}
    is_max = (
        maximized
        if maximized is not None
        else bool(getattr(window, "_jarvis_maximized", False))
    )
    if is_max:
        window.restore()
        window._jarvis_maximized = False
        return {"ok": True, "maximized": False}
    window.maximize()
    window._jarvis_maximized = True
    return {"ok": True, "maximized": True}
