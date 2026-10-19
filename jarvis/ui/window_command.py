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


def native_drag_available(window: Any, *, platform: str | None = None) -> bool:
    """Whether :func:`begin_native_move` can actually work on this host.

    The page asks before it gives up pywebview's drag region, so a host without
    a marshalling UI thread keeps a working drag instead of none. Same three
    conditions the move itself needs: Windows, a form that can invoke, and
    pythonnet to build the delegate.
    """
    if (platform if platform is not None else sys.platform) != "win32":
        return False
    if not callable(getattr(getattr(window, "native", None), "BeginInvoke", None)):
        return False
    try:
        from System import Action  # noqa: F401, PLC0415 — availability probe
    except ImportError:  # no .NET runtime: this probe's answer IS the report
        return False
    return True


def _grab_caption(hwnd: int) -> None:
    """The two Win32 calls, to be run ON THE WINDOW'S OWN THREAD."""
    import ctypes  # noqa: PLC0415 — Win32 only, never imported on other hosts

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    # ReleaseCapture frees the capture of the CALLING thread. While the WebView
    # still owns the mouse, the system's move loop exits immediately and the
    # window cannot be dragged at all.
    user32.ReleaseCapture()
    user32.SendMessageW(ctypes.c_void_p(int(hwnd)), _WM_NCLBUTTONDOWN, _HTCAPTION, 0)


def begin_native_move(
    window: Any, hwnd: int | None, *, platform: str | None = None
) -> dict[str, Any]:
    """Hand an in-progress caption drag to Windows' own window-move loop.

    The page otherwise drags through pywebview's drag region, which repositions
    the window from JavaScript. Windows never learns that a move is happening,
    so Snap does nothing: no edge preview, no half-screen drop, no shake.
    Releasing the mouse capture and posting ``WM_NCLBUTTONDOWN`` with
    ``HTCAPTION`` instead tells the window manager "the user just grabbed your
    title bar", and it runs its native modal move loop with every snap gesture
    included.

    THREAD AFFINITY IS THE WHOLE POINT. Both calls belong to the UI thread:
    ``ReleaseCapture`` only frees the capture of the thread that calls it, and
    the move loop must run where the window pumps messages. A first attempt ran
    them on the request's worker thread, which left the capture with the WebView
    and the window completely unmovable (live 2026-10-19). The WinForms form is
    therefore asked to invoke them: ``BeginInvoke`` so the HTTP request returns
    at once instead of hanging for as long as the user drags.

    The window frame returns ``HTCLIENT`` for the caption strip on purpose
    (``window_frame``), so the page keeps its own caption buttons; this is the
    one place that asks for caption behaviour, and only for the strip the page
    nominates.

    Non-Windows hosts report ``unsupported_platform`` rather than silently
    doing nothing: macOS drags its native title bar itself, and the page never
    draws a drag strip there. A host whose window cannot marshal onto its UI
    thread reports ``no_ui_thread``, so the page can keep the pywebview drag
    rather than losing the gesture altogether.
    """
    plat = platform if platform is not None else sys.platform
    if plat != "win32":
        return {"ok": False, "reason": "unsupported_platform"}
    if not hwnd:
        return {"ok": False, "reason": "no_window"}

    form = getattr(window, "native", None)
    invoke = getattr(form, "BeginInvoke", None)
    if form is None or not callable(invoke):
        return {"ok": False, "reason": "no_ui_thread"}
    try:
        from System import Action  # noqa: PLC0415 — pythonnet, WinForms host only
    except ImportError:
        # Not swallowed: the reason travels back to the page, which then keeps
        # pywebview's drag region. Logging every mousedown on a host without a
        # .NET runtime would be noise, and ``native_drag_available`` already
        # told the page this would happen.
        return {"ok": False, "reason": "no_ui_thread"}

    handle = int(hwnd)
    invoke(Action(lambda: _grab_caption(handle)))
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
