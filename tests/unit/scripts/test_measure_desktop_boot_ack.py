"""The boot harness must reject its own run when the paint ack did not land.

Forensic 2026-10-19: the harness acknowledged the window's first paint with a
bare POST to ``/api/ui/shell-painted``. The surface boundary answers that route
only for a same-origin, browser-shaped request, so the ack was refused with
403, the bootstrap's paint event stayed unset, and every headless run silently
paid the full 12 s ``wait_shell_painted`` fallback — reported as a boot cost.

These checks are deliberately STATUS-based: a timing difference is exactly the
evidence that misled us, so a refused ack must fail on the HTTP status alone,
with no run and no clock involved.
"""

import pytest

from scripts.measure_desktop_boot import check_paint_ack


def test_accepted_ack_passes() -> None:
    # A 204 records nothing, so a clean result carries neither key.
    check_paint_ack({"wall_ms": 1200.0, "phases": {}})


@pytest.mark.parametrize("status", [403, 404, 405, 500])
def test_any_non_204_ack_invalidates_the_run(status: int) -> None:
    with pytest.raises(RuntimeError, match="not accepted"):
        check_paint_ack({"wall_ms": 1200.0, "paint_ack_status": status})


def test_failed_ack_request_invalidates_the_run() -> None:
    with pytest.raises(RuntimeError, match="request failed"):
        check_paint_ack({"wall_ms": 1200.0, "paint_ack_error": "URLError(timeout)"})


def test_the_message_names_the_cause_not_the_symptom() -> None:
    # Whoever reads this failure must look at the headers and the boundary,
    # never at the startup path: the run is invalid, not slow.
    with pytest.raises(RuntimeError) as excinfo:
        check_paint_ack({"paint_ack_status": 403})

    message = str(excinfo.value)
    assert "invalid, not slow" in message
    assert "Origin" in message
    assert "surface_security.py" in message
