"""The bridge idle reaper: it closes an unused bridge and never a busy one."""
from __future__ import annotations

import time

import pytest

from jarvis.plugins.brain import chatgpt_web_bridge as bridge


@pytest.fixture
def sup(monkeypatch) -> bridge.BridgeSupervisor:
    """A fresh supervisor whose _kill is recorded instead of run."""
    s = bridge.BridgeSupervisor()
    killed: list[bool] = []
    monkeypatch.setattr(s, "_kill", lambda _settings: killed.append(True))
    monkeypatch.setattr(bridge, "settings", lambda: None)
    s.killed = killed  # type: ignore[attr-defined]
    s._started_at = time.monotonic()  # pretend it is running
    return s


def test_a_fresh_bridge_is_never_reaped(sup) -> None:
    """No turn has finished yet, so its idle age is unknown, not zero."""
    assert sup.idle_seconds() is None
    assert sup.stop_if_idle(0.01) is False
    assert sup.killed == []


def test_an_idle_bridge_is_stopped_past_the_timeout(sup) -> None:
    with sup.lease():
        pass
    time.sleep(0.05)
    assert sup.stop_if_idle(0.01) is True
    assert sup.killed == [True]
    assert sup._started_at is None


def test_an_idle_bridge_is_kept_below_the_timeout(sup) -> None:
    with sup.lease():
        pass
    assert sup.stop_if_idle(60.0) is False
    assert sup.killed == []


def test_a_running_turn_is_never_interrupted(sup) -> None:
    """The load-bearing guarantee: a held lease blocks the reaper outright."""
    with sup.lease():
        pass
    time.sleep(0.05)
    with sup.lease():  # a new turn starts after the idle period elapsed
        assert sup.in_use() is True
        assert sup.idle_seconds() is None
        assert sup.stop_if_idle(0.01) is False
    assert sup.killed == []


def test_concurrent_turns_keep_the_bridge_until_the_last_one_ends(sup) -> None:
    with sup.lease():
        with sup.lease():
            assert sup.in_use() is True
        # the inner turn ended but the outer one is still streaming
        assert sup.in_use() is True
        assert sup.stop_if_idle(0.0) is False
    assert sup.in_use() is False


def test_a_lease_is_released_even_when_the_turn_raises(sup) -> None:
    """An aborted or failed answer must not pin the browser open forever."""
    with pytest.raises(RuntimeError), sup.lease():
        raise RuntimeError("stream broke mid-answer")
    assert sup.in_use() is False
    assert sup.idle_seconds() is not None


def test_a_zero_timeout_disables_reaping(sup) -> None:
    with sup.lease():
        pass
    time.sleep(0.05)
    assert sup.stop_if_idle(0.0) is False
    assert sup.killed == []


def test_an_already_stopped_bridge_is_not_killed_twice(sup) -> None:
    with sup.lease():
        pass
    time.sleep(0.05)
    assert sup.stop_if_idle(0.01) is True
    assert sup.stop_if_idle(0.01) is False, "a second reap must be a no-op"
    assert sup.killed == [True]


# ---- configuration ---------------------------------------------------------

def test_the_default_idle_timeout_is_fifteen_minutes(monkeypatch) -> None:
    monkeypatch.setattr(bridge, "_provider_extra", lambda _name: "")
    assert bridge.idle_shutdown_seconds() == pytest.approx(15 * 60)


def test_the_idle_timeout_is_configurable(monkeypatch) -> None:
    monkeypatch.setattr(bridge, "_provider_extra", lambda _name: "3")
    assert bridge.idle_shutdown_seconds() == pytest.approx(180.0)


def test_zero_minutes_keeps_the_bridge_for_the_session(monkeypatch) -> None:
    monkeypatch.setattr(bridge, "_provider_extra", lambda _name: "0")
    assert bridge.idle_shutdown_seconds() == 0.0


def test_an_unparsable_timeout_falls_back_to_the_default(monkeypatch) -> None:
    """A typo in the config must not silently disable or wildly change this."""
    monkeypatch.setattr(bridge, "_provider_extra", lambda _name: "fifteen")
    assert bridge.idle_shutdown_seconds() == pytest.approx(15 * 60)


def test_a_negative_timeout_is_clamped_to_disabled(monkeypatch) -> None:
    monkeypatch.setattr(bridge, "_provider_extra", lambda _name: "-5")
    assert bridge.idle_shutdown_seconds() == 0.0
