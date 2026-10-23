"""The chat-seat warm task reads the live config and logs what it did.

Live 2026-10-23: every boot logged "'WebServer' object has no attribute
'config'", because the task read a server attribute that does not exist.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from jarvis.ui.web.server import WebServer


def test_warm_task_reads_the_live_config(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = SimpleNamespace(name="live-config")
    seen: list[object] = []

    async def _fake_warm(config: object, registry: object) -> str:
        seen.append(config)
        return "off"

    monkeypatch.setattr("jarvis.core.config.load_config", lambda: sentinel)
    monkeypatch.setattr("jarvis.brain.seat_warm.warm_chat_seat", _fake_warm)
    server = object.__new__(WebServer)

    async def _run() -> None:
        server._schedule_chat_seat_warm()
        await server._chat_seat_warm_task

    asyncio.run(_run())

    assert seen == [sentinel]
