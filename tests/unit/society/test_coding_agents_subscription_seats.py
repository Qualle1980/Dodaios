"""A signed-in subscription seat is visible where delegation is decided.

Live 2026-11-15: asked to give Grok a task, the General tried the brain switch
for the xAI API row, got "API key missing" and did the work itself. The Grok
Build subscription seat was signed in the whole time; nothing it could read
said so.
"""
from __future__ import annotations

import pytest

from jarvis.agent_chat import seat_login
from jarvis.brain import app_control
from jarvis.society import coding_threads
from jarvis.society.coding_threads import CodingThreadTool


def _seats(signed_in: dict[str, bool | None]):
    def fake(runner: str) -> bool | None:
        return signed_in.get(runner)

    return fake


async def _agents(monkeypatch: pytest.MonkeyPatch, signed_in: dict[str, bool | None]):
    async def installed() -> dict[str, bool]:
        return {}

    monkeypatch.setattr(coding_threads, "_installed_clis", installed)
    monkeypatch.setattr(seat_login, "seat_signed_in", _seats(signed_in))
    tool = CodingThreadTool.__new__(CodingThreadTool)
    return await tool._do_agents(None, None, {})


async def test_grok_build_listed_as_connected_coding_agent_when_subscribed(monkeypatch):
    listing = await _agents(monkeypatch, {"grok-cli": True})
    rows = {row["agent"]: row for row in listing["coding_agents"]}
    assert rows["grok-build"]["signed_in"] is True
    assert "signed_in=true" in listing["routing"]


async def test_a_seat_without_a_readable_login_is_unknown_not_false(monkeypatch):
    listing = await _agents(monkeypatch, {})
    rows = {row["agent"]: row for row in listing["coding_agents"]}
    assert rows["grok-build"]["signed_in"] is None


async def test_missing_api_key_names_the_signed_in_seat(monkeypatch):
    monkeypatch.setattr(seat_login, "seat_signed_in", _seats({"grok-cli": True}))
    hint = await app_control._signed_in_seat_hint("grok")
    assert "Grok Build" in hint and "agent=grok-build" in hint


async def test_no_hint_when_the_seat_is_signed_out(monkeypatch):
    monkeypatch.setattr(seat_login, "seat_signed_in", _seats({"grok-cli": False}))
    assert await app_control._signed_in_seat_hint("grok") == ""


async def test_no_hint_for_a_provider_without_a_cli_twin(monkeypatch):
    monkeypatch.setattr(seat_login, "seat_signed_in", _seats({"grok-cli": True}))
    assert await app_control._signed_in_seat_hint("groq") == ""
