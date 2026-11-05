"""Trusted turn provenance and the immutable chat completion contract."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChatTurn:
    session_id: str
    turn_id: str
    user_text: str
    direct_user: bool
    trace_id: str
    #: The turn that first carried this request, when this one continues it.
    #: A delegated result can come back long after its turn closed, as a turn
    #: of its own (``jarvis/agent_chat/parked_work.py``); budgets and limits
    #: that belong to the REQUEST have to survive that, so they key on this
    #: rather than on ``turn_id``. Empty means this turn is the original.
    origin_turn_id: str = ""

    @property
    def request_id(self) -> str:
        """The identity of the request, across every turn that continues it."""
        return self.origin_turn_id or self.turn_id


current_chat_turn: ContextVar[ChatTurn | None] = ContextVar("current_chat_turn", default=None)


@dataclass(frozen=True, slots=True)
class ChatCompletion:
    turn: ChatTurn
    events_json: str
