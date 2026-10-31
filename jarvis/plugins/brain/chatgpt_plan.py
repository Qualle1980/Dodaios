"""ChatGPT plan brain: turns on the user's ChatGPT plan via Sign in with ChatGPT.

Inference goes to the public Responses API (``api.openai.com/v1/responses``)
with the plan's access token from ``chatgpt_plan_auth``. Contract per turn:

* ``store: false`` and ``stream: true``; ``input`` carries the full history
  (no ``previous_response_id``); the system prompt goes in ``instructions``,
  never a ``role: "system"`` item;
* no sampling or metadata fields; function tools and image input ride the
  same route (probed live 2026-10-24 on the plan token: a ``function_call``
  came back, an ``input_image`` was described), built by the OpenAI brain's
  own Chat -> Responses converters so both providers send one shape;
* only ``response.completed`` is success; a stream that ends without it fails;
* the finished response's ``usage`` is yielded as the turn's last delta, so
  :class:`jarvis.brain.usage_meter.MeteredBrain` writes this route's spend to
  the ledger like every other brain's — it comes out of the person's ChatGPT
  plan window, the same one ``codex`` draws from;
* nothing is ever re-sent: once the request may have left, an unclear outcome
  is ``submission_ambiguous`` and the user decides;
* there is no API-key fallback. ``OPENAI_API_KEY`` is never read here.

Failures are ``ChatGPTPlanError`` with the server's code, HTTP status and
request id, so the UI can explain them (usage limit, ineligible account, ...).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from jarvis.core.protocols import BrainDelta, BrainMessage, BrainRequest

from ._openai_base import (
    _chat_messages_to_responses_input,
    _openai_tool_name_map,
    _to_openai_messages,
    _tools_openai_format,
    _tools_responses_format,
)
from .chatgpt_plan_auth import RESOURCE, USAGE_URL, ChatGPTPlanError, auth

log = logging.getLogger(__name__)

RESPONSES_URL = f"{RESOURCE}/responses"
CLIENT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)
_RETRY_DELAYS = (1.0, 3.0)

# User-facing explanations of the documented plan error codes.
_EXPLAIN = {
    "subscription_sharing_user_not_eligible": (
        "This ChatGPT account cannot share its plan with Dodaios."
    ),
    "subscription_sharing_usage_limit_exceeded": (
        f"Your ChatGPT plan's usage limit is reached; see {USAGE_URL}."
    ),
    "subscription_sharing_usage_unavailable": (
        "ChatGPT plan usage is temporarily unavailable; try again later."
    ),
    "subscription_sharing_unsupported_capability": (
        "The ChatGPT plan does not support part of this request."
    ),
    "subscription_sharing_route_not_supported": (
        "The ChatGPT plan does not support this request route."
    ),
    "subscription_sharing_invalid_user": "The ChatGPT sign-in is no longer valid; reauthorize.",
    "chatpass_v2_scope_not_authorized": "The sign-in lacks plan access; reauthorize and allow it.",
    "chatpass_v2_invalid_authorization_context": (
        "The sign-in is not valid for plan use; reauthorize."
    ),
    "subscription_sharing_user_unavailable": (
        "ChatGPT plan access is temporarily unavailable; try again later."
    ),
}
_RETRYABLE = frozenset(
    {"subscription_sharing_usage_unavailable", "subscription_sharing_user_unavailable"}
)


def _text_of(msg: BrainMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "\n".join(
        block["text"]
        for block in msg.content
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )


def tool_name_map(req: BrainRequest) -> dict[str, str]:
    """Jarvis tool name -> the OpenAI-safe name sent on the wire."""
    return _openai_tool_name_map(tuple(req.tools or ()))


def _log_weight(body: dict[str, Any]) -> None:
    """Record what each part of the request weighs, every turn.

    The ledger says a turn cost 29_694 input tokens; it cannot say which part
    of the request they were. Four characters are roughly one token, which is
    accurate enough to tell a bloated tool catalogue from a long conversation
    -- the question that decides where trimming pays.
    """
    def weight(value: Any) -> int:
        if isinstance(value, str):
            return len(value) // 4
        return len(json.dumps(value, separators=(",", ":"))) // 4 if value else 0

    instructions = weight(body.get("instructions"))
    tools = weight(body.get("tools"))
    conversation = weight(body.get("input"))
    log.info(
        "plan request ~%d tokens: instructions %d, tools %d (%d schemas), conversation %d",
        instructions + tools + conversation,
        instructions,
        tools,
        len(body.get("tools") or ()),
        conversation,
    )


def _cache_key(model: str, instructions: str, tools: Any) -> str:
    """A routing key for the part of the request that never changes.

    Instructions and tool schemas are ~22k tokens and byte-identical from one
    turn to the next, yet a stable prefix is only reused when the request
    reaches the machine that holds it. OpenAI routes on ``prompt_cache_key``,
    so the key is derived from the prefix ALONE: including the conversation
    would send every turn to a fresh machine, which is the behaviour this
    replaces (live install 2026-10-30: 29_694 input tokens per voice turn,
    one cache hit in twenty-four).

    The model is part of the key because a cache entry never crosses models.
    """
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(instructions.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(json.dumps(tools or [], sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return f"jarvis-{digest.hexdigest()[:32]}"


def build_body(req: BrainRequest, model: str) -> dict[str, Any]:
    """The Responses request: full history, instructions, tools, nothing optional.

    History, images, tool calls and tool results go through the OpenAI brain's
    Chat -> Responses converters (one shape for both providers); the
    instructions keep this route's own order: the request's system prompt,
    then system messages.
    """
    names = tool_name_map(req)
    chat = _to_openai_messages(
        tuple(req.messages), None, supports_vision=True, tool_name_map=names,
    )
    _, items = _chat_messages_to_responses_input(chat)
    body: dict[str, Any] = {"model": model, "input": items, "store": False, "stream": True}
    extra_system = [_text_of(m) for m in req.messages if m.role == "system"]
    instructions = "\n\n".join(p for p in (req.system or "", *extra_system) if p)
    if instructions:
        body["instructions"] = instructions
    if req.tools:
        body["tools"] = _tools_responses_format(_tools_openai_format(tuple(req.tools), names))
        # The caller narrows the callable set instead of shortening the list:
        # `tools` stays byte-identical from turn to turn, which is the only way
        # the ~18k-token prefix is ever a cache hit (live 2026-10-31: twelve
        # turns, 197_807 input tokens, zero cached, because a per-turn
        # shortlist changed the array on every request).
        allowed = tuple(dict.fromkeys(req.allowed_tools or ()))
        if allowed and len(allowed) < len(req.tools):
            body["tool_choice"] = {
                "type": "allowed_tools",
                "mode": "auto",
                "tools": [
                    {"type": "function", "name": names.get(name, name)}
                    for name in allowed
                ],
            }
    body["prompt_cache_key"] = _cache_key(model, body.get("instructions", ""), body.get("tools"))
    _log_weight(body)
    return body


def error_from(status: int, payload: Any, request_id: str | None) -> ChatGPTPlanError:
    """Map an admission or stream error, keeping status, code and request id."""
    err: Any = payload.get("error") if isinstance(payload, dict) else None
    code = message = param = ""
    if isinstance(err, dict):
        code, message, param = (
            str(err.get("code") or ""),
            str(err.get("message") or ""),
            str(err.get("param") or ""),
        )
    elif isinstance(payload, dict) and payload.get("detail"):
        message = str(payload["detail"])
    if code in _EXPLAIN:
        text = _EXPLAIN[code]
        if param:
            text = f"{text} ({param})"
    elif status == 401:
        code, text = (
            code or "unauthorized",
            "ChatGPT rejected the sign-in; check the account and reauthorize.",
        )
    elif status == 403:
        code, text = (
            code or "forbidden",
            f"ChatGPT refused the request: {message or 'permission denied'}.",
        )
    elif status == 429:
        code, text = (
            code or "rate_limited",
            f"ChatGPT is rate limiting this account; see {USAGE_URL}.",
        )
    elif status == 503:
        code, text = code or "unavailable", "ChatGPT is temporarily unavailable."
    else:
        code = code or "failed"
        text = f"ChatGPT plan error {status} {code}: {message}".strip()
    return ChatGPTPlanError(code, text, status=status, request_id=request_id)


class ChatGPTPlanBrain:
    name: str = "chatgpt-plan"
    context_window: int = 128_000
    supports_tools: bool = True
    supports_vision: bool = True
    #: ``tool_choice.allowed_tools`` can narrow the callable set per request,
    #: so this route is handed the FULL tool list and keeps a cacheable prefix.
    supports_allowed_tools: bool = True
    #: The caller's system prompt goes in ``instructions``, a real channel.
    native_system_prompt: bool = True
    #: The model the last call actually ran on, for the usage ledger.
    #:
    #: This route resolves its model DURING the call (nothing is pinned, so
    #: ``_prepare`` falls through to the account's first listed model), which
    #: left ``MeteredBrain`` with an empty name to write down: it reads this
    #: attribute for exactly that case and its comment names the incident
    #: ("live 2026-10-23: a chatgpt-plan turn on gpt-6.1-sol was recorded with
    #: no model"). The contract existed; no plugin had ever filled it, so the
    #: most expensive route the product has was also the only one whose
    #: spending could not be attributed to a model. Measured again on
    #: 2026-10-31: twelve turns, 197_807 input tokens, model column empty.
    last_model: str = ""

    def __init__(self, model: str | None = None, structured_prompts: bool = True) -> None:
        # ``structured_prompts`` is accepted for resolve_subscription_brain: an
        # API brain has no conversational wrapper, so every request already
        # carries the caller's contract and full history verbatim.
        del structured_prompts
        model = (model or "").strip()
        self._model = model.removeprefix("chatgpt-plan/")

    @staticmethod
    def subscription_connected() -> bool:
        """Signed in with plan scope: background work may run on this seat."""
        try:
            return auth().connected()
        except ChatGPTPlanError:  # an unreadable store reads as not connected; auth() logs why
            return False

    def can_call_tools(self) -> bool:
        return True

    def estimate_cost(self, req: BrainRequest) -> float:
        return 0.0

    def _prepare(self) -> tuple[str, str]:
        a = auth()
        status = a.status()
        if not status["signed_in"]:
            raise ChatGPTPlanError("not_signed_in", "Sign in with ChatGPT on the Agents tab first.")
        if not status["plan_enabled"]:
            raise ChatGPTPlanError(
                "plan_scope_missing",
                "The ChatGPT sign-in did not grant plan access; reauthorize and allow it.",
            )
        token = a.access_token()
        model = self._model or status["model"]
        if not model:
            models = a.models()
            if not models:
                raise ChatGPTPlanError(
                    "no_models", "Your ChatGPT plan lists no models for Dodaios."
                )
            model = models[0]["slug"]
        # Publish before returning: the meter reads it in a ``finally``, so a
        # call that fails after this point is still attributed to the model it
        # was made against.
        self.last_model = model
        return token, model

    async def complete(self, req: BrainRequest) -> AsyncIterator[BrainDelta]:
        token, model = await asyncio.to_thread(self._prepare)
        body = build_body(req, model)
        reverse_names = {safe: name for name, safe in tool_name_map(req).items()}
        headers = {"Authorization": f"Bearer {token}", "Accept": "text/event-stream"}
        for attempt, delay in enumerate((0.0, *_RETRY_DELAYS)):
            if delay:
                await asyncio.sleep(delay)
            admitted = False
            finished = False
            try:
                async with httpx.AsyncClient(timeout=CLIENT_TIMEOUT) as client:
                    async with client.stream(
                        "POST", RESPONSES_URL, json=body, headers=headers
                    ) as resp:
                        if resp.status_code != 200:
                            raw = await resp.aread()
                            try:
                                payload = json.loads(raw)
                            except ValueError:  # a non-JSON error body still maps by status
                                payload = None
                            exc = error_from(
                                resp.status_code, payload, resp.headers.get("x-request-id")
                            )
                            # A 503 admission refusal was not processed: retrying is no resubmit.
                            if (
                                resp.status_code == 503 or exc.code in _RETRYABLE
                            ) and attempt < len(_RETRY_DELAYS):
                                log.info(
                                    "chatgpt-plan: %s, retrying (request %s)",
                                    exc.code,
                                    exc.request_id,
                                )
                                continue
                            raise exc
                        admitted = True
                        async for delta in _events(resp, reverse_names):
                            if delta.finish_reason:
                                finished = True
                            yield delta
            except ChatGPTPlanError:
                raise
            except httpx.ConnectError as exc:
                raise ChatGPTPlanError(
                    "network", "api.openai.com is not reachable; nothing was sent."
                ) from exc
            except httpx.HTTPError as exc:
                if admitted and not finished:
                    raise ChatGPTPlanError(
                        "submission_ambiguous",
                        "The connection to ChatGPT broke mid-answer; the message may have been "
                        "processed. It was not re-sent.",
                    ) from exc
                raise ChatGPTPlanError(
                    "submission_ambiguous",
                    "The request to ChatGPT failed after it may have been sent "
                    f"({type(exc).__name__}); it was not re-sent.",
                ) from exc
            if not finished:
                raise ChatGPTPlanError(
                    "incomplete_stream",
                    "ChatGPT ended the answer without completing it; it was not re-sent.",
                )
            return


def _tool_call(item: dict[str, Any], reverse_names: dict[str, str], index: int) -> dict[str, Any]:
    """A finished ``function_call`` item as the tool call the tool loop runs."""
    name = str(item.get("name") or "")
    try:
        arguments = json.loads(item.get("arguments") or "{}")
    except ValueError:
        log.warning("chatgpt-plan: tool call %r had unparsable arguments; running it empty", name)
        arguments = {}
    return {
        "id": str(item.get("call_id") or item.get("id") or f"call_{index}"),
        "name": reverse_names.get(name, name),
        "input": arguments if isinstance(arguments, dict) else {},
    }


def _usage_of(payload: dict[str, Any]) -> dict[str, int] | None:
    """What the finished response says it spent, in the canonical shape.

    Every turn on this route spends the person's ChatGPT PLAN window — the
    same window ``codex`` draws from. Until this block existed the route was
    the only brain in the tree that reported nothing, so
    :class:`jarvis.brain.usage_meter.MeteredBrain` wrote nothing down and the
    most expensive calls the product makes were the only ones nobody could
    see.

    ``input_tokens`` is the UNCACHED share, as the cost contract requires and
    as ``_openai_base`` already does for the same API: OpenAI counts cache
    hits inside its input figure, and leaving them there prices every hit at
    the full rate.
    """
    usage = (payload.get("response") or {}).get("usage")
    if not isinstance(usage, Mapping):
        return None

    def count(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            # A non-numeric count says nothing, and metering must never break
            # the call it measures.
            return 0

    details = usage.get("input_tokens_details")
    cached = count(details.get("cached_tokens")) if isinstance(details, Mapping) else 0
    out: dict[str, int] = {
        "input_tokens": max(count(usage.get("input_tokens")) - cached, 0),
        "output_tokens": count(usage.get("output_tokens")),
    }
    if cached > 0:
        out["cache_hit_tokens"] = cached
    return out if any(out.values()) else None


async def _events(
    resp: httpx.Response, reverse_names: dict[str, str] | None = None
) -> AsyncIterator[BrainDelta]:
    """Stream deltas; tool calls are collected and handed out before the finish,
    like the OpenAI brain's Responses path, so the tool loop sees one shape."""
    reverse_names = reverse_names or {}
    calls: list[dict[str, Any]] = []
    event = ""
    async for line in resp.aiter_lines():
        if line.startswith("event:"):
            event = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except ValueError:
            log.debug("chatgpt-plan: skipped an unparsable SSE line")
            continue
        kind = str(payload.get("type") or event)
        if kind == "response.output_text.delta":
            text = payload.get("delta")
            if isinstance(text, str) and text:
                yield BrainDelta(content=text)
        elif kind == "response.output_item.done":
            item = payload.get("item") or {}
            if isinstance(item, dict) and item.get("type") == "function_call":
                calls.append(_tool_call(item, reverse_names, len(calls)))
        elif kind == "response.completed":
            for call in calls:
                yield BrainDelta(tool_call=call)
            yield BrainDelta(finish_reason="tool_calls" if calls else "stop")
            # Usage last, after the finish reason, as the OpenAI brain's
            # Responses path yields it: the meter reads it from the stream.
            spent = _usage_of(payload)
            if spent is not None:
                yield BrainDelta(usage=spent)
            return
        elif kind == "response.incomplete":
            reason = ((payload.get("response") or {}).get("incomplete_details") or {}).get("reason")
            if reason == "max_output_tokens":
                yield BrainDelta(finish_reason="length")
                # A truncated answer was paid for in full; it is still spend.
                spent = _usage_of(payload)
                if spent is not None:
                    yield BrainDelta(usage=spent)
                return
            raise ChatGPTPlanError(
                "incomplete", f"ChatGPT stopped the answer early ({reason or 'unknown'})."
            )
        elif kind in ("response.failed", "error"):
            inner = (
                (payload.get("response") or {}).get("error")
                if kind == "response.failed"
                else payload.get("error", payload)
            )
            raise error_from(200, {"error": inner if isinstance(inner, dict) else {}}, None)
