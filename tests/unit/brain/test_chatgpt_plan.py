"""Sign in with ChatGPT (chatgpt-plan): OAuth lifecycle and the Responses brain.

Every network call goes through ``httpx.MockTransport``; no real endpoint,
browser or token is touched.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from jarvis.core.protocols import BrainMessage, BrainRequest
from jarvis.plugins.brain import chatgpt_plan, chatgpt_plan_auth
from jarvis.plugins.brain.chatgpt_plan_auth import (
    ISSUER,
    MODELS_URL,
    PLAN_SCOPE,
    SCOPES,
    TOKEN_URL,
    Attempt,
    ChatGPTPlanError,
    PlanAuth,
    Store,
    build_authorize_url,
    pkce_pair,
    redact,
    validate_id_token,
)

CLIENT_ID = "oaiapp_test123"
JWKS_URL = "https://auth.example/jwks"
REVOKE_URL = "https://auth.example/revoke"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwks() -> dict[str, Any]:
    jwk = json.loads(RSAAlgorithm.to_jwk(KEY.public_key()))
    jwk.update({"kid": "k1", "alg": "RS256", "use": "sig"})
    return {"keys": [jwk]}


def _id_token(key: Any = KEY, **overrides: Any) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "aud": CLIENT_ID,
        "sub": "user-1",
        "email": "person@example.com",
        "nonce": "n-1",
        "iat": now,
        "exp": now + 600,
    }
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "k1"})


class FakeOpenAI:
    """Routes for discovery, JWKS, token, revoke and models; records each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.token_reply: tuple[int, dict[str, Any]] = (200, {})
        self.models_reply: tuple[int, dict[str, Any]] = (200, {"models": []})
        self.refresh_count = 0
        self._mu = threading.Lock()

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        with self._mu:
            self.calls.append((url, form))
        if url.startswith(f"{ISSUER}/.well-known"):
            return httpx.Response(
                200, json={"jwks_uri": JWKS_URL, "revocation_endpoint": REVOKE_URL}
            )
        if url == JWKS_URL:
            return httpx.Response(200, json=_jwks())
        if url == TOKEN_URL:
            if form.get("grant_type") == "refresh_token":
                with self._mu:
                    self.refresh_count += 1
                time.sleep(0.05)  # widen the race window for the lock test
            status, body = self.token_reply
            return httpx.Response(status, json=body)
        if url == REVOKE_URL:
            return httpx.Response(200)
        if url == MODELS_URL:
            status, body = self.models_reply
            return httpx.Response(status, json=body)
        return httpx.Response(404)


@pytest.fixture
def fake() -> FakeOpenAI:
    return FakeOpenAI()


@pytest.fixture
def plan(tmp_path, fake, monkeypatch) -> PlanAuth:
    monkeypatch.setattr(chatgpt_plan_auth, "_sleep", lambda _s: None)
    client = httpx.Client(transport=httpx.MockTransport(fake.handler))
    opened: list[str] = []
    return PlanAuth(store=Store(tmp_path), client=lambda: client, open_browser=opened.append)


def _attempt(*, reauth: bool = False, client_id: str = "dynamic_agent_client") -> Attempt:
    return Attempt(
        state="s-1",
        nonce="n-1",
        verifier="v" * 43,
        redirect_uri="http://127.0.0.1:5555/auth/callback",
        client_id=client_id,
        reauth=reauth,
    )


def _signed_in(plan: PlanAuth, *, scope: str = SCOPES, expires_in: float = 3600) -> None:
    plan.store.write(
        {
            "client_id": CLIENT_ID,
            "sub": "user-1",
            "email": "person@example.com",
            "scope": scope,
            "tokens": {
                "access_token": "at-old",
                "refresh_token": "rt-old",
                "id_token": "idt",
                "expires_at": time.time() + expires_in,
            },
        }
    )


def _token_body(**extra: Any) -> dict[str, Any]:
    body = {
        "access_token": "at-new",
        "refresh_token": "rt-new",
        "id_token": _id_token(),
        "expires_in": 3600,
        "scope": SCOPES,
    }
    body.update(extra)
    return body


def _query(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


# ---- PKCE and authorize URL ----------------------------------------------


def test_pkce_challenge_is_s256_of_verifier() -> None:
    verifier, challenge = pkce_pair()
    digest = hashlib.sha256(verifier.encode()).digest()
    assert challenge == base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    assert 43 <= len(verifier) <= 128


def test_first_login_url_uses_dynamic_client_and_agent_name() -> None:
    q = _query(build_authorize_url(_attempt(), host_id="urn:uuid:x", challenge="c"))
    assert q["client_id"] == "dynamic_agent_client"
    assert q["code_challenge_method"] == "S256"
    assert PLAN_SCOPE in q["scope"].split()
    assert q["agent_name_hint"] == "Dodaios"
    assert "id_token_hint" not in q


def test_reauth_url_carries_hints_not_agent_name() -> None:
    url = build_authorize_url(
        _attempt(reauth=True, client_id=CLIENT_ID),
        host_id="urn:uuid:x",
        challenge="c",
        id_token_hint="idt",  # noqa: S106
        login_hint="person@example.com",
    )
    q = _query(url)
    assert q["client_id"] == CLIENT_ID
    assert q["id_token_hint"] == "idt"  # noqa: S105
    assert "agent_name_hint" not in q


def test_host_id_persists(tmp_path) -> None:
    first = Store(tmp_path).host_id()
    assert first.startswith("urn:uuid:")
    assert Store(tmp_path).host_id() == first


# ---- callback --------------------------------------------------------------


def test_callback_registers_client_and_stores_account(plan, fake) -> None:
    fake.token_reply = (200, _token_body())
    plan._attempt = _attempt()
    ok, _msg = plan.handle_callback({"state": "s-1", "code": "c-1", "client_id": CLIENT_ID})
    assert ok
    exchange = next(f for u, f in fake.calls if u == TOKEN_URL)
    assert exchange["grant_type"] == "authorization_code"
    assert exchange["client_id"] == CLIENT_ID
    assert exchange["code_verifier"] == "v" * 43
    data = plan.store.read()
    assert data["client_id"] == CLIENT_ID
    assert data["sub"] == "user-1"
    status = plan.status()
    assert status["signed_in"] and status["plan_enabled"] and status["registered"]


def test_state_mismatch_is_rejected_without_exchange(plan, fake) -> None:
    plan._attempt = _attempt()
    ok, msg = plan.handle_callback({"state": "other", "code": "c-1", "client_id": CLIENT_ID})
    assert not ok and "stale" in msg
    assert not fake.calls
    assert plan._attempt is not None  # a stray request does not cancel the real attempt


def test_access_denied_marks_login_denied(plan) -> None:
    plan._attempt = _attempt()
    ok, _ = plan.handle_callback({"state": "s-1", "error": "access_denied"})
    assert not ok
    assert plan.status()["login"] == "denied"


@pytest.mark.parametrize("returned", ["", "dynamic_agent_client"])
def test_first_login_needs_a_registered_client(plan, fake, returned) -> None:
    plan._attempt = _attempt()
    ok, _ = plan.handle_callback({"state": "s-1", "code": "c", "client_id": returned})
    assert not ok
    assert plan.status()["error"] == "registration_incomplete"
    assert not fake.calls


def test_reauth_reuses_stored_client_id(plan, fake) -> None:
    _signed_in(plan)
    fake.token_reply = (200, _token_body())
    plan._attempt = _attempt(reauth=True, client_id=CLIENT_ID)
    ok, _ = plan.handle_callback({"state": "s-1", "code": "c"})
    assert ok
    exchange = next(f for u, f in fake.calls if u == TOKEN_URL)
    assert exchange["client_id"] == CLIENT_ID


def test_reauth_rejects_other_client(plan) -> None:
    _signed_in(plan)
    plan._attempt = _attempt(reauth=True, client_id=CLIENT_ID)
    ok, _ = plan.handle_callback({"state": "s-1", "code": "c", "client_id": "oaiapp_other"})
    assert not ok and plan.status()["error"] == "client_mismatch"


def test_reauth_rejects_other_account(plan, fake) -> None:
    _signed_in(plan)
    fake.token_reply = (200, _token_body(id_token=_id_token(sub="user-2")))
    plan._attempt = _attempt(reauth=True, client_id=CLIENT_ID)
    ok, _ = plan.handle_callback({"state": "s-1", "code": "c"})
    assert not ok and plan.status()["error"] == "account_mismatch"
    assert plan.store.read()["tokens"]["access_token"] == "at-old"  # noqa: S105


def test_nonce_mismatch_fails_login(plan, fake) -> None:
    fake.token_reply = (200, _token_body(id_token=_id_token(nonce="other")))
    plan._attempt = _attempt()
    ok, _ = plan.handle_callback({"state": "s-1", "code": "c", "client_id": CLIENT_ID})
    assert not ok and plan.status()["error"] == "nonce_mismatch"
    assert "tokens" not in plan.store.read()


def test_missing_plan_scope_is_signed_in_without_plan(plan, fake) -> None:
    fake.token_reply = (200, _token_body(scope="openid profile email offline_access"))
    plan._attempt = _attempt()
    ok, _ = plan.handle_callback({"state": "s-1", "code": "c", "client_id": CLIENT_ID})
    assert ok
    status = plan.status()
    assert status["signed_in"] and not status["plan_enabled"]
    assert not plan.connected()


# ---- ID token validation ---------------------------------------------------


@pytest.mark.parametrize(
    ("token", "code"),
    [
        (lambda: _id_token(key=OTHER_KEY), "id_token_invalid"),
        (lambda: _id_token(iss="https://evil.example"), "id_token_invalid"),
        (lambda: _id_token(aud="oaiapp_other"), "id_token_invalid"),
        (lambda: _id_token(exp=int(time.time()) - 60), "id_token_expired"),
        (lambda: _id_token(nonce="other"), "nonce_mismatch"),
    ],
    ids=["bad_signature", "wrong_issuer", "wrong_audience", "expired", "nonce"],
)
def test_id_token_rejections(token, code) -> None:
    with pytest.raises(ChatGPTPlanError) as exc:
        validate_id_token(token(), jwks=_jwks(), client_id=CLIENT_ID, nonce="n-1")
    assert exc.value.code == code


def test_valid_id_token_returns_claims() -> None:
    claims = validate_id_token(_id_token(), jwks=_jwks(), client_id=CLIENT_ID, nonce="n-1")
    assert claims["sub"] == "user-1"


# ---- tokens ----------------------------------------------------------------


def test_fresh_token_needs_no_refresh(plan, fake) -> None:
    _signed_in(plan)
    assert plan.access_token() == "at-old"
    assert fake.refresh_count == 0


def test_refresh_rotates_refresh_token(plan, fake) -> None:
    _signed_in(plan, expires_in=10)
    fake.token_reply = (200, {"access_token": "at-2", "refresh_token": "rt-2", "expires_in": 3600})
    assert plan.access_token() == "at-2"
    sent = next(f for u, f in fake.calls if u == TOKEN_URL)
    assert sent["refresh_token"] == "rt-old" and sent["client_id"] == CLIENT_ID  # noqa: S105
    assert plan.store.read()["tokens"]["refresh_token"] == "rt-2"  # noqa: S105


def test_refresh_without_rotation_keeps_old_refresh_token(plan, fake) -> None:
    _signed_in(plan, expires_in=10)
    fake.token_reply = (200, {"access_token": "at-2", "expires_in": 3600})
    plan.access_token()
    assert plan.store.read()["tokens"]["refresh_token"] == "rt-old"  # noqa: S105


def test_parallel_refresh_posts_once(plan, fake) -> None:
    _signed_in(plan, expires_in=10)
    fake.token_reply = (200, {"access_token": "at-2", "refresh_token": "rt-2", "expires_in": 3600})
    results: list[str] = []
    threads = [
        threading.Thread(target=lambda: results.append(plan.access_token())) for _ in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert results == ["at-2"] * 5
    assert fake.refresh_count == 1


def test_refresh_invalid_grant_requires_reauth(plan, fake) -> None:
    _signed_in(plan, expires_in=10)
    fake.token_reply = (400, {"error": "invalid_grant"})
    with pytest.raises(ChatGPTPlanError) as exc:
        plan.access_token()
    assert exc.value.code == "reauthorization_required"
    assert not plan.status()["signed_in"]
    assert plan.store.read()["client_id"] == CLIENT_ID


def test_refresh_invalid_client(plan, fake) -> None:
    _signed_in(plan, expires_in=10)
    fake.token_reply = (401, {"error": "invalid_client"})
    with pytest.raises(ChatGPTPlanError) as exc:
        plan.access_token()
    assert exc.value.code == "invalid_client"


def test_logout_revokes_and_clears(plan, fake) -> None:
    _signed_in(plan)
    plan.update_meta(model="gpt-x")
    assert plan.logout() is True
    revoke = next(f for u, f in fake.calls if u == REVOKE_URL)
    assert revoke == {
        "token": "rt-old",
        "token_type_hint": "refresh_token",
        "client_id": CLIENT_ID,
    }
    data = plan.store.read()
    for key in ("tokens", "scope", "sub", "email", "model"):
        assert key not in data
    assert data["client_id"] == CLIENT_ID  # the registration is reused by the next sign-in
    assert not plan.status()["signed_in"]


def test_status_and_redaction_never_reveal_tokens(plan) -> None:
    _signed_in(plan)
    text = json.dumps(plan.status())
    for secret in ("at-old", "rt-old", "idt"):
        assert secret not in text
    line = redact("GET /cb?code=abc&state=s&id_token_hint=eyJx refresh_token=rt1&access_token=at1")
    for secret in ("abc", "eyJx", "rt1", "at1"):
        assert secret not in line


# ---- models ----------------------------------------------------------------


def test_model_catalog_keeps_listed_rows(plan, fake) -> None:
    _signed_in(plan)
    fake.models_reply = (
        200,
        {
            "models": [
                {"slug": "gpt-a", "display_name": "GPT A", "visibility": "list"},
                {"slug": "gpt-hidden", "visibility": "hide"},
                {"slug": "gpt-b", "visibility": "list"},
            ]
        },
    )
    assert plan.models() == [
        {"slug": "gpt-a", "display_name": "GPT A"},
        {"slug": "gpt-b", "display_name": "gpt-b"},
    ]


def _two_models() -> tuple[int, dict[str, Any]]:
    return (
        200,
        {
            "models": [
                {"slug": "gpt-a", "display_name": "GPT A", "visibility": "list"},
                {"slug": "gpt-b", "visibility": "list"},
            ]
        },
    )


def _model_fetches(fake: FakeOpenAI) -> int:
    return sum(1 for url, _ in fake.calls if url == MODELS_URL)


def test_the_model_list_is_fetched_once_for_a_run_of_calls(plan, fake) -> None:
    """The conductor resolves a model before EVERY seat call; one order made
    five of them, and the account's plan cannot change between them."""
    _signed_in(plan)
    fake.models_reply = _two_models()

    first = plan.models()
    for _ in range(4):
        assert plan.models() == first

    assert _model_fetches(fake) == 1


def test_an_explicit_refresh_always_asks_the_server(plan, fake) -> None:
    _signed_in(plan)
    fake.models_reply = _two_models()
    plan.models()

    plan.models(refresh=True)

    assert _model_fetches(fake) == 2


def test_a_different_account_never_reads_the_previous_list(plan, fake) -> None:
    """The key is the account, so no TTL has to expire for a switch to land."""
    _signed_in(plan)
    fake.models_reply = _two_models()
    assert [row["slug"] for row in plan.models()] == ["gpt-a", "gpt-b"]

    data = plan.store.read()
    data["sub"] = "user-2"
    plan.store.write(data)
    fake.models_reply = (
        200,
        {"models": [{"slug": "gpt-c", "visibility": "list"}]},
    )

    assert [row["slug"] for row in plan.models()] == ["gpt-c"]
    assert _model_fetches(fake) == 2


def test_a_sign_out_drops_the_cached_list(plan, fake) -> None:
    _signed_in(plan)
    fake.models_reply = _two_models()
    plan.models()

    plan.logout()
    _signed_in(plan)
    plan.models()

    assert _model_fetches(fake) == 2


def test_an_expired_list_is_fetched_again(plan, fake, monkeypatch) -> None:
    _signed_in(plan)
    fake.models_reply = _two_models()
    plan.models()

    clock = time.monotonic() + chatgpt_plan_auth.MODELS_TTL_S + 1.0
    monkeypatch.setattr(chatgpt_plan_auth.time, "monotonic", lambda: clock)
    plan.models()

    assert _model_fetches(fake) == 2


def test_an_empty_list_is_not_held_onto(plan, fake) -> None:
    """Callers treat "no models" as a hard error, so it must stay askable."""
    _signed_in(plan)
    fake.models_reply = (200, {"models": []})
    assert plan.models() == []

    fake.models_reply = _two_models()
    assert [row["slug"] for row in plan.models()] == ["gpt-a", "gpt-b"]
    assert _model_fetches(fake) == 2


def test_a_caller_cannot_mutate_the_cached_list(plan, fake) -> None:
    _signed_in(plan)
    fake.models_reply = _two_models()
    plan.models()[0]["slug"] = "tampered"

    assert [row["slug"] for row in plan.models()] == ["gpt-a", "gpt-b"]


def test_model_catalog_failure(plan, fake) -> None:
    _signed_in(plan)
    fake.models_reply = (403, {})
    with pytest.raises(ChatGPTPlanError) as exc:
        plan.models()
    assert exc.value.code == "models_failed"


# ---- brain -----------------------------------------------------------------


def _sse(*events: dict[str, Any]) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


class FakeResponses:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = replies
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert str(request.url) == chatgpt_plan.RESPONSES_URL
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def brain_env(plan, monkeypatch):
    _signed_in(plan)
    plan.update_meta(model="gpt-a")
    monkeypatch.setattr(chatgpt_plan_auth, "_auth", plan)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-never-be-sent")
    sleeps: list[float] = []

    async def no_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(chatgpt_plan.asyncio, "sleep", no_sleep)
    real = httpx.AsyncClient

    def install(replies: list[Any]) -> FakeResponses:
        server = FakeResponses(replies)

        def factory(**kwargs: Any) -> httpx.AsyncClient:
            return real(transport=httpx.MockTransport(server.handler), **kwargs)

        monkeypatch.setattr(chatgpt_plan.httpx, "AsyncClient", factory)
        return server

    return install, sleeps


def _req() -> BrainRequest:
    return BrainRequest(
        system="Be brief.",
        messages=[
            BrainMessage(role="system", content="Extra rule."),
            BrainMessage(role="user", content="Antworte exakt: DODAIOS-SIWC-OK"),
        ],
    )


async def _collect(brain: chatgpt_plan.ChatGPTPlanBrain) -> list[Any]:
    return [d async for d in brain.complete(_req())]


def _completed() -> httpx.Response:
    return httpx.Response(200, content=_sse({"type": "response.completed", "response": {}}))


def test_body_is_minimal_and_stateless() -> None:
    body = chatgpt_plan.build_body(_req(), "gpt-a")
    # The cache key is asserted on its own below; it is derived, not content.
    body.pop("prompt_cache_key")
    assert body == {
        "model": "gpt-a",
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "Antworte exakt: DODAIOS-SIWC-OK"},  # i18n-allow
        ]}],
        "store": False,
        "stream": True,
        "instructions": "Be brief.\n\nExtra rule.",
    }


def test_tools_images_and_a_tool_round_trip_reach_the_plan_route() -> None:
    """Probed live 2026-10-24 on the plan token: function tools and input_image
    are accepted. The body is the OpenAI brain's own Responses shape."""
    from jarvis.core.protocols import ImageBlock

    req = BrainRequest(
        messages=[
            BrainMessage(role="user", content="What is on the screen?",
                         images=(ImageBlock(mime="image/png", data_b64="AAAA"),)),
            BrainMessage(role="assistant", content=[
                {"type": "tool_use", "id": "c1", "name": "read_file", "input": {"path": "a.md"}},
            ]),
            BrainMessage(role="tool", tool_call_id="c1", content="hello"),
        ],
        tools=({"name": "read_file", "description": "Read a file.",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},),
    )
    body = chatgpt_plan.build_body(req, "gpt-a")
    user, call, result = body["input"]
    assert {"type": "input_image", "image_url": "data:image/png;base64,AAAA"} in user["content"]
    assert call == {"type": "function_call", "call_id": "c1", "name": "read_file",
                    "arguments": '{"path":"a.md"}'}
    assert result == {"type": "function_call_output", "call_id": "c1", "output": "hello"}
    assert body["tools"] == [{
        "type": "function", "name": "read_file", "description": "Read a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }]
    assert chatgpt_plan.ChatGPTPlanBrain().can_call_tools() is True


def test_a_function_call_streams_as_a_tool_call_before_the_finish(brain_env) -> None:
    install, _ = brain_env
    install([httpx.Response(200, content=_sse(
        {"type": "response.output_item.done", "item": {
            "type": "function_call", "call_id": "c9", "name": "spawn_worker",
            "arguments": '{"task": "count lines"}'}},
        {"type": "response.completed", "response": {}},
    ))])
    deltas = asyncio.run(_collect(chatgpt_plan.ChatGPTPlanBrain("chatgpt-plan/gpt-a")))
    assert [d.tool_call for d in deltas if d.tool_call] == [
        {"id": "c9", "name": "spawn_worker", "input": {"task": "count lines"}}
    ]
    assert [d.finish_reason for d in deltas if d.finish_reason] == ["tool_calls"]


def test_streaming_turn_completes(brain_env) -> None:
    install, _ = brain_env
    server = install(
        [
            httpx.Response(
                200,
                content=_sse(
                    {"type": "response.output_text.delta", "delta": "DODAIOS-"},
                    {"type": "response.output_text.delta", "delta": "SIWC-OK"},
                    {"type": "response.completed", "response": {}},
                ),
            )
        ]
    )
    deltas = asyncio.run(_collect(chatgpt_plan.ChatGPTPlanBrain("chatgpt-plan/gpt-a")))
    assert "".join(d.content or "" for d in deltas) == "DODAIOS-SIWC-OK"
    assert deltas[-1].finish_reason == "stop"
    req = server.requests[0]
    assert req.headers["authorization"] == "Bearer at-old"
    sent = json.loads(req.content)
    assert sent["store"] is False and sent["stream"] is True and sent["model"] == "gpt-a"


def test_failed_event_maps_code(brain_env) -> None:
    install, _ = brain_env
    install(
        [
            httpx.Response(
                200,
                content=_sse(
                    {
                        "type": "response.failed",
                        "response": {
                            "error": {"code": "subscription_sharing_usage_limit_exceeded"}
                        },
                    }
                ),
            )
        ]
    )
    with pytest.raises(ChatGPTPlanError) as exc:
        asyncio.run(_collect(chatgpt_plan.ChatGPTPlanBrain()))
    assert exc.value.code == "subscription_sharing_usage_limit_exceeded"


def test_not_eligible_admission_is_not_retried(brain_env) -> None:
    install, sleeps = brain_env
    server = install(
        [
            httpx.Response(
                403,
                json={"error": {"code": "subscription_sharing_user_not_eligible"}},
                headers={"x-request-id": "req-1"},
            )
        ]
    )
    with pytest.raises(ChatGPTPlanError) as exc:
        asyncio.run(_collect(chatgpt_plan.ChatGPTPlanBrain()))
    assert exc.value.code == "subscription_sharing_user_not_eligible"
    assert exc.value.request_id == "req-1"
    assert len(server.requests) == 1 and not sleeps


def test_503_admission_retries_then_succeeds(brain_env) -> None:
    install, sleeps = brain_env
    server = install([httpx.Response(503, json={}), _completed()])
    asyncio.run(_collect(chatgpt_plan.ChatGPTPlanBrain()))
    assert len(server.requests) == 2 and sleeps == [1.0]


def test_stream_without_completed_is_never_resubmitted(brain_env) -> None:
    install, _ = brain_env
    server = install(
        [httpx.Response(200, content=_sse({"type": "response.output_text.delta", "delta": "x"}))]
    )
    with pytest.raises(ChatGPTPlanError) as exc:
        asyncio.run(_collect(chatgpt_plan.ChatGPTPlanBrain()))
    assert exc.value.code == "incomplete_stream"
    assert len(server.requests) == 1


def test_broken_send_is_ambiguous_and_not_resent(brain_env) -> None:
    install, _ = brain_env
    server = install([httpx.ReadError("reset")])
    with pytest.raises(ChatGPTPlanError) as exc:
        asyncio.run(_collect(chatgpt_plan.ChatGPTPlanBrain()))
    assert exc.value.code == "submission_ambiguous"
    assert len(server.requests) == 1


def test_abort_closes_stream_without_resend(brain_env) -> None:
    install, _ = brain_env
    server = install(
        [
            httpx.Response(
                200,
                content=_sse(
                    {"type": "response.output_text.delta", "delta": "a"},
                    {"type": "response.output_text.delta", "delta": "b"},
                    {"type": "response.completed", "response": {}},
                ),
            )
        ]
    )

    async def first_then_abort() -> str:
        gen = chatgpt_plan.ChatGPTPlanBrain().complete(_req())
        first = await gen.__anext__()
        await gen.aclose()
        return first.content or ""

    assert asyncio.run(first_then_abort()) == "a"
    assert len(server.requests) == 1


def test_no_api_key_fallback_when_signed_out(brain_env, plan) -> None:
    install, _ = brain_env
    server = install([])
    plan.logout()
    with pytest.raises(ChatGPTPlanError) as exc:
        asyncio.run(_collect(chatgpt_plan.ChatGPTPlanBrain()))
    assert exc.value.code == "not_signed_in"
    assert not server.requests


def test_a_finished_turn_reports_what_it_spent(brain_env) -> None:
    """The meter can only write down what the stream hands it.

    This route spends the person's ChatGPT plan window — the same one Codex
    draws from — and reported nothing, so its turns were the only spend in
    the product that no ledger and no report could see.
    """
    install, _ = brain_env
    install([httpx.Response(200, content=_sse(
        {"type": "response.output_text.delta", "delta": "hi"},
        {"type": "response.completed", "response": {"usage": {
            "input_tokens": 1200, "output_tokens": 48,
            "input_tokens_details": {"cached_tokens": 1000},
        }}},
    ))])
    deltas = asyncio.run(_collect(chatgpt_plan.ChatGPTPlanBrain("chatgpt-plan/gpt-a")))
    spent = [d.usage for d in deltas if d.usage]
    # Uncached input only: cache hits priced at the full rate was a real bug
    # on the sibling route, so this one states the same shape.
    assert spent == [{"input_tokens": 200, "output_tokens": 48, "cache_hit_tokens": 1000}]
    # And it arrives after the finish reason, where the meter reads it.
    assert [i for i, d in enumerate(deltas) if d.finish_reason] < [
        i for i, d in enumerate(deltas) if d.usage
    ]


def test_a_truncated_answer_still_reports_its_spend(brain_env) -> None:
    """An answer cut off at the output cap was paid for in full."""
    install, _ = brain_env
    install([httpx.Response(200, content=_sse(
        {"type": "response.incomplete", "response": {
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {"input_tokens": 90, "output_tokens": 4096},
        }},
    ))])
    deltas = asyncio.run(_collect(chatgpt_plan.ChatGPTPlanBrain("chatgpt-plan/gpt-a")))
    assert [d.finish_reason for d in deltas if d.finish_reason] == ["length"]
    assert [d.usage for d in deltas if d.usage] == [
        {"input_tokens": 90, "output_tokens": 4096}
    ]


def test_a_response_without_usage_reports_nothing_rather_than_zeroes(brain_env) -> None:
    """Metering must never invent a reading. A missing block is no delta."""
    install, _ = brain_env
    install([_completed()])
    deltas = asyncio.run(_collect(chatgpt_plan.ChatGPTPlanBrain("chatgpt-plan/gpt-a")))
    assert not [d.usage for d in deltas if d.usage]


def test_an_unreadable_usage_block_never_breaks_the_turn(brain_env) -> None:
    """A count that is not a number tells us nothing, and the answer still stands."""
    install, _ = brain_env
    install([httpx.Response(200, content=_sse(
        {"type": "response.output_text.delta", "delta": "ok"},
        {"type": "response.completed", "response": {"usage": {
            "input_tokens": "many", "output_tokens": None, "input_tokens_details": "nope",
        }}},
    ))])
    deltas = asyncio.run(_collect(chatgpt_plan.ChatGPTPlanBrain("chatgpt-plan/gpt-a")))
    assert "".join(d.content or "" for d in deltas) == "ok"
    assert [d.finish_reason for d in deltas if d.finish_reason] == ["stop"]
    assert not [d.usage for d in deltas if d.usage]


def test_a_metered_plan_turn_reaches_the_usage_ledger(brain_env, monkeypatch) -> None:
    """The whole chain, not just the delta: this is what makes the spend visible.

    ``BrainProviderRegistry.instantiate`` hands every provider through
    ``meter_brain``, which writes each usage block to
    :func:`jarvis.costs.ledger.record_usage`. The plan brain yielded no block,
    so that door opened onto nothing for this route.
    """
    from jarvis.brain.usage_meter import meter_brain

    written: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "jarvis.brain.usage_meter.record_usage",
        lambda **kw: written.append(kw),
    )
    install, _ = brain_env
    install([httpx.Response(200, content=_sse(
        {"type": "response.completed", "response": {"usage": {
            "input_tokens": 800, "output_tokens": 120,
            "input_tokens_details": {"cached_tokens": 300},
        }}},
    ))])
    brain = meter_brain(chatgpt_plan.ChatGPTPlanBrain("chatgpt-plan/gpt-a"), "chatgpt-plan")

    async def drain() -> None:
        async for _ in brain.complete(_req()):
            pass

    asyncio.run(drain())
    assert len(written) == 1, written
    entry = written[0]
    assert entry["provider"] == "chatgpt-plan"
    assert (entry["tokens_in"], entry["tokens_out"], entry["tokens_cached"]) == (500, 120, 300)

# -- the ledger must also know WHICH model spent it -------------------------
# Measured on the live install 2026-10-31: twelve plan turns, 197_807 input
# tokens, and the ledger's model column empty on every one. The route resolves
# its model DURING the call (nothing is pinned, so _prepare falls through to
# the account's model and then to its first listed one), so the name
# MeteredBrain knew before the call was "". The meter already had a contract
# for that case -- it reads ``last_model`` off the plugin, and its comment
# names this very incident -- but no plugin had ever published it. Without the
# model, the most expensive route the product has is also the only one whose
# spending cannot be priced.


def _ledger_entry(brain_env, monkeypatch, brain: Any) -> dict[str, Any]:
    from jarvis.brain.usage_meter import meter_brain

    written: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "jarvis.brain.usage_meter.record_usage", lambda **kw: written.append(kw)
    )
    install, _ = brain_env
    install([httpx.Response(200, content=_sse(
        {"type": "response.completed", "response": {"usage": {
            "input_tokens": 800, "output_tokens": 120,
            "input_tokens_details": {"cached_tokens": 300},
        }}},
    ))])
    metered = meter_brain(brain, "chatgpt-plan")

    async def drain() -> None:
        async for _ in metered.complete(_req()):
            pass

    asyncio.run(drain())
    assert len(written) == 1, written
    return written[0]


def test_an_unpinned_plan_turn_still_records_its_model(brain_env, monkeypatch) -> None:
    """The live shape: no model pinned, so the name exists only after _prepare."""
    brain = chatgpt_plan.ChatGPTPlanBrain()
    assert not getattr(brain, "last_model", ""), "nothing is known before the call"
    entry = _ledger_entry(brain_env, monkeypatch, brain)
    assert entry["model"] == "gpt-a", entry
    assert brain.last_model == "gpt-a"


def test_a_pinned_plan_turn_records_the_pin(brain_env, monkeypatch) -> None:
    """A pin must win over the account default, in the ledger as in the call."""
    entry = _ledger_entry(
        brain_env, monkeypatch, chatgpt_plan.ChatGPTPlanBrain("chatgpt-plan/gpt-pinned")
    )
    assert entry["model"] == "gpt-pinned", entry


def test_api_key_is_never_sent(brain_env) -> None:
    install, _ = brain_env
    server = install([_completed()])
    asyncio.run(_collect(chatgpt_plan.ChatGPTPlanBrain()))
    for req in server.requests:
        assert "sk-must-never-be-sent" not in str(req.headers)
        assert b"sk-must-never-be-sent" not in req.content


def test_missing_plan_scope_blocks_turn(brain_env, plan) -> None:
    install, _ = brain_env
    server = install([])
    _signed_in(plan, scope="openid profile email offline_access")
    with pytest.raises(ChatGPTPlanError) as exc:
        asyncio.run(_collect(chatgpt_plan.ChatGPTPlanBrain()))
    assert exc.value.code == "plan_scope_missing"
    assert not server.requests


def test_the_plan_brain_qualifies_as_a_structured_subscription_seat() -> None:
    """resolve_subscription_brain probes ``structured_prompts``; live 2026-10-24
    the probe's TypeError skipped the user's own chat seat for the conductor."""
    brain = chatgpt_plan.ChatGPTPlanBrain("chatgpt-plan/gpt-a", structured_prompts=True)
    assert brain._model == "gpt-a"
    assert chatgpt_plan.ChatGPTPlanBrain.native_system_prompt is True


def test_the_static_prefix_gets_a_stable_cache_key() -> None:
    """The ~22k tokens of instructions + tool schemas are identical every turn.

    Measured on the live install 2026-10-30: 24 voice turns, 712_650 input
    tokens, 29_694 per turn, and ONE cache hit. OpenAI routes a request to a
    cache by ``prompt_cache_key``; without one the stable prefix lands on a
    different machine each turn and is paid in full. The key must therefore
    depend on the prefix ONLY -- a key that moved with the user's question
    would route every turn somewhere new again, which is today's behaviour.
    """
    first = chatgpt_plan.build_body(_req(), "gpt-a")
    assert first["prompt_cache_key"]

    later = BrainRequest(
        system="Be brief.",
        messages=[
            BrainMessage(role="system", content="Extra rule."),
            BrainMessage(role="user", content="Etwas voellig anderes."),  # i18n-allow
        ],
    )
    assert chatgpt_plan.build_body(later, "gpt-a")["prompt_cache_key"] == first["prompt_cache_key"]

    other_tools = BrainRequest(
        system="Be brief.",
        messages=[BrainMessage(role="system", content="Extra rule.")],
        tools=({"name": "read_file", "description": "Read a file.", "input_schema": {}},),
    )
    assert (
        chatgpt_plan.build_body(other_tools, "gpt-a")["prompt_cache_key"]
        != first["prompt_cache_key"]
    )

    assert (
        chatgpt_plan.build_body(_req(), "gpt-b")["prompt_cache_key"]
        != first["prompt_cache_key"]
    )
# -- the prefix must not move between turns ---------------------------------
# Live 2026-10-31: twelve plan turns, 197_807 input tokens, ZERO cached. The
# log showed why -- instructions were byte-stable at 13_720 tokens while the
# tool array changed every single turn (22, 23, 24 schemas; 4_845-5_114
# tokens), because the shortlist adds "the tool a turn mandated" and reveals
# deferred ones from the next round on. OpenAI's prompt-caching guide:
# "Cache hits are only possible for exact prefix matches within a prompt. This
# also applies to images and tools, which must be identical between requests."
# Its documented remedy is `allowed_tools`, which narrows what may be CALLED
# without shortening what is SENT. These tests hold that line, because a
# plausible-looking refactor can quietly cost 79% of the input bill.


def _tool(name: str) -> dict[str, Any]:
    return {"name": name, "description": f"does {name}", "input_schema": {"type": "object"}}


_SURFACE = tuple(_tool(n) for n in ("run-shell", "screen-snapshot", "spawn-worker", "find_tools"))


def _body(allowed: tuple[str, ...]) -> dict[str, Any]:
    return chatgpt_plan.build_body(
        BrainRequest(
            messages=(BrainMessage(role="user", content="hi"),),
            tools=_SURFACE,
            allowed_tools=allowed,
            system="Be brief.",
        ),
        "gpt-6.1-sol",
    )


def test_the_tool_array_is_identical_whatever_is_allowed() -> None:
    """Two turns, two allow-lists, one byte-identical prefix."""
    first = _body(("run-shell", "find_tools"))
    second = _body(("screen-snapshot", "spawn-worker", "find_tools"))
    assert first["tools"] == second["tools"], "the cached prefix moved"
    assert first["instructions"] == second["instructions"]
    assert first["prompt_cache_key"] == second["prompt_cache_key"], (
        "a key that changes routes every turn to a cold machine"
    )


def test_the_allow_list_is_what_changes() -> None:
    narrow = _body(("run-shell",))
    wide = _body(("run-shell", "spawn-worker"))
    assert narrow["tool_choice"]["type"] == "allowed_tools"
    assert [t["name"] for t in narrow["tool_choice"]["tools"]] == ["run-shell"]
    assert [t["name"] for t in wide["tool_choice"]["tools"]] == ["run-shell", "spawn-worker"]
    assert narrow["tools"] == wide["tools"]


def test_no_restriction_sends_no_tool_choice() -> None:
    """Allowing everything is not a restriction, and must not look like one."""
    assert "tool_choice" not in _body(tuple(t["name"] for t in _SURFACE))
    assert "tool_choice" not in _body(()), "an empty allow-list means no restriction"


def test_a_brain_without_the_capability_keeps_a_filtered_list() -> None:
    """The 67_852-token refusal must stay prevented for everyone else."""
    from jarvis.brain.tool_use_loop import ToolUseLoop

    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name
            self.description = f"does {name}"
            self.schema = {"type": "object"}

    loop = ToolUseLoop.__new__(ToolUseLoop)
    loop._tools = {n: _Tool(n) for n in ("a", "b", "c")}
    loop._hidden = {"c"}

    loop._brain = object()                      # declares nothing
    schemas, allowed = loop._tool_surface()
    assert [s["name"] for s in schemas] == ["a", "b"], "a plain brain must stay filtered"
    assert allowed == ()

    class _Capable:
        supports_allowed_tools = True

    loop._brain = _Capable()
    schemas, allowed = loop._tool_surface()
    assert [s["name"] for s in schemas] == ["a", "b", "c"], "the full list is the stable one"
    assert allowed == ("a", "b")
