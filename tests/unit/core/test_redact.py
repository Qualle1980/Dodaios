"""``jarvis.core.redact`` — the redact-and-cap gate for decision-log previews.

Every persisted telemetry preview (a tool's ``output_preview``, the brain's
``rationale``) passes through ``safe_preview`` before it reaches the bus / the
session DB / the local Markdown diary. These tests pin the two guarantees that
make persisting such previews safe: credential shapes are masked, and the length
is capped so a tool cannot smuggle a large blob into the log.
"""
from __future__ import annotations

import logging

import pytest

from jarvis.core.redact import DEFAULT_PREVIEW_CHARS, redact_secrets, safe_preview


def test_plain_text_passes_through_untouched() -> None:
    text = "Listed 3 GCP projects: alpha, beta, gamma. Total spend last month: 12.40 EUR."
    assert safe_preview(text) == text


def test_openai_style_key_is_masked() -> None:
    leaked = "your key is sk-proj-AbCdEf0123456789ghijKLmnopQRstuv and it works"
    out = safe_preview(leaked)
    assert "sk-proj-AbCdEf0123456789ghijKLmnopQRstuv" not in out
    assert "<redacted:openai_key>" in out
    # The surrounding prose survives so the log stays readable.
    assert out.startswith("your key is ")


def test_provider_and_bearer_tokens_are_masked() -> None:
    google = "AIzaSyA1234567890abcdefghijklmnopqrstuvwx"
    bearer = "Bearer abcDEF1234567890ghijKL"
    out = safe_preview(f"google={google}\nauth={bearer}")
    assert google not in out
    assert "abcDEF1234567890ghijKL" not in out
    assert "<redacted:provider_key>" in out
    assert "<redacted:bearer_token>" in out


def test_labelled_secret_keeps_label_drops_value() -> None:
    out = safe_preview("api_key=supersecretvalue123")
    assert "supersecretvalue123" not in out
    # Label survives for readability; value is gone.
    assert "api_key=" in out
    assert "<redacted:labelled_secret>" in out


def test_labelled_secret_masks_python_and_json_mapping_previews() -> None:
    value = "supersecretvalue123"

    python_preview = safe_preview({"api_key": value})
    json_preview = safe_preview('{"access_token": "supersecretvalue123"}')

    assert value not in python_preview
    assert value not in json_preview
    assert "<redacted:labelled_secret>" in python_preview
    assert "<redacted:labelled_secret>" in json_preview


def test_git_sha1_is_not_treated_as_a_secret() -> None:
    # A 40-char hex is a git commit SHA — legitimate, must NOT be masked.
    sha = "a" * 40
    assert sha in safe_preview(f"committed at {sha}")


def test_length_is_capped_with_honest_marker() -> None:
    # Spaced prose so it is long without being one 64+ char credential-shaped run.
    big = "word " * (DEFAULT_PREVIEW_CHARS)  # ~5x over the cap
    out = safe_preview(big)
    assert len(out) <= DEFAULT_PREVIEW_CHARS + 40  # cap + short marker
    assert "more chars)" in out


def test_custom_cap_is_respected() -> None:
    out = safe_preview("abcdefghij", max_chars=4)
    assert out.startswith("abcd")
    assert "more chars)" in out


def test_none_and_non_string_values_are_stringified() -> None:
    assert safe_preview(None) == ""
    assert safe_preview({"projects": 3}) == '{"projects": 3}'
    assert safe_preview(42) == "42"


def test_redact_secrets_is_pure_and_idempotent() -> None:
    once = redact_secrets("token: eyJabcdef.eyJ123456.sigABC123")
    twice = redact_secrets(once)
    assert once == twice
    assert "<redacted:jwt>" in once


def test_redacts_credentials_embedded_in_http_request_urls() -> None:
    google_key = "AQ." + "a" * 40
    telegram_token = "123456789:" + "B" * 35
    text = (
        f"GET https://example.test/models?key={google_key}&page=1 "
        f"POST https://api.telegram.org/bot{telegram_token}/getUpdates"
    )

    redacted = redact_secrets(text)

    assert google_key not in redacted
    assert telegram_token not in redacted
    assert "?key=<redacted:query_secret>&page=1" in redacted
    assert "bot<redacted:telegram_bot_token>/getUpdates" in redacted


def test_structured_values_preview_as_json_not_python_repr() -> None:
    # The reasoning trace turns a result into a sentence; it can read JSON,
    # not ``{'computers': []}``.
    assert safe_preview({"computers": [], "ok": True}) == '{"computers": [], "ok": true}'
    assert safe_preview(["café", 2]) == '["café", 2]'
    # Anything JSON cannot encode still previews instead of raising.
    assert safe_preview({"at": object()}).startswith('{"at": "<object object')


# --- RedactingLogFilter ----------------------------------------------------------

_BOT_TOKEN = "987654321:AAFakeTokenForTestsOnly_abcdefghijklmnop"  # noqa: S105 - a fake


@pytest.mark.parametrize(
    "level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
)
def test_the_redacting_filter_masks_every_level_and_the_logged_exception(
    caplog: pytest.LogCaptureFixture, level: int
) -> None:
    """A record's message, its arguments, and the traceback printed under it
    (the error and the cause it was raised from) all lose the token."""
    from jarvis.core.redact import redact_logger

    name = "jarvis.test.redact_probe_levels"
    logger = logging.getLogger(name)
    redact_logger(name)
    url = f"https://api.telegram.org/bot{_BOT_TOKEN}/getUpdates"
    try:
        try:
            raise ConnectionError(f"cannot reach {url}")
        except ConnectionError as cause:
            raise RuntimeError(f"The token `{_BOT_TOKEN}` was rejected by the server.") from cause
    except RuntimeError:
        with caplog.at_level(logging.DEBUG, logger=name):
            logger.log(level, "polling %s failed", url, exc_info=True, stack_info=True)

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert _BOT_TOKEN not in caplog.text
    assert _BOT_TOKEN not in record.getMessage()
    # No handler can format the original exception again.
    assert record.exc_info is None and _BOT_TOKEN not in (record.exc_text or "")
    assert "RuntimeError" in caplog.text and "ConnectionError" in caplog.text


def test_the_redacting_filter_leaves_a_clean_exception_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from jarvis.core.redact import redact_logger

    name = "jarvis.test.redact_probe_clean"
    logger = logging.getLogger(name)
    redact_logger(name)
    try:
        raise ValueError("nothing secret here")
    except ValueError:
        with caplog.at_level(logging.DEBUG, logger=name):
            logger.exception("it failed")
    assert caplog.records[0].exc_info is not None  # untouched: nothing to mask
    assert "nothing secret here" in caplog.text


# --- tool_args_shape / log_tool_call (INFO keeps diagnosis, drops free text) ---

_DICTATED_BODY = (
    "Please tell Maria that the meeting moved to Thursday at three "
    "and her private note about the clinic stay stays between us."
)


def test_tool_args_shape_keeps_keys_and_hides_dictated_text() -> None:
    from jarvis.core.redact import tool_args_digest, tool_args_shape

    args = {
        "action": "send",
        "to": "maria@example.com",
        "body": _DICTATED_BODY,
    }
    shape = tool_args_shape(args)
    assert "action:str=send" in shape
    assert "body:str:" in shape
    assert "to:str:" in shape  # email has @ — not enum-like, length only
    assert f"digest={tool_args_digest(args)}" in shape
    for word in ("Maria", "Thursday", "clinic", "private", "meeting"):
        assert word not in shape
    assert _DICTATED_BODY not in shape


def test_tool_args_shape_distinguishes_identical_and_different_calls() -> None:
    from jarvis.core.redact import tool_args_digest, tool_args_shape

    first = {"action": "status", "filter": "open"}
    same = {"filter": "open", "action": "status"}  # key order must not matter
    other = {"action": "status", "filter": "done"}
    assert tool_args_digest(first) == tool_args_digest(same)
    assert tool_args_digest(first) != tool_args_digest(other)
    assert tool_args_shape(first) == tool_args_shape(same)
    assert tool_args_shape(first) != tool_args_shape(other)


def test_tool_args_shape_masks_credential_shaped_scalars() -> None:
    from jarvis.core.redact import tool_args_shape

    secret = "sk-proj-AbCdEf0123456789ghijKLmnopQRstuv"
    shape = tool_args_shape({"api_key": secret, "action": "run"})
    assert secret not in shape
    assert "api_key:str:" in shape
    assert "action:str=run" in shape


def test_log_tool_call_info_omits_body_debug_keeps_redacted_preview(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from jarvis.core.redact import log_tool_call

    name = "jarvis.test.tool_args_log"
    logger = logging.getLogger(name)
    secret = "sk-proj-AbCdEf0123456789ghijKLmnopQRstuv"
    args = {"message": _DICTATED_BODY, "token": secret, "action": "draft"}
    with caplog.at_level(logging.DEBUG, logger=name):
        log_tool_call(logger, "compose_note", args)

    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(info_records) == 1
    assert len(debug_records) == 1
    info_msg = info_records[0].getMessage()
    debug_msg = debug_records[0].getMessage()

    assert "compose_note" in info_msg
    assert "message:str:" in info_msg
    assert "action:str=draft" in info_msg
    assert "digest=" in info_msg
    for word in ("Maria", "Thursday", "clinic", "private", "meeting"):
        assert word not in info_msg
    assert secret not in info_msg
    assert secret not in debug_msg
    assert "<redacted:openai_key>" in debug_msg
    # DEBUG still carries the redacted full arguments for local debugging.
    assert "args=" in debug_msg
    assert "draft" in debug_msg
