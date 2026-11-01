from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvis.core.protocols import BrainDelta
from jarvis.speech.pipeline import PipelineState, SpeechPipeline
from jarvis.voice.subscription_profile import (
    CODEX_SUBSCRIPTION_VOICE_PROFILE,
    CodexSubscriptionVoiceBrain,
    configured_voice_profile,
    subscription_voice_capability,
)


def _config(*, profile: str = "", realtime_provider: str = "") -> SimpleNamespace:
    realtime = SimpleNamespace(provider=realtime_provider) if realtime_provider else None
    return SimpleNamespace(
        voice=SimpleNamespace(profile=profile, mode="realtime"),
        brain=SimpleNamespace(reply_language="auto", realtime=realtime),
    )


def test_only_explicit_profile_selects_the_classic_subscription_composition() -> None:
    assert (
        configured_voice_profile(profile := _config(profile=CODEX_SUBSCRIPTION_VOICE_PROFILE))
        == CODEX_SUBSCRIPTION_VOICE_PROFILE
    )
    assert profile.voice.mode == "realtime"
    assert configured_voice_profile(
        _config(realtime_provider="codex-subscription-realtime")
    ) == ""
    assert configured_voice_profile(_config()) == ""


def test_realtime_provider_pin_does_not_change_the_selected_engine() -> None:
    pipeline = SpeechPipeline.__new__(SpeechPipeline)
    pipeline._config = _config(realtime_provider="codex-subscription-realtime")

    assert pipeline._configured_voice_mode() == "realtime"


def test_profile_can_be_applied_and_removed_without_process_restart() -> None:
    delegate = object()
    config = _config()
    pipeline = SpeechPipeline.__new__(SpeechPipeline)
    pipeline._config = config
    pipeline._base_brain = delegate
    pipeline._brain = delegate
    pipeline._state = PipelineState.IDLE

    assert pipeline.apply_voice_profile(CODEX_SUBSCRIPTION_VOICE_PROFILE) is False
    assert isinstance(pipeline._brain, CodexSubscriptionVoiceBrain)
    assert config.voice.mode == "pipeline"

    assert pipeline.apply_voice_profile("") is False
    assert pipeline._brain is delegate


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_capability_uses_the_same_composed_path_on_every_desktop_os(
    platform: str,
) -> None:
    config = _config(profile=CODEX_SUBSCRIPTION_VOICE_PROFILE)
    config.stt = SimpleNamespace(provider="nemotron-local")
    config.tts = SimpleNamespace(provider="piper-local")

    capability = subscription_voice_capability(
        config,
        account_ready=True,
        runtime_attached=True,
        display_present=True,
        platform=platform,
    )

    assert capability.available is True
    assert capability.reason == "ready"


def test_capability_reports_headless_audio_as_unavailable() -> None:
    config = _config(profile=CODEX_SUBSCRIPTION_VOICE_PROFILE)
    config.stt = SimpleNamespace(provider="nemotron-local")
    config.tts = SimpleNamespace(provider="piper-local")

    capability = subscription_voice_capability(
        config,
        account_ready=True,
        runtime_attached=True,
        display_present=False,
        platform="linux",
    )

    assert capability.available is False
    assert capability.reason == "headless_audio_unavailable"


@pytest.mark.asyncio
async def test_conversation_turns_stream_through_subscription_with_history() -> None:
    requests = []

    class Subscription:
        async def complete(self, request):
            requests.append(request)
            yield BrainDelta(content="Hello.")
            yield BrainDelta(finish_reason="stop")

    class Delegate:
        @staticmethod
        def _turn_has_action_intent(_text: str) -> bool:
            return False

        async def generate_stream(self, *_args, **_kwargs):
            pytest.fail("conversational turns must not use the router brain")
            yield ""

    brain = CodexSubscriptionVoiceBrain(Delegate(), _config())
    brain._subscription = Subscription()

    first = [chunk async for chunk in brain.generate_stream("Share a short joke.")]
    second = [chunk async for chunk in brain.generate_stream("Tell me something amusing.")]

    assert first == ["Hello."]
    assert second == ["Hello."]
    assert [message.role for message in requests[1].messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert "REPLY LANGUAGE" in str(requests[0].system)


@pytest.mark.asyncio
async def test_subscription_failure_before_first_delta_uses_provider_chain() -> None:
    delegated = []

    class Subscription:
        async def complete(self, _request):
            raise RuntimeError("subscription transport is busy")
            yield BrainDelta()  # pragma: no cover - preserves async-generator shape

    class Delegate:
        _last_turn_all_failed = False
        _last_turn_suppressed = False
        _last_turn_executed_action_tool = False

        @staticmethod
        def _turn_has_action_intent(_text: str) -> bool:
            return False

        async def generate_stream(self, text, **_kwargs):
            delegated.append(text)
            yield "Fallback answer."

    brain = CodexSubscriptionVoiceBrain(Delegate(), _config())
    brain._subscription = Subscription()

    result = [chunk async for chunk in brain.generate_stream("Tell me a short joke.")]

    assert result == ["Fallback answer."]
    assert delegated == ["Tell me a short joke."]


@pytest.mark.asyncio
async def test_registered_action_overrides_a_native_conversation_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated = []

    class Delegate:
        _last_turn_all_failed = False
        _last_turn_suppressed = False
        _last_turn_executed_action_tool = True

        @staticmethod
        def _turn_has_action_intent(_text: str) -> bool:
            return True

        async def generate_stream(self, text, **_kwargs):
            delegated.append(text)
            yield "Action completed."

    monkeypatch.setattr(
        "jarvis.voice.subscription_profile.plan_turn",
        lambda *_args, **_kwargs: SimpleNamespace(
            requires_orchestrator=False,
            reasons=frozenset(),
        ),
    )
    brain = CodexSubscriptionVoiceBrain(Delegate(), _config())

    result = [
        chunk
        async for chunk in brain.generate_stream("Use the connected service for this request.")
    ]

    assert result == ["Action completed."]
    assert delegated == ["Use the connected service for this request."]


@pytest.mark.asyncio
async def test_action_turns_keep_the_existing_orchestrator() -> None:
    delegated = []

    class Delegate:
        _last_turn_all_failed = False
        _last_turn_suppressed = False
        _last_turn_executed_action_tool = True

        @staticmethod
        def _turn_has_action_intent(_text: str) -> bool:
            return True

        async def generate_stream(self, text, **_kwargs):
            delegated.append(text)
            yield "Done."

    brain = CodexSubscriptionVoiceBrain(Delegate(), _config())
    result = [chunk async for chunk in brain.generate_stream("Open Chrome now.")]

    assert result == ["Done."]
    assert delegated == ["Open Chrome now."]
    assert brain._last_turn_executed_action_tool is True


# --- persona on the voice paths (P5): one character on chat, pipeline and live ---


def test_subscription_voice_prompt_carries_the_persona_and_the_language_pin() -> None:
    """The transport answered as a generic assistant without the persona block."""
    from jarvis.voice.subscription_profile import subscription_system_prompt

    prompt = subscription_system_prompt("de")

    assert "REPLY LANGUAGE" in prompt
    assert "HUMOUR" in prompt.upper()
    assert "Christian" in prompt
    # The persona comes first; the language pin must still win as the last word.
    assert prompt.index("Christian") < prompt.index("REPLY LANGUAGE")


def test_subscription_voice_prompt_uses_the_one_central_persona_loader(monkeypatch) -> None:
    """A custom prompt set in settings must reach voice, not only the chat."""
    import jarvis.brain.persona_loader as loader

    monkeypatch.setattr(loader, "read_custom_prompt", lambda: "You are a laconic test rig.")
    from jarvis.voice.subscription_profile import subscription_system_prompt

    prompt = subscription_system_prompt("en")

    assert "You are a laconic test rig." in prompt
    assert "REPLY LANGUAGE" in prompt
    assert "Christian" not in prompt


def test_persona_fault_still_leaves_a_usable_language_pin(monkeypatch) -> None:
    import jarvis.brain.persona_loader as loader

    def boom(*_args, **_kwargs):
        raise OSError("persona unreadable")

    monkeypatch.setattr(loader, "load_effective_persona_prompt", boom)
    from jarvis.voice.subscription_profile import subscription_system_prompt

    prompt = subscription_system_prompt("de")

    assert "REPLY LANGUAGE" in prompt
    assert "Christian" not in prompt


def test_live_session_instructions_carry_the_persona_on_the_speaking_model() -> None:
    """The voice the user hears gets the character, inside the prompt budget.

    The delegation backend deliberately does not repeat it: one
    ``session.started`` message carries both prompts, the tool declarations and
    the history seed under 20 kB, and spending the persona twice overran that
    by 253 bytes.
    """
    from jarvis.live.config import LiveConfig

    config = LiveConfig(configured=True, backend_model="gpt-5")
    session = config.session_config(language="de", tools=[], identity="You are Jarvis.")

    assert "Christian" in session["instructions"]
    assert "Christian" not in session["delegation"]["responses"]["instructions"]
    # Identity stays the first thing the model reads.
    assert session["instructions"].startswith("You are Jarvis.")
    budget = session["instructions"] + session["delegation"]["responses"]["instructions"]
    assert len(budget.encode("utf-8")) < 20_000


def test_persona_handbook_keeps_the_humour_safety_rules() -> None:
    """The carve-outs are the point: wit is never allowed to override care."""
    from jarvis.brain.persona_loader import (
        load_compact_persona_prompt,
        load_persona_prompt,
    )

    for block in (load_persona_prompt(), load_compact_persona_prompt()):
        lowered = block.lower()
        assert "christian" in lowered
        # No jokes where they would do damage.
        assert "security" in lowered
        assert "destructive" in lowered
        # And never the same gag on repeat.
        assert "never the same joke" in lowered or "never the same one" in lowered
