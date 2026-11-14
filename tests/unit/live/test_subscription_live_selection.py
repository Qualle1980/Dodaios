"""Selecting the ChatGPT-subscription live call, end to end on disk. No network.

Every case writes a throwaway config with the shipped writer and reads the
decision back, so the chain the user actually depends on -- writer, TOML,
provider pin, voice mode -- is proven without opening a call or touching the
real configuration.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.core import config_writer

_MINIMAL = """\
[voice]
mode = "pipeline"

[tts]
provider = "supertonic-local"
"""

SUBSCRIPTION_PROFILE = {
    "configured": True,
    "auth_mode": "chatgpt_subscription",
    "subscription_voice": "cove",
    "subscription_backend_model": "gpt-5",
    "subscription_reasoning_effort": "medium",
}


@pytest.fixture(autouse=True)
def _no_repo_side_effects(tmp_path: Path, monkeypatch) -> None:
    """Keep the writer's drift-sync out of the tracked repo file.

    ``set_live_profile`` mirrors its selection into the drift-sync file beside
    the real config. Redirecting it means these cases cannot leave the working
    tree dirty, which is what a throwaway config is for.
    """
    monkeypatch.setattr(
        config_writer,
        "_config_soll_path",  # i18n-allow: internal drift-sync identifier
        lambda: tmp_path / "config-soll.json",  # i18n-allow: same identifier
    )


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "jarvis.toml"
    path.write_text(_MINIMAL, encoding="utf-8")
    return path


def _read(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8-sig"))


def _cfg_from(doc: dict) -> SimpleNamespace:
    """The shape the realtime factory reads, built from written TOML."""
    return SimpleNamespace(
        voice=SimpleNamespace(
            mode=doc.get("voice", {}).get("mode", "pipeline"),
            profile=doc.get("voice", {}).get("profile", ""),
        ),
        brain=SimpleNamespace(
            realtime=SimpleNamespace(
                provider=doc.get("brain", {}).get("realtime", {}).get("provider", ""),
                fallback_provider=None,
                fallback_provider_2=None,
            )
        ),
    )


def _fake_registry(monkeypatch) -> None:
    """The shipped adapter's capability flags, without needing its entry point."""
    import jarvis.realtime.factory as factory

    class _SubscriptionLive:
        supports_realtime = True
        client_managed_delegation = True
        external_credentials_kind = "chatgpt_oauth"
        credential_candidates = ()
        provider_fallback_allowed = False
        implicit_usage_fallback_allowed = False

    plugins = {"openai-live-subscription": _SubscriptionLive}
    monkeypatch.setattr(factory, "list_plugins", lambda _group: list(plugins))
    monkeypatch.setattr(factory, "load", lambda _group, name, protocol=None: plugins[name])


# ------------------------------------------------------------------ selection


def test_the_writer_selects_the_subscription_provider_and_realtime_mode(
    config_path: Path,
) -> None:
    config_writer.set_live_profile(SUBSCRIPTION_PROFILE, path=config_path)

    doc = _read(config_path)

    assert doc["live"]["auth_mode"] == "chatgpt_subscription"
    assert doc["brain"]["realtime"]["provider"] == "openai-live-subscription"
    assert doc["voice"]["mode"] == "realtime"
    # No API-key model is pinned anywhere by this selection.
    assert doc["brain"]["realtime"]["model"] == "gpt-live-1-codex"


def test_the_written_selection_actually_reaches_the_live_transport(
    config_path: Path, monkeypatch
) -> None:
    """The decision the session builder makes, read back from the written file."""
    from jarvis.realtime.factory import (
        _realtime_is_the_configured_voice_mode,
        subscription_live_pin,
    )

    _fake_registry(monkeypatch)
    config_writer.set_live_profile(SUBSCRIPTION_PROFILE, path=config_path)
    cfg = _cfg_from(_read(config_path))

    assert subscription_live_pin(cfg) == "openai-live-subscription"
    assert _realtime_is_the_configured_voice_mode(cfg) is True


def test_a_leftover_pipeline_profile_does_not_block_the_selection(
    config_path: Path, monkeypatch
) -> None:
    """An install that once used the pipeline composition still gets its call."""
    from jarvis.realtime.factory import _realtime_is_the_configured_voice_mode

    _fake_registry(monkeypatch)
    config_writer.set_voice_profile("codex-subscription-voice", path=config_path)
    config_writer.set_live_profile(SUBSCRIPTION_PROFILE, path=config_path)

    doc = _read(config_path)
    assert doc["voice"]["profile"] == "codex-subscription-voice"
    assert _realtime_is_the_configured_voice_mode(_cfg_from(doc)) is True


# ------------------------------------------------------------------ refusals


def test_an_incomplete_selection_is_refused_and_changes_nothing(
    config_path: Path,
) -> None:
    before = config_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        config_writer.set_live_profile(
            {**SUBSCRIPTION_PROFILE, "subscription_backend_model": ""},
            path=config_path,
        )

    assert config_path.read_text(encoding="utf-8") == before


def test_an_unselected_profile_is_refused(config_path: Path) -> None:
    before = config_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        config_writer.set_live_profile(
            {**SUBSCRIPTION_PROFILE, "configured": False}, path=config_path
        )

    assert config_path.read_text(encoding="utf-8") == before


def test_pipeline_stays_the_mode_until_a_selection_is_written(
    config_path: Path, monkeypatch
) -> None:
    from jarvis.realtime.factory import _realtime_is_the_configured_voice_mode

    _fake_registry(monkeypatch)

    assert _realtime_is_the_configured_voice_mode(_cfg_from(_read(config_path))) is False


# ------------------------------------------------------------------ rollback


def test_the_writer_keeps_a_restorable_copy_of_the_previous_config(
    config_path: Path, monkeypatch
) -> None:
    """The documented way back to the pipeline configuration."""
    from jarvis.realtime.factory import _realtime_is_the_configured_voice_mode

    _fake_registry(monkeypatch)
    before = config_path.read_text(encoding="utf-8")

    config_writer.set_live_profile(SUBSCRIPTION_PROFILE, path=config_path)
    backup = config_path.with_name(config_path.name + ".pre-live.bak")

    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == before

    # Restoring it puts the install back on the pipeline, with no live pin.
    config_path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    doc = _read(config_path)

    assert doc["voice"]["mode"] == "pipeline"
    assert "live" not in doc
    assert _realtime_is_the_configured_voice_mode(_cfg_from(doc)) is False
    # The rollback must not take the speech settings with it.
    assert doc["tts"]["provider"] == "supertonic-local"
