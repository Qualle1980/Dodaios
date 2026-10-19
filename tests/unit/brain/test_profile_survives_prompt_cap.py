"""Who the user is must survive a brain's prompt cap.

A text-only brain keeps only the HEAD of the system prompt: the ChatGPT-web
bridge cuts at ``MAX_SYSTEM_CHARS`` because every character is typed into the
web composer. The profile block used to sit behind the persona and the standing
instructions, so it started around 18_000 characters and was cut away: asked
who the user was, the assistant said it did not know their name while USER.md
held it all along (live 2026-10-19).

These tests pin the position, not the wording, so a future block added to the
prompt cannot silently push identity out of reach again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import jarvis.core.config as core_config
from jarvis.brain.manager import BrainManager
from jarvis.core.config import load_config
from jarvis.memory.user_profile import UserProfile
from jarvis.plugins.brain.chatgpt_web import MAX_SYSTEM_CHARS

_USER_MD = """---
schema_version: 1
subject_type: user
identity:
  name: Testperson
  preferred_address: Chef
  pronouns: Du
  languages:
  - deutsch
  primary_language: de
  timezone: Europe/Berlin
---

Some free-text notes about the user.
"""


def _manager(profile: UserProfile | None) -> BrainManager:
    """A BrainManager with __init__ bypassed — only the attrs the prompt needs."""
    m = BrainManager.__new__(BrainManager)
    m._soul = None
    m._user_profile = profile
    # The prompt offers the profile-update tool only when it is in the tool set;
    # an empty dict keeps that branch off without hiding the profile block.
    m._tools = {}
    m._people = None
    m._core_memory = None
    m._system_prompt_extra = ""
    m._wiki_context_suffix = ""
    m._reply_language = "auto"
    cfg = load_config()
    cfg.performance.cache_optimized_prompt = False
    m._config = cfg
    return m


@pytest.fixture(autouse=True)
def _isolate_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(core_config, "DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def profile(tmp_path: Path) -> UserProfile:
    path = tmp_path / "USER.md"
    path.write_text(_USER_MD, encoding="utf-8")
    return UserProfile.load(path)


def test_the_name_is_still_there_after_the_cap(profile: UserProfile) -> None:
    """The real failure: a clipped prompt no longer contained the name."""
    prompt = _manager(profile)._build_system_prompt()
    assert "Testperson" in prompt[:MAX_SYSTEM_CHARS]


def test_identity_comes_before_the_guidance_blocks(profile: UserProfile) -> None:
    """Guidance is long; identity is short. Short and essential goes first."""
    prompt = _manager(profile)._build_system_prompt()
    assert prompt.index("Testperson") < prompt.index("voice companion")


def test_no_profile_still_builds(profile: UserProfile) -> None:
    prompt = _manager(None)._build_system_prompt()
    assert "Testperson" not in prompt
    assert prompt.strip()
