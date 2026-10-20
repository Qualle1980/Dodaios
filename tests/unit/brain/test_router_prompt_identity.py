"""The router prompt names no user: identity comes from config.

The prompt used to address one maintainer's name; every other install read
someone else's name in its dispatcher rules. The assistant name is rewritten
from the wake word by ``RouterBrain``; the user is addressed neutrally.
"""

from jarvis.brain.router import IDENTITY_OPENER, SYSTEM_PROMPT

_RENAMED = "Du bist Athena."  # i18n-allow: German router prompt text
_ROLE = "Als Router bist du Dispatcher"  # i18n-allow: German router prompt text


def test_prompt_hardcodes_no_personal_name():
    assert "Ruben" not in SYSTEM_PROMPT


def test_prompt_opens_with_the_rewritable_identity_sentence():
    assert SYSTEM_PROMPT.startswith(IDENTITY_OPENER)
    renamed = SYSTEM_PROMPT.replace(IDENTITY_OPENER, _RENAMED, 1)
    assert renamed.startswith(f"{_RENAMED} {_ROLE}")
