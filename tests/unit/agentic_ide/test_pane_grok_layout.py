"""Grok's real composer layout must read as ready (screen captured 2026-11-17).

The composer sits in a box, and a status line follows it. Grok's first fix
required the bottom row to be the marker, so this layout stayed not-ready.
"""

from __future__ import annotations

from jarvis.agentic_ide.fleet_actions import bottom_input_content

GROK_IDLE = [
    "  Help improve Grok [Opt out] [Opt in]",
    "  │ >                                                                       │",
    "  ╰────────────────────────────────────────────────────────── Grok 4.5 (medium) · always-approve ─╯",
    "  scratch-chain-test │ Grok 4.5 │ 1% c",
]

GROK_WITH_OLD_PROMPT = [
    "> build the calc module",
    "  Done: calc.py written.",
    "  scratch-chain-test │ Grok 4.5 │ 1% c",
]

CONSENT_ONLY = [
    "  Help improve Grok                                                  [Opt out] [Opt in]",
    "  Off by default. Opt-in to allow SpaceXAI to retain coding data.",
    "  Read Terms and Privacy Policy.",
]


def test_grok_boxed_composer_with_status_line_below_is_ready() -> None:
    assert bottom_input_content(GROK_IDLE) == ""


def test_unboxed_old_prompt_above_status_line_is_not_ready() -> None:
    assert bottom_input_content(GROK_WITH_OLD_PROMPT) is None


def test_consent_banner_is_not_ready() -> None:
    assert bottom_input_content(CONSENT_ONLY) is None
