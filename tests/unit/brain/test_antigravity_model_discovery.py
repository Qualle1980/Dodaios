"""Antigravity must ask the CLI which model it may run, never assume one.

agy reversed its own contract twice. In 1.0.10 it ignored ``--model`` and
required ``--effort`` beside it; in 1.5.1 it validates ``--model`` against the
account's list and refuses ``--effort`` when the id already carries a level.
A hardcoded id therefore ages into a hard failure, and this provider's did:
``model gemini-3.5-flash is not recognized as a known model``, exit 1 on every
turn.

So the id is resolved from what the binary reports. These tests pin the two
pure pieces of that: parsing the report and choosing from it.
"""
from __future__ import annotations

from jarvis.plugins.brain.antigravity import (
    _agy_effort,
    _build_argv,
    parse_agy_models,
    pick_agy_model,
)
from jarvis.google_cli.resolver import GoogleCli

# What `agy models` printed on a real account, 2026-11-05.
_REAL_OUTPUT = (
    "gemini-3.8-flash-high\tGemini 3.8 Flash (High)\n"
    "gemini-3.8-flash-medium\tGemini 3.8 Flash (Medium)\n"
    "gemini-3.8-flash-low\tGemini 3.8 Flash (Low)\n"
    "claude-sonnet-4-6\tClaude Sonnet 4.6 (Thinking)\n"
    "claude-opus-4-6-thinking\tClaude Opus 4.6 (Thinking)\n"
    "gpt-oss-120b-medium\tGPT-OSS 120B (Medium)\n"
)
_REAL_IDS = (
    "gemini-3.8-flash-high",
    "gemini-3.8-flash-medium",
    "gemini-3.8-flash-low",
    "claude-sonnet-4-6",
    "claude-opus-4-6-thinking",
    "gpt-oss-120b-medium",
)


class TestParsing:
    def test_reads_the_ids_from_a_real_report(self) -> None:
        assert parse_agy_models(_REAL_OUTPUT) == _REAL_IDS

    def test_ignores_the_progress_line_and_blanks(self) -> None:
        noisy = "Fetching available models...\n\n" + _REAL_OUTPUT + "\n"
        assert parse_agy_models(noisy) == _REAL_IDS

    def test_tolerates_extra_columns(self) -> None:
        assert parse_agy_models("some-model\tSome Model\textra\n") == ("some-model",)

    def test_nothing_usable_is_an_empty_result_not_a_crash(self) -> None:
        assert parse_agy_models("") == ()
        assert parse_agy_models("no tabs here\n") == ()


class TestChoosing:
    def test_an_offered_id_is_used_unchanged(self) -> None:
        assert pick_agy_model(_REAL_IDS, "claude-sonnet-4-6", "high") == "claude-sonnet-4-6"

    def test_a_retired_id_falls_back_to_the_preferred_family(self) -> None:
        # The exact failure that broke the provider: a configured id agy no
        # longer knows must not reach the command line.
        assert pick_agy_model(_REAL_IDS, "gemini-3.5-flash", "medium") == (
            "gemini-3.8-flash-medium"
        )

    def test_the_effort_picks_the_level(self) -> None:
        for effort, expected in (
            ("low", "gemini-3.8-flash-low"),
            ("medium", "gemini-3.8-flash-medium"),
            ("high", "gemini-3.8-flash-high"),
        ):
            assert pick_agy_model(_REAL_IDS, "gemini-3.5-flash", effort) == expected

    def test_a_levelless_family_id_gains_the_wanted_level(self) -> None:
        assert pick_agy_model(_REAL_IDS, "gemini-3.8-flash", "low") == "gemini-3.8-flash-low"

    def test_a_wrong_level_on_a_real_family_is_corrected(self) -> None:
        assert pick_agy_model(_REAL_IDS, "gemini-3.8-flash-high", "low") == (
            "gemini-3.8-flash-high"
        ), "an exactly offered id wins over the effort hint"

    def test_an_unknown_family_still_resolves_to_something_runnable(self) -> None:
        chosen = pick_agy_model(_REAL_IDS, "something-nobody-ships", "high")
        assert chosen in _REAL_IDS

    def test_no_report_leaves_the_caller_s_id_alone(self) -> None:
        # Discovery failing is not licence to invent an id: degrade to the old
        # behaviour and let agy speak for itself.
        assert pick_agy_model((), "gemini-3.5-flash", "medium") == "gemini-3.5-flash"


class TestEffortLadder:
    def test_jarvis_levels_map_onto_agy_levels(self) -> None:
        class Req:
            def __init__(self, effort: str) -> None:
                self.reasoning_effort = effort

        assert _agy_effort(Req("none")) == "low"
        assert _agy_effort(Req("low")) == "low"
        assert _agy_effort(Req("medium")) == "medium"
        assert _agy_effort(Req("high")) == "high"
        assert _agy_effort(Req("xhigh")) == "high"
        assert _agy_effort(Req("max")) == "high"
        assert _agy_effort(Req("")) == "medium"


class TestArgv:
    def test_agy_never_receives_the_effort_flag(self) -> None:
        # agy 1.5.1: "--model gemini-3.8-flash-low conflicts with
        # --effort=medium". The level is in the id; the flag must not appear.
        cli = GoogleCli(kind="agy", argv_prefix=["agy"], version=None)
        argv = _build_argv(cli, "prompt", "gemini-3.8-flash-low", "medium")
        assert "--effort" not in argv
        assert argv[argv.index("--model") + 1] == "gemini-3.8-flash-low"

    def test_the_gemini_cli_path_keeps_its_own_flags(self) -> None:
        cli = GoogleCli(kind="gemini", argv_prefix=["gemini"], version=None)
        argv = _build_argv(cli, "prompt", "gemini-3.5-flash", "medium")
        assert "--approval-mode" in argv, "the fallback path is untouched"
