"""Universal claims must not ride on a limited tool sample.

Measured 2026-11-15: ``gh pr list --limit 3`` returned three merged PRs and
the model said there were no open PRs while thirteen drafts were open. The
honesty backstop already catches claims without a tool; it did not catch a
universal inventory claim over a deliberate sample.
"""

from __future__ import annotations

from jarvis.brain.action_honesty import (
    annotate_command_sample,
    has_universal_claim_over_sample,
    replace_unbacked_action_claim,
    sample_limit_from_command,
)


def test_universal_claim_over_limited_gh_sample_is_rewritten() -> None:
    # The German reply measured live on 2026-11-15.
    spoken = "Offene PRs gibt es im Moment keine."  # i18n-allow: quoted live reply
    command = "gh pr list --limit 3"
    result = annotate_command_sample(
        "MERGED\tfeat A\nMERGED\tfeat B\nMERGED\tfeat C\n[exit 0 · 0.4s]",
        command,
    )
    assert "sample of 3" in result
    honest = replace_unbacked_action_claim(
        spoken,
        executed_tools={"RunCommand"},
        language="de",
        tool_commands=(command,),
        tool_results=(result,),
    )
    assert honest != spoken
    assert has_universal_claim_over_sample(
        spoken,
        tool_commands=(command,),
        tool_results=(result,),
    )


def test_universal_claim_over_truncated_hit_list_is_rewritten() -> None:
    spoken = "There are none left; every match was already closed."
    # 10_000 hits exist; the tool only returned 100 and marked truncation.
    result = (
        "hit-1\nhit-2\n…\nhit-100\n"
        "…[truncated, 9900 more characters]\n"
        "(sample of 100; not the complete set)"
    )
    honest = replace_unbacked_action_claim(
        spoken,
        executed_tools={"RunCommand", "Grep"},
        language="en",
        tool_results=(result,),
    )
    assert honest != spoken
    lowered = honest.casefold()
    for banned in ("all", "none", "every", "keine"):  # i18n-allow: banned DE lemma
        assert banned not in lowered


def test_universal_claim_over_complete_result_stays() -> None:
    # Same wording, but over the full inventory.
    spoken = "Offene PRs gibt es im Moment keine."  # i18n-allow: quoted live reply
    command = "gh pr list --state open"
    result = "No open pull requests\n[exit 0 · 0.3s]"
    assert sample_limit_from_command(command) is None
    assert annotate_command_sample(result, command) == result
    honest = replace_unbacked_action_claim(
        spoken,
        executed_tools={"RunCommand"},
        language="de",
        tool_commands=(command,),
        tool_results=(result,),
    )
    assert honest == spoken
    assert not has_universal_claim_over_sample(
        spoken,
        tool_commands=(command,),
        tool_results=(result,),
    )
