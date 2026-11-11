"""Wrong-target: a docs/ edit must not be satisfied by a wiki write.

Live 2026-11-11 E1 asked for a line in docs/; the model wrote a vault note
via wiki_ingest and called it "the documents". F-0 (catalog packing that
stops instead of skipping) makes workspace-orchestrate reachable when it
lands, but that packing fix is not on origin/main for this branch, and even
with it the speech lie needs claim-scoped honesty (part a).
"""

from __future__ import annotations

from jarvis.brain.action_honesty import replace_unbacked_action_claim
from jarvis.live.config import LiveConfig


def test_f0_packing_stop_is_not_on_this_tree() -> None:
    """Document that (c) is not cured by F-0 here: packing still continues."""
    from pathlib import Path

    tools_src = Path(__file__).resolve().parents[3] / "jarvis" / "live" / "tools.py"
    text = tools_src.read_text(encoding="utf-8")
    # Unfixed packing drops an oversized tool with continue and keeps filling.
    assert "used_bytes + size > _CATALOG_BYTE_BUDGET" in text
    budget_block = text.split("used_bytes + size > _CATALOG_BYTE_BUDGET", 1)[1][:200]
    assert "continue" in budget_block


def test_live_instructions_refuse_wiki_as_documents_substitute() -> None:
    wire = LiveConfig(
        configured=True, backend_model="chosen-model"
    ).session_config(language="en", tools=[])
    instructions = wire["delegation"]["responses"]["instructions"]
    assert "never call a vault note 'the documents'" in instructions
    assert "workspace-orchestrate is unavailable" in instructions


def test_wiki_success_does_not_ground_a_docs_claim() -> None:
    spoken = "Your full name was successfully entered into the documents."
    honest = replace_unbacked_action_claim(
        spoken,
        executed_tools={"jarvis_wiki_ingest", "wiki_ingest"},
        language="en",
    )
    assert honest != spoken
