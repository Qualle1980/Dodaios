"""Wrong-target: a docs/ edit must not be satisfied by a wiki write.

Live 2026-11-11 E1 asked for a line in docs/; the model wrote a vault note
via wiki_ingest and called it "the documents". F-0 (catalog packing that
stops instead of skipping) landed via #145; claim-scoped honesty (part a)
still catches the wiki-as-documents speech lie even when packing is fixed.
"""

from __future__ import annotations

from jarvis.brain.action_honesty import replace_unbacked_action_claim
from jarvis.live.config import LiveConfig


def test_f0_packing_stop_is_on_this_tree() -> None:
    """#145 landed F-0: non-promised packing breaks instead of skipping."""
    import re
    from pathlib import Path

    tools_src = Path(__file__).resolve().parents[3] / "jarvis" / "live" / "tools.py"
    text = tools_src.read_text(encoding="utf-8")
    # Promised tools may still continue (short form / fault log); phase-2
    # unpromised packing must stop so a light low-rank tool cannot steal the
    # slot of a heavy high-rank one.
    assert re.search(
        r"used_bytes \+ size > _CATALOG_BYTE_BUDGET:\s*"
        r"if promised:.*?continue.*?break",
        text,
        re.DOTALL,
    ), "phase-2 packing must break after the promised short-path, not skip"


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
