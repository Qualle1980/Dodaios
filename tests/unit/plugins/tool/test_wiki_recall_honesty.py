"""F-3 — wiki-recall distinguishes absent vault, empty vault, and no match."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.memory.wiki.search import VaultSearch
from jarvis.plugins.tool.wiki_recall import WikiRecallTool


@pytest.mark.asyncio
async def test_wiki_recall_absent_vault_names_the_root(tmp_path: Path) -> None:
    missing = (tmp_path / "no-vault-here").resolve()
    tool = WikiRecallTool(VaultSearch(missing))
    result = await tool.execute({"query": "anything"}, ctx=None)
    assert result.success is False
    assert "unavailable" in (result.error or "").lower()
    assert str(missing) in (result.error or "")


@pytest.mark.asyncio
async def test_wiki_recall_no_match_names_the_root(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("# Note\n\nhello\n", encoding="utf-8")
    tool = WikiRecallTool(VaultSearch(tmp_path))
    result = await tool.execute({"query": "zzzz-no-such-term"}, ctx=None)
    assert result.success is True
    assert "No wiki matches" in result.output
    assert str(tmp_path.resolve()) in result.output
