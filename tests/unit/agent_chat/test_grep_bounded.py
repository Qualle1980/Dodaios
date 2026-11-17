"""Grep ends, and says so when it stopped early (2026-11-17)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.agent_chat import tools


def test_a_search_with_no_match_is_bounded_by_the_file_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(30):
        (tmp_path / f"f{i}.txt").write_text("nothing here\n", encoding="utf-8")
    monkeypatch.setattr(tools, "_GREP_MAX_FILES", 10)

    out, is_error = tools._grep({"pattern": "NEEDLE"}, tmp_path)

    assert is_error is False
    assert out.startswith("No matches for 'NEEDLE'")
    assert "search is partial" in out
    assert "after 10 files" in out


def test_a_complete_search_says_nothing_about_partiality(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("MAX_CORRECTIONS = 1\n", encoding="utf-8")

    out, is_error = tools._grep({"pattern": "MAX_CORRECTIONS"}, tmp_path)

    assert is_error is False
    assert "a.txt:1:" in out
    assert "partial" not in out


def test_a_large_file_is_skipped_not_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "big.txt").write_text("NEEDLE\n" * 50, encoding="utf-8")
    monkeypatch.setattr(tools, "_GREP_MAX_FILE_BYTES", 10)

    out, _ = tools._grep({"pattern": "NEEDLE"}, tmp_path)

    assert out.startswith("No matches for 'NEEDLE'")
