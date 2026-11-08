"""The one-time import of the stores the vault replaces (ADR-0044 chunk 4).

Deterministic on purpose: it takes the atomic sentences the old stores already
hold - entity `## Facts` bullets, project goals, the notebook entries, the
identity fields - and no model call distils prose into anything. What cannot be
read atomically stays where it is and stays searchable.

Everything arrives with `origin="imported"`, which the store caps at
`INFERRED_MAX_CONFIDENCE`: nobody said these sentences to the assistant today,
they were found in a file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.memory.vault import MemoryVault
from jarvis.memory.vault.legacy_import import (
    _SENSITIVE,
    CONFIDENCE,
    MAX_CHARS,
    META_KEY,
    clean,
    import_legacy,
    import_once,
)
from jarvis.memory.vault.models import INFERRED_MAX_CONFIDENCE

USER_PAGE = """---
type: entity
entity_kind: person
slug: user
aliases: [User, the user]
---

# User

## Facts

- Lives in Ludwigshafen-Oppau.
- Speaks German, Arabic and English.

## Sources

- [[sessions/2026-10-16]]
"""

FRIEND_PAGE = """---
type: entity
entity_kind: person
slug: christian
---

# Christian

## Facts

- Works at the same company.
"""

PROJECT_PAGE = """---
type: project
slug: dodaios
status: active
last_activity: {last}
---

# Dodaios

## Goal

A local voice assistant that runs on the user's own machine.
"""


def _vault_tree(root: Path, *, project_last: str = "2026-11-01") -> None:
    (root / "entities").mkdir(parents=True, exist_ok=True)
    (root / "projects").mkdir(parents=True, exist_ok=True)
    (root / "entities" / "user.md").write_text(USER_PAGE, encoding="utf-8")
    (root / "entities" / "christian.md").write_text(FRIEND_PAGE, encoding="utf-8")
    (root / "projects" / "dodaios.md").write_text(
        PROJECT_PAGE.format(last=project_last), encoding="utf-8"
    )


def _data_dir(root: Path, *, name: str = "Ahmad") -> None:
    (root / "workspace").mkdir(parents=True, exist_ok=True)
    (root / "workspace" / "USER.md").write_text(
        "---\n"
        "identity:\n"
        f"  name: {name}\n"
        "  preferred_address: Boss\n"
        "  primary_language: de\n"
        "  languages: [de, en]\n"
        "  timezone: Europe/Berlin\n"
        "---\n\n# Profile\n",
        encoding="utf-8",
    )
    (root / "core_memory.json").write_text(
        json.dumps(
            {
                "user_facts": {"general": ["Works evenings."], "hardware": "Has a laptop."},
                "current_projects": {"general": ["Ships the voice assistant."]},
                "persona": {"tone": "short"},
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def vault(tmp_path):
    with MemoryVault(tmp_path / "vault.db") as handle:
        yield handle


# -- what comes across -------------------------------------------------


def test_the_identity_fields_become_profile_memories(tmp_path, vault) -> None:
    _data_dir(tmp_path / "data")

    report = import_legacy(vault, wiki_root=None, data_dir=tmp_path / "data")

    texts = [m.text for m in vault.list()]
    assert "The user's name is Ahmad." in texts
    assert "How the user wants to be addressed: Boss." in texts
    assert "The user's time zone is Europe/Berlin." in texts
    assert report.by_source["user-profile"] == 5


def test_entity_facts_become_memories_naming_their_subject(tmp_path, vault) -> None:
    _vault_tree(tmp_path / "wiki")
    _data_dir(tmp_path / "data")

    import_legacy(vault, wiki_root=tmp_path / "wiki", data_dir=tmp_path / "data")

    texts = [m.text for m in vault.list()]
    # The user's own page needs no name prefix; someone else's does.
    assert "Lives in Ludwigshafen-Oppau." in texts
    assert "Christian: Works at the same company." in texts


def test_the_users_own_page_becomes_profile_and_a_friend_a_relationship(tmp_path, vault) -> None:
    _vault_tree(tmp_path / "wiki")
    _data_dir(tmp_path / "data")

    import_legacy(vault, wiki_root=tmp_path / "wiki", data_dir=tmp_path / "data")

    by_text = {m.text: m for m in vault.list()}
    assert by_text["Lives in Ludwigshafen-Oppau."].kind == "profile"
    assert by_text["Christian: Works at the same company."].kind == "relationship"


def test_an_active_project_becomes_a_project_memory(tmp_path, vault) -> None:
    _vault_tree(tmp_path / "wiki")
    _data_dir(tmp_path / "data")

    import_legacy(vault, wiki_root=tmp_path / "wiki", data_dir=tmp_path / "data")

    projects = [m for m in vault.list() if m.kind == "project"]
    assert any("local voice assistant" in m.text for m in projects)


def test_a_long_dormant_project_arrives_as_history(tmp_path, vault) -> None:
    """The old wiki never closed a project, so stale ones must not read as current."""
    _vault_tree(tmp_path / "wiki", project_last="2020-01-01")
    _data_dir(tmp_path / "data")

    import_legacy(vault, wiki_root=tmp_path / "wiki", data_dir=tmp_path / "data")

    current = [m.text for m in vault.list()]
    everything = [m.text for m in vault.list(include_closed=True)]
    assert not any("local voice assistant" in t for t in current)
    assert any("local voice assistant" in t for t in everything)


def test_core_memory_facts_come_across_with_their_label(tmp_path, vault) -> None:
    _data_dir(tmp_path / "data")

    import_legacy(vault, wiki_root=None, data_dir=tmp_path / "data")

    texts = [m.text for m in vault.list()]
    assert "Works evenings." in texts
    assert "hardware: Has a laptop." in texts
    # Settings are not memories.
    assert not any("tone" in t for t in texts)


# -- how it comes across -----------------------------------------------


def test_everything_imported_is_marked_as_found_not_said(tmp_path, vault) -> None:
    _vault_tree(tmp_path / "wiki")
    _data_dir(tmp_path / "data")

    import_legacy(vault, wiki_root=tmp_path / "wiki", data_dir=tmp_path / "data")

    for memory in vault.list(include_closed=True):
        assert memory.origin == "imported"
        assert memory.confidence <= INFERRED_MAX_CONFIDENCE
        assert memory.confidence == pytest.approx(CONFIDENCE)
        assert memory.source["via"] == "legacy-import"
        assert memory.source["file"], "a claim has to be traceable to its file"


def test_wiki_markup_does_not_reach_a_memory() -> None:
    assert clean("Works with [[entities/christian|Christian]] daily.") == (
        "Works with Christian daily."
    )
    assert clean("Lives in [[places/oppau]].") == "Lives in oppau."
    assert clean("Prefers tea *(inferred)*") == "Prefers tea"


def test_an_over_long_sentence_is_cut_not_dropped(tmp_path, vault) -> None:
    root = tmp_path / "wiki"
    (root / "entities").mkdir(parents=True)
    (root / "entities" / "user.md").write_text(
        "---\ntype: entity\nslug: user\n---\n\n# User\n\n## Facts\n\n- "
        + "word " * 400
        + "\n",
        encoding="utf-8",
    )
    _data_dir(tmp_path / "data")

    import_legacy(vault, wiki_root=root, data_dir=tmp_path / "data")

    long_ones = [m for m in vault.list() if m.text.startswith("word word")]
    assert long_ones, "a long fact is shortened, never silently skipped"
    assert len(long_ones[0].text) <= MAX_CHARS
    assert long_ones[0].text.endswith("…")


@pytest.mark.parametrize(
    "sentence",
    [
        "Started a therapy in March.",
        "Hat im März eine Therapie begonnen.",  # i18n-allow: the notebooks are German
        "Geht regelmäßig in die Moschee.",  # i18n-allow: the notebooks are German
        "Nimmt ein Medikament gegen Angst.",  # i18n-allow: the notebooks are German
    ],
)
def test_a_sensitive_sentence_is_marked_in_both_languages(sentence: str) -> None:
    """An English-only filter would wave every German one into the standing block."""
    assert _SENSITIVE.search(sentence) is not None


def test_an_ordinary_sentence_is_not_marked_sensitive() -> None:
    assert _SENSITIVE.search("Wohnt in Ludwigshafen-Oppau.") is None  # i18n-allow: German note
    assert _SENSITIVE.search("Ships the voice assistant.") is None


def test_a_sensitive_fact_stays_out_of_the_standing_block(tmp_path, vault) -> None:
    root = tmp_path / "wiki"
    (root / "entities").mkdir(parents=True)
    (root / "entities" / "user.md").write_text(
        "---\ntype: entity\nslug: user\n---\n\n# User\n\n## Facts\n\n"
        "- Lives in Oppau.\n- Started a therapy in March.\n",
        encoding="utf-8",
    )
    _data_dir(tmp_path / "data")

    import_legacy(vault, wiki_root=root, data_dir=tmp_path / "data")

    block = vault.profile_block()
    assert "Lives in Oppau." in block
    assert "therapy" not in block
    # Still remembered, still findable.
    assert any("therapy" in h.memory.text for h in vault.search("therapy"))


# -- running it twice --------------------------------------------------


def test_running_it_again_adds_nothing(tmp_path, vault) -> None:
    _vault_tree(tmp_path / "wiki")
    _data_dir(tmp_path / "data")

    first = import_legacy(vault, wiki_root=tmp_path / "wiki", data_dir=tmp_path / "data")
    second = import_legacy(vault, wiki_root=tmp_path / "wiki", data_dir=tmp_path / "data")

    assert first.added > 0
    assert second.added == 0
    assert second.existing == first.added


def test_import_once_runs_once_and_records_it(tmp_path, vault) -> None:
    _data_dir(tmp_path / "data")

    assert import_once(vault, wiki_root=None, data_dir=tmp_path / "data") is True
    assert import_once(vault, wiki_root=None, data_dir=tmp_path / "data") is False

    receipt = json.loads(vault.get_meta(META_KEY))
    # Five identity fields plus the three core-memory facts.
    assert receipt["added"] == 8
    assert receipt["by_source"] == {"user-profile": 5, "core-memory": 3}
    assert receipt["at"]


def test_a_forgotten_memory_is_not_brought_back_by_a_second_import(tmp_path, vault) -> None:
    """`forget` is the user's word; re-running the import must not undo it."""
    _data_dir(tmp_path / "data")
    import_legacy(vault, wiki_root=None, data_dir=tmp_path / "data")
    name = next(m for m in vault.list() if "name is Ahmad" in m.text)
    assert vault.forget(name.id) == 1

    import_legacy(vault, wiki_root=None, data_dir=tmp_path / "data")

    assert not any("name is Ahmad" in m.text for m in vault.list())


# -- missing sources ---------------------------------------------------


def test_no_sources_at_all_imports_nothing(tmp_path, vault) -> None:
    report = import_legacy(vault, wiki_root=None, data_dir=tmp_path / "empty")

    assert report.added == 0
    assert vault.stats()["current"] == 0


def test_an_unreadable_core_memory_skips_only_that_source(tmp_path, vault) -> None:
    data = tmp_path / "data"
    _data_dir(data)
    (data / "core_memory.json").write_text("{not json", encoding="utf-8")

    report = import_legacy(vault, wiki_root=None, data_dir=data)

    assert report.by_source.get("user-profile") == 5
    assert "core-memory" not in report.by_source
