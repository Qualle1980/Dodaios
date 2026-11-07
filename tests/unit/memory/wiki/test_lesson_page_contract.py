"""A lesson page must say who, what, why and on what evidence.

The owner's requirement, translated: everything anyone gets wrong is written
down in the vault, and the shape is always the same -- who, what, why, and on
what grounds. A note that records a symptom and a guess prevents nothing;
the next session reads it, cannot tell cause from hypothesis, and pays for
the same day twice.

So the contract is enforced rather than requested: ``atomic_writer`` already
rolls back a write whose page is ``is_schema_valid=False``, and these tests
pin what makes a lesson valid.

  wer     -> frontmatter ``source``
  was     -> ## Symptom
  warum   -> ## Cause      (confirmed, not the first hypothesis)
  weshalb -> ## Evidence   (the discriminating measurement)
          -> ## Rule       (what must be found next time)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.memory.wiki.page import (
    CANONICAL_SECTIONS,
    DIR_TO_TYPE,
    REQUIRED_KEYS,
    parse_markdown,
)

SECTIONS = ("Symptom", "Cause", "Evidence", "Rule")


def _page(
    *,
    slug: str = "warm-up-covered-one-leg",
    source: str = "claude",
    status: str = "unverified",
    sections: tuple[str, ...] = SECTIONS,
    extra_front: str = "",
) -> str:
    front = [
        "---",
        "type: lesson",
        f"slug: {slug}",
        f"source: {source}",
        f"status: {status}",
    ]
    if extra_front:
        front.append(extra_front)
    front.append("---")
    body = [f"# {slug}", ""]
    for name in sections:
        body += [f"## {name}", "", f"what belongs under {name}.", ""]
    return "\n".join(front) + "\n" + "\n".join(body)


def _path(tmp_path: Path, slug: str = "warm-up-covered-one-leg") -> Path:
    directory = tmp_path / "lessons"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{slug}.md"


def test_lesson_is_a_known_page_type() -> None:
    assert DIR_TO_TYPE["lessons"] == "lesson"
    assert "lesson" in REQUIRED_KEYS
    assert CANONICAL_SECTIONS["lesson"] == SECTIONS


def test_a_complete_lesson_is_valid(tmp_path: Path) -> None:
    page = parse_markdown(_page(), _path(tmp_path))
    assert page.is_schema_valid


@pytest.mark.parametrize("missing", SECTIONS)
def test_a_lesson_missing_any_required_section_is_rejected(
    tmp_path: Path, missing: str
) -> None:
    kept = tuple(name for name in SECTIONS if name != missing)
    page = parse_markdown(_page(sections=kept), _path(tmp_path))
    assert not page.is_schema_valid, f"a lesson without ## {missing} was accepted"


def test_a_lesson_without_a_source_is_rejected(tmp_path: Path) -> None:
    """Who learned it is the half of recall that keeps two workers apart."""
    raw = _page().replace("source: claude\n", "")
    assert not parse_markdown(raw, _path(tmp_path)).is_schema_valid


def test_an_empty_required_section_is_not_a_filled_one(tmp_path: Path) -> None:
    """A heading with nothing under it is the shape of the problem, not a fix."""
    raw = _page().replace(
        "## Cause\n\nwhat belongs under Cause.\n", "## Cause\n\n"
    )
    assert not parse_markdown(raw, _path(tmp_path)).is_schema_valid


def test_sections_may_appear_in_any_order(tmp_path: Path) -> None:
    """The contract is about substance, not about sequence."""
    page = parse_markdown(
        _page(sections=("Rule", "Evidence", "Cause", "Symptom")), _path(tmp_path)
    )
    assert page.is_schema_valid


def test_extra_sections_are_welcome(tmp_path: Path) -> None:
    page = parse_markdown(
        _page(sections=(*SECTIONS, "Change", "Open")), _path(tmp_path)
    )
    assert page.is_schema_valid


def test_the_filename_must_match_the_slug(tmp_path: Path) -> None:
    page = parse_markdown(_page(slug="one-name"), _path(tmp_path, "another-name"))
    assert not page.is_schema_valid


def test_other_page_types_keep_their_contract(tmp_path: Path) -> None:
    """Requiring sections must not start requiring them of everything else."""
    directory = tmp_path / "entities"
    directory.mkdir(parents=True)
    raw = "---\ntype: entity\nslug: groq\n---\n\n# groq\n\nno sections at all.\n"
    page = parse_markdown(raw, directory / "groq.md")
    assert page.is_schema_valid


# ──────────────────────────────────────────────────────────────────────
# End to end: the contract reaches the writer, with the REAL parser.
# The existing rollback tests use a FakePageRepository, so they prove the
# writer honours a verdict -- not that a lesson's own verdict arrives.
# ──────────────────────────────────────────────────────────────────────


def _writer(tmp_path: Path):  # noqa: ANN202
    from jarvis.memory.wiki.atomic_writer import AtomicWriter

    root = tmp_path / "vault"
    (root / "lessons").mkdir(parents=True)
    return (
        AtomicWriter(
            vault_root=root,
            backup_dir=tmp_path / "backups",
            max_backups=5,
        ),
        root,
    )


def _apply(writer, root: Path, raw: str, slug: str):  # noqa: ANN001, ANN202
    import asyncio

    from jarvis.memory.wiki.atomic_writer import PageUpdate
    from jarvis.memory.wiki.page import MarkdownPageRepository

    target = root / "lessons" / f"{slug}.md"
    update = PageUpdate(target_path=target, operation="create", new_body=raw)
    result = asyncio.run(writer.apply([update], repo=MarkdownPageRepository()))
    return result, target


def test_the_writer_refuses_a_lesson_without_a_confirmed_cause(tmp_path: Path) -> None:
    writer, root = _writer(tmp_path)
    raw = _page(sections=("Symptom", "Evidence", "Rule"))
    result, target = _apply(writer, root, raw, "warm-up-covered-one-leg")

    assert result.applied == []
    assert result.failed_validation == [target.resolve()]
    assert not target.exists(), "an unprovable lesson was stored anyway"


def test_the_writer_stores_a_complete_lesson(tmp_path: Path) -> None:
    writer, root = _writer(tmp_path)
    result, target = _apply(writer, root, _page(), "warm-up-covered-one-leg")

    assert result.failed_validation == []
    assert target.exists()
    assert "## Rule" in target.read_text(encoding="utf-8")
