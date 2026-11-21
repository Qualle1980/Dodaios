"""CI wiring tests for fused wiki recall.

The cross-language encoder is deliberately built from the answer key. It proves
that fusion, excerpts, fallback, and output budgets are connected correctly.
It cannot disagree with the supplied translations and therefore makes no claim
about real retrieval quality. Run scripts/wiki_recall_quality_benchmark.py with
the installed local model for that measurement.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

import pytest

from jarvis.memory.wiki import semantic_search
from jarvis.memory.wiki.fts_index import ensure_schema, index_vault
from jarvis.memory.wiki.recall_quality import rank_hits
from jarvis.memory.wiki.search import SearchHit, VaultSearch
from jarvis.plugins.tool.wiki_recall import WikiRecallTool
from tests.fakes.fake_semantic_encoder import FakeSemanticEncoder

_FIXTURES = Path(__file__).resolve().parent
_CASES = [
    json.loads(line)
    for line in (_FIXTURES / "questions.jsonl").read_text(encoding="utf-8").splitlines()
]
_HIT = re.compile(r"^- \*\*.*?\*\* — (.*?) \[.*?\] \((.*?)\)$", re.MULTILINE)


class AnswerKeyWiringEncoder(FakeSemanticEncoder):
    """Map every cross-language fixture from its declared answer key."""

    def __init__(self) -> None:
        super().__init__(
            {
                case["id"]: (case["query"].split()[0], case["semantic_term"])
                for case in _CASES
                if case["mode"] == "cross"
            }
        )

    def encode(self, texts, *, wait=False):
        vectors = super().encode(texts, wait=wait)
        if vectors is not None:
            # Unknown concepts must not give unrelated queries a shared axis.
            vectors[:, -1] = 0.0
        return vectors


@contextmanager
def wiring_recall(tmp_path: Path, *, semantic: bool):
    root = _FIXTURES / "vault"
    with closing(sqlite3.connect(":memory:", check_same_thread=False)) as conn:
        ensure_schema(conn)
        index_vault(root, conn)
        if semantic:
            engine = semantic_search.activate(
                root,
                db_path=tmp_path / "semantic.db",
                encoder=AnswerKeyWiringEncoder(),
            )
            assert engine.sync().failed == 0
        try:
            yield WikiRecallTool(VaultSearch(root, conn=conn))
        finally:
            if semantic:
                semantic_search.deactivate()


async def retrieve(tool, case):
    result = await tool.execute({"query": case["query"], "k": 5}, ctx=None)
    assert result.success
    return _HIT.findall(result.output or "")


@pytest.mark.asyncio
async def test_answer_key_wiring_returns_cross_language_excerpt_in_one_call(
    tmp_path: Path,
) -> None:
    case = next(case for case in _CASES if case["mode"] == "cross")
    with wiring_recall(tmp_path, semantic=True) as tool:
        hits = await retrieve(tool, case)
    excerpt, path = hits[0]
    assert path == case["path"]
    assert case["answer"] in excerpt


@pytest.mark.asyncio
async def test_wiring_without_semantic_runtime_returns_lexical_excerpt(
    tmp_path: Path,
) -> None:
    case = next(case for case in _CASES if case["mode"] == "lexical")
    with wiring_recall(tmp_path, semantic=False) as tool:
        hits = await retrieve(tool, case)
    excerpt, path = next(hit for hit in hits if hit[1] == case["path"])
    assert case["answer"] in excerpt
    assert path == case["path"]


def test_fusion_uses_passage_evidence_when_rrf_scores_tie(tmp_path: Path) -> None:
    thin = SearchHit(
        title="Navigation summary",
        path=tmp_path / "brief.md",
        snippet="A short repeated subject reference with little supporting evidence.",
        score=0.46,
    )
    answer = SearchHit(
        title="Answer note",
        path=tmp_path / "answer.md",
        snippet=(
            "The detailed passage states the requested fact and includes enough "
            "supporting context for a grounded answer to the question."
        ),
        score=0.35,
    )

    ranked = rank_hits(
        exact=[],
        lexical=[thin, answer],
        semantic=[thin, answer],
    )

    assert ranked[0].path == answer.path


@pytest.mark.asyncio
async def test_exact_identifiers_and_titles_survive_competing_lexical_hits(
    tmp_path: Path,
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "primary.md").write_text(
        "---\nid: ref-381\n---\n\n# Transfer Window\n\nStarts at 07:40 UTC.\n",
        encoding="utf-8",
    )
    for i in range(12):
        (root / f"decoy-{i}.md").write_text(
            f"# Transfer Window archive {i}\n\nref-381 ref-381 ref-381.\n",
            encoding="utf-8",
        )
    with closing(sqlite3.connect(":memory:", check_same_thread=False)) as conn:
        ensure_schema(conn)
        index_vault(root, conn)
        tool = WikiRecallTool(VaultSearch(root, conn=conn))
        for identifier in ("ref-381", "Transfer Window", "primary.md"):
            hits = await retrieve(tool, {"query": identifier})
            assert hits[0] == ("Starts at 07:40 UTC.", "primary.md")


@pytest.mark.asyncio
async def test_wire_output_obeys_recall_budgets(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    sentence = "Schedule details fill this sentence with a record of the intended cadence. "
    for i in range(6):
        title = f"Schedule entry {i} " + "long title words " * 7
        (root / f"expanded-schedule-record-{i}.md").write_text(
            f"# {title}\n\n" + sentence * 10,
            encoding="utf-8",
        )
    with closing(sqlite3.connect(":memory:", check_same_thread=False)) as conn:
        ensure_schema(conn)
        index_vault(root, conn)
        tool = WikiRecallTool(VaultSearch(root, conn=conn))
        result = await tool.execute({"query": "schedule", "k": 10}, ctx=None)
    output = result.output or ""
    hits = _HIT.findall(output)
    assert len(hits) == 3
    assert len(output) <= 1800
    assert all(len(excerpt) <= 480 for excerpt, _ in hits)
    assert all(excerpt.endswith((".", ".…")) for excerpt, _ in hits)


def test_uncorroborated_weak_semantic_candidate_is_not_injected(
    tmp_path: Path,
) -> None:
    weak = SearchHit(
        title="Weak semantic neighbor",
        path=tmp_path / "weak.md",
        snippet=(
            "This long passage is semantically nearby but remains below the "
            "meaning-only confidence floor used by recall."
        ),
        score=0.35,
    )

    assert rank_hits(exact=[], lexical=[], semantic=[weak]) == []
    assert rank_hits(exact=[], lexical=[weak], semantic=[weak]) == [weak]
