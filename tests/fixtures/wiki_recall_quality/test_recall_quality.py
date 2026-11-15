"""Synthetic recall benchmark using FTS5 and the real cosine search with a fake encoder."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

import pytest

from jarvis.memory.wiki import semantic_search
from jarvis.memory.wiki.fts_index import ensure_schema, index_vault
from jarvis.memory.wiki.search import VaultSearch
from jarvis.plugins.tool.wiki_recall import WikiRecallTool
from tests.fakes.fake_semantic_encoder import FakeSemanticEncoder

_FIXTURES = Path(__file__).resolve().parent
_CASES = [
    json.loads(line)
    for line in (_FIXTURES / "questions.jsonl").read_text(encoding="utf-8").splitlines()
]
_HIT = re.compile(r"^- \*\*.*?\*\* — (.*?) \[.*?\] \((.*?)\)$", re.MULTILINE)


class FixtureEncoder(FakeSemanticEncoder):
    def __init__(self):
        super().__init__({
            case["id"]: (case["query"].split()[0], case["semantic_term"])
            for case in _CASES
            if case["mode"] == "cross"
        })

    def encode(self, texts, *, wait=False):
        vectors = super().encode(texts, wait=wait)
        if vectors is not None:
            # Unknown concepts must not give unrelated queries a shared axis.
            vectors[:, -1] = 0.0
        return vectors


@contextmanager
def recall(tmp_path, *, semantic):
    root = _FIXTURES / "vault"
    with closing(sqlite3.connect(":memory:", check_same_thread=False)) as conn:
        ensure_schema(conn)
        index_vault(root, conn)
        if semantic:
            engine = semantic_search.activate(
                root, db_path=tmp_path / "semantic.db", encoder=FixtureEncoder()
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
async def test_wiki_recall_cross_language_returns_answer_excerpt_in_one_call(tmp_path):
    case = next(case for case in _CASES if case["mode"] == "cross")
    with recall(tmp_path, semantic=True) as tool:
        hits = await retrieve(tool, case)
    excerpt, path = hits[0]
    assert path == case["path"]
    assert case["answer"] in excerpt


@pytest.mark.asyncio
async def test_wiki_recall_without_semantic_runtime_still_returns_lexical_excerpt(tmp_path):
    case = next(case for case in _CASES if case["mode"] == "lexical")
    with recall(tmp_path, semantic=False) as tool:
        hits = await retrieve(tool, case)
    excerpt, path = next(hit for hit in hits if hit[1] == case["path"])
    assert case["answer"] in excerpt
    assert path == case["path"]


async def measure(tmp_path):
    with recall(tmp_path, semantic=True) as tool:
        hybrid = [await retrieve(tool, case) for case in _CASES]
    with recall(tmp_path, semantic=False) as tool:
        fallback = [await retrieve(tool, case) for case in _CASES]
    scores = dict.fromkeys(
        ("Answer-bearing Hit@1", "Recall@3", "MRR@5", "Excerpt Support Rate",
         "False Injection Rate", "Fallback Parity"), 0.0
    )
    positives = negatives = lexical = retrieved = 0
    for case, hits, lexical_hits in zip(_CASES, hybrid, fallback, strict=True):
        if case["path"] is None:
            negatives += 1
            scores["False Injection Rate"] += bool(hits)
            continue
        positives += 1
        expected = case["path"]
        rank = next((i for i, (_, path) in enumerate(hits, 1) if path == expected), 0)
        supports = bool(rank and case["answer"] in hits[rank - 1][0])
        scores["Answer-bearing Hit@1"] += rank == 1 and supports
        scores["Recall@3"] += 0 < rank <= 3
        scores["MRR@5"] += 1 / rank if 0 < rank <= 5 else 0
        retrieved += bool(rank)
        scores["Excerpt Support Rate"] += supports
        if case["mode"] == "lexical":
            lexical += 1
            scores["Fallback Parity"] += hits[:1] == lexical_hits[:1]
    denominators = [positives, positives, positives, retrieved, negatives, lexical]
    return {
        metric: round(value / count, 4) if count else 0.0
        for (metric, value), count in zip(scores.items(), denominators, strict=True)
    }


@pytest.mark.asyncio
async def test_exact_identifiers_and_titles_survive_competing_lexical_hits(tmp_path):
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
async def test_wire_output_obeys_recall_budgets(tmp_path):
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


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temp:
        print(json.dumps(asyncio.run(measure(Path(temp))), indent=2))
