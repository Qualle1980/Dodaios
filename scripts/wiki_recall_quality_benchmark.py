"""Measure wiki recall quality with the installed local semantic model.

The benchmark reads only the checked-in synthetic fixture vault. It never opens
the user's private vault and never calls a paid provider. Missing model files or
runtime support stop the run without reporting a retrieval score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
import sys
import tempfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from jarvis.memory.wiki import semantic_search  # noqa: E402
from jarvis.memory.wiki.fts_index import ensure_schema, index_vault  # noqa: E402
from jarvis.memory.wiki.search import SearchHit, VaultSearch  # noqa: E402
from jarvis.memory.wiki.semantic_model import (  # noqa: E402
    SemanticEncoder,
    model_dir,
    model_present,
    runtime_available,
)
from jarvis.plugins.tool.wiki_recall import WikiRecallTool  # noqa: E402

_FIXTURES = _ROOT / "tests" / "fixtures" / "wiki_recall_quality"
_HIT = re.compile(r"^- \*\*.*?\*\* — (.*?) \[.*?\] \((.*?)\)$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class RetrievedHit:
    excerpt: str
    path: str


def _cases() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (_FIXTURES / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_hits(output: str) -> list[RetrievedHit]:
    return [RetrievedHit(excerpt, path) for excerpt, path in _HIT.findall(output)]


def _format_ranking(hits: list[SearchHit]) -> str:
    if not hits:
        return "-"
    return ", ".join(f"{hit.path.name}:{hit.score:.4f}" for hit in hits[:5])


async def _retrieve(tool: WikiRecallTool, query: str) -> list[RetrievedHit]:
    result = await tool.execute({"query": query, "k": 3}, ctx=None)
    if not result.success:
        raise RuntimeError(result.error or "wiki recall failed")
    return _parse_hits(result.output or "")


async def _measure(
    *,
    search: VaultSearch,
    engine: semantic_search.SemanticVaultSearch,
    cases: list[dict[str, Any]],
    show_rankings: bool,
) -> dict[str, int]:
    tool = WikiRecallTool(search)
    counts = {
        "positive": 0,
        "lexical": 0,
        "lexical_hit1": 0,
        "cross": 0,
        "cross_hit1": 0,
        "answer_hit1": 0,
        "excerpt_support": 0,
        "negative": 0,
        "false_injection": 0,
        "fallback_lexical": 0,
    }
    rows: list[tuple[str, str, bool, bool, str]] = []

    for case in cases:
        query = str(case["query"])
        if show_rankings:
            lexical = search.search(query, k=10)
            semantic = engine.search(query, k=10, full_passage=True, wait=True)
            print(f"\n{case['id']}")
            print(f"  lexical raw : {_format_ranking(lexical)}")
            print(f"  semantic raw: {_format_ranking(semantic)}")

        hits = await _retrieve(tool, query)
        expected = case.get("path")
        if expected is None:
            counts["negative"] += 1
            counts["false_injection"] += bool(hits)
            continue

        counts["positive"] += 1
        mode = str(case["mode"])
        counts[mode] += 1
        top = hits[0] if hits else RetrievedHit("", "-")
        is_hit = top.path == expected
        has_answer = str(case["answer"]).casefold() in top.excerpt.casefold()
        counts[f"{mode}_hit1"] += is_hit
        counts["answer_hit1"] += is_hit and has_answer
        gold = next((hit for hit in hits if hit.path == expected), None)
        supported = bool(gold and str(case["answer"]).casefold() in gold.excerpt.casefold())
        counts["excerpt_support"] += supported
        rows.append((str(case["id"]), mode, is_hit, supported, top.path))

    semantic_search.deactivate()
    fallback_tool = WikiRecallTool(search)
    for case in cases:
        if case.get("mode") != "lexical":
            continue
        hits = await _retrieve(fallback_tool, str(case["query"]))
        top = hits[0] if hits else RetrievedHit("", "-")
        if top.path == case["path"] and str(case["answer"]).casefold() in top.excerpt.casefold():
            counts["fallback_lexical"] += 1

    print()
    for case_id, mode, is_hit, supported, top_path in rows:
        mark = "OK" if is_hit and supported else "--"
        print(
            f"{mark} {case_id:<28} {mode:<8} top={top_path:<24} excerpt={'y' if supported else 'n'}"
        )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_ROOT / "data",
        help="data directory containing models/wiki-semantic-search",
    )
    parser.add_argument(
        "--show-rankings",
        action="store_true",
        help="print lexical and semantic rankings before fusion",
    )
    args = parser.parse_args()

    print(f"model directory : {model_dir(args.data_dir)}")
    print(f"model present   : {model_present(args.data_dir)}")
    print(f"runtime present : {runtime_available()}")
    if not model_present(args.data_dir) or not runtime_available():
        print("REAL ENCODER UNAVAILABLE: benchmark not run; no quality score reported.")
        return 2

    cases = _cases()
    vault = _FIXTURES / "vault"
    with tempfile.TemporaryDirectory() as temp:
        encoder = SemanticEncoder(data_dir=args.data_dir)
        try:
            with closing(sqlite3.connect(":memory:", check_same_thread=False)) as conn:
                ensure_schema(conn)
                index_vault(vault, conn)
                search = VaultSearch(vault, conn=conn)
                engine = semantic_search.activate(
                    vault,
                    db_path=Path(temp) / "semantic.db",
                    encoder=encoder,
                )
                sync = engine.sync()
                print(f"index sync      : failed={sync.failed}")
                if sync.failed:
                    print("REAL ENCODER INDEX INCOMPLETE: no quality score reported.")
                    return 3
                counts = asyncio.run(
                    _measure(
                        search=search,
                        engine=engine,
                        cases=cases,
                        show_rankings=args.show_rankings,
                    )
                )
        finally:
            semantic_search.deactivate()
            encoder.close()

    print()
    print(f"Answer-bearing Hit@1 : {counts['answer_hit1']}/{counts['positive']}")
    print(f"Lexical Hit@1           : {counts['lexical_hit1']}/{counts['lexical']}")
    print(f"Cross-language Hit@1    : {counts['cross_hit1']}/{counts['cross']}")
    print(f"Excerpt Support        : {counts['excerpt_support']}/{counts['positive']}")
    print(f"False Injection Rate   : {counts['false_injection']}/{counts['negative']}")
    print(f"Lexical Fallback       : {counts['fallback_lexical']}/{counts['lexical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
