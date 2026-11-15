"""Rank wiki recall candidates and select compact body evidence."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from jarvis.memory.wiki.search import SearchHit, _build_match_expr
from jarvis.memory.wiki.semantic_index import passages

log = logging.getLogger(__name__)

# Cormack, Clarke and Buettcher, SIGIR 2009, doi:10.1145/1571941.1572114.
_RRF_K = 60

# A meaning-only candidate needs stronger evidence than one corroborated by FTS.
# This floor is measured against the versioned negative-query fixture.
_SEMANTIC_ONLY_MIN_SCORE = 0.36


def rank_hits(
    exact: Sequence[SearchHit],
    lexical: Sequence[SearchHit],
    semantic: Sequence[SearchHit],
) -> list[SearchHit]:
    """Keep exact candidates first and fuse the remaining two ranked lists."""
    candidates: dict[Path, SearchHit] = {}
    scores: dict[Path, float] = {}
    semantic_scores = {hit.path: hit.score for hit in semantic}
    semantic_evidence = {
        hit.path: hit.score * _passage_evidence_weight(hit.snippet) for hit in semantic
    }
    semantic_by_evidence = sorted(
        semantic,
        key=lambda hit: semantic_evidence[hit.path],
        reverse=True,
    )
    for ranking in (lexical, semantic_by_evidence):
        seen: set[Path] = set()
        for rank, hit in enumerate(ranking, 1):
            if hit.path in seen:
                continue
            seen.add(hit.path)
            candidates.setdefault(hit.path, hit)
            scores[hit.path] = scores.get(hit.path, 0.0) + 1 / (_RRF_K + rank)
    pinned = {hit.path: hit for hit in exact}
    lexical_paths = {hit.path for hit in lexical}
    eligible = (
        path
        for path in candidates
        if path in lexical_paths or semantic_scores.get(path, -1.0) >= _SEMANTIC_ONLY_MIN_SCORE
    )
    ranked = sorted(
        eligible,
        key=lambda path: (scores[path], semantic_evidence.get(path, -1.0)),
        reverse=True,
    )
    return [*pinned.values(), *(candidates[path] for path in ranked if path not in pinned)]


def _passage_evidence_weight(text: str) -> float:
    """Discount thin semantic matches without excluding short factual notes.

    Cosine similarity alone can favor a short navigation blurb that repeats
    the subject over a passage that states the answer. Short evidence keeps
    40 percent of its score and reaches full weight at 16 words.
    """
    word_count = len(text.split())
    return 0.4 + 0.6 * min(word_count, 16) / 16


def lexical_passage(query: str, title: str, body: str) -> str:
    """Rank a note's body passages using their own local BM25 statistics."""
    pieces = passages(title, body)
    if not pieces:
        return ""
    match = _build_match_expr(query.split())
    if match:
        try:
            with closing(sqlite3.connect(":memory:")) as conn:
                conn.execute(
                    "CREATE VIRTUAL TABLE evidence USING fts5("
                    "text, tokenize='unicode61 remove_diacritics 2')"
                )
                conn.executemany(
                    "INSERT INTO evidence(text) VALUES (?)", ((text,) for text, _ in pieces)
                )
                row = conn.execute(
                    "SELECT text FROM evidence WHERE evidence MATCH ? "
                    "ORDER BY bm25(evidence), rowid LIMIT 1",
                    (match,),
                ).fetchone()
                if row:
                    return str(row[0])
        except sqlite3.OperationalError as exc:
            log.debug("wiki-recall: passage scoring unavailable: %s", exc)
    # Exact title and alias matches may have no query words in the body.
    return pieces[0][0]
