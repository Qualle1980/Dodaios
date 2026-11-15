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


def rank_hits(
    exact: Sequence[SearchHit],
    lexical: Sequence[SearchHit],
    semantic: Sequence[SearchHit],
) -> list[SearchHit]:
    """Keep exact candidates first and fuse the remaining two ranked lists."""
    candidates: dict[Path, SearchHit] = {}
    scores: dict[Path, float] = {}
    for ranking in (lexical, semantic):
        seen: set[Path] = set()
        for rank, hit in enumerate(ranking, 1):
            if hit.path in seen:
                continue
            seen.add(hit.path)
            candidates.setdefault(hit.path, hit)
            scores[hit.path] = scores.get(hit.path, 0.0) + 1 / (_RRF_K + rank)
    pinned = {hit.path: hit for hit in exact}
    ranked = sorted(candidates, key=scores.__getitem__, reverse=True)
    return [*pinned.values(), *(candidates[path] for path in ranked if path not in pinned)]


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
