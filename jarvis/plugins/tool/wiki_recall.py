"""``wiki-recall`` tool — keyword search over the long-term Obsidian wiki.

B5 Agent B (recall-tool).  Router-tier read-only tool.

The tool exposes ``VaultSearch`` (``jarvis.memory.wiki.search``) to the
router brain, so the brain can look up information that Jarvis or the
user has previously stored in the wiki vault.

Usage pattern:
    Brain calls ``wiki-recall`` with 1–4 keywords whenever the user asks
    "what do we know about X", "who is Y", or references a past project,
    person, or decision by name.  The tool returns a compact markdown
    block with up to *k* ranked hits, ready to paste into the system
    prompt.

Placement rule:
    This tool is router-visible. Jarvis-Agents may reach the live tool only
    through ADR-0025's mission-scoped supervisor broker. It must **never**
    appear in any ``SUB_TOOLS`` frozenset or direct worker tool set (AP-D9).

Privacy rule (AP-7 / §5 overview):
    Query text is logged only at DEBUG level.  Hit count summary is
    INFO level.

Vault-root resolution (§6 AGENT-B, spec A7):
    ``cfg.wiki_integration.vault_root`` is resolved through the canonical
    :func:`jarvis.memory.wiki.vault_root.resolve_vault_root` — a relative
    root anchors to the repo root, never the process CWD. If the config
    field is unavailable at construction time, the tool falls back to the
    resolver's default vault location and logs a single WARNING.
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.core.protocols import ToolResult
from jarvis.memory.wiki.recall_quality import lexical_passage, rank_hits
from jarvis.memory.wiki.search import SearchHit

if TYPE_CHECKING:
    from jarvis.memory.wiki.search import VaultSearch

log = logging.getLogger(__name__)

_MAX_HITS = 3
_CANDIDATE_DEPTH = 10
_MAX_EXCERPT_CHARS = 480
_MAX_BLOCK_CHARS = 1800
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_NO_EXCERPT = "[no excerpt available; open the note for content]"


class WikiRecallTool:
    """Router-tier keyword search over the long-term Obsidian wiki vault."""

    read_only = True

    name: str = "wiki-recall"
    description: str = (
        "Search the user's long-term Obsidian wiki for notes matching the query. "
        "Returns a compact markdown summary with up to 3 ranked hits and excerpts. Use this "
        "when the user asks 'what do we know about X', 'who is Y', or references "
        "a past project, person, or decision by name."
    )
    risk_tier: str = "safe"
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "1-4 keywords; matched against frontmatter + body",
            },
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": 3,
            },
        },
        "required": ["query"],
    }
    input_examples: list[dict[str, Any]] = [
        {"query": "Harald birthday"},
        {"query": "Dodaios architecture"},
    ]

    def __init__(self, search: VaultSearch) -> None:
        self._search = search

    async def execute(self, args: dict[str, Any], ctx: Any) -> ToolResult:
        """Run the vault search and return a compact markdown block.

        Output format::

            ## Wiki hits for "<query>"
            - **<title>** — <snippet>… (10-notes/Harald.md)
            - …

        Paths are rendered relative to ``vault_root`` so the brain sees
        ``10-notes/Harald.md``, not an absolute Windows path.

        Returns
        -------
        ToolResult(success=True, output=<markdown>)
            When hits are found.
        ToolResult(success=True, output="No matches …")
            When the memory is available but nothing matches.
        ToolResult(success=False, error="vault unavailable")
            When the vault root does not exist at execute time.
        """
        query = str(args.get("query", "")).strip()
        k = int(args.get("k", _MAX_HITS))
        # Clamp defensively — schema validation may not have run upstream.
        k = max(1, min(k, _MAX_HITS))

        vault_root = self._search._root  # type: ignore[attr-defined]

        # Privacy: query is DEBUG only.
        log.debug("wiki-recall: query=%r k=%d vault=%s", query, k, vault_root)

        if not vault_root.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"vault unavailable at {vault_root}",
            )

        lexical = self._search.search(query, k=_CANDIDATE_DEPTH)
        exact_lookup = getattr(self._search, "search_exact", None)
        exact = exact_lookup(query, k=k) if exact_lookup else []
        semantic = await asyncio.to_thread(_semantic_hits, query, vault_root)
        hits = rank_hits(exact, lexical, semantic)[:k]
        semantic_passages = {hit.path: hit.snippet for hit in semantic}

        log.info("wiki-recall: %d hit(s) for query (k=%d)", len(hits), k)

        display_query = " ".join(query.split())[:120]
        if not hits:
            return ToolResult(
                success=True,
                output=(
                    f'No wiki matches found for "{display_query}" in {vault_root}.'
                ),
            )

        lines = [f'## Wiki hits for "{display_query}"']
        length = len(lines[0])
        indexed_body = getattr(self._search, "indexed_body", None)
        for hit in hits:
            try:
                rel_path = hit.path.relative_to(vault_root)
            except ValueError:
                rel_path = hit.path

            excerpt = semantic_passages.get(hit.path, "")
            if not excerpt:
                body = indexed_body(hit) if indexed_body else None
                if body is not None:
                    excerpt = await asyncio.to_thread(lexical_passage, query, hit.title, body)
                elif indexed_body is None:
                    excerpt = hit.snippet or hit.preview

            if not excerpt:
                log.warning(
                    "wiki-recall: hit %r has empty snippet and preview; "
                    "no body excerpt available",
                    hit.title,
                )
                excerpt = _NO_EXCERPT
            prefix = f"- **{' '.join(hit.title.split())[:160]}** — "
            suffix = f" [{_trust_label(hit)}] ({rel_path})"
            remaining = _MAX_BLOCK_CHARS - length - len(prefix) - len(suffix) - 1
            available = min(_MAX_EXCERPT_CHARS, remaining)
            if available < min(len(excerpt), 80):
                break
            line = f"{prefix}{_clip_excerpt(excerpt, available)}{suffix}"
            lines.append(line)
            length += len(line) + 1

        return ToolResult(success=True, output="\n".join(lines))


_TRUST_MARKERS: tuple[tuple[str, str], ...] = (
    ("__fm_status_verified__", "verified"),
    ("__fm_status_proposed__", "unconfirmed"),
    ("__fm_status_superseded__", "superseded"),
)


def _trust_label(hit: Any) -> str:
    """How much weight this hit deserves, in words the model can repeat.

    Recall reaches the whole vault on purpose: a lesson exists so a mistake is
    not repeated, and a gate that hid every note without machine-confirmed
    evidence hid 209 of 216 of them, which defeats the point. Honesty is
    therefore carried per hit instead of by exclusion — the model is told what
    it is reading and can say "an unconfirmed note from the vault says ...".

    The marker strings come from ``jarvis/memory/wiki/fts_index.py``, which
    indexes ``status: <value>`` as ``__fm_status_<value>__`` exactly so a
    status cannot be confused with a field that merely contains the word.
    A note with no status at all is reported as unknown, never as verified.
    """
    frontmatter = str(getattr(hit, "frontmatter", "") or "")
    for marker, label in _TRUST_MARKERS:
        if marker in frontmatter:
            return label
    return "status unknown"


def _clip_excerpt(excerpt: str, limit: int) -> str:
    excerpt = " ".join(excerpt.split())
    if len(excerpt) <= limit:
        return excerpt
    boundaries = (
        match.start() for match in _SENTENCE_END.finditer(excerpt[:limit + 1])
        if match.start() < limit
    )
    end = max(boundaries, default=limit - 1)
    return excerpt[:end].rstrip() + "…"


def _semantic_hits(query: str, vault_root: Path) -> list[SearchHit]:
    try:
        from jarvis.memory.wiki.semantic_search import get_active

        active = get_active()
        if active is not None and active.vault_root == vault_root:
            return active.search(query, k=_CANDIDATE_DEPTH, full_passage=True)
    except Exception as exc:  # noqa: BLE001 - lexical recall stands on its own
        log.debug("wiki-recall: semantic search skipped: %s", exc)
    return []


def _build_search_instance() -> VaultSearch:
    """Build a ``VaultSearch`` with the configured vault root.

    Resolves through :func:`jarvis.memory.wiki.vault_root.resolve_vault_root`
    (spec A7), so a relative root anchors to the repo root, never the
    process CWD.  Falls back to the resolver's default vault location when
    the config field is absent, logging a single WARNING in that case.
    """
    from jarvis.memory.wiki.external_sources import register_from_config
    from jarvis.memory.wiki.search import VaultSearch
    from jarvis.memory.wiki.vault_root import resolve_vault_root

    raw: str | Path | None = None
    sources: tuple = ()
    verified_only = False
    try:
        from jarvis.core import config as cfg

        loaded = cfg.load_config()
        sources = register_from_config(loaded)
        # Agent A defines wiki_integration.vault_root; it may not exist yet.
        wiki_cfg = getattr(loaded, "wiki_integration", None)
        if wiki_cfg is not None:
            raw = wiki_cfg.vault_root
            verified_only = bool(getattr(wiki_cfg, "verified_only", False))
    except Exception as exc:  # noqa: BLE001
        log.debug("wiki-recall: config load skipped: %s", exc)

    resolved = resolve_vault_root(raw).path
    if raw is None:
        log.warning(
            "wiki-recall: cfg.wiki_integration.vault_root not found; "
            "defaulting to %s",
            resolved,
        )

    return VaultSearch(
        resolved,
        external_sources=sources,
        verified_only=verified_only,
    )
