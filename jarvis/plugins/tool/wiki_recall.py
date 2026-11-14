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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.core.protocols import ToolResult

if TYPE_CHECKING:
    from jarvis.memory.wiki.search import VaultSearch

log = logging.getLogger(__name__)


class WikiRecallTool:
    """Router-tier keyword search over the long-term Obsidian wiki vault."""

    read_only = True

    name: str = "wiki-recall"
    description: str = (
        "Search the user's long-term Obsidian wiki for notes matching keywords. "
        "Returns a compact markdown summary with up to 5 ranked hits. Use this "
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
                "default": 5,
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
        k = int(args.get("k", 5))
        # Clamp defensively — schema validation may not have run upstream.
        k = max(1, min(k, 10))

        vault_root = self._search._root  # type: ignore[attr-defined]

        # Privacy: query is DEBUG only.
        log.debug("wiki-recall: query=%r k=%d vault=%s", query, k, vault_root)

        if not vault_root.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"vault unavailable at {vault_root}",
            )

        hits = self._search.search(query, k=k)
        # With semantic search switched on, add pages that mean the same but
        # share no word with the query. Never waits for the model, and runs
        # off the event loop: encoding the query is CPU work.
        hits = await asyncio.to_thread(_with_semantic_hits, hits, query, vault_root)

        log.info("wiki-recall: %d hit(s) for query (k=%d)", len(hits), k)

        if not hits:
            return ToolResult(
                success=True,
                output=(
                    f'No wiki matches found for "{query}" in {vault_root}.'
                ),
            )

        lines: list[str] = [f'## Wiki hits for "{query}"']
        for hit in hits:
            # Render path relative to vault_root so the brain sees a short,
            # portable path rather than an absolute Windows path.
            try:
                rel_path = hit.path.relative_to(vault_root)
            except ValueError:
                rel_path = hit.path

            snippet_part = f" — {hit.snippet}" if hit.snippet else ""
            lines.append(
                f"- **{hit.title}**{snippet_part} "
                f"[{_trust_label(hit)}] ({rel_path})"
            )

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


def _with_semantic_hits(hits: list[Any], query: str, vault_root: Path) -> list[Any]:
    try:
        from jarvis.memory.wiki.semantic_search import with_semantic_hits

        return with_semantic_hits(hits, query, vault_root=vault_root)
    except Exception as exc:  # noqa: BLE001 - the word search result stands on its own
        log.debug("wiki-recall: semantic search skipped: %s", exc)
        return hits


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
