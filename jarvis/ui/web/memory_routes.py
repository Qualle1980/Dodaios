"""HTTP surface for the personal memory vault (ADR-0044).

Why this exists. ADR-0044 opens by naming three questions no earlier store
could answer: *what do you know about me?*, *how did that fact change?*, and
*give me my memory as a file I can take with me.* The vault answers all three,
and ``jarvis/memory/vault/portable.py`` has carried ``export_markdown``,
``export_bundle`` and ``import_bundle`` since the port — tested in
``tests/unit/memory/vault/test_portable.py`` — with **no caller anywhere**, in
this tree or upstream. Until now the only way into the vault was a model turn
through the ``remember`` / ``memory-search`` tools, so the third promise could
not be kept by anyone: not the owner, not an operator, not another agent.

Note the name collision that hides this: ``memory_vault_routes.py`` serves
``/api/wiki`` and is the page-shaped wiki, not this vault. This module is the
vault's own surface and uses ``/api/memory``.

Every write goes through ``MemoryVault.remember``, so the regex-only secret
guard (AP-2), the one-statement length limit, the kind/origin validation, the
content-hash idempotence and the tombstone rule all apply unchanged. A route
that bypassed them would be a second write path, and a second write path is how
a guard stops being a guard.

``origin`` is deliberately **not** defaulted to ``user``. The vault caps the
confidence of a non-user origin and refuses to resurrect a forgotten fact
unless the user asks, so a caller that claims ``user`` is claiming the owner
said it. An agent writing on its own reading says ``inferred`` and is believed
less, which is the honest arrangement.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from jarvis.memory.vault import runtime as vault_runtime
from jarvis.memory.vault.models import KINDS, ORIGINS, SENSITIVITIES
from jarvis.memory.vault.store import MemoryVault

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["memory"])

#: An import bundle is a zip of JSONL files. A real export of a long-lived
#: vault is far smaller than this; the cap exists so a wrong upload fails fast
#: instead of being read into memory.
MAX_BUNDLE_BYTES = 64 * 1024 * 1024


def _vault(request: Request) -> MemoryVault:
    """The open vault, or a 404 that says which of the two reasons applies."""
    config = getattr(request.app.state, "config", None)
    if config is None:  # pragma: no cover - the app always carries its config
        raise HTTPException(503, "The backend is still starting.")
    if not vault_runtime.vault_enabled(config):
        raise HTTPException(404, "The personal memory vault is switched off ([memory.vault]).")
    vault = vault_runtime.active(config)
    if vault is None:
        raise HTTPException(404, "The personal memory vault is not available.")
    return vault


@router.get("/stats")
async def memory_stats(request: Request) -> dict[str, Any]:
    """How much the vault holds, per kind — the cheap answer to "what is in there"."""
    import asyncio

    vault = _vault(request)
    return await asyncio.to_thread(vault.stats)


@router.get("/export", openapi_extra={"x-jarvis-risk-tier": "monitor"})
async def memory_export(
    request: Request,
    fmt: str = Query(default="markdown", pattern="^(markdown|bundle)$"),
) -> Any:
    """Hand the whole memory over: readable Markdown, or a re-importable bundle.

    ``markdown`` returns ``{path: text}`` — ``MEMORY.md`` plus one file per
    kind, which opens in any Markdown editor. ``bundle`` returns the zip
    ``import`` accepts, so a vault can be moved between machines.

    This reads EVERY memory, including the ones a profile block deliberately
    leaves out, so it is a monitored action rather than an ordinary read.
    """
    import asyncio

    from jarvis.memory.vault.portable import export_zip_bytes, markdown_files

    vault = _vault(request)
    if fmt == "bundle":
        data = await asyncio.to_thread(export_zip_bytes, vault)
        return Response(
            content=data,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="memory-bundle.zip"'},
        )
    files = await asyncio.to_thread(markdown_files, vault)
    return {"format": "markdown", "files": files}


@router.post("/import", openapi_extra={"x-jarvis-risk-tier": "monitor"})
async def memory_import(request: Request) -> dict[str, Any]:
    """Apply a bundle produced by ``/export?fmt=bundle``. Importing twice is a no-op.

    The body is the raw zip. ``import_bundle`` applies tombstones first and then
    inserts only memories whose id and content are unknown, so a repeated import
    changes nothing — which is what makes this safe to retry after a failure.
    """
    import asyncio

    from jarvis.memory.vault.portable import BundleError, import_bundle

    vault = _vault(request)
    body = await request.body()
    if not body:
        raise HTTPException(400, "The request body must be the bundle.")
    if len(body) > MAX_BUNDLE_BYTES:
        raise HTTPException(413, "The bundle is larger than this endpoint accepts.")
    try:
        report = await asyncio.to_thread(import_bundle, vault, body)
    except BundleError as exc:
        # Not swallowed (AP-30): a malformed bundle is the caller's error and
        # its reason is the useful part of the answer.
        raise HTTPException(400, f"The bundle could not be read: {exc}") from exc
    return report.to_dict()


class RememberBody(BaseModel):
    """One statement. Long text is a sign it is several memories, not one."""

    text: str = Field(min_length=1, max_length=2000)
    kind: str = Field(default="profile")
    origin: str = Field(default="inferred")
    entities: list[str] = Field(default_factory=list)
    scope: str = Field(default="global")
    sensitivity: str = Field(default="normal")
    pinned: bool = False
    source: dict[str, Any] | None = None
    valid_to: str | None = None


@router.post("/remember", openapi_extra={"x-jarvis-risk-tier": "monitor"})
async def memory_remember(payload: RememberBody, request: Request) -> dict[str, Any]:
    """Store one atomic memory, or report the identical one already current.

    The store's own ``action`` is returned verbatim rather than flattened to
    ok/not-ok, because these are different facts about the vault and a caller
    that cannot tell them apart will write the same thing again: ``added``
    stored it, ``existing`` means it was already current, ``refused`` means the
    secret guard rejected the text, and ``blocked`` means the owner has
    forgotten it and only the owner may bring it back.
    """
    import asyncio

    vault = _vault(request)
    if payload.kind not in KINDS:
        raise HTTPException(400, f"unknown kind {payload.kind!r}; one of {list(KINDS)}")
    if payload.origin not in ORIGINS:
        raise HTTPException(400, f"unknown origin {payload.origin!r}; one of {list(ORIGINS)}")
    if payload.sensitivity not in SENSITIVITIES:
        raise HTTPException(400, f"unknown sensitivity {payload.sensitivity!r}")
    try:
        result = await asyncio.to_thread(
            vault.remember,
            payload.text,
            payload.kind,
            entities=tuple(payload.entities),
            origin=payload.origin,
            source=payload.source,
            scope=payload.scope,
            sensitivity=payload.sensitivity,
            pinned=payload.pinned,
            valid_to=payload.valid_to,
        )
    except ValueError as exc:
        # The store validates its own invariants; passing its words on beats
        # inventing a vaguer message here.
        raise HTTPException(400, str(exc)) from exc
    memory = result.memory
    return {
        "action": result.action,
        "reason": result.reason,
        "memory_id": None if memory is None else memory.id,
    }


__all__ = ["MAX_BUNDLE_BYTES", "router"]
