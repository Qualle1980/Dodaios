"""The personal memory vault finally has a door (ADR-0044).

ADR-0044 opens by naming three questions: what do you know about me, how did
that fact change, and give me my memory as a file I can take with me. The third
could not be kept by anyone: ``portable.py`` carried ``export_markdown``,
``export_bundle`` and ``import_bundle``, tested, with no caller in this tree or
upstream, and the only way into the vault was a model turn through the
``remember`` tool. These tests pin the surface that opens it.

Watch the name collision: ``memory_vault_routes.py`` serves ``/api/wiki`` and is
the page-shaped wiki. This module's routes are the vault's own, under
``/api/memory``.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from jarvis.memory.vault import MemoryVault
from jarvis.memory.vault import runtime as vault_runtime
from jarvis.ui.web import memory_routes as routes


def _clock() -> str:
    return "2026-03-01T10:00:00+00:00"


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryVault:
    """A vault with one superseded fact, one pinned preference and one forgotten."""
    v = MemoryVault(tmp_path / "vault.db", clock=_clock)
    old = v.remember("I live in Hamburg.", "profile", entities=["Hamburg"]).memory
    v.supersede(old.id, "I live in Berlin.", entities=["Berlin"])
    v.remember("Prefers German replies.", "preference", pinned=True)
    monkeypatch.setattr(vault_runtime, "_active", v, raising=False)
    monkeypatch.setattr(vault_runtime, "active", lambda _cfg: v)
    monkeypatch.setattr(vault_runtime, "vault_enabled", lambda _cfg: True)
    yield v
    v.close()


def _request() -> SimpleNamespace:
    """Routes only need ``app.state.config``; the vault handle is process-wide."""
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(config=object())))


# --- the vault is reachable at all --------------------------------------------


async def test_stats_answers_what_is_in_there(vault: MemoryVault) -> None:
    stats = await routes.memory_stats(_request())

    assert isinstance(stats, dict)
    assert stats  # the shape is the store's; that it answers is the contract here


async def test_a_switched_off_vault_says_so_instead_of_failing_obscurely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vault_runtime, "vault_enabled", lambda _cfg: False)

    with pytest.raises(HTTPException) as caught:
        await routes.memory_stats(_request())

    assert caught.value.status_code == 404
    # The owner has to be able to tell "switched off" from "broken".
    assert "switched off" in str(caught.value.detail)


async def test_a_missing_vault_is_a_404_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vault_runtime, "vault_enabled", lambda _cfg: True)
    monkeypatch.setattr(vault_runtime, "active", lambda _cfg: None)

    with pytest.raises(HTTPException) as caught:
        await routes.memory_stats(_request())

    assert caught.value.status_code == 404


# --- "give me my memory as a file I can take with me" -------------------------


async def test_markdown_export_is_readable_and_names_its_files(vault: MemoryVault) -> None:
    out = await routes.memory_export(_request(), fmt="markdown")

    assert out["format"] == "markdown"
    files = out["files"]
    assert any(name.endswith("MEMORY.md") for name in files), files
    joined = "\n".join(files.values())
    assert "Berlin" in joined
    # The superseded fact must not read as still true in a human-facing export.
    assert "Hamburg" not in joined or "Berlin" in joined


async def test_bundle_export_is_a_zip_that_import_accepts(vault: MemoryVault) -> None:
    response = await routes.memory_export(_request(), fmt="bundle")

    assert response.media_type == "application/zip"
    assert "attachment" in response.headers["Content-Disposition"]
    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        names = archive.namelist()
    assert "manifest.json" in names, names


# --- a bundle goes back in, and twice changes nothing -------------------------


async def test_import_restores_into_an_empty_vault(
    vault: MemoryVault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exported = await routes.memory_export(_request(), fmt="bundle")
    body = exported.body

    target = MemoryVault(tmp_path / "target.db", clock=_clock)
    monkeypatch.setattr(vault_runtime, "active", lambda _cfg: target)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(config=object())),
        body=_body_of(body),
    )

    report = await routes.memory_import(request)

    assert report["added"] > 0
    assert not report["errors"]
    assert any("Berlin" in m.text for m in target.list())
    target.close()


async def test_importing_the_same_bundle_twice_adds_nothing(
    vault: MemoryVault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Idempotence is what makes a failed import safe to retry.
    body = (await routes.memory_export(_request(), fmt="bundle")).body
    target = MemoryVault(tmp_path / "target.db", clock=_clock)
    monkeypatch.setattr(vault_runtime, "active", lambda _cfg: target)

    first = await routes.memory_import(
        SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(config=object())), body=_body_of(body)
        )
    )
    second = await routes.memory_import(
        SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(config=object())), body=_body_of(body)
        )
    )

    assert first["added"] > 0
    assert second["added"] == 0
    target.close()


async def test_an_empty_body_is_a_400_not_an_unreadable_bundle(vault: MemoryVault) -> None:
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(config=object())), body=_body_of(b"")
    )

    with pytest.raises(HTTPException) as caught:
        await routes.memory_import(request)

    assert caught.value.status_code == 400


async def test_rubbish_is_refused_with_its_reason_not_swallowed(vault: MemoryVault) -> None:
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(config=object())),
        body=_body_of(b"this is not a zip"),
    )

    with pytest.raises(HTTPException) as caught:
        await routes.memory_import(request)

    assert caught.value.status_code == 400
    assert "could not be read" in str(caught.value.detail)


# --- writing one fact ---------------------------------------------------------


async def test_remember_stores_one_fact_and_reports_added(vault: MemoryVault) -> None:
    out = await routes.memory_remember(
        routes.RememberBody(text="The ChatGPT subscription ends on 2026-12-01.", kind="project"),
        _request(),
    )

    assert out["action"] == "added"
    assert out["memory_id"]
    assert any("2026-12-01" in m.text for m in vault.list())


async def test_the_same_fact_twice_reports_existing_not_a_second_row(
    vault: MemoryVault,
) -> None:
    body = routes.RememberBody(text="Calls the owner Ahmad.", kind="profile")

    first = await routes.memory_remember(body, _request())
    second = await routes.memory_remember(body, _request())

    assert first["action"] == "added"
    assert second["action"] == "existing"


async def test_origin_defaults_to_inferred_so_a_route_cannot_claim_the_owner_said_it(
    vault: MemoryVault,
) -> None:
    # The vault caps a non-user origin's confidence and refuses to resurrect a
    # forgotten fact unless the user asks. Defaulting to "user" here would let
    # any caller borrow the owner's authority.
    assert routes.RememberBody(text="x").origin == "inferred"

    out = await routes.memory_remember(routes.RememberBody(text="Likes terse answers."), _request())

    stored = next(m for m in vault.list() if "terse" in m.text)
    assert out["action"] == "added"
    assert stored.origin == "inferred"


async def test_a_secret_is_refused_by_the_store_not_stored(vault: MemoryVault) -> None:
    # AP-2: the regex-only guard lives in the store, and this route must not
    # become a second write path that goes around it.
    out = await routes.memory_remember(
        routes.RememberBody(text="My API key is sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"),
        _request(),
    )

    assert out["action"] == "refused"
    assert out["memory_id"] is None
    assert not any("sk-ant" in m.text for m in vault.list())


async def test_an_unknown_kind_is_a_400_with_the_allowed_list(vault: MemoryVault) -> None:
    with pytest.raises(HTTPException) as caught:
        await routes.memory_remember(
            routes.RememberBody(text="Something.", kind="not-a-kind"), _request()
        )

    assert caught.value.status_code == 400
    assert "unknown kind" in str(caught.value.detail)


async def test_an_unknown_origin_cannot_be_smuggled_in(vault: MemoryVault) -> None:
    with pytest.raises(HTTPException) as caught:
        await routes.memory_remember(
            routes.RememberBody(text="Something.", origin="owner"), _request()
        )

    assert caught.value.status_code == 400
    assert "unknown origin" in str(caught.value.detail)


def _body_of(data: bytes):
    """Stand in for Starlette's ``await request.body()``."""

    async def _read() -> bytes:
        return data

    return _read
