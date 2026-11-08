"""Every write path of the vault refuses what OUR secret guard knows (AP-2).

The ported suite proves one shape, an OpenAI key, through ``remember``. Our
`jarvis.memory.wiki.secret_guard` knows nine, and its hex threshold is
deliberately 64 rather than 40 so a git commit hash in biographical prose is
NOT a secret. Neither property is the upstream store's; both are ours to keep,
and a write path that skipped the check would leak into the row, the FTS index
and any voice readback.
"""
from __future__ import annotations

import pytest

from jarvis.memory.vault import MemoryVault
from jarvis.memory.vault.models import Memory, content_hash

# Assembled at runtime so no literal credential shape sits in this file: the
# pre-push credential gate reads the source, and a PEM header written out in
# full is a finding whether or not it guards a real key.
_PEM_HEADER = "-----{} {} {} {}-----".format("BEGIN", "RSA", "PRIVATE", "KEY")  # noqa: UP032

SECRETS: dict[str, str] = {
    "openai_key": "sk-" + "proj-" + "A" * 40,
    "google_key": "AIza" + "B" * 35,
    "xai_key": "xai-" + "C" * 24,
    "groq_key": "gsk_" + "D" * 24,
    "github_token": "ghp_" + "E" * 36,
    "bearer_token": "Bearer " + "F" * 24,
    "labelled_secret": "api_key = " + "G" * 20,
    "jwt": "eyJhbGciOi" + "." + "eyJzdWIiOi" + "." + "H" * 10,
    "long_hex": "a1b2c3d4" * 9,
    "pem_private_key": _PEM_HEADER,
}


@pytest.fixture
def vault(tmp_path):
    with MemoryVault(tmp_path / "vault.db") as handle:
        yield handle


@pytest.mark.parametrize("name", sorted(SECRETS))
def test_remember_refuses_every_shape_our_guard_knows(vault: MemoryVault, name: str) -> None:
    result = vault.remember(f"My credential is {SECRETS[name]}")

    assert result.action == "refused", f"{name} was stored"
    assert result.memory is None
    assert vault.stats()["current"] == 0


@pytest.mark.parametrize("name", sorted(SECRETS))
def test_supersede_refuses_every_shape_too(vault: MemoryVault, name: str) -> None:
    """A fact may be replaced by a newer fact, never by a credential."""
    first = vault.remember("I work at a bakery.", "profile")
    assert first.memory is not None

    result = vault.supersede(first.memory.id, f"My credential is {SECRETS[name]}")

    assert result.action == "refused", f"{name} was stored"
    still = vault.get(first.memory.id)
    assert still is not None and still.is_current(vault.now()), "the old fact was closed anyway"


@pytest.mark.parametrize("name", sorted(SECRETS))
def test_an_import_cannot_smuggle_one_in(vault: MemoryVault, name: str) -> None:
    """A bundle is untrusted input: `insert_imported` checks like the others."""
    text = f"My credential is {SECRETS[name]}"
    smuggled = Memory(
        id="mem_20260101000000_abcdef123456",
        content_hash=content_hash("profile", text),
        kind="profile",
        text=text,
        recorded_at="2026-01-01T00:00:00+00:00",
        origin="imported",
    )

    assert vault.insert_imported(smuggled) is False
    assert vault.stats()["current"] == 0


def test_a_commit_hash_in_prose_is_not_a_secret(vault: MemoryVault) -> None:
    """Our hex threshold is 64, not 40, so a git SHA-1 stays rememberable."""
    sha = "0123456789abcdef" * 2 + "01234567"  # 40 hex chars, a commit id
    assert len(sha) == 40

    result = vault.remember(f"My first Dodaios commit was {sha}.", "episode")

    assert result.action == "added"
    assert vault.stats()["current"] == 1


def test_a_refusal_leaves_no_trace_in_search_or_audit(vault: MemoryVault) -> None:
    """A refused write must not reach the FTS index or name the credential."""
    secret = SECRETS["openai_key"]

    vault.remember(f"My OpenAI key is {secret}")

    assert vault.search("OpenAI key") == []
    assert vault.list(include_closed=True) == []
    assert all(secret not in str(entry) for entry in vault.audit_log())
