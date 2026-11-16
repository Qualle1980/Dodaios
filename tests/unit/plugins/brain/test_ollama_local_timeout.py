"""Local Ollama: loopback read budget and CPU-only context planning.

A CPU-bound local prefill can take minutes (measured ~240 s for a ~23k-token
prompt). The shared network-shaped read ceiling of 120 s is wrong for
loopback; the 2 s connect timeout that feeds the fallback chain must stay.
When the host has no vouchable accelerator, ``num_ctx`` must follow a CPU
profile (8k–16k), never a window sized from misread system RAM as
"accelerator memory".
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from jarvis.core.config import JarvisConfig
from jarvis.plugins.brain.ollama import CLIENT_TIMEOUT, OllamaBrain, client_timeout_for
from tests.fakes.provider_config import install_provider_config


@pytest.fixture(autouse=True)
def _forget_context_windows():
    from jarvis.brain import local_context
    from jarvis.plugins.brain import ollama as plugin

    plugin.reset_context_memo()
    local_context.reset_registry()
    yield
    plugin.reset_context_memo()
    local_context.reset_registry()


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _TagsClient:
    """Minimal AsyncClient stand-in for ``/api/tags`` + ``/api/show``."""

    tags: dict[str, Any] = {"models": [{"name": "qwen3.5:9b", "size": 6 * 1024**3}]}
    show: dict[str, Any] = {
        "capabilities": ["completion", "tools"],
        "model_info": {
            "general.architecture": "qwen3",
            "qwen3.context_length": 262_144,
        },
    }

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _TagsClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(self, url: str) -> _FakeResponse:
        if url.endswith("/api/show"):
            return _FakeResponse(self.show)
        return _FakeResponse(self.tags)

    async def post(self, url: str, json: dict[str, Any] | None = None) -> _FakeResponse:
        return _FakeResponse(self.show)


def _local_config(monkeypatch: pytest.MonkeyPatch) -> None:
    install_provider_config(monkeypatch, JarvisConfig())
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


def test_local_ollama_read_timeout_exceeds_network_ceiling() -> None:
    """Loopback may stream for minutes; a remote host keeps today's budget.

    Connect stays at 2 s either way so an unreachable endpoint still fails
    into the fallback chain (and PR #107's first-sign lead budget remains a
    separate bound on setup, not this read value).
    """
    local = client_timeout_for("http://127.0.0.1:11434")
    localhost = client_timeout_for("http://localhost:11434")
    remote = client_timeout_for("http://gpu.lan:11434")

    assert local.connect <= 2.0
    assert localhost.connect <= 2.0
    assert remote.connect <= 2.0
    assert local.read >= 300.0
    assert localhost.read >= 300.0
    assert remote.read == CLIENT_TIMEOUT.read
    assert CLIENT_TIMEOUT.connect <= 2.0


@pytest.mark.asyncio
async def test_cpu_only_host_does_not_plan_from_accelerator_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No vouchable accelerator → CPU profile (8k–16k), never accelerator copy."""
    from jarvis.brain import ollama_profiles, ollama_pull

    monkeypatch.setattr(httpx, "AsyncClient", _TagsClient)
    monkeypatch.setattr(ollama_pull, "accelerator_gb", lambda: (0.0, "none"))
    # Plenty of RAM would otherwise push automatic_context to the 32k ceiling.
    monkeypatch.setattr(ollama_pull, "total_memory_gb", lambda: 64.0)
    _local_config(monkeypatch)

    tokens, reason = ollama_profiles.automatic_context(
        size_gb=6.0,
        native_context=262_144,
        accelerator_gb=0.0,
        ram_gb=64.0,
    )
    assert "accelerator memory" not in reason

    brain = OllamaBrain(model="qwen3.5:9b")
    planned = await brain._automatic_context("qwen3.5:9b")
    assert 8_192 <= planned <= 16_384
    # The plugin must clamp the RAM-sized ceiling down to the CPU profile.
    assert planned <= 16_384
    assert planned < tokens or tokens <= 16_384


@pytest.mark.asyncio
async def test_real_accelerator_keeps_current_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real card still reaches the automatic ceiling; CPU clamp stays off."""
    from jarvis.brain import ollama_pull

    monkeypatch.setattr(httpx, "AsyncClient", _TagsClient)
    monkeypatch.setattr(ollama_pull, "accelerator_gb", lambda: (24.0, "nvidia-smi"))
    monkeypatch.setattr(ollama_pull, "total_memory_gb", lambda: 64.0)
    _local_config(monkeypatch)

    brain = OllamaBrain(model="qwen3.5:9b")
    planned = await brain._automatic_context("qwen3.5:9b")
    assert planned == 32_768
