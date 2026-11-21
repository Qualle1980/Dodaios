"""Voice runtime_location tool wraps the shared deployment facts."""

from __future__ import annotations

import pytest

from jarvis.core import runtime_refs
from jarvis.plugins.tool.runtime_location import RuntimeLocationTool
from jarvis.ui.web import surface_security


@pytest.fixture(autouse=True)
def _clear_listen_stamp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_refs, "_LISTEN_HOST", [])
    monkeypatch.setattr(runtime_refs, "_LISTEN_PORT", [])
    monkeypatch.setattr(runtime_refs, "_API_BASE_URL", [])
    monkeypatch.delenv("JARVIS_BIND_HOST", raising=False)
    monkeypatch.delenv("JARVIS_INSTANCE", raising=False)
    surface_security.reset_browser_login_required()
    surface_security.set_browser_login_required(False)
    yield
    surface_security.reset_browser_login_required()


@pytest.mark.asyncio
async def test_tool_returns_deployment_facts() -> None:
    result = await RuntimeLocationTool().execute({}, None)  # type: ignore[arg-type]
    assert result.success
    assert result.output["ok"] is True
    assert result.output["instance"] in {"default", "dev"}
    assert result.output["bind_scope"] in {"loopback", "external"}
    assert "control_key_path" in result.output
    assert "hostname" not in result.output
