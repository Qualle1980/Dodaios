"""CLI auth probes must never run on the event loop.

``ClaudeAuthService.status`` / ``CodexAuthService.status`` /
``GoogleCliAuthService.status`` and the matching ``logout_blocking`` SPAWN the
real CLI binary. That costs a few hundred milliseconds (and after the agy
keyring probe, up to several seconds), and on the event loop it freezes
everything else this process is doing.

Live forensic 2026-07-27 (realtime voice, gemini-live): the UI's provider
panels poll these endpoints while a reply is being spoken. Each inline probe
stalled the loop 311-375 ms, the realtime audio socket went unread, the device
buffer drained, and the maintainer heard a hole in the middle of a sentence —
logged by ``RealtimeSession._note_audio_flow`` as "this process's event loop
stalled ... the audio likely sat unread in the socket".

This is a structural guard, not a behavioural one: it walks the route modules
and fails on any ``async def`` that *calls* a blocking ``.status()`` /
``.logout_blocking()`` on the coroutine itself. A sibling ``to_thread`` for
``start_login`` does not excuse an inline ``status()`` — that was the
``antigravity_login`` regression after the keyring probe made ``status()``
expensive. Nested sync helpers passed to ``asyncio.to_thread`` are fine;
adding a new provider CLI is exactly when this regresses.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: Auth services whose probes spawn a subprocess.
BLOCKING_SERVICES = (
    "ClaudeAuthService",
    "CodexAuthService",
    "GoogleCliAuthService",
)

#: Method names that must not be *called* on the event-loop coroutine body.
BLOCKING_METHODS = frozenset({"status", "logout_blocking"})

#: Route modules that expose those services to the UI.
ROUTE_MODULES = (
    "jarvis/ui/web/provider_routes.py",
    "jarvis/ui/web/antigravity_routes.py",
    "jarvis/ui/web/claude_routes.py",
    "jarvis/ui/web/server.py",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _async_defs_touching_a_blocking_service(path: Path):
    """Yield (function name, node) for async defs naming a blocking service."""
    # utf-8-sig: several route modules are BOM-prefixed (the config writer's
    # BOM-safe contract, AP-7), which plain utf-8 parsing rejects.
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        body = ast.get_source_segment(source, node) or ""
        if any(service in body for service in BLOCKING_SERVICES):
            yield node.name, node


class _DirectBlockingCallFinder(ast.NodeVisitor):
    """Find ``.status()`` / ``.logout_blocking()`` calls on the async body.

    Nested ``def`` / ``async def`` / ``lambda`` bodies are skipped: those are
    the helpers a route correctly hands to ``asyncio.to_thread``.
    """

    def __init__(self) -> None:
        self.offenders: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        return

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in BLOCKING_METHODS:
            self.offenders.append(func.attr)
        self.generic_visit(node)


@pytest.mark.parametrize("module", ROUTE_MODULES)
def test_route_handlers_offload_cli_auth_probes(module: str) -> None:
    path = _REPO_ROOT / module
    if not path.exists():  # pragma: no cover - module renamed
        pytest.skip(f"{module} is gone")

    offenders: list[str] = []
    for name, node in _async_defs_touching_a_blocking_service(path):
        finder = _DirectBlockingCallFinder()
        for stmt in node.body:
            finder.visit(stmt)
        if finder.offenders:
            offenders.append(f"{name}({', '.join(finder.offenders)})")

    assert not offenders, (
        f"{module}: {offenders} call a CLI auth probe inline. The probe spawns "
        "a binary and blocks the event loop for hundreds of milliseconds to "
        "seconds, which is heard as a hole in the middle of a spoken answer. "
        "Wrap it: `await asyncio.to_thread(service.status)`. A sibling "
        "`to_thread` for start_login does not cover status()."
    )
