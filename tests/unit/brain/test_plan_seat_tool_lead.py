"""The tool lead a plan-seat talker is actually allowed to have.

A plan-seat talker — a signed-in subscription with no key slot, e.g. the
ChatGPT plan/web brains — cannot emit ``tool_calls``, so a tool turn needs
another provider to LEAD it and pick the tool. Two bugs made that lead
unreachable:

* ``_first_tool_capable_provider`` walked a closed list of key-billed
  providers, so a keyless install (a local runtime only) found no lead at all;
* for a plan-seat talker every candidate on that list was then dropped by the
  seat filter in ``_build_fallback_chain``, which permits only the seat itself
  and keyless local runtimes. The lead was chosen and discarded, the turn ran
  toolless, and the delegation tools that hand coding work to Codex CLI /
  Claude Code were never reachable from a ChatGPT plan/web talker.

The lead is now enumerated from the registry and narrowed with the SAME seat
policy the chain filter uses, so the provider picked here is one the chain can
really reach. The seat guard itself is unchanged: no stored API key stands in
for the user's plan.

Deterministic — no live provider, no network, no model call.
"""
from __future__ import annotations

from jarvis.brain.manager import BrainManager
from jarvis.core.bus import EventBus
from jarvis.core.config import JarvisConfig


class _ToolCapableBrain:
    supports_tools = True


def _manager(active: str, available: dict[str, object]) -> BrainManager:
    mgr = BrainManager(config=JarvisConfig(), bus=EventBus(), tools={})
    mgr._active_name = active  # type: ignore[attr-defined]
    mgr._fast_model = lambda name: f"{name}-fast"  # type: ignore[method-assign]
    mgr._registry._loaded = True
    mgr._registry._classes = dict(available)  # type: ignore[attr-defined]
    return mgr


def test_plan_seat_talker_is_led_by_a_keyless_local_runtime() -> None:
    """ollama leads, not the API provider the chain's seat filter would drop."""
    mgr = _manager("chatgpt-web", {"gemini": object, "ollama": object})
    mgr._brain_cache[("gemini", "gemini-fast")] = _ToolCapableBrain()
    mgr._brain_cache[("ollama", "ollama-fast")] = _ToolCapableBrain()

    assert mgr._first_tool_capable_provider("fast") == ("ollama", "ollama-fast")


def test_a_plan_seat_lead_survives_the_chain_seat_filter() -> None:
    """The whole point: the lead must still be there after the chain is built."""
    mgr = _manager("chatgpt-web", {"gemini": object, "ollama": object})
    mgr._brain_cache[("gemini", "gemini-fast")] = _ToolCapableBrain()
    mgr._brain_cache[("ollama", "ollama-fast")] = _ToolCapableBrain()
    mgr._active_can_call_tools = lambda: False  # type: ignore[assignment]
    mgr._turn_substantive = True

    chain = mgr._build_fallback_chain("fast")

    assert mgr._router_lead_key == ("ollama", "ollama-fast")
    assert ("ollama", "ollama-fast") in chain
    assert all(name != "gemini" for name, _ in chain), (
        "a stored API key must never stand in for the user's plan seat"
    )


def test_no_key_billed_lead_for_a_plan_seat_talker() -> None:
    """Without a local runtime the seat guard wins: no lead, honest toolless turn."""
    mgr = _manager("chatgpt-web", {"gemini": object, "claude-api": object})
    mgr._brain_cache[("gemini", "gemini-fast")] = _ToolCapableBrain()
    mgr._brain_cache[("claude-api", "claude-api-fast")] = _ToolCapableBrain()

    assert mgr._first_tool_capable_provider("fast") is None


def test_keyless_install_finds_a_local_lead() -> None:
    """Not seat-bound, but key-billed providers were the only candidates before."""
    mgr = _manager("codex", {"ollama": object})
    mgr._brain_cache[("ollama", "ollama-fast")] = _ToolCapableBrain()

    assert mgr._first_tool_capable_provider("fast") == ("ollama", "ollama-fast")


def test_an_unconfigured_local_runtime_does_not_win_the_lead() -> None:
    """Two local runtimes registered, one configured: the configured one leads.

    Both are keyless and tool-capable, so the seat policy permits either. The
    install's second local runtime has no model configured, and a lead with no
    model can never answer — plain alphabetical order handed it the turn.
    """
    mgr = _manager("chatgpt-web", {"local-openai": object, "ollama": object})
    models = {"ollama": "qwen3.5:4b", "local-openai": ""}
    mgr._fast_model = lambda name: models.get(name, "")  # type: ignore[method-assign]
    mgr._deep_model = lambda name: models.get(name, "")  # type: ignore[method-assign]
    mgr._brain_cache[("ollama", "qwen3.5:4b")] = _ToolCapableBrain()
    mgr._brain_cache[("local-openai", "")] = _ToolCapableBrain()

    assert mgr._first_tool_capable_provider("fast") == ("ollama", "qwen3.5:4b")
