"""A dead ChatGPT browser bridge must not be reported as a network problem.

Live forensic (2026-10-30, the maintainer's own install): ``deep_brain`` was
``chatgpt-web``, the browser-session bridge. Eleven times in two and a half
hours a question that routed to the deep tier ended in

    Alle 1 Provider-Versuche fehlgeschlagen. Letzter Fehler:  # i18n-allow
    ChatGPT bridge error 200 browser_stream_inconsistent
    -> "Provider chatgpt-web unerreichbar. Netzwerk pruefen."

The network was fine the whole time; the browser tab holding the ChatGPT
session was. Sent to look at their router, the maintainer reported the app as
"no connection" and lost the afternoon to it. The bridge failure is its own
user-actionable category -- reopen or re-authorize the ChatGPT tab -- and the
spoken cause has to say so, the way ``context_overflow`` already refuses to
blame the network for a request that was simply too big.
"""
from __future__ import annotations

from jarvis.brain.manager import _DEAD_LIST_KINDS, _classify_provider_error

# Both wordings seen in the live log on 2026-10-30.
BRIDGE_STREAM = (
    "ChatGPT bridge error 200 browser_stream_inconsistent: ChatGPT channel "
    "produced an inconsistent stream"
)
BRIDGE_CHANNEL = (
    "ChatGPT bridge error 200 chatgpt_channel_closed: the browser channel closed"
)


def test_a_bridge_failure_is_its_own_kind_not_a_generic_one() -> None:
    for msg in (BRIDGE_STREAM, BRIDGE_CHANNEL):
        assert _classify_provider_error(msg, default="call_fail") == "browser_bridge"


def test_a_bridge_failure_is_transient_so_it_never_dead_lists() -> None:
    """Reopening the tab fixes it; the provider must come back on its own."""
    assert "browser_bridge" not in _DEAD_LIST_KINDS


def test_the_spoken_cause_names_the_tab_and_never_the_network() -> None:
    from jarvis.brain.manager import _format_provider_chain_error

    cause = _format_provider_chain_error(
        [("chatgpt-web", "chatgpt-web/instant", "browser_bridge", "")]
    )
    assert "chatgpt-web" in cause
    assert "Netzwerk" not in cause
    # The action the person can actually take.
    assert "ChatGPT" in cause


def test_a_real_network_failure_still_reads_as_one() -> None:
    from jarvis.brain.manager import _format_provider_chain_error

    cause = _format_provider_chain_error([("ollama", "llama3", "call_fail", "")])
    assert "Netzwerk" in cause
