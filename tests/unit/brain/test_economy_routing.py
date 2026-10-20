"""What a turn costs before any model has spoken (phase 5).

The economy question is not "which model is cheapest" but "how many paid
executors does an ordinary turn wake up". Everything asserted here is a
decision the code makes with no model call at all, measured on the shape of a
real install (a ChatGPT seat as the talker, codex as the worker):

* a smalltalk turn builds a chain of exactly ONE provider — the one the user
  picked — so a greeting or a clock question cannot add a second model call;
* a substantive turn on a seat talker is LED by a keyless local runtime, never
  by a key-billed family: a seat cannot emit tool calls, so something must lead
  it, and ``seat_successors`` decides that the something is free;
* an explicit heavy-work trigger dispatches deterministically, so the most
  expensive route (a worker) is the one route that needs no routing round;
* an ordinary coding sentence does NOT force-spawn — delegation happens on a
  named vehicle or a measured effort signal, not on the word "fix";
* the retry bound is a constant, not a loop that re-probes a capped plan.

Deterministic — no network, no provider call, no model inference.
"""
from __future__ import annotations

from typing import Any

import pytest

from jarvis.brain.background_policy import seat_bound
from jarvis.brain.intent_router import classify
from jarvis.brain.manager import BrainManager
from jarvis.core.bus import EventBus
from jarvis.core.config import BrainTierConfig, JarvisConfig
from jarvis.core.protocols import ExecutionContext, ToolResult

#: Families that bill a stored API key per token. A turn that only chats must
#: not reach one of these on its own.
_KEY_BILLED = frozenset(
    {"claude-api", "gemini", "openai", "openrouter", "grok", "nvidia", "groq", "vertex"}
)

_SMALLTALK = ("danke", "wie spaet ist es", "hallo")
_SUBSTANTIVE = (
    "fix the typo in the README",
    "add a health endpoint to the gateway",
    "fuehre die tests aus",  # i18n-allow: simulated German user utterance
)


class _SpawnTool:
    name = "spawn_worker"
    risk_tier = "safe"
    description = "Hand heavy work to a worker"
    schema: dict[str, Any] = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(self, args, ctx: ExecutionContext) -> ToolResult:
        self.calls.append(args)
        return ToolResult(success=True, output="worker finished")


class _Executor:
    def __init__(self, tool: _SpawnTool) -> None:
        self.tool = tool

    async def execute(self, tool, args, *a: Any, **k: Any) -> ToolResult:
        return await self.tool.execute(args, None)  # type: ignore[arg-type]


def _manager(primary: str = "chatgpt-web") -> tuple[BrainManager, _SpawnTool]:
    config = JarvisConfig()
    config.brain.primary = primary
    config.brain.worker = BrainTierConfig(provider="openai-codex", model="")
    spawn = _SpawnTool()
    mgr = BrainManager(
        config=config,
        bus=EventBus(),
        tools={spawn.name: spawn},
        tool_executor=_Executor(spawn),  # type: ignore[arg-type]
    )
    mgr._active_name = primary  # type: ignore[attr-defined]
    return mgr, spawn


def _chain(mgr: BrainManager, text: str) -> list[tuple[str, str | None]]:
    """Build the turn's chain exactly as ``_generate_untagged`` does."""
    smalltalk = mgr._is_smalltalk(text)
    mgr._turn_substantive = not smalltalk  # type: ignore[attr-defined]
    mgr._turn_needs_tools = (not smalltalk) and mgr._turn_has_action_intent(text)
    mgr._router_lead_key = None  # type: ignore[attr-defined]
    return mgr._build_fallback_chain(classify(text).level)


# --------------------------------------------------------------------------
# No extra model call for a turn that needs none.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", _SMALLTALK)
def test_a_smalltalk_turn_wakes_exactly_one_provider(text: str) -> None:
    mgr, _ = _manager()

    chain = _chain(mgr, text)

    assert [name for name, _ in chain] == ["chatgpt-web"], (
        "a greeting or a clock question must not add a second model call"
    )
    assert mgr._router_lead_key is None


@pytest.mark.parametrize("text", _SMALLTALK)
def test_a_smalltalk_turn_is_not_delegated(text: str) -> None:
    mgr, spawn = _manager()

    assert mgr._should_force_spawn(text) is False
    assert spawn.calls == []


# --------------------------------------------------------------------------
# A substantive turn needs a lead, and the lead is free.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", _SUBSTANTIVE)
def test_the_tool_lead_of_a_seat_turn_is_never_key_billed(text: str) -> None:
    """A seat talker cannot call tools, so a lead is unavoidable — but not paid."""
    mgr, _ = _manager()
    assert seat_bound("chatgpt-web"), "premise: the talker holds a plan seat"

    chain = _chain(mgr, text)

    assert len(chain) >= 2, "a seat talker needs a lead to pick the tool"
    lead = chain[0][0]
    assert lead not in _KEY_BILLED, f"{lead} bills a stored key for a chat turn"
    assert chain[-1][0] == "chatgpt-web", "the user's brain still answers"


@pytest.mark.parametrize("text", _SUBSTANTIVE)
def test_no_entry_of_a_seat_turn_bills_a_key(text: str) -> None:
    mgr, _ = _manager()

    chain = _chain(mgr, text)

    assert {name for name, _ in chain} & _KEY_BILLED == set()


# --------------------------------------------------------------------------
# Delegation: explicit, deterministic, and not on a keyword.
# --------------------------------------------------------------------------

def test_an_explicit_trigger_dispatches_without_a_routing_round() -> None:
    mgr, _ = _manager()

    assert mgr._should_force_spawn(
        "mach bitte einen subagent der die landingpage baut"  # i18n-allow
    ) is True


@pytest.mark.parametrize("text", _SUBSTANTIVE)
def test_an_ordinary_coding_sentence_is_not_force_spawned(text: str) -> None:
    """Measured 2026-10-20: these reach the routing round, not the worker.

    Delegation costs a worktree, a subscription turn and a critic loop. It is
    reached by naming the vehicle or by the effort test, never by the presence
    of a verb like "fix" — otherwise every mention of a bug would start one.
    """
    mgr, spawn = _manager()

    assert mgr._should_force_spawn(text) is False
    assert spawn.calls == []


# --------------------------------------------------------------------------
# Bounded retries.
# --------------------------------------------------------------------------

def test_the_critic_retry_bound_is_a_constant() -> None:
    """No loop may re-probe a capped plan until it answers."""
    from jarvis.missions.critic.runner import MAX_CRITIC_LOOPS

    assert isinstance(MAX_CRITIC_LOOPS, int)
    assert 1 <= MAX_CRITIC_LOOPS <= 5


# --------------------------------------------------------------------------
# The selected provider is tried before any other family.
# --------------------------------------------------------------------------

def test_a_provider_without_a_resolved_model_still_leads_its_own_chain() -> None:
    """Measured 2026-10-20: it did not, and a stored key inherited the turn.

    ``brain.primary`` pointed at a local runtime with no model tag configured
    yet. Both ``_fast_model`` and ``_deep_model`` returned "", every branch of
    step 1 is guarded by them, and the provider therefore never reached its own
    chain — which then began with claude-api. The registry lists a provider
    because its plugin imported, not because a credential exists, so nothing
    further down refused the escalation either.
    """
    mgr, _ = _manager("ollama")
    assert not mgr._fast_model("ollama") and not mgr._deep_model("ollama"), (
        "premise: this provider resolves no model in a default config"
    )

    chain = _chain(mgr, "was ist die hauptstadt von frankreich")  # i18n-allow

    assert chain[0][0] == "ollama"
    billed_before = [n for n, _ in chain[: _index_of(chain, "ollama")] if n in _KEY_BILLED]
    assert billed_before == []


def _index_of(chain: list[tuple[str, str | None]], name: str) -> int:
    for i, (n, _) in enumerate(chain):
        if n == name:
            return i
    raise AssertionError(f"{name} is not on the chain: {[n for n, _ in chain]}")
