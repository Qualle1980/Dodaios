"""Direct-tool orchestration for Gemini and compatible local voice servers."""

from __future__ import annotations

import array
import asyncio
import base64
import json
import logging
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Literal
from uuid import uuid4

from jarvis.audio import echo_reference
from jarvis.brain.identity import name_directive
from jarvis.core.agent_brief import AGENT_BRIEF_RULE
from jarvis.core.paths import user_data_dir
from jarvis.core.runtime_refs import get_supervisor_tool_gateway
from jarvis.core.tool_budget import VOICE_TOOL_BUDGET_S
from jarvis.cu.direct import COMPUTER_CONTROL_RULES
from jarvis.live.coordination import (
    NAMED_TOOLS,
    ON_SCREEN_RULE,
    TEAM_TALK_RULE,
    WORK_STATUS_RULE,
    conversation_rules,
)
from jarvis.live.product import PRODUCT_BRIEF
from jarvis.live.runtime import claim, register, unregister
from jarvis.live.session import LiveVoiceSession, _identity
from jarvis.live.state import LiveLedger, TranscriptFragment
from jarvis.live.tasks import BackgroundTask, report_prompt
from jarvis.live.tools import LiveTools, compact_declaration, take_images
from jarvis.realtime.audio import StreamingPcm16Resampler
from jarvis.realtime.protocol import RealtimeSessionConfig, RealtimeUnavailableError
from jarvis.speech.echo_guard import SelfEchoGuard

log = logging.getLogger(__name__)

# Mic frames quieter than this fraction of the recent played peak while the
# echo reference is armed are held back so Gemini never hears our own
# speakers. Louder frames still travel — that is a real barge-in.
_ECHO_MIC_HOLD_RATIO = 0.5
_ECHO_MIC_HOLD_FLOOR = 0.02
_ECHO_REFERENCE_ACTIVE_RMS = 0.01
_ECHO_REFERENCE_WINDOW_S = 2.0


def _pcm16_rms(pcm: bytes) -> float:
    """RMS of little-endian int16 PCM, normalized to 0..1."""
    if len(pcm) < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0
    mean_sq = sum(int(sample) * int(sample) for sample in samples) / len(samples)
    return (mean_sq ** 0.5) / 32768.0


def _record_played_pcm(pcm: bytes, sample_rate: int) -> None:
    """Append one outbound PCM block to the process-local echo envelope.

    Live plays through the browser, not ``AudioPlayer``, so the player tap
    never sees these bytes. Recording them here is what makes
    ``echo_reference`` reachable on the native live path (BUG-101 shape).
    """
    if sample_rate <= 0 or len(pcm) < 2:
        return
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return
    mean_sq = sum(int(sample) * int(sample) for sample in samples) / len(samples)
    rms = (mean_sq ** 0.5) / 32768.0
    echo_reference.record(rms, len(samples) / float(sample_rate))


def _echo_reference_recently_active(
    *, window_s: float = _ECHO_REFERENCE_WINDOW_S
) -> bool:
    """True when the speakers recently emitted audible energy."""
    try:
        reference = echo_reference.snapshot(window_s)
    except Exception:  # noqa: BLE001 — echo gate must never break the call
        log.debug("live.native echo-reference snapshot failed", exc_info=True)
        return False
    return any(rms >= _ECHO_REFERENCE_ACTIVE_RMS for _ts, _dur, rms in reference)

# How long a provider's readiness probe may take before the call is refused
# (release SLO: an honest refusal within one second, spoken and visible).
_DUPLEX_PROBE_BUDGET_S = 1.0
# Spoken only when the provider gives no reason of its own (or its probe hangs).
_DUPLEX_REFUSAL_FALLBACK = "The selected voice engine cannot take a call right now."
# Ceiling for ONE native tool call before the live model is released with an
# honest "still running" result; the tool itself keeps running. Ported from
# ``_NATIVE_TOOL_DEADLINE_S`` in ``jarvis/realtime/session.py``. A module
# attribute so tests can pin it low.
_TOOL_DEADLINE_S = VOICE_TOOL_BUDGET_S
# Session built-ins of ``LiveTools``: always declared, whatever the budget.
_SESSION_TOOLS = frozenset({
    "end_call", "discover_tools", "call_tool", "confirm_action", "task_status",
    "computer", "take_appshot",
})
# Under a declaration budget (a provider's ``tool_declaration_budget_tokens``
# or ``[voice].realtime_tool_declaration_budget_tokens``, the smaller wins), at
# most this many catalog tools are declared directly; every other tool stays
# reachable through discover_tools/call_tool. A small local model pays a full
# LLM round per discovery step, so the frequent tools come first. Every tool the
# conversation rules call by name comes before them: a small model that does
# not see a named tool declares it missing instead of discovering it.
_DIRECT_TOOL_LIMIT = 12
_DIRECT_TOOL_PREFERENCE = (
    *(name for name in NAMED_TOOLS if name not in _SESSION_TOOLS),
    # The personal memory vault, read side (ADR-0044). First of the catalog
    # tools because the budget cuts from the tail: with 12 direct slots and a
    # byte budget that the mid-weight steering tools already fill (measured
    # 15833 of 24000 B ahead of them), an unranked memory tool is declared on
    # no live call at all, and "what do you know about me" is among the most
    # frequent spoken turns. Their schema field was the 2026-11-09 loop
    # (fixed in #142); being absent HERE is why a correct schema still never
    # reached a live model.
    "memory-search",
    "memory-history",
    "memory-topics",
    "find-app-action",
    "run-app-action",
    "search_web",
    "open_app",
    "take_appshot",
    "screen_snapshot",
    "google_calendar",
    "gmail",
    "youtube_music",
    "spotify",
    "home_assistant",
    "wiki-recall",
    "product_help",
    "runtime_location",
)


# How the browser must play this path's audio. The native path never
# negotiates WebRTC (``requires_webrtc_answer`` is False above), so its PCM
# travels as untimed binary frames and the playback worklet is the only sink.
# Without this announcement the client leaves ``timedOutput`` false and plays
# every arriving frame AT ONCE, over whatever is already scheduled — audible
# as words running over each other, worst on a long answer (reported
# 2026-11-13). #115 built the ordered timeline for exactly these untimed
# frames (``handleUntimedAudio``: a frame starts where the queued audio ends),
# but it only joins that timeline inside a timed session. ``source_timed_audio``
# is NOT the switch to flip for it: that flag claims the provider hands us
# intervals, which Gemini Live does not, and it would also start emitting
# speech-timing frames in ``jarvis/live/session.py``.
_OUTPUT_TRANSPORT = "timed_pcm"


def _declared_name(declaration: dict) -> str:
    """The canonical tool name behind a declaration (catalog tools are aliased)."""
    name = str(declaration.get("name", ""))
    if name in _SESSION_TOOLS:
        return name
    return str(declaration.get("description", "")).split(": ", 1)[0]


def _fit_declarations(
    declarations: list[dict],
    budget_chars: int,
    *,
    fillers: frozenset[str] | None = None,
) -> tuple[dict, ...]:
    """Session built-ins plus a curated direct set that fits ``budget_chars``.

    Slots the preferred tools leave free go to other catalog tools; with
    ``fillers`` only to those, so a tool with an effect of its own (restarting
    the app) is never one of a small model's few direct tools by catalog order
    alone. It stays reachable through discover_tools.
    """
    session = [d for d in declarations if d.get("name") in _SESSION_TOOLS]
    rank = {name: index for index, name in enumerate(_DIRECT_TOOL_PREFERENCE)}
    catalog = sorted(
        (
            d
            for d in declarations
            if d.get("name") not in _SESSION_TOOLS
            and (fillers is None or _declared_name(d) in rank or _declared_name(d) in fillers)
        ),
        key=lambda d: rank.get(_declared_name(d), len(rank)),
    )
    used = sum(len(json.dumps(d)) for d in session)
    direct: list[dict] = []
    for declaration in catalog:
        if len(direct) >= _DIRECT_TOOL_LIMIT:
            break
        if _declared_name(declaration) in NAMED_TOOLS:
            # A tool the rules call by name is always declared, in short:
            # workspace-orchestrate alone outgrew a small model's whole budget.
            declaration = compact_declaration(declaration)
        size = len(json.dumps(declaration))
        if used + size > budget_chars:
            continue
        used += size
        direct.append(declaration)
    return (*session, *direct)


def _log_late_probe(task: asyncio.Future) -> None:
    """Retrieve a readiness probe that outlived its refusal, so nothing is lost."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.warning("Native voice readiness probe failed after the refusal: %s", exc)
    else:
        log.info("Native voice readiness probe answered late: %s", task.result())



def split_new_input_tokens(
    usage: dict[str, int], previous_total: int
) -> tuple[int, int, int]:
    """What this Live turn ADDED to the context, split into text and audio.

    A Live session holds its context on the server: the client never re-sends
    the system prompt or the tool declarations. ``prompt_token_count`` still
    reports everything the model attended to, so it grows with every turn, and
    adding those readings together bills the same context over and over.
    Measured 2026-11-07 against Google's own quota counter: 138 339 input
    tokens out of 12 turns where Google recorded 12 370 for the whole day.

    Growth, not the reading, is what the turn added. The user's speech is
    always new, so it is taken out of the growth first and whatever remains is
    new text. A total that SHRANK means the context was compacted and rebuilt,
    and the model read all of it again, so that reading counts whole.

    Returns ``(new_total, text_in, audio_in)``.
    """
    total = max(0, int(usage.get("input_total", 0) or 0))
    audio = max(0, int(usage.get("input_audio", 0) or 0))
    new_total = total if total < previous_total else total - previous_total
    audio_in = min(audio, new_total)
    return new_total, new_total - audio_in, audio_in

class NativeLiveVoiceSession(LiveVoiceSession):
    """Reuse native turn mechanics, without the legacy heuristic delegate loop."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._native_turn_complete = True
        # Same text backstop the realtime session uses (BUG-089): register
        # every assistant transcript and drop provider-transcribed self-echo
        # before it becomes a user turn. The acoustic envelope
        # (``echo_reference``) arms the short-utterance judgment.
        self._echo_guard = SelfEchoGuard()

    async def _start(self, message: dict) -> None:
        self._adopt_desktop_session()
        gateway = get_supervisor_tool_gateway()
        if gateway is None:
            raise RuntimeError("Dodaios tools are still starting.")
        root = user_data_dir()
        await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
        self._ledger = await asyncio.to_thread(LiveLedger, root / "live.sqlite3")
        self._tools = LiveTools(
            gateway, self._ledger, self.session_id, language=self._language, backend_model=""
        )
        self._tools.ask_hangup = self._ask_voice_hangup
        try:
            claim(self.session_id)
            self._initial_seed = self._take_initial_context()
            settings = getattr(self._config.brain, "providers", {}).get(self.active_provider)
            # The active provider's own model only. ``[brain.realtime].model``
            # belongs to whichever card last wrote it (the OpenAI Live card)
            # and survives a provider switch, so Gemini or the local engine
            # was handed another provider's model id. Empty = adapter default.
            model = getattr(settings, "model", "") or ""
            self._active_model = model
            language_rule = (
                "Use the user's language and follow explicit language changes. "
                if getattr(self._config.brain, "reply_language", "auto") == "auto"
                else f"Speak {self._language}. "
            )
            cfg = RealtimeSessionConfig(
                model=model,
                language=self._language,
                voice=getattr(settings, "voice", "") or "",
                instructions=(
                    (_identity(self._config) or name_directive(""))
                    + "\n\n"
                    + PRODUCT_BRIEF
                    + " "
                    + language_rule
                    + conversation_rules("your tools")
                    + "Use your tools directly "
                    "for actions, private information and current facts. Use discover_tools and "
                    "call_tool for any tool not declared directly. "
                    + (
                        # Only when Computer Use is on: the tool is then in the catalog.
                        COMPUTER_CONTROL_RULES
                        if any(d.name == "computer" for d in self._tools.catalog())
                        else ""
                    )
                    + " When the user asks for an appshot, call take_appshot. "
                    "Request confirmation for pending approvals. Use confirm_action only after "
                    "explicit approval. A started job is not complete. Never invent tool results. "
                    "Several orders can run at once: a function answered as pending with a "
                    "task_id keeps running in the background and its result is handed to you "
                    "later. Say that it is running, take the next request, never call it again; "
                    "use task_status when the user asks how far a task is or wants one stopped. "
                    "Use workspace-orchestrate for coding tasks: inspect and resolve project, "
                    "workspace and agent references, then send to the returned stable IDs. "
                    "For a NEW coding agent call workspace-orchestrate create (cli, count, "
                    "prompt) in the named or visible workspace; never spawn_worker. "
                    "Explicit references override the visible workspace; ask on ambiguity. "
                    "Do not switch the UI to address another workspace. Reuse request_id on "
                    "retries and never replay uncertain delivery. "
                    + WORK_STATUS_RULE
                    + ON_SCREEN_RULE
                    + TEAM_TALK_RULE
                    + AGENT_BRIEF_RULE
                ),
                history=tuple(
                    {"role": item["role"], "text": item["delta"]} for item in self._initial_seed
                ),
                tools=self._tool_declarations(),
            )
            self._native_config = cfg
            await self._refuse_unless_duplex_ready()
            from jarvis.live.recovery import connection_permit

            await connection_permit()
            self._connection = await self._provider.open_session(cfg)
            self._active_model = getattr(self._connection, "model", "") or model
            from jarvis.core.events import (
                RealtimeSessionReady,
                VoiceSessionStarted,
                VoiceTurnStarted,
            )

            if self._bus is not None:
                if not self._parent_owned:
                    await self._bus.publish(
                        VoiceSessionStarted(
                            session_id=self.session_id,
                            language=self._language,
                        )
                    )
                await self._bus.publish(
                    VoiceTurnStarted(
                        session_id=self.session_id,
                        turn_id=self._archive_turn_id,
                    )
                )
                await self._bus.publish(
                    RealtimeSessionReady(
                        session_id=self.session_id,
                        provider=self.active_provider,
                        model=self._active_model,
                        language=self._language,
                        surface=self._surface,
                    )
                )
            rate = int(self._provider.input_sample_rate)
            self._resampler = StreamingPcm16Resampler(int(message.get("sample_rate", 48000)), rate)
            register(self)
            self._pump_task = asyncio.create_task(self._pump(), name="native-live-events")
            await self._take_startup_input(message)
            self._watch_input_mute()
            await self._send_json(
                {
                    "type": "audio_ready",
                    "sound_effects": bool(
                        getattr(getattr(self._config, "ui", None), "sound_effects", True)
                    ),
                    "provider": self.active_provider,
                    "model": model,
                    "input_sample_rate": rate,
                    "output_sample_rate": self._provider.output_sample_rate,
                    "language": self._language,
                    "requires_webrtc_answer": False,
                    "output_transport": _OUTPUT_TRANSPORT,
                    "input_muted": self._input_muted,
                }
            )
            # Audio flows over this socket: results that finished before the
            # call may be offered at the first pause, which is now.
            self._notify_pause()
        except BaseException:
            await self.end(reason="error")
            raise

    def _declaration_budget_tokens(self) -> int:
        """The smaller of the config bound and the provider's own budget; 0 = none.

        Same rule as ``RealtimeVoiceSession._declaration_budget_chars``: the
        provider's budget is a capability (AP-21), never a name check.
        """
        configured = int(
            getattr(
                getattr(self._config, "voice", None),
                "realtime_tool_declaration_budget_tokens",
                0,
            )
            or 0
        )
        declared = int(getattr(self._provider, "tool_declaration_budget_tokens", 0) or 0)
        budgets = [budget for budget in (configured, declared) if budget > 0]
        return min(budgets) if budgets else 0

    def _tool_declarations(self) -> tuple[dict, ...]:
        assert self._tools is not None
        declarations = [
            {key: value for key, value in declaration.items() if key != "type"}
            for declaration in self._tools.declarations()
        ]
        budget = self._declaration_budget_tokens()
        if budget <= 0:
            # AP-30 again, one level up: the early return skipped the only
            # line the live path says about its tool surface, and an unbounded
            # budget is the DEFAULT. So on a real install the call declared
            # its tools in silence -- 2026-11-07 was spent inferring what a
            # live call had offered while the text brain plainly logged
            # "Tool surface for gemini: 23 of 89 tools shown". An unbounded
            # surface is a fact worth the same line.
            self._report_tool_surface(len(declarations), len(declarations), 0)
            return tuple(declarations)
        safe = frozenset(
            descriptor.name
            for descriptor in self._tools.catalog()
            if str(getattr(descriptor, "risk_tier", "")) == "safe"
        )
        fitted = _fit_declarations(declarations, budget * 4, fillers=safe)
        self._report_tool_surface(len(fitted), len(declarations), budget)
        return fitted

    def _report_tool_surface(self, shown: int, total: int, budget: int) -> None:
        """Say what the call offered — once per distinct surface, not per call.

        AP-30: a trimmed tool set looks exactly like a complete one, and so
        does an UNBOUNDED one. The budget-0 path used to return before the log
        entirely, and 0 is the default, so on a real install the live call
        declared its tools in silence: 2026-11-07 was spent inferring what a
        live call had offered while the text brain plainly logged
        "Tool surface for gemini: 23 of 89 tools shown".

        Reporting only on change is not just tidiness. _tool_declarations runs
        on every rebuild, and logging there on every call perturbed the timing
        of an async subscription test enough to make it fail about one run in
        three (5/5 green without it). Work in a hot path is never free.
        """
        state = (shown, total, budget, self.active_provider)
        if getattr(self, "_tool_surface_reported", None) == state:
            return
        self._tool_surface_reported = state
        if budget > 0:
            log.info(
                "Native voice declares %d of %d tools under a %d-token budget "
                "for %s; the rest stay reachable through discover_tools",
                shown, total, budget, self.active_provider,
            )
            return
        # "nothing is behind discover_tools" was FALSE, and it cost two days.
        # Both numbers here come from the SAME list -- what ``declarations()``
        # already returned -- so with no budget of our own they are always
        # equal and say nothing about the catalog. Underneath, the catalog's
        # own byte budget had silently dropped 24 of 47 tools, and the line
        # read as proof that it had not. The catalog reports its own trim now
        # (``LiveTools._report_catalog_surface``); this line no longer claims
        # anything about it.
        log.info(
            "Native voice declares all %d tools it was given, with no "
            "declaration budget of its own for %s",
            shown, self.active_provider,
        )

    async def _refuse_unless_duplex_ready(self) -> None:
        """Refuse within a second, in the provider's own words, when it cannot talk.

        A provider without the probe is assumed ready. A probe that outlives
        the budget keeps running in the background: it may be starting an
        engine, which must not be torn in half.
        """
        probe = getattr(self._provider, "can_open_duplex_session", None)
        if not callable(probe):
            return
        check = asyncio.ensure_future(probe())
        try:
            ready = bool(await asyncio.wait_for(asyncio.shield(check), _DUPLEX_PROBE_BUDGET_S))
        except TimeoutError:  # a late probe is logged by its callback
            check.add_done_callback(_log_late_probe)
            ready = False
        except Exception:
            log.warning("Native voice readiness probe failed", exc_info=True)
            ready = False
        if ready:
            return
        reason = (
            str(getattr(self._provider, "duplex_unavailable_reason", "") or "").strip()
            or _DUPLEX_REFUSAL_FALLBACK
        )
        self._detail = reason
        log.warning("Native voice refused to start (%s): %s", self.active_provider, reason)
        try:
            await self._send_json(
                {
                    "type": "error_spoken",
                    "text": reason,
                    "detail": reason,
                    "language": self._language,
                    "spoken_kind": "reply",
                    "provider": self.active_provider,
                }
            )
        except Exception:  # noqa: BLE001 — the refusal still propagates
            log.warning("Native voice refusal notice could not be sent", exc_info=True)
        raise RealtimeUnavailableError(reason)

    async def _interrupt_reply(self) -> None:
        """Barge-in: stop the provider's reply and flush what the browser holds."""
        interrupt = getattr(self._connection, "interrupt", None)
        if callable(interrupt):
            try:
                await interrupt()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning(
                    "Native voice interrupt failed; flushing playback anyway", exc_info=True
                )
        self.playback_active = False
        await self._emit_indicator({"type": "audio_clear"})

    def _echo_should_hold_mic(self, pcm: bytes) -> bool:
        """Hold quiet mic frames while our own speakers are in the envelope.

        Adapts the desktop barge-in energy floor (``desktop.py``): without
        local candidate-frame correlation on this path, low mic energy during
        recent played output is treated as speaker bleed and never reaches
        Gemini. A louder frame fails open — that is the barge-in path.
        """
        try:
            reference = echo_reference.snapshot(_ECHO_REFERENCE_WINDOW_S)
        except Exception:  # noqa: BLE001 — never block the mic on a snapshot error
            log.debug("live.native echo-reference mic hold failed", exc_info=True)
            return False
        peaks = [rms for _ts, _dur, rms in reference if rms >= _ECHO_REFERENCE_ACTIVE_RMS]
        if not peaks:
            return False
        mic_rms = _pcm16_rms(pcm)
        threshold = max(_ECHO_MIC_HOLD_FLOOR, max(peaks) * _ECHO_MIC_HOLD_RATIO)
        return mic_rms < threshold

    async def handle_audio_frame(self, pcm: bytes) -> None:
        if (
            self._connection is not None
            and not self._closing
            and not self._recovering
            and not self._input_muted
        ):
            if self._echo_should_hold_mic(pcm):
                return
            audio = self._resampler.process(pcm)
            if audio:
                try:
                    await self._connection.send_audio(
                        SimpleNamespace(
                            pcm=audio,
                            sample_rate=self._provider.input_sample_rate,
                        )
                    )
                except Exception:
                    log.debug("Native audio send failed; closing transport", exc_info=True)
                    await self._connection.close()

    def _is_provider_self_echo(self, text: str) -> bool:
        """True when a final input transcript is our own recent speech.

        ``judge_short`` is armed only while the echo envelope (or the session's
        speaking/playback flags) says we were just audible — that is the
        truncated-echo window from BUG-101. Novel user words fail open.
        """
        probe = str(text or "").strip()
        if not probe:
            return False
        judge_short = self._output_was_running() or _echo_reference_recently_active()
        try:
            return bool(self._echo_guard.is_echo(probe, judge_short=judge_short))
        except Exception:  # noqa: BLE001 — echo guard must never break barge-in
            log.debug("live.native self-echo judgment failed", exc_info=True)
            return False

    async def _drop_self_echo_transcript(self, text: str) -> None:
        """Swallow a provider-transcribed echo and stop any auto-answer to it."""
        log.info(
            "live.native[%s] dropped provider-transcribed self-echo "
            "before it became a turn: %r",
            self.session_id,
            text[:80],
        )
        # Gemini Live creates responses automatically: the server may already
        # be answering its own echo. Interrupt that generation; a genuine
        # later user turn opens a fresh one.
        await self._interrupt_reply()

    async def _open_replacement(self, history: list[dict]) -> None:
        native_history = tuple(
            {"role": item["role"], "text": item["content"][0]["text"]} for item in history
        )
        cfg = replace(
            self._native_config,
            history=native_history,
            instructions=self._native_config.instructions
            + " Connection restored. Wait for the user before starting another task.",
        )
        self._connection = await self._provider.open_session(cfg)
        await self._send_json(
            {
                "type": "audio_ready",
                "provider": self.active_provider,
                "model": self._active_model,
                "language": self._language,
                "input_sample_rate": self._provider.input_sample_rate,
                "output_sample_rate": self._provider.output_sample_rate,
                "requires_webrtc_answer": False,
                "output_transport": _OUTPUT_TRANSPORT,
                "reconnected": True,
            }
        )

    async def _apply_planner_grounding(self, text: str) -> None:
        """When the planner requires product_help, run it and inject the result.

        Gemini Live cannot force a tool call (``required_tool`` is dropped on
        ``request_response``), and the measured B2 failure was exactly
        ``function_calls=0`` with a textbook answer. The app therefore owns the
        lookup once per user turn and hands the trusted guide to the model as
        context before it answers. Weather and empty plans leave the wire alone.
        """
        if self._tools is None or self._connection is None or self._closing:
            return
        try:
            result = await self._tools.ground_self_knowledge_once(text)
        except Exception:  # noqa: BLE001 — never end the call for a planner miss
            log.warning(
                "Native voice planner grounding failed; answering without it",
                exc_info=True,
            )
            return
        if result is None:
            return
        directive = (
            "This turn needs the built-in product guide. A trusted product_help "
            "result follows as application context; answer from it and do not "
            "call product_help again for this turn."
        )
        update = getattr(self._connection, "update_session", None)
        if callable(update):
            try:
                await update(turn_directive=directive)
            except TypeError:
                # Adapter predates turn_directive — context injection below still helps.
                pass
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning(
                    "Native voice turn_directive for product_help failed; "
                    "injecting guide context anyway",
                    exc_info=True,
                )
        prompt = self._tools.format_self_knowledge_context(result)
        try:
            await self._connection.send_text(prompt)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning(
                "Native voice could not inject product_help context",
                exc_info=True,
            )
            return
        log.info(
            "Native voice injected planner product_help grounding (%d chars) "
            "before the model answer",
            len(prompt),
        )

    async def _ground_then_maybe_respond(self, language: str, text: str) -> None:
        """Ground self-knowledge first; request a response only when the provider waits."""
        await self._apply_planner_grounding(text)
        if self._closing or self._connection is None:
            return
        if getattr(self._connection, "creates_responses_automatically", True):
            return
        await self._request_native_response(language)

    async def _request_native_response(self, language: str = "") -> None:
        # The turn's language is decided once (``turn_language``); a provider
        # that can take it per session hears it before it answers.
        update = getattr(self._connection, "update_session", None)
        if language and callable(update):
            try:
                await update(language=language)
            except asyncio.CancelledError:
                raise
            except Exception:
                if not self._closing:
                    log.warning(
                        "Native voice language update failed; answering anyway", exc_info=True
                    )
        try:
            await self._connection.request_response()
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self._closing:
                log.exception("Native voice response request failed")

    async def _pump(self) -> None:
        try:
            while not self._closing:
                events = self._connection.receive().__aiter__()
                while not self._closing:
                    try:
                        event = await anext(events)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # Shutdown is quiet; active failures enter the recovery loop.
                        if self._closing:
                            return
                        if await self._wait_for_connection():
                            break
                        return
                    await self._native_event(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Native live connection terminated")
            self._failed = True
            await self._send_json(
                {
                    "type": "provider_error",
                    "error": "Voice connection lost; actions were not replayed.",
                    # Stable code the desktop UI translates; ``error`` stays.
                    "message_code": "voice_connection_lost",
                }
            )
        finally:
            unregister(self.session_id)
            self._notify_ended()
            self._closed.set()

    def _output_was_running(self) -> bool:
        """Whether the assistant was audibly producing output right now."""
        return bool(self._speaking or self.playback_active)

    async def _native_event(self, event: Any) -> None:
        assert self._tools is not None and self._ledger is not None
        if event.type in {
            "audio_delta", "input_transcript", "output_transcript_delta", "tool_call",
        }:
            self._native_turn_complete = False
        if event.type == "audio_delta" and event.audio is not None:
            self._report_started()
            await self._note_speaking()
            pcm = event.audio.pcm
            rate = int(
                getattr(event.audio, "sample_rate", 0)
                or getattr(self._provider, "output_sample_rate", 0)
                or 24_000
            )
            _record_played_pcm(pcm, rate)
            # Date the text echo guard forward by this chunk's duration so
            # short trailing echoes after the last transcript still match.
            duration_s = (len(pcm) / 2) / float(rate) if rate > 0 else 0.0
            if duration_s > 0.0:
                self._echo_guard.touch(
                    time.time_ns() + int(duration_s * 1e9)
                )
            await self._send_binary(pcm)
        elif event.type in {"input_transcript", "output_transcript_delta"}:
            role: Literal["user", "assistant"] = (
                "user" if event.type == "input_transcript" else "assistant"
            )
            if role == "user":
                # Diagnose speaker-echo turns from the log alone: previously
                # these only appeared as chat "user" lines (e.g. garbled
                # fragments of our own reply) with no matching log entry.
                text = event.text or ""
                log.info(
                    "live.native[%s] input_transcript chars=%d final=%s "
                    "output_running=%s speaking=%s playback_active=%s "
                    "echo_ref_active=%s text=%r",
                    self.session_id,
                    len(text),
                    bool(event.is_final),
                    self._output_was_running(),
                    bool(self._speaking),
                    bool(self.playback_active),
                    _echo_reference_recently_active(),
                    text[:120],
                )
                if event.is_final and self._is_provider_self_echo(text):
                    await self._drop_self_echo_transcript(text)
                    return
                self._tools.user_text = text
                if event.is_final:
                    self._tools.revision += 1
                    if self._bus is not None:
                        from jarvis.core.events import BrainTurnStarted

                        await self._bus.publish(BrainTurnStarted(
                            source_layer="live.native", trace_id=self._indicator_trace_id,
                            provider=self.active_provider, model=self._active_model,
                        ))
                    if not self._closing and not self._recovering:
                        self._resume_needs_input = False
                        self._tools.accept_new_input()
                        self._reconnect_attempts = 0
                    from jarvis.core.turn_language import resolve_output_language

                    self._language = resolve_output_language(
                        getattr(self._config.brain, "reply_language", "auto"),
                        "auto",
                        event.text or "",
                        conversation_language=self._language,
                    )
                    self._tools.language = self._language
                    # Plan the final transcript before the model answers: when
                    # product_help is required, the app runs it once and injects
                    # the guide (Gemini Live cannot be forced to call a tool).
                    task = asyncio.create_task(
                        self._ground_then_maybe_respond(
                            self._language, event.text or ""
                        )
                    )
                    self._control_tasks.add(task)
                    task.add_done_callback(self._control_tasks.discard)
            if role == "assistant":
                self._report_started()
                assistant_text = event.text or ""
                if assistant_text.strip():
                    self._echo_guard.register(assistant_text, slot="assistant")
            if role == "assistant" or event.is_final:
                stamp = time.monotonic_ns() // 1_000_000
                await asyncio.to_thread(
                    self._ledger.append,
                    TranscriptFragment(
                        self.session_id,
                        str(uuid4()),
                        role,
                        event.text or "",
                        stamp,
                        stamp,
                    ),
                )
            await self._send_json(
                {
                    "type": "transcript",
                    "role": role,
                    "text": event.text or "",
                    "is_final": event.is_final,
                }
            )
            stamp = time.monotonic_ns() // 1_000_000
            caption = self._transcript.feed(
                session_id=self.session_id, trace_id=self._indicator_trace_id,
                event_id=str(uuid4()), role=role, text=event.text or "",
                start_ms=stamp, end_ms=stamp, snapshot=role == "user",
            )
            # What the call said, where the rest of the session looks for it.
            # ``LiveVoiceSession._event`` has always kept this; the native
            # handler built the same caption and dropped it, so every reader
            # of ``_captions`` saw an empty string on the path the app
            # actually runs -- the honesty check among them.
            self._captions[role] = caption.text
            if self._bus is not None:
                await self._bus.publish(caption)
            if role == "user" and event.is_final:
                self._transcript.finish("user")
        elif event.type == "tool_call":
            # The model is answering the report, starting with a tool.
            self._report_started()
            await self._note_thinking()
            task = asyncio.create_task(self._call(event, self._tools.revision))
            self._track_job(task)
        elif event.type in {"interrupted", "speech_started"}:
            # A barge-in into a report: the user heard its start and chose to
            # talk. The report is in the model's context for follow-ups.
            superseded = bool(getattr(event, "superseded", False))
            self_initiated = bool(getattr(event, "self_initiated", False))
            output_running = self._output_was_running()
            barge_in = event.type == "speech_started" and output_running
            # Name the trigger so a chopped reply can be attributed from the
            # log: provider barge-in, our own cancel, or a superseded regen.
            if event.type == "interrupted":
                if self_initiated:
                    cause = "self_initiated_cancel"
                elif superseded:
                    cause = "provider_superseded_regen"
                else:
                    cause = "provider_interrupted"
            else:
                cause = "speech_started_barge_in" if barge_in else "speech_started_idle"
            log.info(
                "live.native[%s] abort cause=%s event=%s superseded=%s "
                "self_initiated=%s output_running=%s speaking=%s "
                "playback_active=%s",
                self.session_id,
                cause,
                event.type,
                superseded,
                self_initiated,
                output_running,
                bool(self._speaking),
                bool(self.playback_active),
            )
            self._report_finished(delivered=True)
            self._speaking = False
            self._thinking = False
            if barge_in:
                await self._interrupt_reply()
            await self._emit_indicator({"type": "tts_cancel"})
        elif event.type == "turn_complete":
            self._native_turn_complete = True
            self._report_finished(delivered=True)
            self._transcript.finish("assistant")
            await self._note_turn_end()
            self._notify_pause()
        elif event.type == "usage":
            usage = event.usage or {}
            await asyncio.to_thread(
                self._ledger.backend_usage,
                self.session_id,
                str(uuid4()),
                self._active_model,
                usage,
            )
            if self._bus is not None:
                from jarvis.brain.cost import calculate_realtime_cost_usd
                from jarvis.core.events import BrainTurnCompleted

                new_in, text_in, audio_in = split_new_input_tokens(
                    usage, self._live_input_high_water
                )
                self._live_input_high_water = max(
                    0, int(usage.get("input_total", 0) or 0)
                )
                tokens_out = int(usage.get("output_total", 0) or 0)
                turn_cost = calculate_realtime_cost_usd(
                    self._active_model,
                    text_in,
                    usage.get("output_text", 0),
                    audio_in,
                    usage.get("output_audio", 0),
                )
                self._live_spend["tokens_in"] += new_in
                self._live_spend["tokens_out"] += tokens_out
                self._live_spend["cost_usd"] += turn_cost
                await self._bus.publish(
                    BrainTurnCompleted(
                        provider=self.active_provider,
                        model=self._active_model,
                        tokens_in=new_in,
                        tokens_out=tokens_out,
                        tokens_cached=usage.get("input_cached", 0),
                        tokens_in_audio=audio_in,
                        tokens_out_audio=int(usage.get("output_audio", 0) or 0),
                        cost_usd=turn_cost,
                        finish_reason="realtime_usage",
                    )
                )
            await self._send_json({"type": "live_backend_usage", "usage": event.usage})
        elif event.type == "error":
            if not getattr(event, "recoverable", False):
                raise RuntimeError("Native voice connection failed")
            log.warning("Native voice rejected a recoverable operation")

    async def _call(self, event: Any, revision: int) -> None:
        assert self._tools is not None
        try:
            # The live model is blocked on this call. It is released with a
            # pending receipt once the call outlasts the voice budget, or once
            # the user moves on while an action still runs; the tool itself
            # finishes as a background task and keeps its receipt.
            result = await self._tools.execute_releasable(
                f"{self._wire_epoch}:{event.call_id or uuid4()}",
                event.tool_name,
                event.tool_args or {},
                revision,
                deadline_s=_TOOL_DEADLINE_S,
            )
            if self._closing:
                return
            result = await self._send_tool_images(result)
            from jarvis.live.tools import annotate_failed_tool_result

            result = annotate_failed_tool_result(result)
            if result.get("success") is False or result.get("speak_failure"):
                self._turn_failed_tools.add(str(event.tool_name or ""))
            self._native_turn_complete = False
            await self._connection.send_tool_result(event.call_id, event.tool_name, result)
            if self._tools.end_requested:
                asyncio.create_task(self.end(reason="voice_pattern"), name="native-live-hangup")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Native tool call failed; receipt retained")

    async def _send_tool_images(self, result: dict) -> dict:
        """Keep the execution receipt even when its visual evidence cannot be delivered."""
        images = take_images(result)
        if not images:
            return result
        send_image = getattr(self._connection, "send_image", None)
        try:
            if not callable(send_image):
                raise RuntimeError("This voice server cannot receive screen images.")
            for image in images:
                await send_image(base64.b64decode(image["data"]), image["mime"])
        except Exception:
            log.warning("Native tool screenshot delivery failed", exc_info=True)
            return {
                **result,
                "success": False,
                "verified": False,
                "error": (
                    str(result.get("error") or "") + " The tool's screenshot could not be "
                    "delivered. Its execution receipt is preserved below; an action may "
                    "already have run. Do not claim visual success or repeat that action "
                    "without checking the current screen."
                ).strip(),
            }
        return result

    def _notify_pause(self) -> None:
        if self._native_turn_complete and not self._has_pending_work():
            self._thinking = False
        super()._notify_pause()

    async def _deliver_task_report(self, task: BackgroundTask) -> bool:
        # The original function call already received its pending answer. The
        # final receipt is a new report at a pause, never a second function
        # response and never a repeated action; its screenshot goes first.
        if not self._floor_open:
            return False
        result = await self._send_tool_images(dict(task.result or {}))
        if not self._task_report_current(task):
            return True  # Overtaken while the image went out; the ledger keeps it.
        if not self._floor_open:
            return False
        return await self._submit_report(
            "[Application event, not the user speaking]\n"
            + report_prompt(task, result, language=self._language),
            {"task_report": True},
        )

    async def _submit_report(self, prompt: str, kwargs: dict[str, Any]) -> bool:
        del kwargs  # a native model always holds its functions
        self._report_sent()
        try:
            await self._connection.send_text(prompt)
        except BaseException:
            self._cancel_report_timeout()
            self._report_state = ""
            raise
        return True

    async def handle_control(self, message: dict) -> None:
        if self._closing:
            return
        if message.get("type") == "text_input" and self._connection is not None:
            self._native_turn_complete = False
            text = str(message.get("text", ""))
            if self._tools is not None:
                self._tools.user_text = text
                self._tools.revision += 1
                if not self._recovering:
                    self._resume_needs_input = False
                    self._tools.accept_new_input()
                    self._reconnect_attempts = 0
                await self._apply_planner_grounding(text)
            await self._connection.send_text(text)
        else:
            await super().handle_control(message)

    async def deliver_announcement(
        self, text: str, *, report: str | None = None, **kwargs: Any
    ) -> bool:
        if not self.is_active:
            return False
        if str(report or "").strip():
            # The model reasons over the agent's full report before speaking
            # (``report_prompt``); refused mid-turn so the caller retries at
            # the next pause instead of talking over anyone.
            if not self._report_floor(kwargs):
                return False
            from jarvis.realtime.report_prompt import report_update_prompt

            text = report_update_prompt(
                text,
                str(report),
                language=str(kwargs.get("language") or self._language),
                kind=str(kwargs.get("spoken_kind") or "completion"),
            )
            return await self._submit_report(text, kwargs)
        await self._connection.send_text(text)
        return True

    async def attach_appshot(self, image: bytes, mime: str, note: str) -> bool:
        """Hand the appshot to the native model as a video frame, silently.

        A text turn would make the model answer right away; the frame alone is
        what it looks at when the user asks. Servers without image input
        decline, and the caller parks the appshot for the next message.
        """
        send_image = getattr(self._connection, "send_image", None)
        if not self.is_active or not callable(send_image):
            return False
        await send_image(image, mime)
        # Native transports have no silent text-input contract. The workspace
        # tool asks the model to select this scoped ID before it can hand off work.
        from jarvis.core.image_references import appshot_context

        appshot_context(self.session_id, image, mime, self._config)
        return True

    async def end(self, *, reason: str = "client_stop") -> None:
        if self._ended:
            return
        self._ended = True
        self._stop_watching_input_mute()
        self._clear_media_levels()
        self._closing = True
        self._task_reports.clear()
        await self._publish_phase("idle")
        self._hangup_reason = reason
        unregister(self.session_id)
        self._notify_ended()
        if self._tools is not None:
            await self._tools.close()
        for task in list(self._control_tasks):
            task.cancel()
        await asyncio.gather(*self._control_tasks, return_exceptions=True)
        if self._connection is not None:
            await self._connection.close()
        if self._pump_task is not None:
            self._pump_task.cancel()
            await asyncio.gather(self._pump_task, return_exceptions=True)
        if self._ledger is not None:
            from jarvis.live.recording import archive_session

            await archive_session(self, reason)
            if self._jobs or self._background_work:
                from jarvis.live.runtime import retain_work

                retain_work((*self._jobs, *self._background_work), self._ledger)
            else:
                await asyncio.to_thread(self._ledger.close)
        self._closed.set()
        await self._send_json({"type": "audio_closed"})
