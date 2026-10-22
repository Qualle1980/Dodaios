"""Per-turn JSONL latency log writer (LATENCY_REPORT_001 deliverable).

Subscribes to ``LatencyTurnComplete`` and appends one self-contained row per
voice turn to ``state/latency_log.jsonl``. The row carries the anchor (turn t0
wall clock + monotonic), every stage offset that was marked, derived metrics
(TTFW, total), and best-effort context fields (token counts, audio length, TTS
char count).

Design constraints (mirrors AP-9 / AP-18 from AGENTS.md):
  * Subscriber callback never blocks: write is enqueued, flushed in a daemon
    thread. The hot path returns instantly.
  * Stdlib only — runs unchanged on the €5/month VPS doctrine.
  * Append-only JSONL so ``jq``, ``tail -f``, and the aggregation CLI can read
    while writes are in flight. No locking visible to readers.
  * If the file path is unwritable the writer logs once and degrades silently
    (telemetry must never break the hot path).
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from jarvis.core.events import BrainTurnProvenance, LatencyTurnComplete

if TYPE_CHECKING:
    from jarvis.core.bus import EventBus

__all__ = ["LatencyLogWriter"]

logger = logging.getLogger(__name__)

# Cap the in-memory queue — under pathological pressure (writer thread
# stuck on a hung disk) we drop rather than balloon RAM. 10 000 rows is
# ~2 MB of JSONL, plenty of headroom for a normal session.
_QUEUE_CAPACITY = 10_000

# Sentinel placed on the queue to ask the flusher thread to exit.
_STOP = object()
# Brain provenance waiting for its turn's latency row; oldest drops first.
_MAX_PENDING_PROVENANCE = 64


class LatencyLogWriter:
    """Bus subscriber that materializes LatencyTurnComplete events as JSONL.

    Lifecycle::

        writer = LatencyLogWriter(Path("state/latency_log.jsonl"))
        writer.attach(bus)
        # ... voice turns happen ...
        writer.close()        # drains queue, joins thread

    The flusher runs in a dedicated daemon thread to honour the "<2 ms per
    stage" budget — the bus callback path stays free of disk I/O.
    """

    def __init__(self, log_path: Path | str) -> None:
        self._log_path = Path(log_path)
        # Parent directory creation is best-effort; if it fails we degrade
        # to "writes are dropped" rather than crashing the hot path.
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning(
                "latency log dir not writable: %s", self._log_path.parent,
            )
        self._queue: queue.Queue[object] = queue.Queue(maxsize=_QUEUE_CAPACITY)
        self._thread: threading.Thread | None = None
        self._dropped = 0
        self._written = 0
        self._lock = threading.Lock()
        self._attached_bus: EventBus | None = None
        self._handler_ref = self._on_event  # keep ref so unsubscribe matches
        self._provenance_ref = self._on_provenance
        # turn_id hex -> brain provenance, held until the turn's row is built.
        self._provenance: OrderedDict[str, dict[str, object]] = OrderedDict()
        self._start_thread()

    @property
    def written(self) -> int:
        return self._written

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def path(self) -> Path:
        return self._log_path

    # ----------------------------------------------------------------------
    # Bus binding
    # ----------------------------------------------------------------------
    def attach(self, bus: EventBus) -> None:
        """Subscribe to ``LatencyTurnComplete`` on ``bus``. Idempotent.

        ``EventBus.subscribe`` takes a per-event-type callback; we register
        narrowly (not via ``subscribe_all``) so the wildcard FlightRecorder
        path isn't duplicated.
        """
        if self._attached_bus is bus:
            return
        self._attached_bus = bus
        bus.subscribe(LatencyTurnComplete, self._handler_ref)
        bus.subscribe(BrainTurnProvenance, self._provenance_ref)

    async def _on_provenance(self, event: BrainTurnProvenance) -> None:
        if not isinstance(event, BrainTurnProvenance):
            return
        self._provenance[event.trace_id.hex] = {
            "provider": event.provider,
            "requested_model": event.requested_model,
            "responded_model": event.responded_model,
            "success": event.success,
            "error_kind": event.error_kind,
            "fallback_used": event.fallback_used,
            "chatgpt_bridge_used": event.chatgpt_bridge_used,
            "attempts": [
                {"provider": p, "model": m, "outcome": o} for p, m, o in event.attempts
            ],
            "tokens_in": event.tokens_in,
            "tokens_out": event.tokens_out,
            "tokens_cached": event.tokens_cached,
        }
        # A brain turn with no voice row (chat, agents) must not grow this.
        while len(self._provenance) > _MAX_PENDING_PROVENANCE:
            self._provenance.popitem(last=False)

    async def _on_event(self, event: LatencyTurnComplete) -> None:
        # Guard wrongly-routed events (defensive — subscribe is narrowed).
        if not isinstance(event, LatencyTurnComplete):
            return
        row = self._build_row(event)
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self._dropped += 1
            logger.warning(
                "latency log queue full (dropped=%d) — writer thread stuck?",
                self._dropped,
            )

    # ----------------------------------------------------------------------
    # Row construction
    # ----------------------------------------------------------------------
    def _build_row(self, event: LatencyTurnComplete) -> dict[str, object]:
        stages = dict(event.stages_ms)  # defensive copy
        # Derived metrics. Use TURN_TO_FIRST_AUDIO as the canonical TTFW
        # because BRAIN_FIRST_AUDIO + ACK_FIRST_AUDIO race depending on
        # whether the ack-brain fired first.
        ttfw_ms = stages.get("turn_to_first_audio")
        total_ms = stages.get("tts_stream_done")
        durations_ms = _derive_durations(stages)
        invalid_durations = _invalid_durations(stages)
        return {
            "turn_id": event.trace_id.hex,
            "iso_timestamp": _iso(event.timestamp_ns),
            "anchor_ns": event.anchor_ns,
            "stages_ms": stages,
            "durations_ms": durations_ms,
            "ttfw_ms": ttfw_ms,
            "total_ms": total_ms,
            "stt_input_audio_ms": (
                event.stt_input_audio_ms
                if event.stt_input_audio_ms >= 0
                else None
            ),
            "brain_input_tokens": (
                event.brain_input_tokens if event.brain_input_tokens >= 0 else None
            ),
            "brain_output_tokens": (
                event.brain_output_tokens if event.brain_output_tokens >= 0 else None
            ),
            "tts_input_chars": (
                event.tts_input_chars if event.tts_input_chars >= 0 else None
            ),
            "errors": list(event.errors),
            "diagnostics": _build_diagnostics(event, invalid_durations),
            "brain": self._provenance.pop(event.trace_id.hex, None),
        }

    # ----------------------------------------------------------------------
    # Flusher thread
    # ----------------------------------------------------------------------
    def _start_thread(self) -> None:
        thread = threading.Thread(
            target=self._flush_loop,
            name="latency-log-writer",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def _flush_loop(self) -> None:
        # Each pass writes up to N rows in one append to amortize fsync
        # cost. Small batches keep latency for the aggregation CLI low.
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            batch: list[dict[str, object]] = [item]  # type: ignore[list-item]
            # Drain anything else already queued — keep the batch under 256
            # to bound write time.
            while len(batch) < 256:
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is _STOP:
                    self._write_batch(batch)
                    return
                batch.append(nxt)  # type: ignore[arg-type]
            self._write_batch(batch)

    def _write_batch(self, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        try:
            with self._log_path.open("a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False))
                    fh.write("\n")
            with self._lock:
                self._written += len(rows)
        except OSError as exc:
            logger.warning("latency JSONL write failed: %s", exc)

    # ----------------------------------------------------------------------
    # Shutdown
    # ----------------------------------------------------------------------
    def close(self) -> None:
        """Drain the queue and join the writer thread.

        Safe to call multiple times. After close ``attach()`` is invalid;
        construct a new writer for a second pipeline.
        """
        thread = self._thread
        if thread is None:
            return
        try:
            self._queue.put(_STOP, timeout=1.0)
        except queue.Full:
            # Queue jammed — best-effort: thread will see _STOP eventually
            # if we put without a timeout, but we don't block on shutdown.
            logger.warning("latency log shutdown: queue full, leaving thread")
            return
        thread.join(timeout=5.0)
        self._thread = None


def _iso(ts_ns: int) -> str:
    """ns since epoch (wall clock) → ISO-8601 UTC string."""
    seconds = ts_ns / 1_000_000_000
    return datetime.fromtimestamp(seconds, tz=UTC).isoformat()


# Stage pairs whose ms-difference is meaningful for the bottleneck ranking.
# Order matters: each derived duration is one row in the report table.
_DURATION_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("vad_to_stt_first", "stt_first_partial", ""),
    ("stt_streaming", "stt_finalize", "stt_first_partial"),
    ("stt_to_brain_request", "brain_request_sent", "stt_finalize"),
    ("brain_ttft", "brain_first_token", "brain_request_sent"),
    ("brain_streaming", "brain_last_token", "brain_first_token"),
    ("brain_to_tts_request", "tts_request_sent", "brain_last_token"),
    ("tts_ttfb", "tts_first_chunk", "tts_request_sent"),
    ("tts_to_audio_out", "turn_to_first_audio", "tts_first_chunk"),
    ("tts_tail", "tts_stream_done", "turn_to_first_audio"),
)


def _derive_durations(stages: dict[str, float]) -> dict[str, float | None]:
    """Per-stage deltas in ms, computed from the stage offsets."""
    out: dict[str, float | None] = {}
    for label, end_phase, start_phase in _DURATION_PAIRS:
        end = stages.get(end_phase)
        if end is None:
            out[label] = None
            continue
        if not start_phase:
            out[label] = end
            continue
        start = stages.get(start_phase)
        if start is None:
            out[label] = None
            continue
        delta = round(end - start, 3)
        out[label] = delta if delta >= 0 else None
    return out


def _invalid_durations(stages: dict[str, float]) -> list[dict[str, object]]:
    invalid: list[dict[str, object]] = []
    for label, end_phase, start_phase in _DURATION_PAIRS:
        if not start_phase:
            continue
        end = stages.get(end_phase)
        start = stages.get(start_phase)
        if end is None or start is None:
            continue
        delta = round(end - start, 3)
        if delta < 0:
            invalid.append(
                {
                    "label": label,
                    "start_phase": start_phase,
                    "end_phase": end_phase,
                    "delta_ms": delta,
                    "reason": "end_before_start",
                }
            )
    return invalid


def _build_diagnostics(
    event: LatencyTurnComplete,
    invalid_durations: list[dict[str, object]],
) -> dict[str, object]:
    has_tokens = event.brain_input_tokens >= 0 or event.brain_output_tokens >= 0
    durations = _derive_durations(dict(event.stages_ms))
    stt_ms = durations.get("stt_streaming")
    bridge_ms = durations.get("stt_to_brain_request")
    tts_provider = event.diagnostics.get("tts_provider")
    tts_voice = event.diagnostics.get("tts_voice")
    tts_ttfb = durations.get("tts_ttfb")
    has_tts = tts_provider is not None or tts_ttfb is not None
    return {
        "event": dict(event.diagnostics),
        "turn": {
            "event_type": type(event).__name__,
            "event_timestamp_ns": event.timestamp_ns,
            "input_event_id": None,
            "output_event_id": None,
            "continuation_events": [],
            "dedup_decision": {
                "availability": "unavailable",
                "reason": "not_present_on_latency_event",
            },
        },
        "stt": {
            "provider": None,
            "detected_language": None,
            "final_completed": None,
            "duration_ms": stt_ms,
            "error": {
                "availability": "unavailable",
                "reason": "not_present_on_latency_event",
            },
            "availability": "partial" if stt_ms is not None else "unavailable",
            "reason": (
                "duration_only_from_stage_timestamps"
                if stt_ms is not None
                else "not_present_on_latency_event"
            ),
        },
        "routing_bridge": {
            "intent_category": None,
            "route": None,
            "provider": None,
            "bridge_start_ns": None,
            "bridge_end_ns": None,
            "bridge_duration_ms": bridge_ms,
            "error": {
                "availability": "unavailable",
                "reason": "not_present_on_latency_event",
            },
        },
        "tts": {
            "provider": tts_provider,
            "language": None,
            "voice": tts_voice,
            "ttfb_ms": tts_ttfb,
            "tail_ms": durations.get("tts_tail"),
            "tts_start_ns": None,
            "first_audio_ns": None,
            "playback_handoff_ns": None,
            "playback_start_ns": None,
            "playback_end_ns": None,
            "queue_wait_ms": None,
            "availability": "partial" if has_tts else "unavailable",
            "reason": (
                "provider_voice_and_durations_only"
                if has_tts
                else "not_present_on_latency_event"
            ),
        },
        "timing": {
            "clock": "perf_counter_ns",
            "anchor_ns": event.anchor_ns,
            "invalid_durations": invalid_durations,
        },
        "tokens": {
            "input": event.brain_input_tokens if event.brain_input_tokens >= 0 else None,
            "output": event.brain_output_tokens if event.brain_output_tokens >= 0 else None,
            "provider": None,
            "model": None,
            "source": None,
            "availability": "partial" if has_tokens else "unavailable",
            "reason": (
                "counts_present_but_provider_metadata_not_present"
                if has_tokens
                else "token_metadata_not_present_on_event"
            ),
        },
    }
