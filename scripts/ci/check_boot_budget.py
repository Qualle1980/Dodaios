#!/usr/bin/env python
"""Startup-budget regression guard (TTU) — fail when boot gets slow again.

Why this exists (TTU forensic 2026-07-02, docs/diagnostics/BOOT-TTU-NOTES.md):
boot regressions land silently — a stale custom wake model put a 114.7 s
model-load cascade on the critical path and nobody noticed until the user did.
This guard runs ONE isolated cold boot through the committed harness
(``scripts/measure_desktop_boot.py``) and fails when a measured anchor exceeds
its budget, so a feature that sneaks heavy work onto the startup path breaks
CI/pre-push instead of the user's day.

Anchors and budgets (override via env for slower CI boxes):
- window: spawn -> the UI shell serves (``median_wall_ms``).
  Budget ``JARVIS_BOOT_BUDGET_WINDOW_MS`` (default 8000; measured median on
  the reference box: ~1.2-2.2 s).
- voice TTU: spawn -> ``VOICE_USABLE_MS`` (wake model warmed + VAD + TTS up).
  Budget ``JARVIS_BOOT_BUDGET_VOICE_MS`` (default 20000; clean 5-run median
  on the reference box after the prefetch+prime work: 5.1 s). Checked only
  when the host can run the voice stack (audio device present) — a headless
  CI box checks the window anchor and skips voice honestly instead of faking
  a pass.
- app-interactive: spawn -> ``APP_INTERACTIVE_MS`` (``set_app``: the UI's held
  data requests are answered). Budget ``JARVIS_BOOT_BUDGET_INTERACTIVE_MS``
  (default 20000). This anchor sits behind the wake-model gate BY DESIGN, so
  it moves whenever voice does.
- post-voice work: app-interactive MINUS voice TTU — ``prepare_app`` and the
  route mounting, i.e. the part a new feature actually regresses. Budget
  ``JARVIS_BOOT_BUDGET_POST_VOICE_MS`` (default 8000). When app-interactive
  misses its budget while this one holds, the cause is voice, not the path.

Sampling: ``JARVIS_BOOT_BUDGET_RUNS`` cold boots (default 3) run SERIALLY
after one discarded warmup, and every sample is printed next to the median.
One sample cannot separate a regression from boot noise — the measured spread
for one unchanged commit on the reference box was 9.5 s to 41.5 s. A spread
above 2x prints an INVESTIGATE line and makes any before/after comparison on
those numbers inconclusive; outliers are never silently dropped.

Budgets are deliberately ~4x the measured medians: loose enough that machine
variance never flakes, tight enough that every regression class we have
actually seen (114.7 s load cascade, 12 s gate timeout, 60 s overlay fallback)
blows through them.

Exit codes: 0 = within budget, 1 = budget exceeded, 78 = skipped (no python /
harness prerequisites) so callers can treat "could not measure" as neutral.

How to keep a feature off the critical startup path (the doctrine this guard
enforces): do NOT add work before the speech pipeline / VOICE_READY. New
subsystems belong behind ``_heavy_backend_bg`` (desktop_app), a deferred
registry scan, or a fire-and-forget task AFTER voice-ready. When in doubt run
``scripts/measure_desktop_boot.py --voice --runs 3`` before/after your change.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "scripts" / "measure_desktop_boot.py"
TTU_LATEST = REPO_ROOT / "desktop-ttu-latest.json"
LATEST = REPO_ROOT / "desktop-boot-latest.json"

SKIP_EXIT = 78  # EX_CONFIG-ish: "could not measure" — neutral, not a failure

DEFAULT_WINDOW_BUDGET_MS = 8_000.0
DEFAULT_VOICE_BUDGET_MS = 20_000.0
# APP_INTERACTIVE = set_app hands the UI's held data requests to the real app
# (the user-facing "usable" moment; measured median 5.3 s isolated). Checked in
# voice mode (the run must live past the window anchor to see it).
DEFAULT_INTERACTIVE_BUDGET_MS = 20_000.0
# APP_INTERACTIVE sits behind the wake-model gate BY DESIGN
# (``_heavy_backend_bg`` waits for ``_wake_model_loaded``, bounded at 12 s,
# before ``prepare_app``), so the absolute anchor above moves whenever voice
# does and cannot tell "voice got slower" from "someone put work on the
# startup path". This budget is the DIFFERENCE — the work that actually runs
# after voice is usable — and that is the one a feature regresses.
DEFAULT_POST_VOICE_BUDGET_MS = 8_000.0

# One sample cannot separate a regression from boot noise: the measured spread
# on the reference box is 9.5 s to 41.5 s for the same commit. Run several
# cold boots SERIALLY (never concurrently — parallel boots compete for CPU,
# disk and the GPU probe and measure the contention, not the app) and judge
# the median. Overridable so a slow CI box can spend less.
DEFAULT_RUNS = 3
# Ratio between the slowest and fastest run above which the numbers are not a
# usable basis for a decision. Reported loudly, never silently discarded: an
# outlier is a lead, and dropping it is how the 403 ack survived for months.
UNSTABLE_SPREAD_RATIO = 2.0


def _macos_microphone_granted() -> bool:
    """Return whether this process may measure voice without a TCC prompt."""
    if sys.platform != "darwin":
        return True
    try:
        from jarvis.platform.permissions import (  # noqa: PLC0415
            PermissionId,
            PermissionState,
            SystemPermissionPort,
        )

        return SystemPermissionPort().state(PermissionId.MICROPHONE) is PermissionState.GRANTED
    except Exception:  # noqa: BLE001 — an uncertain TCC state fails closed
        return False


def _audio_capable() -> bool:
    """True when the host can plausibly run the voice stack (mic present)."""
    if not _macos_microphone_granted():
        return False
    try:
        import sounddevice  # noqa: PLC0415

        devices = sounddevice.query_devices()
        return any(int(d.get("max_input_channels", 0)) > 0 for d in devices)
    except Exception:  # noqa: BLE001 — no PortAudio / headless box
        return False


def _report_spread(summary: dict, label: str, runs_key: str) -> None:
    """Print every sample for *label*, plus median and spread.

    A gate that prints one number teaches whoever reads it to trust one
    number. Printing the samples is what makes "the ranges overlap" visible
    at the moment of the decision instead of in a later forensic.
    """
    samples = [float(v) for v in summary.get(runs_key) or []]
    if not samples:
        return
    median = statistics.median(samples)
    low, high = min(samples), max(samples)
    line = (
        f"boot-budget: {label} samples "
        f"{', '.join(f'{v:.0f}' for v in samples)} ms "
        f"-> median {median:.0f} ms (min {low:.0f}, max {high:.0f})"
    )
    print(line, flush=True)
    if len(samples) > 1 and low > 0 and high / low >= UNSTABLE_SPREAD_RATIO:
        # Not a failure: an unstable box must not block a correct change. But
        # it is also not a basis for claiming an improvement or a regression.
        print(
            f"boot-budget: INVESTIGATE — {label} spread is {high / low:.1f}x "
            f"(max/min). Treat any before/after comparison on these numbers as "
            f"inconclusive and find the cause of the slow run (contention from "
            f"a parallel build or agent session, a cold cache, a bounded gate "
            f"timing out) before drawing a conclusion from it.",
            flush=True,
        )


def main() -> int:
    window_budget = float(
        os.environ.get("JARVIS_BOOT_BUDGET_WINDOW_MS", DEFAULT_WINDOW_BUDGET_MS)
    )
    voice_budget = float(
        os.environ.get("JARVIS_BOOT_BUDGET_VOICE_MS", DEFAULT_VOICE_BUDGET_MS)
    )
    interactive_budget = float(
        os.environ.get(
            "JARVIS_BOOT_BUDGET_INTERACTIVE_MS", DEFAULT_INTERACTIVE_BUDGET_MS
        )
    )
    post_voice_budget = float(
        os.environ.get("JARVIS_BOOT_BUDGET_POST_VOICE_MS", DEFAULT_POST_VOICE_BUDGET_MS)
    )
    runs = max(1, int(os.environ.get("JARVIS_BOOT_BUDGET_RUNS", DEFAULT_RUNS)))
    if not HARNESS.exists():
        print(f"boot-budget: harness missing ({HARNESS}) — skipping", flush=True)
        return SKIP_EXIT

    voice = _audio_capable()
    # One discarded warmup boot: without it the first run also pays the OS file
    # cache for the model and bundle reads, which is the single largest source
    # of the spread this guard used to trip over.
    args = [
        sys.executable,
        str(HARNESS),
        "--runs",
        str(runs),
        "--warmup",
        "1",
    ]
    if voice:
        args.append("--voice")
    print(
        f"boot-budget: measuring {runs} isolated cold boot(s) serially, "
        f"1 discarded warmup (voice={voice}) ...",
        flush=True,
    )
    proc = subprocess.run(  # noqa: S603 — our own harness, fixed argv
        args,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300 + runs * 180,
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:], flush=True)
        print(proc.stderr[-2000:], flush=True)
        print("boot-budget: harness run FAILED — treating as budget failure", flush=True)
        return 1

    latest = TTU_LATEST if voice else LATEST
    summary = json.loads(latest.read_text(encoding="utf-8"))

    failures: list[str] = []
    _report_spread(summary, "window", "wall_ms_runs")
    window_ms = summary.get("median_wall_ms")
    if window_ms is None:
        failures.append("window anchor missing from harness output")
    elif window_ms > window_budget:
        failures.append(
            f"window anchor {window_ms:.0f} ms > budget {window_budget:.0f} ms"
        )

    if voice:
        _report_spread(summary, "voice TTU", "voice_usable_wall_ms_runs")
        _report_spread(summary, "app-interactive", "app_interactive_wall_ms_runs")
        voice_ms = summary.get("median_voice_usable_wall_ms")
        if voice_ms is None:
            failures.append("VOICE_USABLE anchor missing from harness output")
        elif voice_ms > voice_budget:
            failures.append(
                f"voice TTU {voice_ms:.0f} ms > budget {voice_budget:.0f} ms"
            )
        else:
            print(
                f"boot-budget: voice TTU {voice_ms:.0f} ms <= "
                f"{voice_budget:.0f} ms OK",
                flush=True,
            )
        interactive_ms = summary.get("median_app_interactive_wall_ms")
        if interactive_ms is None:
            failures.append("APP_INTERACTIVE anchor missing from harness output")
        else:
            # Separate the two things the absolute anchor confuses: the time
            # the backend DELIBERATELY waits for the wake model, and the work
            # that runs after it (prepare_app + the route mounting).
            post_voice_ms = (
                interactive_ms - voice_ms if isinstance(voice_ms, (int, float)) else None
            )
            if post_voice_ms is not None:
                if post_voice_ms > post_voice_budget:
                    failures.append(
                        f"post-voice work {post_voice_ms:.0f} ms > budget "
                        f"{post_voice_budget:.0f} ms (voice usable at "
                        f"{voice_ms:.0f} ms, app interactive at "
                        f"{interactive_ms:.0f} ms) — this is work on the "
                        f"startup path, not the wake gate"
                    )
                else:
                    print(
                        f"boot-budget: post-voice work {post_voice_ms:.0f} ms <= "
                        f"{post_voice_budget:.0f} ms OK",
                        flush=True,
                    )
            if interactive_ms > interactive_budget:
                cause = (
                    " — but the post-voice work is within budget, so the cause "
                    "is the voice anchor / the wake-model gate it waits on, "
                    "NOT new work on the startup path"
                    if post_voice_ms is not None and post_voice_ms <= post_voice_budget
                    else ""
                )
                failures.append(
                    f"app-interactive {interactive_ms:.0f} ms > budget "
                    f"{interactive_budget:.0f} ms{cause}"
                )
            else:
                print(
                    f"boot-budget: app-interactive {interactive_ms:.0f} ms <= "
                    f"{interactive_budget:.0f} ms OK",
                    flush=True,
                )
    else:
        print(
            "boot-budget: no usable audio input or permission — voice anchor skipped (honest)",
            flush=True,
        )

    if window_ms is not None and window_ms <= window_budget:
        print(
            f"boot-budget: window {window_ms:.0f} ms <= {window_budget:.0f} ms OK",
            flush=True,
        )

    if failures:
        for f in failures:
            print(f"boot-budget FAILED: {f}", flush=True)
        print(
            "A change put work on the critical startup path. Keep new features "
            "behind _heavy_backend_bg / deferred scans (see the module docstring "
            "and docs/diagnostics/BOOT-TTU-NOTES.md).",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
