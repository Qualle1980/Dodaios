"""Antigravity Brain — answer over the user's Google subscription, no API key.

This provider drives the **official** Google agent CLI (Antigravity ``agy`` or
the Gemini CLI, resolved by :func:`jarvis.google_cli.resolver.resolve_google_cli`)
as a subprocess over the existing "Sign in with Google" login. It is the Google
sibling of :class:`jarvis.plugins.brain.codex.CodexBrain` (which does the same
over the ChatGPT subscription via ``codex exec``).

OAuth-only: there is no API-key path here — that is what the existing ``gemini``
brain provider is for. The conversational brain runs the CLI in read-only
``--approval-mode plan`` so it cannot write files or run commands.

Google ToS (hard): we only ever invoke the official binary. We never read the
stored OAuth token to make our own HTTP request. The CLI is slow (the agent
spins up per turn), so this is a deliberate, user-opted path — not the default.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress

from jarvis.core.process_utils import NO_WINDOW_CREATIONFLAGS
from jarvis.core.protocols import BrainDelta, BrainRequest
from jarvis.google_cli.isolated_home import (
    ensure_isolated_home as _ensure_isolated_home,
)
from jarvis.google_cli.isolated_home import (
    iso_home_root as _iso_home_root,
)
from jarvis.google_cli.isolated_home import (
    real_gemini_dir as _real_gemini_dir,
)
from jarvis.google_cli.isolated_home import redirect_home_env
from jarvis.google_cli.pty_runner import repair_agy_path
from jarvis.google_cli.resolver import GoogleCli, resolve_google_cli

from .cli_prompt_context import (
    extract_reply_language_directive,
    render_cli_standing_instructions,
    render_structured_prompt,
)

log = logging.getLogger(__name__)

# Fallback only — the active model comes from [brain.providers.antigravity].model.
# Flash is the fast default for the gemini-CLI fallback path, which takes a
# plain model name. It is NOT used verbatim for agy any more.
#
# History, because the behaviour reversed twice and a stale assumption broke the
# provider outright: agy 1.0.10 IGNORED ``--model`` and REQUIRED ``--effort``
# beside it. agy 1.5.1 does the opposite — it validates ``--model`` against the
# account's own list and REFUSES ``--effort`` when the id already carries a
# level ("--model gemini-3.8-flash-low conflicts with --effort=medium"). A
# hardcoded id therefore ages into a hard failure: this one did, with
# "model gemini-3.5-flash is not recognized as a known model".
#
# So the agy path ASKS the binary what it can run (``agy models``) instead of
# naming a model from memory — the capability gate AP-21 requires.
DEFAULT_MODEL = "gemini-3.5-flash"

#: Family preferred when resolving an agy model, when nothing better is asked
#: for. A family, never a version: versions retire, families do not.
_AGY_PREFERRED_FAMILY = "gemini"

# Hard cap for a single CLI brain turn. The agent CLI is slow (cold start +
# 20k-token system prompt); 120 s leaves headroom without hanging the brain
# coroutine forever if the subscription is unreachable.
_CLI_TIMEOUT_S: float = 120.0

#: ``agy models`` does a short network fetch; it must never hold up a turn for
#: long, because failing it only costs us the model check.
_MODELS_TIMEOUT_S: float = 20.0

# Env keys dropped from the child so the subscription login wins and an
# accidental API key can never bill the wrong account / break the OAuth path.
_DROP_ENV: tuple[str, ...] = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_AISTUDIO_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
)

_CLI_SYSTEM = (
    "You are the user's concise and friendly voice assistant. Answer the user's "
    "message directly in one to three short sentences. Reply in plain text only "
    "— do not run any commands, do not read or edit files, do not use tools."
)


def _parse_cli_answer(stdout: str) -> str:
    """Extract the answer text from the CLI's stdout.

    The Gemini CLI ``-o json`` emits a JSON object with a ``response`` field;
    ``agy`` variants differ. We try JSON first (``response``/``text``/``output``/
    ``content``) and fall back to the raw, trimmed stdout so a plain-text or
    schema-changed output still yields an answer.
    """
    text = (stdout or "").strip()
    if not text:
        return ""
    try:
        obj = json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return text
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, dict):
        for key in ("response", "text", "output", "content", "result"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""
    return text


def _build_cli_prompt(req: BrainRequest) -> str:
    """Flatten the last few conversational turns into one prompt for ``-p``.

    The heavy router system prompt (full of tool definitions) is dropped — it
    would make the agent CLI slow and confused. We send a light conversational
    instruction plus the last ~6 user/assistant turns for context.
    """
    lines: list[str] = [_CLI_SYSTEM, ""]
    prefs = render_cli_standing_instructions(req.system)
    convo = [
        m
        for m in req.messages
        if getattr(m, "role", None) in ("user", "assistant")
        and isinstance(getattr(m, "content", None), str)
    ][-6:]
    for m in convo:
        speaker = "User" if m.role == "user" else "Assistant"
        lines.append(f"{speaker}: {m.content}")
    if prefs:
        lines.extend(["", prefs])
    # The reply-language directive rides LAST (highest recency) so the CLI model
    # answers in the turn's resolved language instead of anchoring to the German
    # persona — without this the directive is dropped and an English request is
    # answered in German (live bug 2026-06-21).
    lang_directive = extract_reply_language_directive(req.system)
    if lang_directive:
        lines.extend(["", lang_directive])
    lines.append("Assistant:")
    return "\n".join(lines)


# Reasoning effort on the agy CLI rides INSIDE the model id as a level suffix
# (``gemini-3.8-flash-low`` / ``-medium`` / ``-high``), and passing ``--effort``
# beside such an id is an error. So the ladder maps onto the suffix instead of a
# flag, and ``--effort`` is never sent to agy.
_DEFAULT_AGY_EFFORT = "medium"
#: The levels agy spells in its model ids. A caller asking to disable thinking
#: maps to the lowest level agy offers, since it has no "off".
_AGY_EFFORTS: tuple[str, ...] = ("low", "medium", "high")


def _agy_effort(req: BrainRequest | None) -> str:
    """The level an agy model id should carry, from the caller's own hint."""
    requested = str(getattr(req, "reasoning_effort", "") or "").strip().lower()
    if requested == "none":
        return "low"
    if requested in {"xhigh", "max"}:
        return "high"
    return requested if requested in _AGY_EFFORTS else _DEFAULT_AGY_EFFORT


def _strip_level(model: str) -> str:
    """``gemini-3.8-flash-low`` -> ``gemini-3.8-flash``; others unchanged."""
    for level in _AGY_EFFORTS:
        suffix = f"-{level}"
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def pick_agy_model(available: Sequence[str], requested: str, effort: str) -> str:
    """Choose a model id agy will actually accept.

    ``available`` is what the binary reported. The order of preference is the
    caller's exact id, then the caller's family at the wanted level, then the
    preferred family at that level, then anything at that level, then anything
    at all. Returns ``requested`` unchanged when nothing was reported, so a
    discovery failure degrades to the old behaviour rather than inventing an id.
    """
    if not available:
        return requested
    if requested in available:
        return requested

    def at_level(candidates: Sequence[str]) -> str | None:
        for candidate in candidates:
            if candidate.endswith(f"-{effort}"):
                return candidate
        return candidates[0] if candidates else None

    base = _strip_level(requested)
    if base:
        same_family = [m for m in available if _strip_level(m) == base]
        if same_family:
            chosen = at_level(same_family)
            if chosen:
                return chosen

    preferred = [m for m in available if m.startswith(_AGY_PREFERRED_FAMILY)]
    chosen = at_level(preferred) or at_level(list(available))
    # ``available`` is non-empty, so at_level cannot return None here.
    assert chosen is not None
    return chosen


#: What agy says when it will not run the model it was handed. Matched on the
#: wording rather than the exit status, because the status also covers real
#: outages and a retry must not paper over those.
_MODEL_REJECTION_MARKERS: tuple[str, ...] = (
    "invalid model selection",
    "is not recognized as a known model",
    "conflicts with --effort",
)


def _is_model_rejection(message: str) -> bool:
    """Whether a failed run failed because of the MODEL, not the subscription."""
    lowered = message.lower()
    return any(marker in lowered for marker in _MODEL_REJECTION_MARKERS)


def parse_agy_models(stdout: str) -> tuple[str, ...]:
    """Model ids from ``agy models`` output (``id<TAB>display name`` per line).

    Tolerant on purpose: the command prints a progress line to stderr and may
    gain columns, so only the first tab-separated field of plausible lines is
    read.
    """
    ids: list[str] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or "	" not in line:
            continue
        candidate = line.split("	", 1)[0].strip()
        if candidate and " " not in candidate:
            ids.append(candidate)
    return tuple(ids)


#: Model ids reported by ``agy models``, cached per process. The command does a
#: network fetch (~2 s), and nothing may run it on the boot path (AP-26), so it
#: is populated lazily on the first turn and reused afterwards.
_agy_models_lock = asyncio.Lock()
_agy_models_cache: tuple[str, ...] | None = None


async def _agy_models(argv_prefix: Sequence[str], env: dict[str, str]) -> tuple[str, ...]:
    """Ask agy which models this account may run. Empty tuple on any failure."""
    global _agy_models_cache
    async with _agy_models_lock:
        if _agy_models_cache is not None:
            return _agy_models_cache
        creationflags = NO_WINDOW_CREATIONFLAGS if sys.platform == "win32" else 0
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv_prefix,
                "models",
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=creationflags,
            )
            raw, _ = await asyncio.wait_for(proc.communicate(), timeout=_MODELS_TIMEOUT_S)
        except (TimeoutError, OSError) as exc:
            # Not cached: a transient failure must not pin an empty list for the
            # life of the process. Degrades to the caller's configured id.
            log.warning("AntigravityBrain: could not list models (%s)", exc)
            return ()
        found = parse_agy_models(raw.decode("utf-8", "replace"))
        if not found:
            log.warning("AntigravityBrain: 'agy models' reported nothing usable")
            return ()
        log.info("AntigravityBrain: agy offers %s", ", ".join(found))
        _agy_models_cache = found
        return found


def _build_argv(
    cli: GoogleCli, prompt: str, model: str, effort: str = _DEFAULT_AGY_EFFORT
) -> list[str]:
    """Build read-only JSON-output commands; the complete prompt travels on stdin.

    Antigravity print mode activates on piped input without ``--print``.
    Keeping input out of argv avoids both Windows and POSIX argument limits.
    """
    if cli.kind == "agy":
        # No ``--effort``: the level is already inside ``model`` and agy rejects
        # the combination. See the note on DEFAULT_MODEL.
        return [
            *cli.argv_prefix,
            "--output-format",
            "json",
            "--mode",
            "plan",
            "--model",
            model,
        ]
    return [
        *cli.argv_prefix,
        "-p",
        "Follow the instructions supplied on stdin.",
        "-m",
        model,
        "--approval-mode",
        "plan",
        "--skip-trust",
        "-o",
        "json",
    ]


class AntigravityBrain:
    name: str = "antigravity"
    context_window: int = 1_048_576
    supports_tools: bool = True  # ignored on the CLI path (mirrors CodexBrain)
    supports_vision: bool = False
    # Neither agy nor the Gemini CLI takes a real system prompt in headless
    # mode: structured mode PREPENDS the caller's contract to the user prompt
    # (render_structured_prompt), where the agent may read it as text to react
    # to rather than orders to follow. Live 2026-08-11: every Agentic IDE pane
    # title became a chat acknowledgement ("Understood! I see …") the moment
    # this provider wrote the recaps. The subscription resolver uses this flag
    # to prefer a sibling with a dedicated system channel when one is signed in.
    native_system_prompt: bool = False

    def __init__(
        self,
        model: str | None = None,
        structured_prompts: bool = False,
        cli_timeout_s: float | None = None,
    ) -> None:
        self._model = model or DEFAULT_MODEL
        # Background/structured callers (the wiki curator tier) set this so
        # their JSON contract reaches the CLI model verbatim instead of the
        # conversational "answer in 1-3 plain-text sentences" wrapper — which
        # made structured output impossible by instruction (live 2026-07-18:
        # every wiki extraction died with "no JSON array found in response").
        self._structured_prompts = bool(structured_prompts)
        # Slow background callers (wiki Stage-2 judge) pass their own per-call
        # budget so the internal cap cannot kill a turn the caller is still
        # willing to wait for; the voice-tier default stays the tight cap.
        try:
            budget = float(cli_timeout_s) if cli_timeout_s is not None else 0.0
        except (TypeError, ValueError):
            budget = 0.0
        self._cli_timeout_s = budget if budget > 0 else _CLI_TIMEOUT_S

    @staticmethod
    def subscription_connected() -> bool:
        """Whether this provider's Google subscription login is usable now.

        The subscription resolver asks the provider class rather than importing
        an auth service per family, so each brain answers for itself. Only a
        personal OAuth login counts: an API key configured for this family bills
        per token, which is precisely what a subscription caller is avoiding.
        Never raises — a failed probe means "not connected", never a broken turn.
        """
        try:
            from jarvis.google_cli.auth_service import GoogleCliAuthService

            status = GoogleCliAuthService().status()
            return bool(
                getattr(status, "connected", False)
                and getattr(status, "mode", "") == "oauth-personal"
            )
        except Exception:  # noqa: BLE001 - a probe degrades, never raises
            return False

    def can_call_tools(self) -> bool:
        """Runtime tool-calling capability (NOT the static ``supports_tools``).

        Antigravity drives the Google-subscription CLI (``agy`` or Gemini) over
        pipes with a flattened prompt — it has no
        chat-completions/function-calling path and drops every tool. The caller
        (``BrainManager``) uses this to delegate tool/Computer-Use turns to a
        tool-capable provider instead of letting the CLI silently no-op."""
        return False

    def _render_prompt(self, req: BrainRequest) -> str:
        """Conversational flattening for voice turns; verbatim for structured."""
        if self._structured_prompts:
            return render_structured_prompt(req)
        return _build_cli_prompt(req)

    async def complete(self, req: BrainRequest) -> AsyncIterator[BrainDelta]:
        cli = resolve_google_cli()
        if cli is None:
            raise RuntimeError(
                "No Google CLI found — install Antigravity (agy) or the Gemini "
                "CLI and sign in with Google."
            )

        prompt = self._render_prompt(req)
        effort = _agy_effort(req)
        model = self._model
        # The configured id is tried FIRST and the binary is only questioned
        # when it refuses that id. Asking up front would spend a subprocess and
        # a network fetch on every single turn to answer a question that is
        # almost always "yes".
        for attempt in (1, 2):
            try:
                async for delta in self._complete_via_pipe(
                    _build_argv(cli, prompt, model, effort), prompt
                ):
                    yield delta
                return
            except RuntimeError as exc:
                retryable = (
                    attempt == 1 and cli.kind == "agy" and _is_model_rejection(str(exc))
                )
                if not retryable:
                    raise
                env = await asyncio.to_thread(self._build_child_env, harden_path=True)
                replacement = pick_agy_model(
                    await _agy_models(cli.argv_prefix, env), model, effort
                )
                if replacement == model:
                    raise
                log.info(
                    "AntigravityBrain: %r was refused, retrying once with %r",
                    model,
                    replacement,
                )
                model = replacement
                # Remember it: otherwise every later turn would pay the same
                # refusal again, roughly three seconds each, on a voice path.
                self._model = replacement

    def _build_child_env(self, *, harden_path: bool) -> dict[str, str]:
        """Child env: drop API keys (subscription login wins), redirect HOME to an
        isolated hook/mcp-free CLI home (the lag fix), and on Windows repair PATH
        so agy's internal ``cmd.exe``/``npm`` spawns
        resolve even from a degraded launch env."""
        env = {k: v for k, v in os.environ.items() if k not in _DROP_ENV}
        if harden_path and sys.platform == "win32":
            node = shutil.which("node") or shutil.which("node.exe")
            node_dir = os.path.dirname(node) if node else None
            env["PATH"] = repair_agy_path(env.get("PATH", ""), node_dir=node_dir)
        # Redirect HOME so the CLI reads our minimal settings.json (no per-turn
        # PowerShell hook storm, no npm MCP boot) instead of the user's ~/.gemini.
        iso = _ensure_isolated_home(
            real_dir=_real_gemini_dir(), dest_root=_iso_home_root(), model=self._model
        )
        if iso:
            redirect_home_env(env, iso)
        return env

    async def _complete_via_pipe(self, argv: list[str], prompt: str) -> AsyncIterator[BrainDelta]:
        """Drive the official Google CLI with UTF-8 stdin and JSON stdout."""
        env = await asyncio.to_thread(self._build_child_env, harden_path=True)
        workdir = tempfile.mkdtemp(prefix="jarvis-antigravity-brain-")
        creationflags = NO_WINDOW_CREATIONFLAGS if sys.platform == "win32" else 0
        log.info(
            "AntigravityBrain(pipe): spawning %s (model=%s, prompt=%d chars)",
            argv[0],
            # From argv, not from self._model: a retry runs a DIFFERENT id, and
            # logging the configured one made a successful retry read as though
            # it had run the refused model.
            argv[argv.index("--model") + 1] if "--model" in argv else "default",
            len(prompt),
        )

        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=workdir,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=creationflags,
            )
        except (FileNotFoundError, OSError) as exc:
            with suppress(OSError):
                shutil.rmtree(workdir, ignore_errors=True)
            log.warning("AntigravityBrain: spawn failed: %s", exc)
            raise RuntimeError(f"Google CLI could not be launched: {exc}") from exc

        # communicate drains both output pipes while sending the full prompt;
        # the same deadline covers a child that never consumes its input.
        comm_task = asyncio.create_task(proc.communicate(input=prompt.encode("utf-8")))
        deadline = t0 + self._cli_timeout_s
        stdout_bytes = b""
        stderr_bytes = b""

        async def _kill() -> None:
            if not comm_task.done():
                comm_task.cancel()
            pid = getattr(proc, "pid", None)
            if sys.platform == "win32" and isinstance(pid, int) and pid > 0:
                with suppress(Exception):
                    killer = await asyncio.create_subprocess_exec(
                        "taskkill",
                        "/PID",
                        str(pid),
                        "/T",
                        "/F",
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                        creationflags=creationflags,
                    )
                    await asyncio.wait_for(killer.wait(), timeout=3.0)
            with suppress(Exception):
                proc.kill()
            with suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            with suppress(asyncio.CancelledError, Exception):
                await comm_task

        try:
            while True:
                slice_timeout = min(3.0, deadline - time.monotonic())
                if slice_timeout <= 0:
                    raise TimeoutError
                done, _ = await asyncio.wait({comm_task}, timeout=slice_timeout)
                if done:
                    stdout_bytes, stderr_bytes = comm_task.result()
                    break
                # No-text progress tick: keeps the caller's no-progress deadline
                # alive through the slow agent spin-up (yields nothing visible).
                yield BrainDelta(content="")
        except asyncio.CancelledError:
            await _kill()
            log.info("AntigravityBrain: cancelled (killed)")
            raise
        except TimeoutError as exc:
            await _kill()
            log.warning(
                "AntigravityBrain: no answer within %.0fs (killed)",
                self._cli_timeout_s,
            )
            raise RuntimeError(
                f"Antigravity (Google login) did not answer within {self._cli_timeout_s:.0f}s."
            ) from exc
        finally:
            with suppress(OSError):
                shutil.rmtree(workdir, ignore_errors=True)

        if proc.returncode:
            # agy reports WHY on both channels, as JSON on stdout and plain text
            # on stderr. Carrying it into the exception is what lets the caller
            # tell a rejected model apart from a real outage.
            complaint = " ".join(
                part
                for part in (
                    stdout_bytes.decode("utf-8", errors="replace").strip(),
                    stderr_bytes.decode("utf-8", errors="replace").strip(),
                )
                if part
            )[:500]
            log.warning(
                "AntigravityBrain: CLI exited with status %s: %s",
                proc.returncode,
                complaint or "(no detail)",
            )
            raise RuntimeError(
                f"Google CLI failed with exit status {proc.returncode}. {complaint}".strip()
            )
        decoded = stdout_bytes.decode("utf-8", errors="replace")
        try:
            envelope = json.loads(decoded)
        except ValueError:
            envelope = None  # The parser below supports older plain-text CLI output.
        if isinstance(envelope, dict) and (
            envelope.get("error") or envelope.get("status") in {"ERROR", "CANCELED", "CANCELLED"}
        ):
            log.warning("AntigravityBrain: CLI reported a failed result")
            raise RuntimeError("Google CLI reported a failed result.")
        answer = _parse_cli_answer(decoded)
        elapsed = time.monotonic() - t0
        if not answer:
            detail = stderr_bytes.decode("utf-8", errors="replace").strip()[:300]
            log.warning(
                "AntigravityBrain: empty answer after %.1fs rc=%s detail=%s",
                elapsed,
                proc.returncode,
                detail[:200],
            )
            raise RuntimeError(
                "Antigravity (Google login) returned no answer" + (f": {detail}" if detail else ".")
            )

        log.info("AntigravityBrain turn ok: %d chars in %.1fs", len(answer), elapsed)
        yield BrainDelta(content=answer)
        yield BrainDelta(finish_reason="stop")

    def estimate_cost(self, req: BrainRequest) -> float:
        # Billed against the Google subscription, not per-call — report ~0.
        return 0.0
