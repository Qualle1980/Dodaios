"""Lifecycle supervisor for the local ChatGPT web bridge behind ``chatgpt-web``.

The bridge is a separate local process (``chatgpt-web-provider serve``) that
drives the user's own ChatGPT web session in a dedicated browser profile and
exposes an OpenAI-Responses-shaped API on loopback. This module owns exactly
that one process: it reports a typed status, starts it lazily (never on the
boot critical path, AP-26), bounds restarts, keeps a PID/lock file so two
callers never spawn twice, and stops it on request.

Configuration lives on ``[brain.providers."chatgpt-web"]``:

* ``base_url`` — the bridge root, loopback only (default ``http://127.0.0.1:17852``)
* ``home``     — the bridge home (its ``config.json`` holds the bearer token;
  the browser session stays inside it and never leaves it)
* ``cli``      — path to the built ``provider-cli.mjs``
* ``runtime``  — the JavaScript runtime that runs it (default ``bun``)
* ``log_dir``  — where the size-capped bridge log goes (default ``<home>/logs``)

The token is read from the secret store first (``chatgpt_web_token``) and
otherwise from the bridge's own ``config.json``; it is never logged.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from jarvis.core import config as cfg
from jarvis.core.http_pool import SyncHttpClientPool
from jarvis.core.process_utils import NO_WINDOW_CREATIONFLAGS, resolve_executable

log = logging.getLogger(__name__)

PROVIDER_ID = "chatgpt-web"
DEFAULT_BASE_URL = "http://127.0.0.1:17852"

BridgeStatus = Literal[
    "not_installed",
    "not_configured",
    "login_required",
    "starting",
    "ready",
    "offline",
    "session_invalid",
    "failed",
]

# How long an unused bridge is kept alive before the reaper closes it. The
# browser profile costs real memory, and the next complex question pays only
# the measured ~6.5 s start to get it back.
DEFAULT_IDLE_SHUTDOWN_MINUTES = 15.0

# How often the idle watcher looks. Coarse on purpose: the timeout is minutes,
# so a tighter poll would only burn wakeups.
REAP_POLL_S = 60.0

# Bounded restarts: a bridge that keeps dying must surface as ``failed``
# instead of being respawned in a loop.
MAX_STARTS = 3
START_WINDOW_S = 600.0
STARTUP_GRACE_S = 45.0
LOG_CAP_BYTES = 5 * 1024 * 1024

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_probe_pool = SyncHttpClientPool(timeout_s=3.0)


class ChatGPTWebError(RuntimeError):
    """A typed bridge failure. ``code`` is one of the status/turn codes below.

    ``login_required`` · ``offline`` · ``session_invalid`` ·
    ``bridge_unavailable`` · ``submission_ambiguous`` · ``failed``
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class BridgeSettings:
    base_url: str
    home: Path | None
    cli: Path | None
    runtime: str
    log_dir: Path | None


@dataclass(frozen=True)
class BridgeState:
    status: BridgeStatus
    detail: str


def _provider_extra(name: str) -> str:
    try:
        provider = cfg.load_config().brain.providers.get(PROVIDER_ID)
    except Exception as exc:  # noqa: BLE001 — an unreadable config reads as "not configured"
        log.warning("chatgpt-web: config unreadable: %s", exc)
        return ""
    if provider is None:
        return ""
    if name == "base_url":
        return (provider.base_url or "").strip()
    return str((provider.model_extra or {}).get(name) or "").strip()


def settings() -> BridgeSettings:
    home = _provider_extra("home")
    cli = _provider_extra("cli")
    log_dir = _provider_extra("log_dir")
    home_path = Path(home) if home else None
    return BridgeSettings(
        base_url=(_provider_extra("base_url") or DEFAULT_BASE_URL).rstrip("/"),
        home=home_path,
        cli=Path(cli) if cli else None,
        runtime=_provider_extra("runtime") or "bun",
        log_dir=Path(log_dir) if log_dir else (home_path / "logs" if home_path else None),
    )


def idle_shutdown_seconds() -> float:
    """How long the bridge may sit unused before the reaper closes it.

    From ``[brain.providers."chatgpt-web"] idle_shutdown_minutes``, defaulting
    to :data:`DEFAULT_IDLE_SHUTDOWN_MINUTES`. ``0`` (or anything unparsable
    read as 0) disables reaping and keeps the bridge up for the session, which
    is the old behaviour.

    The bridge holds a browser profile open: measured 2026-10-21 it accounted
    for ~1.5 GB of the 4.1 GB resident in Chrome on a box with 2.3 GB free, so
    leaving it running after a single complex question is a real cost.
    """
    raw = _provider_extra("idle_shutdown_minutes")
    if raw == "":
        return DEFAULT_IDLE_SHUTDOWN_MINUTES * 60.0
    try:
        minutes = float(raw)
    except (TypeError, ValueError):
        log.warning(
            "chatgpt-web: idle_shutdown_minutes=%r is not a number; "
            "falling back to %s minutes.", raw, DEFAULT_IDLE_SHUTDOWN_MINUTES,
        )
        return DEFAULT_IDLE_SHUTDOWN_MINUTES * 60.0
    return max(minutes, 0.0) * 60.0


def reap_if_idle() -> bool:
    """Close the bridge if it has been idle past the configured timeout.

    Safe to call on any timer: it is a no-op unless the bridge is running AND
    provably unused (no lease held). Returns whether it stopped the bridge.
    """
    return supervisor.stop_if_idle(idle_shutdown_seconds())


def is_loopback(base_url: str) -> bool:
    return (urlparse(base_url).hostname or "") in _LOOPBACK_HOSTS


def bridge_token(s: BridgeSettings | None = None) -> str | None:
    """The bridge bearer token. Never log the return value."""
    token = cfg.get_secret("chatgpt_web_token", "CHATGPT_WEB_PROVIDER_TOKEN")
    if token:
        return token
    s = s or settings()
    if s.home is None:
        return None
    try:
        data = json.loads((s.home / "config.json").read_text(encoding="utf-8-sig"))
    except FileNotFoundError:  # an unconfigured bridge has no token yet
        return None
    except (OSError, ValueError) as exc:
        log.warning("chatgpt-web: bridge config.json unreadable: %s", type(exc).__name__)
        return None
    token = str(data.get("apiToken") or "").strip()
    return token or None


class BridgeSupervisor:
    """One process, one lock, bounded restarts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._starts: deque[float] = deque()
        self._started_at: float | None = None
        self._session_invalid = False
        # Idle accounting for the reaper. ``_leases`` counts turns currently
        # streaming through the bridge; ``_released_at`` is when the last one
        # finished. stop() itself kills unconditionally, so a timer that calls
        # it without consulting these would cut a live answer in half.
        self._leases = 0
        self._released_at: float | None = None
        self._reaper: threading.Thread | None = None

    # ---- idle accounting ---------------------------------------------------

    @contextlib.contextmanager
    def lease(self) -> Iterator[None]:
        """Hold the bridge open for one turn.

        Every path that sends a request through the bridge takes a lease, so
        :meth:`stop_if_idle` can tell "nobody is using it" from "an answer is
        still streaming". Released in a ``finally``, so an abort, a timeout or
        a raised error frees it too — a leaked lease would pin the browser
        open forever, which is the failure mode this whole feature removes.
        """
        with self._lock:
            self._leases += 1
        try:
            yield
        finally:
            with self._lock:
                self._leases = max(self._leases - 1, 0)
                if self._leases == 0:
                    self._released_at = time.monotonic()

    def in_use(self) -> bool:
        """True while at least one turn is streaming through the bridge."""
        with self._lock:
            return self._leases > 0

    def idle_seconds(self) -> float | None:
        """Seconds since the last turn finished, or None if that is unknown.

        None means "do not reap": either a turn is in flight, or the bridge has
        been started but has not served a turn yet, and guessing an idle age
        for it would let the reaper close a bridge the user just opened.
        """
        with self._lock:
            if self._leases > 0 or self._released_at is None:
                return None
            return time.monotonic() - self._released_at

    def start_reaper(self) -> None:
        """Start the idle watcher, once, for as long as the bridge runs.

        Deliberately owned by the bridge rather than the boot sequence: it
        exists only while there is a bridge to close, so nothing initialises on
        the boot critical path (AP-26) and an install that never asks a complex
        question never runs this at all. A daemon thread nobody joins, like the
        realtime supervisor's, so it cannot hold shutdown up.
        """
        with self._lock:
            if self._reaper is not None and self._reaper.is_alive():
                return
            thread = threading.Thread(
                target=self._reap_loop, name="chatgpt-web-idle-reaper", daemon=True
            )
            self._reaper = thread
        thread.start()

    def _reap_loop(self) -> None:
        while True:
            time.sleep(REAP_POLL_S)
            try:
                if self._started_at is None:
                    return  # the bridge is gone; this watcher has no job left
                if reap_if_idle():
                    return
            except Exception:  # noqa: BLE001 — a watcher must not die silently
                log.warning("chatgpt-web: idle reaper check failed", exc_info=True)

    def stop_if_idle(self, idle_timeout_s: float) -> bool:
        """Stop the bridge only if it is provably idle. Returns whether it did.

        Refuses while a lease is held, so a running turn is never interrupted.
        A non-positive timeout disables reaping entirely.
        """
        if idle_timeout_s <= 0:
            return False
        idle = self.idle_seconds()
        if idle is None or idle < idle_timeout_s:
            return False
        with self._lock:
            # Re-check under the lock: a turn may have taken a lease between
            # the idle_seconds() read and here, and that turn must not die.
            if self._leases > 0:
                return False
            if self._started_at is None:
                return False  # already stopped; nothing to do
            self._kill(settings())
            self._started_at = None
            self._released_at = None
        log.info(
            "chatgpt-web: bridge closed after %.0f s idle; the next complex "
            "question starts it again.", idle,
        )
        return True

    # ---- paths -------------------------------------------------------------

    @staticmethod
    def _run_dir(s: BridgeSettings) -> Path | None:
        return s.home / "run" if s.home else None

    def _pid_file(self, s: BridgeSettings) -> Path | None:
        run = self._run_dir(s)
        return run / "bridge.pid" if run else None

    def _read_pid(self, s: BridgeSettings) -> int | None:
        pid_file = self._pid_file(s)
        if pid_file is None:
            return None
        try:
            return int(pid_file.read_text(encoding="utf-8").strip())
        except FileNotFoundError:  # no PID file means nothing was started
            return None
        except (OSError, ValueError) as exc:
            log.warning("chatgpt-web: stale pid file ignored: %s", exc)
            return None

    @staticmethod
    def _pid_alive(pid: int | None) -> bool:
        if not pid:
            return False
        import psutil  # noqa: PLC0415 — lazy (AP-26)

        try:
            proc = psutil.Process(pid)
            return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        except psutil.Error:  # a vanished or foreign PID is not our bridge
            return False

    # ---- status ------------------------------------------------------------

    @staticmethod
    def login_saved(s: BridgeSettings) -> bool:
        """The bridge's own verified-login marker; ``serve`` exits without it."""
        if s.home is None:
            return False
        marker = s.home / "browser" / "storage-state.json.verified.json"
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
        except FileNotFoundError:  # never signed in
            return False
        except (OSError, ValueError) as exc:
            log.warning("chatgpt-web: login marker unreadable: %s", type(exc).__name__)
            return False
        return isinstance(data, dict) and data.get("authenticated") is True

    def mark_session_invalid(self) -> None:
        self._session_invalid = True

    def status(self) -> BridgeState:
        s = settings()
        if not is_loopback(s.base_url):
            return BridgeState("failed", "The bridge URL must be a loopback address.")
        if s.cli is None or not s.cli.is_file():
            return BridgeState("not_installed", "The ChatGPT bridge is not installed.")
        if s.home is None or not (s.home / "config.json").is_file():
            return BridgeState("not_configured", "The ChatGPT bridge has no home configured.")
        token = bridge_token(s)
        if not token:
            return BridgeState("not_configured", "The ChatGPT bridge has no access token.")
        client = _probe_pool.client()
        try:
            client.get(f"{s.base_url}/healthz").raise_for_status()
        except Exception as exc:  # noqa: BLE001 — any probe failure is "not up"
            log.debug("chatgpt-web: healthz failed: %s", type(exc).__name__)
            if self._pid_alive(self._read_pid(s)) and self._within_grace():
                return BridgeState("starting", "The ChatGPT bridge is starting.")
            if not self.login_saved(s):
                return BridgeState("login_required", "Sign in to ChatGPT in the bridge browser.")
            if self._exhausted():
                return BridgeState("failed", "The ChatGPT bridge keeps stopping; restarts paused.")
            return BridgeState("offline", "The ChatGPT bridge is not running.")
        try:
            ready = client.get(
                f"{s.base_url}/readyz", headers={"Authorization": f"Bearer {token}"}
            )
        except Exception as exc:  # noqa: BLE001 — up a moment ago, gone now
            log.debug("chatgpt-web: readyz failed: %s", type(exc).__name__)
            return BridgeState("offline", "The ChatGPT bridge is not responding.")
        if ready.status_code == 401:
            return BridgeState("failed", "The bridge rejected the Dodaios access token.")
        if ready.status_code == 503:
            return BridgeState("login_required", "Sign in to ChatGPT in the bridge browser.")
        if ready.status_code != 200:
            return BridgeState("failed", f"The bridge answered HTTP {ready.status_code}.")
        if self._session_invalid:
            return BridgeState("session_invalid", "The ChatGPT session expired; sign in again.")
        return BridgeState("ready", "ChatGPT is ready.")

    def _within_grace(self) -> bool:
        started = self._started_at
        return started is not None and time.monotonic() - started < STARTUP_GRACE_S

    def _exhausted(self) -> bool:
        now = time.monotonic()
        while self._starts and now - self._starts[0] > START_WINDOW_S:
            self._starts.popleft()
        return len(self._starts) >= MAX_STARTS

    # ---- lifecycle ---------------------------------------------------------

    def _env(self, s: BridgeSettings) -> dict[str, str]:
        env = dict(os.environ)
        env["CHATGPT_WEB_PROVIDER_HOME"] = str(s.home)
        # The bridge must never see a paid key it could fall back to.
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            env.pop(name, None)
        return env

    def _open_log(self, s: BridgeSettings, name: str) -> Any:
        assert s.log_dir is not None
        s.log_dir.mkdir(parents=True, exist_ok=True)
        path = s.log_dir / name
        if path.exists() and path.stat().st_size > LOG_CAP_BYTES:
            path.replace(path.with_suffix(path.suffix + ".1"))
        return path.open("ab")

    def start(self) -> BridgeState:
        """Start the bridge unless it already runs. Idempotent, bounded."""
        with self._lock:
            state = self.status()
            if state.status in ("ready", "login_required", "starting", "session_invalid"):
                return state
            if state.status in ("not_installed", "not_configured"):
                return state
            s = settings()
            if self._pid_alive(self._read_pid(s)):
                # Our process lives but does not answer: replace it once.
                self._kill(s)
            if self._exhausted():
                return BridgeState("failed", "The ChatGPT bridge keeps stopping; restarts paused.")
            runtime = resolve_executable(s.runtime) or s.runtime
            log_file = self._open_log(s, "bridge.log")
            try:
                proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
                    [runtime, str(s.cli), "serve"],
                    cwd=str(s.home),
                    env=self._env(s),
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=NO_WINDOW_CREATIONFLAGS,
                )
            except OSError as exc:
                log.error("chatgpt-web: bridge failed to start: %s", exc)
                return BridgeState("failed", "The ChatGPT bridge could not be started.")
            finally:
                log_file.close()
            self._starts.append(time.monotonic())
            self._started_at = time.monotonic()
            self._session_invalid = False
            pid_file = self._pid_file(s)
            assert pid_file is not None
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(str(proc.pid), encoding="utf-8")
            log.info("chatgpt-web: bridge started (pid %d)", proc.pid)
            return BridgeState("starting", "The ChatGPT bridge is starting.")

    def wait_until_up(self, timeout_s: float = STARTUP_GRACE_S) -> BridgeState:
        deadline = time.monotonic() + timeout_s
        state = self.status()
        while state.status == "starting" and time.monotonic() < deadline:
            time.sleep(0.5)
            state = self.status()
        return state

    def ensure_running(self) -> BridgeState:
        state = self.status()
        if state.status in ("offline", "failed"):
            self.start()
            state = self.wait_until_up()
        elif state.status == "starting":
            state = self.wait_until_up()
        if state.status == "ready":
            # The watcher only makes sense once there is a running bridge, and
            # it is a no-op when idle_shutdown_minutes is 0.
            self.start_reaper()
        return state

    def _kill(self, s: BridgeSettings) -> None:
        pid = self._read_pid(s)
        if self._pid_alive(pid):
            import psutil  # noqa: PLC0415 — lazy (AP-26)

            try:
                proc = psutil.Process(pid)
                for child in proc.children(recursive=True):
                    child.kill()
                proc.kill()
                proc.wait(timeout=10)
            except psutil.Error as exc:
                log.warning("chatgpt-web: could not stop bridge pid %s: %s", pid, exc)
        pid_file = self._pid_file(s)
        if pid_file is not None:
            pid_file.unlink(missing_ok=True)

    def stop(self) -> BridgeState:
        with self._lock:
            self._kill(settings())
            self._started_at = None
        return BridgeState("offline", "The ChatGPT bridge is stopped.")

    def restart(self) -> BridgeState:
        self.stop()
        self.start()
        return self.wait_until_up()

    def login(self) -> BridgeState:
        """Open the bridge's own headed browser profile for a one-time sign-in."""
        s = settings()
        if s.cli is None or s.home is None:
            return self.status()
        runtime = resolve_executable(s.runtime) or s.runtime
        log_file = self._open_log(s, "login.log")
        try:
            subprocess.Popen(  # noqa: S603 — fixed argv, no shell
                [runtime, str(s.cli), "login"],
                cwd=str(s.home),
                env=self._env(s),
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=NO_WINDOW_CREATIONFLAGS,
            )
        finally:
            log_file.close()
        self._session_invalid = False
        return BridgeState("login_required", "Finish the ChatGPT sign-in in the opened browser.")


supervisor = BridgeSupervisor()
