"""Google agent CLI auth service — status / login / logout.

Reports an honest snapshot of the official Google CLI login (Antigravity ``agy``
or the Gemini CLI) used to bill the Brain/Subagents against the user's Google
subscription. Mirrors :mod:`jarvis.codex_auth` for the Google side.

Cross-platform (CLOUD.md Rule #1): pure stdlib, ``pathlib``-only, honors
``$GEMINI_HOME``, degrades to a clean "not installed" snapshot when no binary
resolves. **No token value is ever read into business logic or logged** — only
on-disk login-file presence, the public ``selectedType``, and (for agy 1.5+
keyring logins) a capability probe via ``agy models`` decide the connection
state (Google ToS: never scrape the token into our own client; AP-21).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.core.interactive_terminal import (
    InteractiveTerminalLaunch,
    InteractiveTerminalUnavailable,
    launch_interactive_terminal,
)
from jarvis.core.process_utils import NO_WINDOW_CREATIONFLAGS
from jarvis.google_cli.agy_models import parse_agy_models
from jarvis.google_cli.resolver import GoogleCli, resolve_google_cli

log = logging.getLogger(__name__)

def antigravity_install_command(platform: str | None = None) -> str:
    """Official Antigravity installer for the current OS."""
    target = platform or sys.platform
    if target == "win32":
        return "irm https://antigravity.google/cli/install.ps1 | iex"
    return "curl -fsSL https://antigravity.google/cli/install.sh | bash"


def antigravity_install_hint(platform: str | None = None) -> str:
    """Display-safe Antigravity command plus the cross-platform Gemini fallback."""
    return (
        f"Install Antigravity with: {antigravity_install_command(platform)} "
        "(Gemini CLI alternative: npm i -g @google/gemini-cli)."
    )


def _gemini_home() -> Path:
    override = os.environ.get("GEMINI_HOME")
    return Path(override) if override else (Path.home() / ".gemini")


# Known on-disk OAuth login files, relative to ``~/.gemini``. The Gemini CLI
# writes ``oauth_creds.json``; the Antigravity ``agy`` CLI (the official
# successor) writes its token under a dedicated subdir instead — verified live
# 2026-06-26 with agy 1.0.12: ``~/.gemini/antigravity-cli/antigravity-oauth-token``
# (JSON ``{auth_method, token}``). Reading only ``oauth_creds.json`` made a
# fully-logged-in agy report "Installed but not logged in". Presence + non-empty
# only — the token VALUE is never read into business logic (Google ToS).
_OAUTH_LOGIN_FILES: tuple[tuple[str, ...], ...] = (
    ("oauth_creds.json",),
    ("antigravity-cli", "antigravity-oauth-token"),
)


def _oauth_login_paths(home: Path) -> tuple[Path, ...]:
    """Absolute paths of every known Google CLI OAuth login file under ``home``."""
    return tuple(home.joinpath(*rel) for rel in _OAUTH_LOGIN_FILES)


def _oauth_login_present(home: Path) -> bool:
    """True when ANY known OAuth login file exists and is non-empty.

    Covers both the Gemini CLI and the Antigravity ``agy`` CLI token locations.
    A zero-byte file (interrupted login) does not count as connected.
    """
    for path in _oauth_login_paths(home):
        try:
            if path.is_file() and path.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


# agy 1.5+ keeps its OAuth login in the OS keyring and writes none of the files
# above (verified live 2026-11-10 with agy 1.5.4: ``ChainedAuth: authenticated
# via keyring``). Presence of a Credential Manager target would still guess
# where the credential lives (upstream #1029). Instead we ask the CLI for a
# capability: a non-empty ``agy models`` list means the account can fetch
# models, which requires a live personal login (AP-21). Exit 0 with an empty
# list is the unauthenticated case — so exit code alone is not the signal.
_AGY_AUTH_PROBE_TTL_S = 30.0
_AGY_AUTH_PROBE_TIMEOUT_S = 8.0
_agy_auth_probe_lock = threading.Lock()
_agy_auth_probe_cache: dict[str, tuple[float, bool]] = {}

# Windows Credential Manager target used by agy for the keyring login. Used
# only on logout so Disconnect clears the entry; status never reads it.
_AGY_KEYRING_TARGET = "gemini:antigravity"
_CRED_TYPE_GENERIC = 1
_ERROR_NOT_FOUND = 1168


def clear_agy_auth_probe_cache() -> None:
    """Drop cached ``agy models`` auth probes (tests / login or logout)."""
    with _agy_auth_probe_lock:
        _agy_auth_probe_cache.clear()


def _agy_auth_probe_cache_key(cli: GoogleCli) -> str:
    return "\0".join(cli.argv_prefix) + "\0" + os.environ.get("GEMINI_HOME", "")


def _probe_agy_subscription_via_models(cli: GoogleCli) -> bool:
    """True when ``agy models`` returns a non-empty model list.

    Cached with a short TTL: the status endpoint is polled by the UI and the
    command hits the network (~seconds). Failures (missing binary, timeout,
    non-zero exit, empty list) all report not-authenticated — never a crash
    and never a false connected.
    """
    if cli.kind != "agy":
        return False
    key = _agy_auth_probe_cache_key(cli)
    now = time.monotonic()
    with _agy_auth_probe_lock:
        cached = _agy_auth_probe_cache.get(key)
        if cached is not None and now - cached[0] < _AGY_AUTH_PROBE_TTL_S:
            return cached[1]

    authenticated = False
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv from resolved CLI
            [*cli.argv_prefix, "models"],
            capture_output=True,
            timeout=_AGY_AUTH_PROBE_TIMEOUT_S,
            encoding="utf-8",
            errors="replace",
            text=True,
            creationflags=NO_WINDOW_CREATIONFLAGS,
        )
        if proc.returncode == 0:
            authenticated = bool(parse_agy_models(proc.stdout or ""))
        else:
            log.debug(
                "agy models auth probe exited %s (stderr=%r)",
                proc.returncode,
                (proc.stderr or "")[:200],
            )
    except subprocess.TimeoutExpired as exc:
        log.warning(
            "agy models auth probe timed out after %.1fs: %s",
            _AGY_AUTH_PROBE_TIMEOUT_S,
            exc,
        )
    except OSError as exc:
        log.warning("agy models auth probe failed to spawn: %s", exc)
    except Exception as exc:  # noqa: BLE001 — probe degrades, never raises into status
        log.warning("agy models auth probe unexpected failure: %s", exc)

    with _agy_auth_probe_lock:
        _agy_auth_probe_cache[key] = (now, authenticated)
    return authenticated


def _advapi32() -> Any:
    import ctypes

    return ctypes.WinDLL("advapi32", use_last_error=True)


def _delete_agy_keyring_login() -> None:
    """Remove agy's Credential Manager login entry; absent is not an error."""
    if sys.platform != "win32":
        return
    import ctypes

    advapi32 = _advapi32()
    if advapi32.CredDeleteW(ctypes.c_wchar_p(_AGY_KEYRING_TARGET), _CRED_TYPE_GENERIC, 0):
        return
    err = ctypes.get_last_error()
    if err != _ERROR_NOT_FOUND:
        raise OSError(err, f"could not remove the {_AGY_KEYRING_TARGET} credential")


def _derive_google_auth(
    *, creds_present: bool, settings: dict[str, Any]
) -> tuple[bool, str]:
    """Return ``(connected, mode)`` from the on-disk Gemini login state.

    * OAuth creds on disk -> ``(True, "oauth-personal")`` — the Google
      subscription path (the explicit ``selectedType`` only confirms it).
    * No creds but ``selectedType`` names an API-key/Vertex auth
      -> ``(True, "api_key")``.
    * Neither -> ``(False, "unknown")``.
    """
    sel = None
    if isinstance(settings, dict):
        sel = settings.get("security", {}).get("auth", {}).get("selectedType")
    if creds_present:
        return True, "oauth-personal"
    if sel in ("gemini-api-key", "vertex-ai"):
        return True, "api_key"
    return False, "unknown"


def _email_from_accounts(home: Path) -> str | None:
    """Active Google account email from ``google_accounts.json`` (display only)."""
    try:
        data = json.loads((home / "google_accounts.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    active = data.get("active") if isinstance(data, dict) else None
    return active if isinstance(active, str) and active else None


#: Every ``GoogleCliAuthStatus.message_code``. The desktop UI translates each
#: one (``apikeys_antigravity.status_msg.<code>``);
#: ``tests/unit/ui/test_backend_message_codes.py`` keeps both sides in step.
GOOGLE_CLI_STATUS_MESSAGE_CODES: tuple[str, ...] = (
    "not_installed",
    "not_logged_in",
    "connected_subscription",
    "connected_subscription_email",
    "connected_api_key",
)


@dataclass(frozen=True)
class GoogleCliAuthStatus:
    """Snapshot of the Google CLI login state for the UI + provider routes."""

    installed: bool = False
    connected: bool = False
    mode: str = "unknown"  # "oauth-personal" | "api_key" | "unknown"
    cli_kind: str | None = None  # "agy" | "gemini"
    message: str = ""
    version: str | None = None
    user_email: str | None = None
    binary_path: str = ""
    error: str | None = None
    # Stable code + fill-ins for ``message`` (``GOOGLE_CLI_STATUS_MESSAGE_CODES``)
    # so the desktop UI shows it in the user's language; ``message`` stays English.
    message_code: str = ""
    message_params: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "connected": self.connected,
            "mode": self.mode,
            "cli_kind": self.cli_kind,
            "message": self.message,
            "message_code": self.message_code,
            "message_params": dict(self.message_params),
            "version": self.version,
            "user_email": self.user_email,
            "binary_path": self.binary_path,
            "error": self.error,
        }


def antigravity_provider_ready(
    status: GoogleCliAuthStatus,
    *,
    api_key_present: bool,
) -> bool:
    """Whether the Antigravity CLI provider can be selected honestly.

    The separate Google Gemini provider owns key-only execution. Antigravity is
    a CLI provider, so even its optional API-key billing path must not paint the
    card ready or permit selection when no executable is installed.
    """
    oauth_connected = status.connected and status.mode == "oauth-personal"
    return status.installed and (oauth_connected or api_key_present)


class GoogleCliAuthService:
    """Status / login / logout for the official Google agent CLI.

    ``_resolve`` is split out as a seam so unit tests can stub binary discovery
    while exercising the real ``~/.gemini`` parsing against a temp ``$GEMINI_HOME``.
    """

    def _resolve(self) -> GoogleCli | None:
        # A CLI installed AFTER app start (or into a dir the GUI PATH never
        # had) must still be found — idempotent stat probes, no subprocess.
        try:
            from jarvis.core.path_augment import ensure_cli_paths

            ensure_cli_paths()
        except Exception as exc:  # noqa: BLE001 — probe failure must not break status
            log.debug("CLI PATH augmentation failed during Google discovery: %s", exc)
        return resolve_google_cli()

    def _read_json(self, name: str) -> dict[str, Any]:
        try:
            data = json.loads((_gemini_home() / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def status(self) -> GoogleCliAuthStatus:
        cli = self._resolve()
        if cli is None:
            return GoogleCliAuthStatus(
                message=f"No Google CLI found. {antigravity_install_hint()}",
                message_code="not_installed",
                message_params={"command": antigravity_install_command(None)},
                error="no google cli binary",
            )

        creds_present = _oauth_login_present(_gemini_home())
        connected, mode = _derive_google_auth(
            creds_present=creds_present, settings=self._read_json("settings.json")
        )
        # File absence is not enough for agy 1.5+: the login may live only in
        # the OS keyring. Ask the CLI whether it can list models (capability).
        if not connected and _probe_agy_subscription_via_models(cli):
            connected, mode = True, "oauth-personal"
        email = _email_from_accounts(_gemini_home()) if connected else None

        # Name WHAT was detected: the card is titled "Antigravity", but the
        # resolver also accepts a Gemini-CLI install (incl. a PATH-less npm
        # bundle). Without the kind in the message, a machine that never
        # installed agy shows a bare "Installed" and the user rightly asks
        # "installed WHAT?" (test-machine report 2026-07-18).
        kind_label = "Antigravity (agy)" if cli.kind == "agy" else "Gemini CLI"
        message_params: dict[str, str] = {"cli": kind_label}
        if not connected:
            message = f"{kind_label} installed but not logged in — run the Google login."
            message_code = "not_logged_in"
        elif mode == "oauth-personal":
            message = (
                f"Connected via Google subscription ({email})."
                if email
                else "Connected via Google subscription."
            )
            message_code = "connected_subscription_email" if email else "connected_subscription"
            if email:
                message_params["email"] = email
        else:
            message = "Connected via a Google API key."
            message_code = "connected_api_key"

        log.info(
            "google cli status: installed=True connected=%s mode=%s kind=%s",
            connected,
            mode,
            cli.kind,
        )
        return GoogleCliAuthStatus(
            installed=True,
            connected=connected,
            mode=mode,
            cli_kind=cli.kind,
            message=message,
            message_code=message_code,
            message_params=message_params,
            version=cli.version,
            user_email=email,
            binary_path=(cli.argv_prefix[0] if cli.argv_prefix else ""),
        )

    def start_login(self) -> InteractiveTerminalLaunch:
        """Spawn the official CLI login in a visible console. Raises if absent.

        Neither CLI has a dedicated ``login`` subcommand — verified 2026-06-21,
        ``agy login`` simply HANGS (it is not a real subcommand; ``agy help``
        lists only changelog/help/install/models/plugin/update). Both ``agy`` and
        the Gemini CLI drop into the interactive "Sign in with Google" flow on a
        bare run, so we launch the bare binary in a real external terminal. A
        headless host fails honestly instead of starting an invisible process.
        """
        cli = self._resolve()
        if cli is None:
            raise FileNotFoundError(
                f"No Google CLI found. {antigravity_install_hint()}"
            )
        # Next status must re-probe after the interactive login finishes.
        clear_agy_auth_probe_cache()
        argv = list(cli.argv_prefix)  # bare interactive run — no `login` subcommand
        log.info("Starting Google CLI login (interactive bare run, kind=%s)", cli.kind)
        try:
            return launch_interactive_terminal(argv, title="Google sign-in")
        except InteractiveTerminalUnavailable as exc:
            manual = "agy" if cli.kind == "agy" else "gemini"
            raise InteractiveTerminalUnavailable(
                f"{exc} Open a terminal and run: {manual}"
            ) from exc

    def logout_blocking(self) -> tuple[bool, str | None]:
        """Disconnect by removing every on-disk OAuth login file.

        Neither CLI has a ``logout`` subcommand (verified 2026-06-21) — ``agy
        logout`` would hang like ``agy login``. Removing the OAuth login files IS
        the disconnect; this clears BOTH the Gemini CLI's ``oauth_creds.json`` and
        the agy token under ``antigravity-cli/`` (else Disconnect is a no-op for an
        agy login). The isolated brain home re-syncs to the absent login on the
        next turn (see ``isolated_home``), so agy logs out there too. agy 1.5+
        logs in through the OS keyring instead, so its Credential Manager entry
        is removed as well (adapted from upstream #1029); the capability-probe
        cache is cleared so the next status does not reuse a stale "connected".

        Returns ``(ok, error)``; ``ok`` is True when the login files are gone.
        """
        try:
            for path in _oauth_login_paths(_gemini_home()):
                path.unlink(missing_ok=True)
            _delete_agy_keyring_login()
            clear_agy_auth_probe_cache()
            return True, None
        except OSError as exc:
            return False, str(exc)
