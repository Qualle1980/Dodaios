"""Status derivation for the Google CLI auth service (no real binary/network)."""
from __future__ import annotations

import json
import subprocess
import time

import pytest

import jarvis.google_cli.auth_service as auth_mod
from jarvis.google_cli.auth_service import (
    GoogleCliAuthService,
    _derive_google_auth,
)
from jarvis.google_cli.resolver import GoogleCli
from tests.fakes.fake_agy_models_run import FakeAgyModelsRun

# What a live ``agy models`` printed for a signed-in personal account (2026-11-10).
_AGY_MODELS_STDOUT = (
    "gemini-3.8-flash-high\tGemini 3.8 Flash (High)\n"
    "gemini-3.8-flash-medium\tGemini 3.8 Flash (Medium)\n"
    "gemini-3.8-flash-low\tGemini 3.8 Flash (Low)\n"
)


def _clear_probe_cache() -> None:
    clearer = getattr(auth_mod, "clear_agy_auth_probe_cache", None)
    if callable(clearer):
        clearer()


def _patch_agy_models_run(monkeypatch: pytest.MonkeyPatch, fake: FakeAgyModelsRun) -> None:
    """Install *fake* as ``auth_mod.subprocess.run``.

    On the fixed code the module owns a ``subprocess`` import. On origin/main it
    does not — we still attach a stand-in so the test collects and the
    connected/mode assertions fail honestly instead of erroring on setattr.
    """
    if getattr(auth_mod, "subprocess", None) is None:
        monkeypatch.setattr(auth_mod, "subprocess", subprocess, raising=False)
    monkeypatch.setattr(auth_mod.subprocess, "run", fake)


@pytest.fixture(autouse=True)
def _isolate_agy_side_effects(monkeypatch):
    """Keep every test off the developer's real agy binary and keyring.

    A machine with agy 1.5 signed in via the OS keyring would otherwise make
    ``status()`` report connected through the capability probe, and logout
    would delete the real Credential Manager entry.
    """
    _clear_probe_cache()
    _patch_agy_models_run(
        monkeypatch, FakeAgyModelsRun(raise_exc=FileNotFoundError("agy probe blocked"))
    )
    monkeypatch.setattr(
        auth_mod, "_delete_agy_keyring_login", lambda: None, raising=False
    )
    yield
    _clear_probe_cache()


def test_derive_oauth_personal():
    settings = {"security": {"auth": {"selectedType": "oauth-personal"}}}
    assert _derive_google_auth(creds_present=True, settings=settings) == (
        True,
        "oauth-personal",
    )


def test_derive_creds_without_type_still_connected():
    assert _derive_google_auth(creds_present=True, settings={}) == (
        True,
        "oauth-personal",
    )


def test_derive_api_key_type_without_creds():
    settings = {"security": {"auth": {"selectedType": "gemini-api-key"}}}
    assert _derive_google_auth(creds_present=False, settings=settings) == (
        True,
        "api_key",
    )


def test_derive_unknown_when_nothing():
    assert _derive_google_auth(creds_present=False, settings={}) == (False, "unknown")


def _seed_gemini_home(tmp_path):
    gem = tmp_path / ".gemini"
    gem.mkdir()
    (gem / "oauth_creds.json").write_text(
        json.dumps({"access_token": "x", "refresh_token": "y"})
    )
    (gem / "settings.json").write_text(
        json.dumps(
            {
                "security": {"auth": {"selectedType": "oauth-personal"}},
                "model": {"name": "gemini-3.1-pro-preview"},
            }
        )
    )
    (gem / "google_accounts.json").write_text(json.dumps({"active": "user@example.com"}))
    return gem


def test_status_connected(tmp_path, monkeypatch):
    gem = _seed_gemini_home(tmp_path)
    monkeypatch.setenv("GEMINI_HOME", str(gem))
    svc = GoogleCliAuthService()
    svc._resolve = lambda: GoogleCli(  # type: ignore[method-assign]
        kind="gemini", argv_prefix=["gemini"], version="0.47.0"
    )
    st = svc.status()
    assert st.installed and st.connected
    assert st.mode == "oauth-personal"
    assert st.cli_kind == "gemini"
    assert st.user_email == "user@example.com"
    assert "google subscription" in st.message.lower()
    assert st.message_code == "connected_subscription_email"
    assert st.message_params["email"] == "user@example.com"
    assert st.to_dict()["message_code"] == "connected_subscription_email"


def test_status_not_installed(monkeypatch):
    svc = GoogleCliAuthService()
    svc._resolve = lambda: None  # type: ignore[method-assign]
    st = svc.status()
    assert not st.installed and not st.connected
    d = st.to_dict()
    assert d["mode"] == "unknown"
    assert d["installed"] is False


def test_status_installed_but_logged_out(tmp_path, monkeypatch):
    gem = tmp_path / ".gemini"
    gem.mkdir()  # no oauth_creds.json, no agy token
    monkeypatch.setenv("GEMINI_HOME", str(gem))
    svc = GoogleCliAuthService()
    svc._resolve = lambda: GoogleCli(  # type: ignore[method-assign]
        kind="agy", argv_prefix=["agy"], version=None
    )
    st = svc.status()
    assert st.installed and not st.connected
    assert st.cli_kind == "agy"


def _seed_agy_home(tmp_path):
    """A real-world agy login: the OAuth token lives under ``antigravity-cli/``,
    NOT as ``oauth_creds.json`` (verified live 2026-06-26 with agy 1.0.12)."""
    gem = tmp_path / ".gemini"
    (gem / "antigravity-cli").mkdir(parents=True)
    (gem / "antigravity-cli" / "antigravity-oauth-token").write_text(
        json.dumps({"auth_method": "personal", "token": {"access_token": "x"}})
    )
    # agy shares ~/.gemini for the account + the (clean) selected-type marker.
    (gem / "settings.json").write_text(
        json.dumps({"security": {"auth": {"selectedType": "oauth-personal"}}})
    )
    (gem / "google_accounts.json").write_text(json.dumps({"active": "user@example.com"}))
    return gem


def test_status_connected_via_agy_oauth_token(tmp_path, monkeypatch):
    # The reported bug: agy is logged in (token under antigravity-cli/) but the
    # status read the wrong path (~/.gemini/oauth_creds.json) and showed
    # "Installed but not logged in". It must now report connected.
    gem = _seed_agy_home(tmp_path)
    monkeypatch.setenv("GEMINI_HOME", str(gem))
    svc = GoogleCliAuthService()
    svc._resolve = lambda: GoogleCli(  # type: ignore[method-assign]
        kind="agy", argv_prefix=["agy.exe"], version="1.0.12"
    )
    st = svc.status()
    assert st.installed and st.connected
    assert st.mode == "oauth-personal"
    assert st.cli_kind == "agy"
    assert st.user_email == "user@example.com"


def test_status_agy_empty_token_file_is_not_connected(tmp_path, monkeypatch):
    # A zero-byte token file (interrupted login) must not read as connected.
    gem = tmp_path / ".gemini"
    (gem / "antigravity-cli").mkdir(parents=True)
    (gem / "antigravity-cli" / "antigravity-oauth-token").write_text("")
    monkeypatch.setenv("GEMINI_HOME", str(gem))
    svc = GoogleCliAuthService()
    svc._resolve = lambda: GoogleCli(kind="agy", argv_prefix=["agy"])  # type: ignore[method-assign]
    assert not svc.status().connected


def test_status_connected_via_agy_models_probe_without_token_files(tmp_path, monkeypatch):
    # agy 1.5 stores the login in the OS keyring and writes no token file; the
    # capability probe (non-empty ``agy models``) is what proves the personal
    # subscription, and mode must be oauth-personal for subscription_connected.
    gem = tmp_path / ".gemini"
    gem.mkdir()
    monkeypatch.setenv("GEMINI_HOME", str(gem))
    fake = FakeAgyModelsRun(stdout=_AGY_MODELS_STDOUT)
    _patch_agy_models_run(monkeypatch, fake)
    svc = GoogleCliAuthService()
    svc._resolve = lambda: GoogleCli(  # type: ignore[method-assign]
        kind="agy", argv_prefix=["agy.exe"], version="1.5.4"
    )
    st = svc.status()
    assert st.installed and st.connected
    assert st.mode == "oauth-personal"
    assert st.cli_kind == "agy"
    assert st.message_code == "connected_subscription"
    assert fake.calls == [["agy.exe", "models"]]


def test_status_not_connected_when_agy_models_returns_empty_list(tmp_path, monkeypatch):
    # Exit 0 with an empty model list is the unauthenticated case (agy logs
    # "Auth mode is unspecified … returning empty response") — never "connected".
    gem = tmp_path / ".gemini"
    gem.mkdir()
    monkeypatch.setenv("GEMINI_HOME", str(gem))
    fake = FakeAgyModelsRun(stdout="")
    _patch_agy_models_run(monkeypatch, fake)
    svc = GoogleCliAuthService()
    svc._resolve = lambda: GoogleCli(kind="agy", argv_prefix=["agy"])  # type: ignore[method-assign]
    st = svc.status()
    assert st.installed and not st.connected
    assert st.mode == "unknown"
    assert st.message_code == "not_logged_in"
    # origin/main never probes; the fix must still invoke ``agy models``.
    assert fake.calls == [["agy", "models"]]


def test_status_not_connected_when_agy_binary_absent(tmp_path, monkeypatch):
    gem = tmp_path / ".gemini"
    gem.mkdir()
    monkeypatch.setenv("GEMINI_HOME", str(gem))
    fake = FakeAgyModelsRun(raise_exc=FileNotFoundError("agy"))
    _patch_agy_models_run(monkeypatch, fake)
    svc = GoogleCliAuthService()
    svc._resolve = lambda: GoogleCli(kind="agy", argv_prefix=["agy"])  # type: ignore[method-assign]
    st = svc.status()
    assert st.installed and not st.connected
    assert st.message_code == "not_logged_in"
    assert fake.calls == [["agy", "models"]]


def test_status_not_connected_when_agy_models_times_out(tmp_path, monkeypatch):
    gem = tmp_path / ".gemini"
    gem.mkdir()
    monkeypatch.setenv("GEMINI_HOME", str(gem))
    fake = FakeAgyModelsRun(
        raise_exc=subprocess.TimeoutExpired(cmd=["agy", "models"], timeout=8.0)
    )
    _patch_agy_models_run(monkeypatch, fake)
    svc = GoogleCliAuthService()
    svc._resolve = lambda: GoogleCli(kind="agy", argv_prefix=["agy"])  # type: ignore[method-assign]
    st = svc.status()
    assert st.installed and not st.connected
    assert st.message_code == "not_logged_in"
    assert fake.calls == [["agy", "models"]]


def test_agy_models_auth_probe_is_cached_within_ttl(tmp_path, monkeypatch):
    gem = tmp_path / ".gemini"
    gem.mkdir()
    monkeypatch.setenv("GEMINI_HOME", str(gem))
    fake = FakeAgyModelsRun(stdout=_AGY_MODELS_STDOUT)
    _patch_agy_models_run(monkeypatch, fake)
    # Keep the TTL window open for the second call without waiting wall time.
    monkeypatch.setattr(auth_mod, "_AGY_AUTH_PROBE_TTL_S", 60.0, raising=False)
    clock = {"t": 1000.0}
    if getattr(auth_mod, "time", None) is None:
        monkeypatch.setattr(auth_mod, "time", time, raising=False)
    monkeypatch.setattr(auth_mod.time, "monotonic", lambda: clock["t"])
    svc = GoogleCliAuthService()
    svc._resolve = lambda: GoogleCli(kind="agy", argv_prefix=["agy"])  # type: ignore[method-assign]

    assert svc.status().connected
    clock["t"] += 10.0  # still inside the TTL
    assert svc.status().connected
    assert len(fake.calls) == 1

    clock["t"] += 60.0  # past the TTL — must re-probe
    assert svc.status().connected
    assert len(fake.calls) == 2


def test_logout_clears_agy_keyring_login_and_probe_cache(tmp_path, monkeypatch):
    # Disconnect must clear the keyring entry and the capability-probe cache,
    # else a keyring login keeps reporting connected after Disconnect.
    gem = tmp_path / ".gemini"
    gem.mkdir()
    monkeypatch.setenv("GEMINI_HOME", str(gem))
    keyring = {"present": True}
    monkeypatch.setattr(
        auth_mod,
        "_delete_agy_keyring_login",
        lambda: keyring.update(present=False),
        raising=False,
    )
    fake = FakeAgyModelsRun(stdout=_AGY_MODELS_STDOUT)
    _patch_agy_models_run(monkeypatch, fake)
    svc = GoogleCliAuthService()
    svc._resolve = lambda: GoogleCli(kind="agy", argv_prefix=["agy.exe"])  # type: ignore[method-assign]
    assert svc.status().connected
    ok, err = svc.logout_blocking()
    assert ok and err is None
    assert keyring["present"] is False
    # After logout the probe cache is empty; with an empty models reply the
    # card must flip to not connected.
    fake.stdout = ""
    assert not svc.status().connected


def test_logout_removes_agy_token(tmp_path, monkeypatch):
    # Disconnect must clear the agy token too, not only oauth_creds.json — else
    # the Disconnect button is a no-op for an agy login.
    gem = _seed_agy_home(tmp_path)
    monkeypatch.setenv("GEMINI_HOME", str(gem))
    svc = GoogleCliAuthService()
    svc._resolve = lambda: GoogleCli(kind="agy", argv_prefix=["agy.exe"])  # type: ignore[method-assign]
    ok, err = svc.logout_blocking()
    assert ok and err is None
    assert not (gem / "antigravity-cli" / "antigravity-oauth-token").is_file()
    assert not svc.status().connected


def test_start_login_uses_bare_agy_not_login_subcommand(monkeypatch):
    # agy has NO `login` subcommand (verified 2026-06-21: `agy login` hangs
    # forever). The login is an interactive bare run, so start_login must NOT
    # append "login" — that would spawn a hung process behind the Connect button.
    svc = GoogleCliAuthService()
    svc._resolve = lambda: GoogleCli(kind="agy", argv_prefix=["agy.exe"])  # type: ignore[method-assign]
    captured: dict = {}

    def _fake_launch(argv, **kw):
        captured["argv"] = list(argv)
        captured["kwargs"] = kw
        return auth_mod.InteractiveTerminalLaunch(10, "test-terminal")

    monkeypatch.setattr(auth_mod, "launch_interactive_terminal", _fake_launch)
    svc.start_login()
    assert captured["argv"] == ["agy.exe"]  # bare binary, no "login"
    assert captured["kwargs"] == {"title": "Google sign-in"}


def test_provider_ready_requires_cli_even_with_api_key():
    missing = auth_mod.GoogleCliAuthStatus(installed=False)
    assert not auth_mod.antigravity_provider_ready(missing, api_key_present=True)

    installed = auth_mod.GoogleCliAuthStatus(installed=True)
    assert auth_mod.antigravity_provider_ready(installed, api_key_present=True)

    oauth = auth_mod.GoogleCliAuthStatus(
        installed=True,
        connected=True,
        mode="oauth-personal",
    )
    assert auth_mod.antigravity_provider_ready(oauth, api_key_present=False)


def test_install_command_is_platform_specific():
    assert auth_mod.antigravity_install_command("win32").startswith("irm ")
    assert auth_mod.antigravity_install_command("darwin").startswith("curl ")
    assert auth_mod.antigravity_install_command("linux").startswith("curl ")


def test_logout_removes_creds_without_calling_agy_logout(tmp_path, monkeypatch):
    # agy has NO `logout` subcommand either; removing the on-disk OAuth creds IS
    # the disconnect. Trying `agy logout` would hang/time out for nothing.
    gem = _seed_gemini_home(tmp_path)
    monkeypatch.setenv("GEMINI_HOME", str(gem))
    svc = GoogleCliAuthService()
    svc._resolve = lambda: GoogleCli(kind="agy", argv_prefix=["agy.exe"])  # type: ignore[method-assign]

    ok, err = svc.logout_blocking()
    assert ok and err is None
    assert not (gem / "oauth_creds.json").is_file()  # creds actually removed
