# The dev instance — two desktop apps from one checkout

**Problem.** Seeing a backend change means restarting the desktop app, and a
restart takes the Agentic-IDE panes, their coding-agent sessions and the voice
stack down with it. Restoring them afterwards is the most annoying part of
working on the app from inside the app.

**Answer.** Run two desktop apps from the same working tree:

| | **Dodaios** (default instance) | **Dodaios Dev** (dev instance) |
|---|---|---|
| Purpose | The app you live in: voice, hotkeys, channels, the Agentic-IDE sessions you vibe-code from. Never restarted for a code change. | A second window you restart as often as you like to look at the current working tree. |
| Start | `run.bat` / `./run.sh`, the Start-Menu / Desktop shortcut, autostart | `run-dev.bat` / `./run-dev.sh`, the **Personal Jarvis Dev** shortcut, or `python -m jarvis.ui.web.launcher --instance dev` |
| Frontend | The built bundle; a rebuilt bundle is picked up by the window itself (`bundleWatch`) | Same built bundle, same pick-up — so UI changes show up in BOTH without a restart |
| Backend (Python) changes | Only after a restart — which you now do on the dev app | Restart from the tray / in-app Restart / the launcher as usual |
| Data directory | `data/` | `data-dev/` (created on first start, seeded once with `setup_state.json`, `identity_card.json`, `core_memory.json` and a snapshot of `chats.db`) |
| Ports | `[ui].admin_api_port` etc. as configured (47821 …) | every configured port **+100** (47921 …) |
| Config, keys, agent accounts, skills | shared (`jarvis.toml`, keyring, `%LOCALAPPDATA%\Jarvis` / `~/.jarvis`) | shared — same providers, same logins, so "everything works" |
| Window title / taskbar / tray | Dodaios, logo icon | Dodaios **Dev**, logo with a yellow **DEV** tag, own taskbar group, own tray entry; the sidebar shows a small **DEV** badge |
| Wake word, global hotkeys | as configured | **off** — the microphone and the key combos belong to the default app (the in-window mic button and chat still work) |
| Chat channels (Telegram/Discord), autostart | as configured | **off** — a second poller on the same bot token would collide; the dev app never touches the login autostart entry |
| On-screen overlay (Jarvis Bar / orb) | as configured | **off** — the dev app draws no bar or orb and its Settings refuse to switch the style (HTTP 409): both apps read ONE `jarvis.toml`, and a second bar on the same spot that someone then switched off once took the default app's bar away on its next restart (2026-08-23) |
| Agentic-IDE resume offer | `last_session.json` | `last_session.dev.json` — a restarted dev app never offers to resume panes that are alive in the default app |

## How it is selected

`JARVIS_INSTANCE=dev` in the environment, or `--instance dev` on the launcher
(which sets the variable before anything else is imported). The variable is a
*single* underscore on purpose: the relauncher refreshes every `JARVIS__*`
config override on an in-app restart, and the instance must survive exactly
that restart. Children — the relauncher, the branded re-exec, the unelevated
copy, the in-app restart chain — inherit it.

Only `default` and `dev` are known names; anything else fails the boot with a
sentence, because a typo must not start a second *default* app that fights the
live one over its lock and port.

`jarvis/core/instance.py` is the one place that knows what differs; everything
else asks it (`current_instance()`): `jarvis.core.config.DATA_DIR`,
`load_config` (port offsets, `memory.data_dir`), `jarvis.ui.icon_utils` (AUMID,
shortcut, branded exe, icon), `jarvis.ui.desktop_app.WINDOW_TITLE`, the tray,
`jarvis.setup.state`, the Agentic-IDE resume store, `SpeechPipeline` (wake word,
hotkeys), the Friends stack (channels), the launcher (autostart), and
`jarvis.ui.desktop_app.boot_overlay_style` + the overlay-style settings route
(on-screen overlay).

## Shortcuts

```
python scripts/install_shortcuts.py --dev
```

writes `Desktop\Personal Jarvis Dev.lnk` (DEV icon, `--instance dev`, its own
AppUserModelID). The first dev start also writes the Start-Menu entry
`Personal Jarvis Dev` through the same code path the default app uses, so the
taskbar button is named and iconed from the first launch.

The dev icon is rendered by `scripts/make_app_icon.py` into both icon homes
(`jarvis/assets/icons/` ships with the package, `assets/icons/` is the
build-tool copy).

## Known sharing

Two things are deliberately shared and fine: `jarvis.toml` (settings you change
in the dev app apply to the default app too — same user, same preferences) and
the per-user directory (agent accounts, skills, board, recents). A few
CWD-relative paths still point at `data/` from both instances — the flight
recorder, the wiki curator lock (a cross-process lock, so this is correct), the
review / self-mod audit logs, computer-use blobs. They are logs and locks, not
stores; none of them makes the two apps answer for each other.

## Running it from a linked worktree

A linked `git worktree` is the natural home for the dev instance: a second
checkout the default app never reads. Three things differ from the main
checkout, and each one bites once.

**The interpreter.** A worktree has no `.venv` of its own, so the launchers
fall back to the main checkout's through `git rev-parse --git-common-dir`.
Activating it re-pins nothing — the working directory still wins on import, so
the code that runs is the worktree's own, which `scripts/preflight.ps1`
asserts and reports as `jarvis imports from worktree`. Never `pip install -e`
from a worktree: that pin is the user's desktop app (AP-8 / BUG-219). The same
trap makes *test results* lie — a suite run on a bare PATH interpreter fails by
the dozen on missing optional dependencies, and those failures read exactly
like real regressions.

**The data directory.** `ensure_instance_data_dir` seeds `data-dev/` from
`<repo_root>/data`, and a worktree has none, so the instance comes up as a
blank install: first-run guide, no identity, no chats. Pass the main
checkout's data directory as `default_data_dir` once and it looks like the
product it is a second window of.

**The vault and the config.** `vault_root` is relative, so `/api/wiki/health`
reports `vault_root_source: repo_root` and the worktree's wiki starts nearly
empty; the configured read-only external sources are unaffected and still
answer recall. Give the worktree its own copy of `jarvis.toml` rather than
pointing `JARVIS_CONFIG` at the live one — the dev app writes settings, and
both instances otherwise share a single file.

A workspace needs at least one agent pane; a request with an empty terminal
list is refused. A freshly opened pane stays `pending` until the UI attaches,
so a pane in the API is not yet a running CLI.

## Cross-platform

Built OS-neutral; the Windows identity layer is the one verified live. See
`docs/os-parity.md` P-37 for macOS (no separate bundle, dock icon follows the
default app) and Linux (own `.desktop` entry, unverified live).
