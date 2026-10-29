#!/usr/bin/env sh
# Personal Jarvis - POSIX launcher (macOS + Linux). Parity twin of run.bat.
#
# Usage: ./run.sh [--debug|--headless|--dev]
#   --debug     verbose logging, foreground console
#   --headless  API/WS backend only, no desktop window or voice
#   --dev       frontend served from the Vite dev server (port 5173)
#
# Runs in the foreground (Ctrl+C stops Jarvis); append `&` to detach.
set -u
cd "$(dirname "$0")" || exit 1

# Activate the project venv when there is one (mirrors run.bat).
#
# A linked git worktree has no .venv of its own -- the interpreter that carries
# the dependencies lives in the main checkout, so fall back to it. Activating it
# is safe: the working directory still wins on import, so the code that runs is
# THIS tree's (scripts/preflight.ps1 asserts exactly that). Never
# `pip install -e` from a worktree -- that pin is the user's desktop app and
# their next restart would boot this branch (AP-8 / BUG-219).
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . ".venv/bin/activate"
else
    jarvis_git_common=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)
    if [ -n "${jarvis_git_common:-}" ] && [ -f "${jarvis_git_common%/*}/.venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        . "${jarvis_git_common%/*}/.venv/bin/activate"
    fi
fi

# Pick the interpreter - Debian/Ubuntu ship python3 without a python alias.
if command -v python >/dev/null 2>&1; then
    PY=python
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    echo "error: no Python interpreter found on PATH" >&2
    exit 1
fi

# Pre-boot drift check (optional, needs PowerShell 7+). Mirrors run.bat:
# always continues on failure, never blocks boot, skipped without pwsh.
if [ -f "scripts/check-working-tree.ps1" ] && command -v pwsh >/dev/null 2>&1; then
    pwsh -NoProfile -File "scripts/check-working-tree.ps1" || true
fi

case "${1:-}" in
    --debug)
        # --debug is launcher-local: visible console + verbose logging.
        # Forwarding it to the launcher would trip argparse (see run.bat).
        JARVIS_DEBUG=1 exec "$PY" -m jarvis.ui.web.launcher
        ;;
    --headless)
        exec "$PY" -m jarvis.ui.web.launcher --headless
        ;;
    --dev)
        JARVIS_DEV=1 exec "$PY" -m jarvis.ui.web.launcher --dev
        ;;
    *)
        exec "$PY" -m jarvis.ui.web.launcher
        ;;
esac
