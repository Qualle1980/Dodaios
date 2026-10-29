#!/usr/bin/env sh
# Personal Jarvis - DEV INSTANCE launcher (macOS + Linux). Parity twin of run-dev.bat.
#
# Starts a SECOND desktop app beside the regular one, from this same checkout:
# "Personal Jarvis Dev" - own data dir (data-dev/), own ports (+100), DEV-badged
# icon, no wake word / global hotkeys / chat channels / autostart (those stay
# with the regular app). Restart it as often as you like; the regular app and
# the coding sessions inside it are never touched.
#
# Usage: ./run-dev.sh [--debug|--headless]
# Same as: JARVIS_INSTANCE=dev ./run.sh
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

if command -v python >/dev/null 2>&1; then
    PY=python
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    echo "error: no Python interpreter found on PATH" >&2
    exit 1
fi

case "${1:-}" in
    --debug)
        JARVIS_DEBUG=1 exec "$PY" -m jarvis.ui.web.launcher --instance dev
        ;;
    --headless)
        exec "$PY" -m jarvis.ui.web.launcher --instance dev --headless
        ;;
    *)
        exec "$PY" -m jarvis.ui.web.launcher --instance dev
        ;;
esac
