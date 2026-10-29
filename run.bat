@echo off
REM Personal Jarvis — Windows Launcher
REM Starts Jarvis in the user context (no administrator rights needed).

setlocal
cd /d "%~dp0"

REM Activate the project venv when there is one.
REM
REM A linked git worktree has no .venv of its own -- the interpreter that
REM carries the dependencies lives in the main checkout, so fall back to it.
REM Activating it is safe: the working directory still wins on import, so the
REM code that runs is THIS tree's (scripts\preflight.ps1 asserts exactly that).
REM Never 'pip install -e' from a worktree -- that pin is the user's desktop
REM app and their next restart would boot this branch (AP-8 / BUG-219).
set "JARVIS_ACTIVATE=.venv\Scripts\activate.bat"
if not exist "%JARVIS_ACTIVATE%" (
    for /f "delims=" %%G in ('git rev-parse --path-format^=absolute --git-common-dir 2^>nul') do set "JARVIS_GIT_COMMON=%%G"
)
if defined JARVIS_GIT_COMMON for %%I in ("%JARVIS_GIT_COMMON%\..") do set "JARVIS_ACTIVATE=%%~fI\.venv\Scripts\activate.bat"
if exist "%JARVIS_ACTIVATE%" call "%JARVIS_ACTIVATE%"

REM Pre-Boot-Drift-Check: restore any tracked files that vanished from
REM the working tree (see scripts/check-working-tree.ps1 docblock for the
REM 2026-05-14 incident this guards against). Always exits 0; never blocks
REM boot. Skipped silently when PowerShell is unavailable.
if exist "scripts\check-working-tree.ps1" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\check-working-tree.ps1"
)

REM Start the Jarvis desktop app (FastAPI + pywebview + React).
REM --debug -> visible console, --headless -> backend only, no window,
REM --dev -> frontend from the Vite dev server (port 5173) instead of dist/.
if "%1"=="--debug" (
    REM --debug is batch-local: visible console + verbose logging.
    REM Passing it on to the launcher would trigger an argparse error.
    set JARVIS_DEBUG=1
    python -m jarvis.ui.web.launcher
) else if "%1"=="--headless" (
    python -m jarvis.ui.web.launcher --headless
) else if "%1"=="--dev" (
    set JARVIS_DEV=1
    python -m jarvis.ui.web.launcher --dev
) else (
    start "" pythonw -m jarvis.ui.web.launcher
)

endlocal
