@echo off
REM Personal Jarvis — DEV INSTANCE launcher (Windows).
REM
REM Starts a SECOND desktop app beside the regular one, from this same checkout:
REM "Personal Jarvis Dev" — own data dir (data-dev\), own ports (+100), DEV-badged
REM icon, no wake word / global hotkeys / chat channels / autostart (those stay
REM with the regular app). Restart it as often as you like; the regular app and
REM the coding sessions inside it are never touched.
REM
REM Not the same as dev.bat / run.bat --dev (those load the frontend from a Vite
REM HMR server). This one runs the built frontend exactly like run.bat does.
REM Same as: set JARVIS_INSTANCE=dev && run.bat

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

if exist "scripts\check-working-tree.ps1" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\check-working-tree.ps1"
)

if "%1"=="--debug" (
    set JARVIS_DEBUG=1
    python -m jarvis.ui.web.launcher --instance dev
) else if "%1"=="--headless" (
    python -m jarvis.ui.web.launcher --instance dev --headless
) else (
    start "" pythonw -m jarvis.ui.web.launcher --instance dev
)

endlocal
