"""Codex-fleet actions shared by voice and REST entry points."""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

READY_TIMEOUT_S = 45.0
READY_POLL_S = 0.25
#: How long after its colour question was answered such a CLI gets to read the
#: answer. The reply is written before the app even sees the question, so this
#: only covers the CLI's own parsing.
COLOUR_PROBE_SETTLE_S = 0.3
#: How long a pane that declares the colour question may go without asking it
#: before readiness stops waiting — a CLI version that dropped the question, or
#: an adopted agent whose question was cut out of the replay.
COLOUR_PROBE_WAIT_S = 10.0
#: How long a starting CLI must have stopped asking its terminal questions —
#: and have shown its input line — before a prompt is typed. Covers the
#: questions answered by the viewer one round trip later, and Claude Code's
#: version question about a second after its composer appears.
STARTUP_QUIET_S = 1.5
#: How long after its input line appeared (or, without one, after it
#: started) a pane is still "starting". Past it questions no longer hold a
#: prompt back, so a CLI that keeps asking cannot block typing for good.
STARTUP_WINDOW_S = 15.0
_INPUT_MARKERS = ("›", "❯", ">")
#: Box-drawing characters a TUI may put around its composer row. Grok draws
#: ``│ >   │``; Claude Code puts ``─`` rules above and below ``❯``. Matching
#: only at column 0 never sees those rows.
_BOX_BORDER_CHARS = frozenset("│┃║╎╭╮╰╯─")
#: Claude Code's live input row uses a no-break space after ``❯``; past prompts
#: in the transcript use a plain space, so the NBSP is part of the marker.
_INPUT_NBSP = "\u00a0"
log = logging.getLogger(__name__)


def unwrap_box_line(line: str) -> str:
    """Strip leading/trailing box-drawing borders so the marker is visible."""
    text = line.strip()
    while text and text[0] in _BOX_BORDER_CHARS:
        text = text[1:].lstrip()
    while text and text[-1] in _BOX_BORDER_CHARS:
        text = text[:-1].rstrip()
    return text


def _is_pure_border_line(line: str) -> bool:
    text = line.strip()
    return bool(text) and all(ch in _BOX_BORDER_CHARS or ch.isspace() for ch in text)


def match_input_marker(line: str, markers: tuple[str, ...] | None = None) -> str | None:
    """Return the text after a prompt marker, or ``None`` when the row is not one.

    Sees through box borders and treats a no-break space after ``❯`` as part of
    the marker (Claude Code's live composer).
    """
    text = unwrap_box_line(line)
    if not text:
        return None
    for marker in markers or _INPUT_MARKERS:
        if not text.startswith(marker):
            continue
        rest = text[len(marker) :]
        if marker == "❯" and rest.startswith(_INPUT_NBSP):
            rest = rest[len(_INPUT_NBSP) :]
        return rest.lstrip()
    return None


def bottom_input_content(
    lines: Sequence[str], markers: tuple[str, ...] | None = None
) -> str | None:
    """Content after the marker on the bottom-most live input row, or ``None``.

    Walks from the bottom and skips empty and pure-border rows. A marked row
    counts when it sits inside a box (``│ > … │``), which is how Grok draws its
    composer; a status line may follow it. An unboxed marked row counts only when
    it is the bottom content row, so a past ``> prompt`` above a plain status
    row does not count as ready.
    """
    content = [
        line for line in reversed(tuple(lines)) if line.strip() and not _is_pure_border_line(line)
    ]
    for index, line in enumerate(content[:4]):
        found = match_input_marker(line, markers)
        if found is None:
            continue
        if index == 0 or _is_boxed(line):
            return found
        return None
    return None


def _is_boxed(line: str) -> bool:
    text = line.strip()
    return len(text) >= 2 and text[0] in _BOX_BORDER_CHARS and text[-1] in _BOX_BORDER_CHARS


def ready_for_prompt(term: Any) -> bool:
    """Is this pane far enough along to be typed into?

    Every CLI observed so far swallows a prompt that arrives while it is still
    booting — the keystrokes go nowhere and the user is told the work was sent.
    So waiting for a real input line is the DEFAULT, and an entry only skips it
    by declaring so: one CLI has a launch-and-prompt path measured to be stable
    without the wait, and making it wait would slow down behaviour that already
    works.

    Getting that default the wrong way round is what a new provider cannot
    afford. Opting in silently means every CLI registered later loses its first
    prompt until somebody notices; opting out silently costs one CLI a little
    speed. The cheap mistake is the one worth defaulting to.
    """
    if getattr(term, "status", "") != "live" or not getattr(term, "pty_id", None):
        return False
    from jarvis.workspace import agents as workspace_agents

    spec = workspace_agents.get_agent(getattr(term, "agent", ""))
    line_visible = input_line_visible(term, spec)
    if line_visible:
        _note_input_line(term)
    if spec is not None and not spec.needs_input_line_wait:
        # Typing before the composer is safe for this CLI; typing into its
        # startup questions is not.
        return terminal_questions_settled(term)
    if not line_visible or not terminal_questions_settled(term):
        return False
    if spec is not None and spec.asks_colours_after_input_line:
        return colour_probe_settled(term)
    return True


def _note_input_line(term: Any) -> None:
    watch = getattr(term, "terminal_queries", None)
    if watch is not None:
        watch.note_input_line(time.time())


def terminal_questions_settled(term: Any, now: float | None = None) -> bool:
    """Has this starting CLI finished asking its terminal questions?

    Coding CLIs ask their terminal for colours, version, keyboard protocol
    and the like while they start, and read the answers from their keyboard
    input — some of them right up to, or just after, the moment their input
    line appears. A prompt typed into that window is read where an answer was
    expected, and the answer is then typed into the composer as text such as
    ``]10;rgb:…\\]11;rgb:…\\`` (maintainer report 2026-10-02, every coding
    agent). So a prompt waits until the input line has been up, and no
    question has come, for ``STARTUP_QUIET_S``. Only while the pane is
    starting (``STARTUP_WINDOW_S``): a pane without a watch, or one long past
    its start, is never held back.
    """
    watch = getattr(term, "terminal_queries", None)
    if watch is None:
        return True
    moment = time.time() if now is None else now
    line_at = getattr(watch, "input_line_at", None)
    began = line_at if line_at is not None else getattr(term, "started_at", None)
    if not began or moment - began >= STARTUP_WINDOW_S:
        return True
    for at in (getattr(watch, "asked_at", None), line_at):
        if at is not None and moment - at < STARTUP_QUIET_S:
            return False
    return True


def colour_probe_settled(term: Any, now: float | None = None) -> bool:
    """Has this pane's CLI asked for its colours and had time to read the reply?

    Codex paints its composer, then asks the terminal for its colours one to
    several seconds later — no quiet window can be sure to cover that — and
    reads the answer from its keyboard input. A prompt typed in between is
    read where the answer was expected, and the answer is then typed into the
    composer as ``]10;rgb:…\\]11;rgb:…\\`` (maintainer report 2026-10-02).
    A pane without a watch, or one that never asks within
    ``COLOUR_PROBE_WAIT_S`` of starting, is not held back.
    """
    watch = getattr(term, "terminal_queries", None)
    if watch is None:
        return True
    moment = time.time() if now is None else now
    asked_at = getattr(watch, "colour_asked_at", None)
    if asked_at is not None:
        return moment - asked_at >= COLOUR_PROBE_SETTLE_S
    started_at = getattr(term, "started_at", None)
    return not started_at or moment - started_at >= COLOUR_PROBE_WAIT_S


def input_line_visible(term: Any, spec: Any | None = None) -> bool:
    """Does the bottom of this pane's screen show the CLI's input line?

    The bare observation, without :func:`ready_for_prompt`'s opt-out. A CLI
    that safely accepts typed text before it has painted its composer is
    still LOADING until the composer is there — and "has it finished loading"
    is the question a cold-start slot asks, whatever the CLI's typing policy
    (BUG-189). ``spec`` may be passed in to save the registry lookup.
    """
    if getattr(term, "status", "") != "live" or not getattr(term, "pty_id", None):
        return False
    if spec is None:
        from jarvis.workspace import agents as workspace_agents

        spec = workspace_agents.get_agent(getattr(term, "agent", ""))
    transcript = getattr(term, "transcript", None)
    if transcript is None:
        return False
    try:
        # Read the visible bottom of the current screen when it is available.
        # Transcript history can contain old prompt sigils from the conversation
        # being resumed; treating one of those as the new input box is the same
        # early-write bug with a more convincing false positive.
        screen = getattr(transcript, "screen", None)
        display = getattr(screen, "display", None)
        lines = display()[-10:] if callable(display) else transcript.tail(10)
    except Exception:  # noqa: BLE001 - a test double may expose no real buffer
        return False
    markers = (spec.input_markers if spec else ()) or _INPUT_MARKERS
    if spec is not None and spec.requires_visible_input_cursor:
        # Codex renders the same sigil when its composer is disabled, but its
        # Ratatui frame hides the cursor in that state. The visible cursor is
        # placed back on the editable row only when key events are accepted.
        try:
            visible_cursor = getattr(screen, "visible_cursor", None)
            row_text = getattr(screen, "row_text", None)
            if visible_cursor is None or not callable(row_text):
                return False
            row, column = visible_cursor
            line = row_text(row)
        except Exception:  # noqa: BLE001 - malformed extensions fail closed
            return False
        if match_input_marker(line, markers) is None:
            return False
        # Column check still uses the raw row: borders shift the marker right,
        # and the cursor must sit after that on-screen glyph.
        unwrapped = unwrap_box_line(line)
        leading = line.find(unwrapped) if unwrapped else 0
        if leading < 0:
            leading = 0
        for marker in markers:
            if unwrapped.startswith(marker):
                marker_column = leading
                return column > marker_column
        return False
    return bottom_input_content(lines, markers) is not None


def _ready_for_prompt(term: Any) -> bool:
    """Backward-compatible private name for older callers and extensions."""
    return ready_for_prompt(term)


async def wait_for_prompt_ready(
    session: Any,
    names: Sequence[str],
    *,
    timeout_s: float = READY_TIMEOUT_S,
) -> tuple[str, ...]:
    """Wait for newly mounted panes to display their real input line."""
    return await _wait_for_panes(session, names, ready_for_prompt, timeout_s=timeout_s)


async def wait_for_input_line(
    session: Any,
    names: Sequence[str],
    *,
    timeout_s: float = READY_TIMEOUT_S,
) -> tuple[str, ...]:
    """Wait for panes to paint their input line — loading done, whatever the
    CLI's typing policy (see :func:`input_line_visible`)."""
    return await _wait_for_panes(session, names, input_line_visible, timeout_s=timeout_s)


async def _wait_for_panes(
    session: Any,
    names: Sequence[str],
    predicate: Any,
    *,
    timeout_s: float,
) -> tuple[str, ...]:
    wanted = tuple(dict.fromkeys(name for name in names if name))
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        ready = tuple(
            name
            for name in wanted
            if (term := session.find(name)) is not None and predicate(term)
        )
        if len(ready) == len(wanted) or time.monotonic() >= deadline:
            return ready
        await asyncio.sleep(READY_POLL_S)


async def close_agent_terminals(registry: Any, agent: str | None) -> list[Any]:
    """Close every matching terminal in the front workspace."""
    session = registry.session
    if session is None:
        return []
    names = [
        term.name
        for term in session.terminals
        if agent is None or term.agent == agent
    ]
    closed: list[Any] = []
    # Use the long-standing one-pane operation so this remains compatible with
    # registries that do not yet expose the newer batch convenience method.
    for name in names:
        try:
            closed.append(await registry.close_terminal(name))
        except Exception:  # noqa: BLE001 - continue closing the remaining fleet
            log.warning("Agentic IDE: could not close terminal %s", name, exc_info=True)
    return closed


def terminals_closed_event(
    session: Any,
    closed: Sequence[Any],
    *,
    source_layer: str,
) -> Any:
    """Build the typed bus event that refreshes workspace clients."""
    from jarvis.core.events import AgenticIdeTerminalsClosed

    return AgenticIdeTerminalsClosed(
        session_id=session.id,
        names=tuple(term.name for term in closed),
        folder=session.folder,
        source_layer=source_layer,
    )


__all__ = [
    "READY_POLL_S",
    "READY_TIMEOUT_S",
    "close_agent_terminals",
    "input_line_visible",
    "unwrap_box_line",
    "match_input_marker",
    "bottom_input_content",
    "ready_for_prompt",
    "terminals_closed_event",
    "wait_for_input_line",
    "wait_for_prompt_ready",
]
