"""Session-mode gate for GitHub / git write commands.

Reads are always free. A recognised write needs Ahmad's explicit OK when:

* the session is in ``chat`` mode, or
* the session is in ``project`` mode and ``cwd`` is a main git checkout
  (``.git`` is a directory).

A project session whose ``cwd`` is a linked worktree (``.git`` is a file)
may write without that OK. Distinct from ``permission_mode`` / the runner
ladder: ``auto`` / ``bypass`` still stop here for these writes.
"""

from __future__ import annotations

import logging
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from jarvis.agent_chat.store import DEFAULT_SESSION_MODE

log = logging.getLogger(__name__)

#: Tool names that can carry a shell write.
_SHELL_TOOLS: Final[frozenset[str]] = frozenset({"RunCommand", "run_shell", "Bash"})

#: ``gh`` subcommands that mutate (token after ``gh``).
_GH_WRITE_TOP: Final[frozenset[str]] = frozenset(
    {
        "issue",
        "pr",
        "release",
        "repo",
        "api",
        "gist",
        "project",
        "label",
        "milestone",
        "workflow",
        "run",
        "secret",
        "variable",
        "ssh-key",
        "gpg-key",
        "alias",
        "extension",
        "ruleset",
        "attestation",
        "cache",
        "codespace",
        "org",
        "team",
        "sponsor",
    }
)

#: Read-only ``gh`` top-level commands (always free).
_GH_READ_TOP: Final[frozenset[str]] = frozenset(
    {
        "auth",
        "browse",
        "completion",
        "config",
        "help",
        "search",
        "status",
        "version",
        "preview",
    }
)

#: Mutating verbs under ``gh issue`` / ``gh pr`` / ``gh repo`` / ``gh release``.
_GH_WRITE_VERBS: Final[frozenset[str]] = frozenset(
    {
        "create",
        "edit",
        "close",
        "reopen",
        "delete",
        "comment",
        "merge",
        "ready",
        "review",
        "lock",
        "unlock",
        "pin",
        "unpin",
        "transfer",
        "develop",
        "mark",
        "upload",
        "download",  # release download is read; kept out below
        "publish",
        "set-default",
        "archive",
        "unarchive",
        "rename",
        "sync",
        "fork",
        "clone",  # clone writes a folder; treat as write for gate
        "deploy",
        "cancel",
        "rerun",
        "watch",  # watch is read-ish; leave as write-safe via allow-list of reads
        "enable",
        "disable",
        "update",
        "add",
        "remove",
        "set",
    }
)

#: Explicit read verbs under issue/pr/repo/release (free).
_GH_READ_VERBS: Final[frozenset[str]] = frozenset(
    {
        "list",
        "view",
        "status",
        "checks",
        "diff",
        "checks",
        "files",
        "commits",
        "view",
        "list",
        "download",  # release asset download
    }
)

#: HTTP methods that mutate via ``gh api``.
_MUTATING_METHODS: Final[frozenset[str]] = frozenset(
    {"POST", "PUT", "PATCH", "DELETE"}
)


@dataclass(frozen=True, slots=True)
class WriteGateDecision:
    """Outcome of :func:`evaluate_write_gate`."""

    is_write: bool
    needs_ok: bool
    reason: str = ""


def is_main_checkout(cwd: Path | str | None) -> bool:
    """True when ``cwd`` is a primary git checkout (``.git`` is a directory)."""
    root = _cwd_path(cwd)
    if root is None:
        return False
    git = root / ".git"
    try:
        return git.is_dir()
    except OSError as exc:
        log.warning("write_gate: cannot stat %s: %s", git, exc)
        return False


def is_linked_worktree(cwd: Path | str | None) -> bool:
    """True when ``cwd`` is a linked worktree (``.git`` is a file)."""
    root = _cwd_path(cwd)
    if root is None:
        return False
    git = root / ".git"
    try:
        return git.is_file()
    except OSError as exc:
        log.warning("write_gate: cannot stat %s: %s", git, exc)
        return False


#: Words that mark a script or one-liner as mutating a GitHub endpoint
#: (for example ``python -c "requests.post('https://api.github.com/...')"``).
_SCRIPT_WRITE_MARKERS: Final[tuple[str, ...]] = (
    "post",
    "put",
    "patch",
    "delete",
    "method",
    "-x ",
    "--data",
    "--request",
)


def command_is_write(command: str) -> bool:
    """Whether ``command`` is a recognised GitHub / git mutating action."""
    for segment in _segments(command):
        if _segment_is_write(segment):
            return True
    # Heuristic for scripts that call the GitHub API themselves.
    if _mentions_github_api(command):
        low = (command or "").lower()
        if any(marker in low for marker in _SCRIPT_WRITE_MARKERS):
            return True
    return False


def evaluate_write_gate(
    *,
    tool_name: str,
    args: dict[str, Any] | None,
    mode: str | None,
    cwd: Path | str | None,
) -> WriteGateDecision:
    """Decide whether this tool call needs Ahmad's explicit OK.

    Non-shell tools and read commands return ``needs_ok=False``.
    """
    name = (tool_name or "").strip()
    if name not in _SHELL_TOOLS:
        return WriteGateDecision(False, False)
    command = str((args or {}).get("command") or "")
    if not command.strip():
        return WriteGateDecision(False, False)
    if not command_is_write(command):
        return WriteGateDecision(False, False, "read or unknown non-write")

    session_mode = (mode or "").strip() or DEFAULT_SESSION_MODE
    if session_mode == "chat":
        return WriteGateDecision(
            True,
            True,
            "chat mode: GitHub/git writes need Ahmad's explicit OK",
        )
    if session_mode == "project" and is_main_checkout(cwd):
        return WriteGateDecision(
            True,
            True,
            "project mode on main checkout: writes need Ahmad's explicit OK",
        )
    # project + linked worktree, or project with no .git: allow
    return WriteGateDecision(
        True,
        False,
        "project mode on linked worktree (or non-main): write allowed",
    )


def needs_write_ok(
    tool_name: str,
    args: dict[str, Any] | None,
    *,
    mode: str | None,
    cwd: Path | str | None,
) -> bool:
    """Convenience wrapper for ``force_ask`` hooks."""
    return evaluate_write_gate(
        tool_name=tool_name, args=args, mode=mode, cwd=cwd
    ).needs_ok


def force_ask_for_session(session: Any) -> Any:
    """``force_ask(name, args)`` bound to a session's mode and cwd.

    Composes with society rules when the surface is ``society``.
    """
    mode = str(getattr(session, "mode", "") or DEFAULT_SESSION_MODE)
    cwd = getattr(session, "cwd", "") or ""
    surface = str(getattr(session, "surface", "") or "")

    def _write(name: str, args: dict[str, Any] | None) -> bool:
        return needs_write_ok(name, args, mode=mode, cwd=cwd)

    if surface != "society":
        return _write

    def _combined(name: str, args: dict[str, Any] | None) -> bool:
        if _write(name, args):
            return True
        try:
            from jarvis.society.surface import requires_explicit_approval
        except Exception as exc:  # noqa: BLE001 — log and fall through to write-only
            log.warning("write_gate: society approval lookup failed: %s", exc)
            return False
        return bool(requires_explicit_approval(session.session_id, name, args or {}))

    return _combined


def denial_message(reason: str) -> str:
    text = (reason or "").strip() or "This write needs Ahmad's explicit OK."
    return (
        f"Denied by write gate: {text} "
        "Ask for approval before retrying this GitHub/git write."
    )


# ---------------------------------------------------------------- internals


def _cwd_path(cwd: Path | str | None) -> Path | None:
    if cwd is None:
        return None
    text = str(cwd).strip()
    if not text:
        return None
    try:
        return Path(os.path.expanduser(text)).resolve()
    except OSError as exc:
        log.warning("write_gate: cannot resolve cwd %r: %s", text, exc)
        return None


def _segments(command: str) -> list[list[str]]:
    """Split a shell line into argv segments at ``|`` / ``&&`` / ``;``."""
    import re

    parts = re.split(r"\|\||&&|;|\|", command or "")
    out: list[list[str]] = []
    for part in parts:
        tokens = _tokenize(part)
        if tokens:
            out.append(tokens)
    return out


def _tokenize(part: str) -> list[str]:
    text = (part or "").strip()
    if not text:
        return []
    posix = sys.platform != "win32"
    try:
        return shlex.split(text, posix=posix)
    except ValueError:
        # Unbalanced quotes: fall back to whitespace so we still classify.
        log.debug("write_gate: shlex failed on %r; using whitespace split", text)
        return text.split()


def _segment_is_write(tokens: list[str]) -> bool:
    if not tokens:
        return False
    # Skip env assignments and common wrappers.
    i = 0
    while i < len(tokens) and "=" in tokens[i] and not tokens[i].startswith("-"):
        i += 1
    while i < len(tokens) and tokens[i].lower() in {
        "sudo",
        "time",
        "nohup",
        "env",
        "command",
        "builtin",
        "exec",
    }:
        i += 1
    if i >= len(tokens):
        return False
    head = tokens[i].lower()
    # Strip Windows path / .exe
    base = Path(head).name
    if base.endswith(".exe"):
        base = base[:-4]
    rest = tokens[i + 1 :]

    if base == "gh":
        return _gh_is_write(rest)
    if base == "git":
        return _git_is_write(rest)
    if base.lower() in _HTTP_CLIENTS:
        return _http_is_write(base.lower(), rest)
    return False


def _gh_is_write(args: list[str]) -> bool:
    if not args:
        return False
    # Flags before the subcommand: ``gh --repo x issue edit``
    i = 0
    while i < len(args):
        tok = args[i]
        if tok.startswith("-"):
            # flags that take a value
            if tok in {"-R", "--repo", "-h", "--hostname", "--jq", "-q", "--template", "-t"} or (
                tok.startswith("--") and "=" not in tok and i + 1 < len(args)
                and not args[i + 1].startswith("-")
            ):
                # consume value when present for known value-taking flags
                if tok in {"-R", "--repo", "--hostname", "--jq", "-q", "--template", "-t"} and i + 1 < len(
                    args
                ):
                    i += 2
                    continue
                if tok.startswith("--") and "=" not in tok and i + 1 < len(args) and not args[
                    i + 1
                ].startswith("-"):
                    i += 2
                    continue
            i += 1
            continue
        break
    if i >= len(args):
        return False
    top = args[i].lower()
    sub = [a.lower() for a in args[i + 1 :]]

    if top in _GH_READ_TOP:
        return False
    if top == "api":
        return _gh_api_is_write(args[i + 1 :])
    if top not in _GH_WRITE_TOP:
        # Unknown gh top-level: treat as write to be safe for gate purposes.
        return True

    # Find the verb after optional number / flags
    verb = ""
    for tok in sub:
        if tok.startswith("-"):
            continue
        if tok.isdigit():
            continue
        verb = tok
        break
    if not verb:
        # e.g. ``gh issue`` alone — not a write by itself
        return False
    if verb in _GH_READ_VERBS:
        return False
    if verb in _GH_WRITE_VERBS:
        return True
    # Unknown verb under a write-capable top: conservative write
    return True


def _gh_api_is_write(args: list[str]) -> bool:
    method = "GET"
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in {"-X", "--method"} and i + 1 < len(args):
            method = args[i + 1].upper()
            i += 2
            continue
        if tok.startswith("-X") and len(tok) > 2:
            method = tok[2:].upper()
            i += 1
            continue
        if tok.startswith("--method="):
            method = tok.split("=", 1)[1].upper()
            i += 1
            continue
        # --input / -f / -F / --raw-field imply a body → write
        if tok in {"--input", "-f", "-F", "--raw-field", "--field"} or tok.startswith(
            ("-f", "-F")
        ):
            return True
        i += 1
    return method in _MUTATING_METHODS


#: git global options that take a separate value (``git -C <path> push``).
_GIT_VALUE_OPTIONS: Final[frozenset[str]] = frozenset(
    {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}
)

#: git verbs that change repository state or the remote.
_GIT_WRITE_VERBS: Final[frozenset[str]] = frozenset(
    {
        "push",
        "commit",
        "merge",
        "rebase",
        "reset",
        "clean",
        "tag",
        "checkout",
        "switch",
        "stash",
        "cherry-pick",
        "revert",
        "am",
        "pull",
        "restore",
        "rm",
        "mv",
        "add",
        "worktree",
        "branch",
        "remote",
        "config",
        "gc",
        "prune",
    }
)


def _git_is_write(args: list[str]) -> bool:
    if not args:
        return False
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in _GIT_VALUE_OPTIONS:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        break
    if i >= len(args):
        return False
    verb = args[i].lower()
    if verb in _GIT_WRITE_VERBS:
        return True
    return False


#: HTTP clients whose mutating calls reach GitHub directly.
_HTTP_CLIENTS: Final[frozenset[str]] = frozenset(
    {"curl", "wget", "http", "https", "invoke-restmethod", "invoke-webrequest", "irm", "iwr"}
)

#: Flags that send a body, which makes a plain client call a write.
_HTTP_BODY_FLAGS: Final[frozenset[str]] = frozenset(
    {"-d", "--data", "--data-raw", "--data-binary", "--data-urlencode", "-F", "--form",
     "-T", "--upload-file", "--body", "--post-data", "--post-file"}
)


def _http_is_write(base: str, args: list[str]) -> bool:
    """A plain HTTP client call is a write when it sends a mutating method or a body."""
    for i, tok in enumerate(args):
        low = tok.lower()
        if low in {"-x", "--request", "-method", "-m"} and i + 1 < len(args):
            if args[i + 1].upper() in _MUTATING_METHODS:
                return True
        if low.startswith("-x") and len(tok) > 2 and tok[2:].upper() in _MUTATING_METHODS:
            return True
        if low.startswith("--request=") and tok.split("=", 1)[1].upper() in _MUTATING_METHODS:
            return True
        if tok in _HTTP_BODY_FLAGS or low.startswith(("--data", "--form", "--upload", "--body")):
            return True
        if low.startswith("-d") and len(tok) > 2 and base == "curl":
            return True
    return False


def _mentions_github_api(command: str) -> bool:
    low = (command or "").lower()
    return "api.github.com" in low or "github.com/api" in low


__all__ = [
    "WriteGateDecision",
    "command_is_write",
    "denial_message",
    "evaluate_write_gate",
    "is_linked_worktree",
    "is_main_checkout",
    "needs_write_ok",
]
