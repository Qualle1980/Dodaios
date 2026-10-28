"""Reuse of per-task git worktrees instead of building one per task.

Measured on 2026-10-27, model-free, 3 runs per variant, same commit
(11 677 tracked files) on the project's HDD:

    fresh `git worktree add`   27.2 s median (18.9-53.6, cold is the top end)
    reuse of a clean worktree   0.67 s

The checkout is the dominant cost of mission setup, so a task that can be
handed an already-checked-out tree starts roughly forty times faster. Sparse
checkout was measured too (1.6 s) and rejected: its 368 files can be missing
exactly the file an unknown task needs, while reuse is both faster and
complete.

How a slot comes into existence: **nothing is pre-built.** A pool slot is a
worktree a finished task did not need any more, kept instead of removed
(:meth:`WorkspacePool.adopt`). There is no background warming and no boot
work (AP-26); the first mission after an install pays today's price, every
later one pays 0.67 s.

Safety, which is the whole difficulty here — a reused tree is reset with
``git reset --hard`` and ``git clean``, and those two commands pointed at the
wrong directory would delete the user's own uncommitted work:

* Every slot lives under ``pool_root``, a directory this class owns, placed
  OUTSIDE the mission outputs root. ``missions.cleanup.startup_sweep`` deletes
  any child of the outputs root that is older than its retention and holds no
  deliverables, with no name guard — a pool kept in there would eventually be
  ``rmtree``-d out from under git, leaving stale worktree registrations.
* No destructive command runs before :meth:`_validate_slot` has confirmed, for
  that one call, that the path is inside ``pool_root``, that ``workspace/.git``
  is the worktree LINK FILE (a main checkout and a lean repo both have a
  directory there, so this alone excludes them), and that git agrees the tree
  belongs to this repository.
* A slot is leased exclusively through an ``O_EXCL`` lease file holding the
  owner's pid, so two threads — and two app instances sharing one checkout —
  cannot be handed the same tree.
* A lease whose owning process is gone is NEVER reset and never reused. A dead
  owner means a crashed task whose results may never have been archived, so
  the slot is quarantined under its own name and left on disk for recovery.
* Any failure at all is answered with ``None``, and the caller falls back to a
  fresh ``git worktree add``. The pool is an optimisation; it may never be the
  reason a mission fails.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from jarvis.core.process_utils import NO_WINDOW_CREATIONFLAGS

logger = logging.getLogger(__name__)

#: Lease marker inside a slot directory (next to ``workspace/``, so it is
#: outside the worktree and can never show up in a ``git add -A`` capture).
LEASE_NAME = ".lease"

#: Slot directory name shape. Only a directory matching this is ever a
#: reuse candidate; a quarantined slot is renamed out of this shape on
#: purpose, so it is never picked up again.
_SLOT_RE = re.compile(r"^ws-\d{2}$")

#: Cap on every git call the pool makes. A reset of a clean tree is
#: sub-second; anything near this limit means the tree is wedged, and the
#: honest answer is to quarantine it rather than to keep waiting in the
#: setup window a mission is blocked on.
_GIT_TIMEOUT_S = 120.0


def resolve_pool_root(outputs_root: Path) -> Path:
    """Return the workspace-pool root for a given mission outputs root.

    A sibling of the outputs root rather than a child: see the module
    docstring on ``startup_sweep``.
    """
    outputs_root = outputs_root.resolve()
    return outputs_root.parent / "jarvis-agent-workspace-pool"


class WorkspacePool:
    """A small set of reusable task worktrees, leased one task at a time.

    Thread-safe. ``WorktreeManager`` calls it while holding the orchestrator's
    process-wide worktree lock, but the lease bookkeeping takes its own lock
    anyway: a lease outlives the create call by the whole length of the task,
    so it cannot be protected by the caller's lock.
    """

    def __init__(
        self,
        *,
        repo_root: Path,
        pool_root: Path,
        size: int,
    ) -> None:
        """
        Inputs:
            repo_root: the main checkout (the one holding ``.git``).
            pool_root: directory this pool owns exclusively.
            size: how many slots may exist. ``0`` disables the pool entirely.
        """
        self._repo_root = repo_root.resolve()
        self._pool_root = pool_root.resolve()
        self._size = max(0, int(size))
        self._lock = threading.Lock()
        #: workspace path -> slot directory, for the leases this process holds.
        self._leased: dict[Path, Path] = {}

    @property
    def enabled(self) -> bool:
        """True when the pool may hold slots at all."""
        return self._size > 0

    @property
    def size(self) -> int:
        """Configured slot count."""
        return self._size

    # --- Public API --------------------------------------------------------

    def owns(self, workspace: Path) -> bool:
        """True when ``workspace`` is a slot this process currently leases."""
        with self._lock:
            return self._resolve(workspace) in self._leased

    def acquire(self, *, base_ref: str, branch: str) -> Path | None:
        """Lease a reusable workspace reset to ``base_ref`` on ``branch``.

        Output:
            The ``workspace/`` path, ready for a worker — or ``None`` when no
            slot is free or usable, which means the caller must build a fresh
            worktree as before. Never raises.
        """
        if not self.enabled:
            return None
        for slot in self._candidate_slots():
            if not self._take_lease(slot):
                continue
            workspace = slot / "workspace"
            try:
                if not self._validate_slot(slot):
                    self._quarantine(slot, "validation failed")
                    continue
                self._reset_slot(workspace, base_ref=base_ref, branch=branch)
            except (subprocess.SubprocessError, OSError) as exc:
                # A tree that will not reset is not repaired here: a forced
                # retry is exactly how a half-broken worktree becomes a
                # silently wrong one. Set it aside and let the caller do a
                # fresh checkout.
                logger.warning(
                    "workspace pool: slot %s could not be prepared (%s) — "
                    "quarantined, falling back to a fresh checkout",
                    slot.name, exc,
                )
                self._quarantine(slot, "reset failed")
                continue
            with self._lock:
                self._leased[self._resolve(workspace)] = slot
            logger.info(
                "workspace pool: reusing slot %s at %s", slot.name, base_ref,
            )
            return workspace
        return None

    def adopt(self, workspace: Path) -> bool:
        """Keep a finished task's worktree as a pool slot instead of removing it.

        This is how the pool fills up: no slot is ever built on speculation.

        Inputs:
            workspace: the ``workspace/`` path of a finished task, already
                archived by the caller.

        Output:
            True when the worktree now belongs to the pool and must NOT be
            removed by the caller. False when it was left untouched and the
            caller's normal teardown still owes a removal. Never raises.
        """
        if not self.enabled:
            return False
        workspace = self._resolve(workspace)
        if self.owns(workspace):
            # Already a slot — this is a release, not an adoption.
            return self.release(workspace)
        slot = self._free_slot_name()
        if slot is None:
            return False
        try:
            if not self._is_worktree_link(workspace):
                # A lean workspace is a standalone repo, not a worktree of
                # this checkout. Reusing one would hand the next task another
                # mission's files with no git relationship to reset against.
                return False
            # `git worktree move` wants the destination's PARENT to exist and
            # the destination itself not to (verified: it answers "No such
            # file or directory" otherwise).
            slot.mkdir(parents=True, exist_ok=True)
            # A slot path whose directory is gone - quarantined earlier, or
            # deleted by something outside the app - stays REGISTERED in git,
            # and `worktree move` then refuses that destination for good
            # ("missing but already registered worktree", caught by the setup
            # benchmark on 2026-10-28). Prune forgets exactly those missing
            # entries and nothing else, so a slot name is reusable again.
            self._git(["git", "worktree", "prune"], cwd=self._repo_root)
            # `git worktree move` keeps git's registration correct. A plain
            # rename would leave the admin file pointing at the old path and
            # every later git call in the slot would fail.
            self._git(
                ["git", "worktree", "move", str(workspace), str(slot / "workspace")],
                cwd=self._repo_root,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            logger.info(
                "workspace pool: could not adopt %s (%s) — removing it normally",
                workspace, exc,
            )
            # A half-finished move must not leave an empty slot behind that a
            # later acquire would validate and reset.
            if slot.is_dir() and not (slot / "workspace").exists():
                shutil.rmtree(slot, ignore_errors=True)
            return False
        logger.info(
            "workspace pool: kept a finished worktree as slot %s (was at %s)",
            slot.name, workspace,
        )
        return True

    def release(self, workspace: Path) -> bool:
        """Drop the lease on a slot this process holds.

        The tree is deliberately left dirty: the next :meth:`acquire` resets
        it, so teardown — which a cancelled mission is waiting on — stays
        instant, and a crash between release and the next acquire leaves the
        worker's files on disk rather than deleting them.

        Output:
            True when a lease was dropped (the caller must not remove the
            directory), False when ``workspace`` is not a slot of this pool.
        """
        workspace = self._resolve(workspace)
        with self._lock:
            slot = self._leased.pop(workspace, None)
        if slot is None:
            return False
        lease = slot / LEASE_NAME
        try:
            lease.unlink()
        except FileNotFoundError:
            # The lease is already gone, which is exactly the state release wants.
            pass
        except OSError as exc:
            # The lease could not be dropped, so no other task may ever take
            # this slot again. Quarantining it is the only honest outcome.
            logger.warning(
                "workspace pool: lease of %s could not be released (%s) — "
                "quarantining the slot", slot.name, exc,
            )
            self._quarantine(slot, "lease stuck")
            return True
        logger.debug("workspace pool: released slot %s", slot.name)
        return True

    # --- Slot bookkeeping --------------------------------------------------

    def _candidate_slots(self) -> list[Path]:
        """Existing slot directories, in a stable order."""
        try:
            children = sorted(self._pool_root.iterdir())
        except OSError:
            # No pool directory yet means no slots; a first run is not an error.
            return []
        return [
            child for child in children
            if child.is_dir() and _SLOT_RE.match(child.name)
        ]

    def _free_slot_name(self) -> Path | None:
        """Return a slot path that does not exist yet, or ``None`` when full."""
        taken = {slot.name for slot in self._candidate_slots()}
        for index in range(self._size):
            name = f"ws-{index:02d}"
            if name not in taken:
                return self._pool_root / name
        return None

    def _take_lease(self, slot: Path) -> bool:
        """Claim ``slot`` exclusively, or report that someone else holds it.

        ``O_CREAT | O_EXCL`` is the whole mechanism: it is atomic on Windows
        and POSIX alike, and it works between two app instances sharing this
        checkout, which an in-process set would not.
        """
        lease = slot / LEASE_NAME
        try:
            fd = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # Someone else holds the lease. Whose, and whether they live, is the next line's job.
            self._inspect_stale_lease(slot, lease)
            return False
        except OSError as exc:
            logger.debug("workspace pool: cannot lease %s: %s", slot.name, exc)
            return False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(f"{os.getpid()}\n{time.time():.0f}\n")
        except OSError:
            # The marker exists, which is what makes the lease exclusive; an
            # unreadable pid only costs us the stale-lease recovery below.
            logger.debug("workspace pool: lease of %s has no pid", slot.name)
        return True

    def _inspect_stale_lease(self, slot: Path, lease: Path) -> None:
        """Quarantine a slot whose lease owner is no longer running.

        A live owner is a task in flight and nothing happens. A dead owner is
        a crashed task: its worktree may still hold work that was never
        archived, so the slot is set aside intact. Resetting it here is the
        one thing that would destroy the user's results.
        """
        pid = self._lease_pid(lease)
        if pid is None or _pid_alive(pid):
            return
        logger.warning(
            "workspace pool: slot %s was leased by pid %s, which is gone — "
            "quarantining it with its contents instead of reusing it",
            slot.name, pid,
        )
        self._quarantine(slot, f"owner pid {pid} died")

    @staticmethod
    def _lease_pid(lease: Path) -> int | None:
        """First line of a lease file as a pid, or ``None``."""
        try:
            first = lease.read_text(encoding="utf-8").splitlines()[0].strip()
        except (OSError, IndexError):
            # An unreadable or empty lease cannot name an owner; the caller then treats it as live.
            return None
        return int(first) if first.isdigit() else None

    def _quarantine(self, slot: Path, reason: str) -> None:
        """Move a suspect slot out of the reusable name shape, keeping its files.

        Never deletes: whatever is in there was written by a worker, and this
        code cannot tell a leftover from a deliverable. The directory is
        unregistered from git so a later ``worktree add`` at the same path
        cannot collide, and the files stay for the user.
        """
        target = slot.parent / f"quarantine-{time.strftime('%Y%m%dT%H%M%S')}-{slot.name}"
        try:
            slot.rename(target)
        except OSError as exc:
            logger.warning(
                "workspace pool: slot %s could not be quarantined (%s) — "
                "leaving it alone; it stays leased and unused",
                slot.name, exc,
            )
            return
        try:
            self._git(["git", "worktree", "prune"], cwd=self._repo_root)
        except (subprocess.SubprocessError, OSError) as exc:
            logger.debug("workspace pool: prune after quarantine failed: %s", exc)
        logger.info(
            "workspace pool: quarantined %s as %s (%s)",
            slot.name, target.name, reason,
        )

    # --- Validation and reset ---------------------------------------------

    def _validate_slot(self, slot: Path) -> bool:
        """True when ``slot`` is safe to reset. Checked before every reuse.

        Three independent facts, all required, none of them inferred from
        bookkeeping this process could have got wrong:

        1. the slot resolves inside ``pool_root`` (no symlink or junction out
           of it, which is how a reset could ever reach the main checkout);
        2. ``workspace/.git`` is a FILE — the worktree link. The main checkout
           and a lean workspace both have a ``.git`` directory, so this single
           check excludes both;
        3. git itself says the tree's common dir is this repository's.
        """
        workspace = slot / "workspace"
        try:
            resolved = workspace.resolve(strict=True)
            resolved.relative_to(self._pool_root)
        except (OSError, ValueError):
            # A path that cannot be resolved, or sits outside the pool root, fails the check by definition.
            return False
        if not self._is_worktree_link(workspace):
            return False
        slot_common = self._common_dir(workspace)
        repo_common = self._common_dir(self._repo_root)
        # Both sides come from git. Comparing against ``repo_root / ".git"``
        # instead looks right and is wrong whenever the repo root is itself a
        # linked worktree: its common dir then lives in the MAIN checkout, so
        # every slot failed validation and nothing was ever reused (caught by
        # the setup benchmark, 2026-10-28).
        return slot_common is not None and slot_common == repo_common

    def _common_dir(self, tree: Path) -> Path | None:
        """Absolute git common dir of ``tree``, or ``None`` when git refuses.

        The common dir is the one thing every worktree of a repository shares
        and no unrelated repository can claim, which makes it the right
        identity to compare two trees on.
        """
        try:
            out = self._git(
                ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                cwd=tree,
            )
        except (subprocess.SubprocessError, OSError):
            # git cannot answer for this tree, so it cannot be shown to belong
            # to this repository - and only proof may unlock a reset.
            return None
        raw = (out.stdout or "").strip()
        if not raw:
            return None
        try:
            return Path(raw).resolve()
        except OSError:
            # An unresolvable path is no proof of ownership either.
            return None

    @staticmethod
    def _is_worktree_link(workspace: Path) -> bool:
        """True when ``workspace/.git`` is the link file a worktree has."""
        try:
            return (workspace / ".git").is_file()
        except OSError:
            # An unreadable .git is not the worktree link file this needs to see.
            return False

    def _reset_slot(self, workspace: Path, *, base_ref: str, branch: str) -> None:
        """Bring a leased tree back to a pristine checkout of ``base_ref``.

        Order matters. ``reset --hard`` first, because ``checkout -B`` refuses
        to clobber modified tracked files; then ``clean -fd`` for the worker's
        untracked leftovers; only then the task's own branch.

        ``-x`` is deliberately absent from the clean: ignored trees
        (``node_modules``, ``.venv``, build caches) survive from the previous
        task, which is a second large saving, and they cannot leak into the
        next task's result because the diff capture runs ``git add -A``, which
        skips ignored paths. Dropping the previous branch keeps one branch per
        slot instead of one per task forever.
        """
        self._git(["git", "reset", "--hard", "--quiet"], cwd=workspace)
        self._git(["git", "clean", "-fdq"], cwd=workspace)
        previous = self._current_branch(workspace)
        self._git(["git", "checkout", "-B", branch, base_ref, "--quiet"], cwd=workspace)
        if previous and previous != branch:
            try:
                self._git(["git", "branch", "-D", previous], cwd=workspace)
            except (subprocess.SubprocessError, OSError) as exc:
                # A leftover branch costs nothing but a ref; the workspace is
                # already correct, so this must not fail the task.
                logger.debug(
                    "workspace pool: stale branch %s not deleted: %s", previous, exc,
                )

    def _current_branch(self, workspace: Path) -> str | None:
        """Branch the slot is on, or ``None`` when detached/unknown."""
        try:
            out = self._git(
                ["git", "symbolic-ref", "--quiet", "--short", "HEAD"], cwd=workspace,
            )
        except (subprocess.SubprocessError, OSError):
            # A tree whose branch cannot be read is simply treated as detached.
            return None
        name = (out.stdout or "").strip()
        return name or None

    # --- Plumbing ----------------------------------------------------------

    @staticmethod
    def _resolve(path: Path) -> Path:
        """Best-effort absolute form, so dict keys compare reliably."""
        try:
            return path.resolve()
        except OSError:
            # A path that will not resolve is still usable as a dict key in its absolute form.
            return path.absolute()

    def _git(
        self, cmd: list[str], *, cwd: Path
    ) -> subprocess.CompletedProcess[str]:
        """One git call: bounded, windowless, UTF-8, long-path tolerant."""
        patched = ["git", "-c", "core.longpaths=true", *cmd[1:]]
        try:
            return subprocess.run(  # noqa: S603 — fixed git argv, no shell
                patched,
                cwd=str(cwd),
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_GIT_TIMEOUT_S,
                creationflags=NO_WINDOW_CREATIONFLAGS,
            )
        except subprocess.CalledProcessError as exc:
            # Without git's own words a pool failure reads as a bare exit
            # code, and the caller silently falls back to a slow checkout
            # forever with nothing in the log to explain why.
            logger.info(
                "workspace pool: git failed (exit=%s): %s | stderr: %s",
                exc.returncode, " ".join(patched), (exc.stderr or "").strip(),
            )
            raise


def _pid_alive(pid: int) -> bool:
    """True when a process with ``pid`` is running.

    Deliberately conservative: anything this cannot determine counts as
    alive, because the consequence of a wrong "dead" is quarantining — and in
    the worst case resetting — a workspace a live task is still writing to.
    """
    if pid <= 0:
        return True
    if os.name == "nt":
        try:
            out = subprocess.run(  # noqa: S603 — fixed argv, no shell
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20.0,
                creationflags=NO_WINDOW_CREATIONFLAGS,
            )
        except (OSError, subprocess.SubprocessError):
            # A pid this cannot check counts as alive: the cost of a wrong 'dead' is quarantining a live task's workspace.
            return True
        return f'"{pid}"' in (out.stdout or "")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        # No such process: the owner really is gone.
        return False
    except PermissionError:
        return True  # someone else's process, but a live one
    except OSError:
        # Anything unexpected counts as alive, for the same reason as above.
        return True
    return True
