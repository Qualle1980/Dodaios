---
title: A worktree created inside a sandbox points at a path the host does not have
status: verified
tags: [git, worktree, sandbox]
created: 2026-11-16T12:00:00+01:00
updated: 2026-11-16T12:00:00+01:00
source: claude
---

**Symptom.** `git worktree list` marks the worktree `prunable`; inside it every
git command fails with `fatal: not a git repository: (NULL)`. `git status` from
the host lists the whole tree as untracked, and `.git/worktrees/<name>/` holds a
zero-byte `index.lock`.

**Cause.** The worktree was added by an agent running in a Linux sandbox that
mounts the repository under a different path (`/sessions/.../mnt/<repo>`). The
`.git` file in the worktree and `.git/worktrees/<name>/gitdir` store that
sandbox path. The checkout was cut off half way, leaving the lock behind.

**Fix.** From the main checkout: `git worktree repair <worktree-path>` rewrites
both links to host paths. Confirm no git process is running, delete the
zero-byte lock, then `git reset --hard origin/main` inside the worktree (only
when the branch holds no commits of its own; check `git log origin/main..`).
Recreating the worktree is the fallback.
