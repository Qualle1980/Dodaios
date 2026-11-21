# Agent context: current state

Short and current, at most 8 KB. Rules live in `AGENTS.md`; this file is
context. Replace stale lines instead of appending history.

Updated: 2026-11-16

## Layout

| Path | Holds | Versioned |
| --- | --- | --- |
| `CURRENT.md` | this page: what is in flight, what to know first | yes |
| `lessons/*.md` | one confirmed lesson per file (symptom, cause, fix) | yes |
| `decisions/NNNN-*.md` | numbered decisions with their reasons | yes |
| `logs/` | scratch output; `tools/cleanup.ps1` prunes it | `.gitkeep` only |
| `.cache/lessons.sqlite` | FTS5 index over `lessons/`, rebuilt on demand | no |

## Commands

```bash
python .agent-context/tools/context.py search "worktree lock"    # verified only
python .agent-context/tools/context.py search "worktree" --all    # include drafts
python .agent-context/tools/context.py rebuild                    # optional, search rebuilds when stale
```

```powershell
powershell -File .agent-context/tools/cleanup.ps1                 # dry run, files older than 14 days
powershell -File .agent-context/tools/cleanup.ps1 -Days 3 -Apply  # delete
```

## Lesson format

```markdown
---
title: One line naming the failure
status: verified          # verified | draft | superseded
tags: [git, worktree]
created: 2026-11-16T12:00:00+01:00
updated: 2026-11-16T12:00:00+01:00
source: claude            # the agent that confirmed it
---
**Symptom.** ... **Cause.** ... **Fix.** ... (with the command or test that proved it)
```

`verified` means a test, a command output or a live run proved it. Anything
else stays `draft` and is hidden from the default search. Files starting with
`_` are ignored by the index. No secrets, keys or private personal data.

## Destructive-command guard (proposal, not installed)

`tools/guard-destructive.ps1` is a Claude Code `PreToolUse` hook. For a Bash or
PowerShell call matching a destructive pattern (`rm -rf`, `Remove-Item
-Recurse`, `git reset --hard`, `git push --force`, `git clean -f`,
`git branch -D`, `git stash drop|clear`, `git worktree remove --force`, SQL
`DROP TABLE`, ...) it returns `permissionDecision: "ask"`; otherwise it prints
nothing. `.claude/*` is not versioned, so each machine registers it in its own
user or local settings:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|PowerShell",
        "hooks": [
          {
            "type": "command",
            "command": "powershell -NoProfile -ExecutionPolicy Bypass -File \"<repo>/.agent-context/tools/guard-destructive.ps1\""
          }
        ]
      }
    ]
  }
}
```

Use an absolute path for `<repo>`; whether `$CLAUDE_PROJECT_DIR` expands in a
Windows hook command is not verified.

**Measured 2026-11-16** (Claude Code 2.1.325, headless `claude -p`, Haiku, one
turn each, hook payload reported `"permission_mode":"bypassPermissions"`):

- `rm -rf victim`: the hook answered `ask`, the call was recorded in
  `permission_denials`, the folder survived. `ask` is NOT overridden by
  `bypassPermissions`.
- `ls -d victim`: the hook printed nothing, the command ran, no denial.
- In a headless run there is nobody to ask, so `ask` acts as a hard deny.
  Registered for delegated or headless agents, the guard blocks those commands
  outright. The interactive prompt under bypass was not observed.
- Trap: with `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB` set, a child `claude -p`
  silently drops `--permission-mode bypassPermissions` to `default` (one stderr
  line). A bypass test run in such an environment measures `default`.

## Open

- Interactive confirmation under `bypassPermissions` (desktop app, terminal)
  is unverified.
- No hook is registered on any machine yet.
