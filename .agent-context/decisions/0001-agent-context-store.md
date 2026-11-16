# 0001: Agent context store in `.agent-context/`

Status: accepted, 2026-11-16

## Decision

Agent working context lives in `.agent-context/` as plain Markdown:
`CURRENT.md` (short current state), `lessons/` (one confirmed lesson per file),
`decisions/` (numbered decisions like this one) and `logs/` (scratch output,
never versioned). `tools/context.py` searches `lessons/` through SQLite FTS5.

- **Markdown is the source of truth; the SQLite file is a cache.** It sits in
  the ignored `.cache/` folder and is rebuilt whenever a lesson changes.
- **Only `status: verified` is returned by default.** A draft is a hypothesis;
  serving it as a lesson repeats the guess. `--all` shows drafts on request.
- **Rules stay in `AGENTS.md`.** This folder holds context, never instructions
  that compete with it, and no `CLAUDE.md` is added.
- **Hooks are proposed, not installed.** `.claude/*` is not versioned; each
  machine registers `tools/guard-destructive.ps1` in its own settings.

## Consequences

Standard library only, so it runs on every OS that has Python 3.11. A lesson
is ordinary text that any agent can read without the tool.
