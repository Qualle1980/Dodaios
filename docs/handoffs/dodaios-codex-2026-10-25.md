# Dodaios engineering handoff — 2026-10-25

**Status:** Research only. No changes to Dodaios runtime. Dedicated branch `handoff/chatgpt-review-20261025`. Before implementation inspect local `D:\Projects\Dodaios-codex-worktree-voice` and branch `codex/worktree-timeout-voice`, which are NOT verified to be clean. Latest observed main SHA `ca87fe1e`.

## Claude / Codex handoff evidence
Claude's 2026-10-24 handoff: `handoff-2026-10-24.md` indexed in `MEMORY.md`; Obsidian `AI-Memory-Vault/60-Projekte/Agent-Learnings/Dodaios/22d33c49fba51715.md`. The voice knowledge/screen tests succeeded; voice file request failed, searching too widely and hitting timeouts. Worker launched, but `git worktree add` took ~10.5 minutes, expected 15–30 sec. Disk I/O observed, antivirus NOT established as cause. Codex created an isolated worktree and researched two fixes, but no completed implementation/PR was established from retrieved handoff logs.

## Existing code (observed on GitHub main)
- `jarvis/missions/isolation/worktree.py`: `WorktreeManager.create()` calls `_run_git(["git","worktree","add",...])`; `_run_git` approx. line 792 uses `subprocess.run(..., check=True, capture_output=True, text=True, encoding="utf-8", creationflags=NO_WINDOW_CREATIONFLAGS)` **without timeout**. `_run_git_in` also has no timeout; assess separately.
- `jarvis/brain/tool_gateway.py`: `BrainSupervisorToolGateway._voice_tools()` populates speech-only tool catalog; `voice_catalog()` and `execute()` use this path for realtime.
- `jarvis/agent_chat/folder_tools.py`: `folder_tools(cwd)` already exposes read tools `Read/Ls/Glob/Grep` through `FolderTool` and `ToolExecutor`. No reason to add another MCP server.
- `jarvis/agent_chat/tools.py`: `_read()` honors offset/limit and caps output; existing tools have read-only subset and search limits.
- `AGENTS.md`: isolated feature worktree mandatory, never reset/stash shared main, preflight and import isolation, one real live turn max when approved, avoid unnecessary billed probes.

## External reference projects and design findings
1. [CPython subprocess documentation](https://github.com/python/cpython/blob/main/Doc/library/subprocess.rst): timeout on `Popen.communicate()` does not itself kill child. Terminate, reap and bound cleanup. On Windows consider entire process tree, not only parent; integrate with existing Dodaios process lifecycle.
2. [MCP Filesystem](https://github.com/modelcontextprotocol/servers/blob/main/src/filesystem/lib.ts), [path validation](https://github.com/modelcontextprotocol/servers/blob/main/src/filesystem/path-validation.ts), [README](https://github.com/modelcontextprotocol/servers/blob/main/src/filesystem/README.md): allowlisted roots, path canonicalization, symlink/junction escape detection, no unrestricted cwd/root fallback, explicitly read-only tools. **Do not directly copy third-party code without exact license/version audit.**

## Microtask A — Worktree timeout ONLY
After confirming current local status and no existing partial fix, edit solely `jarvis/missions/isolation/worktree.py` plus focused tests. Bound the `git worktree add` runtime with explicit named failure and elapsed duration. On timeout stop/reap process *and children* safely on Windows; ensure failure does not leave dangerous partial worktree state. Preserve existing Git behavior for unrelated commands. Add mocked tests for success, timeout, cleanup, error clarity, and cancellation race. No live model calls. Run focused tests plus mandatory project gates; report changed files, test results and PR. Stop for review.

## Microtask B — Voice read tools ONLY, separate follow-up
After A is reviewed: reuse `FolderTool`/existing Read/Ls/Glob/Grep in `BrainSupervisorToolGateway` for voice `realtime` requests, with an explicit resolved project checkout root and no global filesystem access. Use existing executor and risk policies; NEVER add Write/Edit/RunCommand. Reject out-of-root paths, `..`, symlink/junction escapes, missing or ambiguous roots. Keep bounded read/search outputs and avoid whole-drive traversal. Add voice catalog + execution tests, traversal-denial tests, symlink tests, no-root tests. Run focused tests/gates; no paid provider turns. Stop.

## Minimal prompts to give Codex
**A:** “In your EXISTING isolated worktree, check only git status and whether the worktree timeout fix already exists. Implement Microtask A exactly as in `docs/handoffs/dodaios-codex-2026-10-25.md`. No further research. Only the worktree file and targeted tests. Run relevant tests/gates; return commit/PR and concise evidence. Stop.”

**B (later, separate turn):** “Implement Microtask B from `docs/handoffs/dodaios-codex-2026-10-25.md` only after confirming A merged. Only voice gateway/adapters + tests. No wide reads, no new libraries, no live turn. Return concise diff/test evidence. Stop.”

**Cost policy:** cheapest available Codex model, low reasoning, no Astra, no unlimited exploratory reads, no repetitive context compaction. ChatGPT owns comparisons/review; Codex only edits and tests.
