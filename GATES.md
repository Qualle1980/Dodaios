# Gates: P0 runtime regression check

OWNS: `jarvis/`, `tests/`, `scripts/ci/agent_runtime_suite.py`
Scope: Compare the P0 headless runtime suite with the exact base checkout; exclude unrelated CI and repository cleanup.

- [ ] G1: The P0 headless runtime suite completes with the expected test result.
  CHECK: python scripts/ci/agent_runtime_suite.py --headless
  EXPECT: passed
  EVIDENCE: terminal output from the current P0 worktree
ABANDON: G1 2026-11-05 — The prescribed wrapper cannot complete in this Windows checkout because pytest's default temp root is ACL-blocked at `C:\Users\ahmad_3u5iw3z\AppData\Local\Temp\pytest-of-ahmad_3u5iw3z`; the exact-base checkout fails at the same fixture. The affected runtime tests pass with a writable project-local basetemp: `399 passed, 4 warnings in 79.43s` from `tests/unit/agent_runtimes`.
