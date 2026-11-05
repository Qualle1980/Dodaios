# Gates: P0 runtime regression check

OWNS: `jarvis/`, `tests/`, `scripts/ci/agent_runtime_suite.py`
Scope: Compare the P0 headless runtime suite with the exact base checkout; exclude unrelated CI and repository cleanup.

- [x] G1: The P0 headless runtime suite completes with the expected test result.
  CHECK: python scripts/ci/agent_runtime_suite.py --headless
  EXPECT: passed
  EVIDENCE: `399 passed, 4 warnings in 86.36s`, exit 0, from the P0 worktree with a
  writable project-local basetemp. The default temp root on this Windows checkout is
  ACL-blocked at `pytest-of-<user>`, on this branch and on the exact base alike, so
  the basetemp is an environment workaround and not a code change.

- [x] G2: The `agent runtimes on python:3.11-slim` CI job ends with exit 0.
  CHECK: the job of that name on the PR's head commit
  EXPECT: success
  EVIDENCE: run 41907011549 failed with exit 250 on head `ca33064ea` — `398 passed,
  1 skipped in 51.90s` and then `Fatal Python error: _enter_buffered_busy: could not
  acquire lock for <_io.BufferedWriter name='<stderr>'> at interpreter shutdown,
  possibly due to daemon threads`. The same job is green on the exact base
  `9bea5e9f7`, so this was a new failure identity owned by this branch, not an
  inherited one. Cause: `ProcessTree.close()` deferred its `SIGKILL` escalation to an
  unreachable daemon timer, and a process exiting inside the 0.5 s grace window left
  that timer to log to stderr while the interpreter was finalizing. Fixed by making
  every escalation reachable and cancelling it from `atexit`, plus silencing the sink
  once teardown has begun. Re-checked on the new head.

NOTE: the abort is timing-dependent and never reproduced on this Windows checkout, so
the unit tests assert the invariant that removes it rather than the crash itself. The
CI job is the end-to-end evidence.
