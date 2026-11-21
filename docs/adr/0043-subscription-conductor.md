# ADR-0043 — The subscription chat conducts every order

**Status:** Proposed · **Date:** 2026-10-23 · **Amends:**
[ADR-0009](0009-self-healing-worker-critic.md), [ADR-0011](0011-router-pure-dispatcher.md)

## Context

An order reaches Dodaios through several separate paths: the router's spawn
tool, the mission Kontrollierer (decompose once, run steps in parallel, review
each step, up to three revisions), the team-thread coordinator (ADR-0042) and
the subagent fan-out. None of them owns the whole order end to end:

- The decomposer runs once. Under 200 characters it never splits; above it, an
  LLM plan is fixed for the whole mission. A step that stalls is retried with
  the same instruction until the loop bound, then the mission fails.
- A step names only `worker_cli` (`claude` | `codex`). A part that the
  subscription chat could answer itself in seconds still starts a worker in a
  fresh worktree (live 2026-10-23: a one-file read took 158 s and ~49k tokens).
- Cost and speed decide nothing per step: the worker provider is one global
  setting, and effort is fixed (`model_reasoning_effort=medium`).
- Nothing watches the order as a whole: no record says whether the request is
  satisfied, whether work is looping, or whether progress stopped.
- The result reaches the user only by voice or as a file.

## Decision (proposed)

1. **One conductor per order: the subscription chat seat** (`chatgpt-plan`,
   zero marginal cost). It keeps two records for the order:
   - *Task record*: the request, known facts, facts to look up, the plan
     (short bullets, one executor per bullet).
   - *Progress record*, refreshed after every step result, as strict JSON:
     `is_request_satisfied`, `is_in_loop`, `is_progress_being_made`,
     `next_executor`, `instruction` — each with a one-line reason.
   On no progress or a loop for N consecutive refreshes (default 2) the
   conductor rewrites facts and plan once ("what went wrong, new plan that
   avoids it"); after R rewrites (default 2) it stops and reports honestly.
2. **Executors are capability lanes, chosen cheapest-first:**

   | Lane | Executor | Used for |
   |---|---|---|
   | `answer` | the conductor itself, no worker | knowledge questions, summaries, plans |
   | `lookup` | Codex, `--sandbox read-only`, low effort | read/search files, count, list |
   | `change` | Codex, `workspace-write`, medium effort | edits, fixes, small features |
   | `deep` | Claude Code (subscription) | large refactors, review-heavy work |
   | `review` | the other family's subscription | independent check of a `change`/`deep` result |

   Each lane is an ordered fallback chain over *connected subscriptions*
   (capability-gated, AP-21/AP-22); a lane with no connected subscription is
   unavailable, never a paid key (background policy).
3. **Deterministic checks before any LLM review** (read-only sandbox, file
   evidence, dump detection — PRs #42/#43) stay; the LLM reviewer runs only for
   `change`/`deep` lanes.
4. **The conductor writes the final answer** into the chat that started the
   order (and speaks it when a voice session is open).
5. The Kontrollierer keeps isolation, budgets, event log and resume; it becomes
   the executor of one lane step, not the planner.

## Consequences

- One model call per progress refresh on the subscription seat (no key cost,
  some latency). A one-bullet `answer` plan finishes with no worker at all.
- `Step.worker_cli` widens to a lane; the router and `test_routing.py` change
  (ADR-0011 amendment).
- Rollout behind `missions.conductor = off | shadow | on`, shadow first
  (records only, compares with today's path), like `central_lead_policy`.

## Open questions

- N and R defaults; per-lane effort values.
- Whether team threads (ADR-0042) become a `deep` lane implementation.
