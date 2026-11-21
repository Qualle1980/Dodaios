# ADR-0043 — One conductor, four domain leads

**Status:** Accepted · **Date:** 2026-11-06

## Context

A mission used to be a plan split into steps that each ran once and came back.
Nothing looked at an order as a whole while it was running, so a step that
answered the wrong question still counted as progress, and a worker that could
not reach a file stalled the order without anyone noticing.

The conductor answered that: one subscription chat owns an order from the
request to the answer, keeping a task record (facts and plan) and refreshing a
progress record after every result — satisfied, looping, moving, who acts next,
with what instruction. Its control flow and prompts are adapted from
Magentic-One in `microsoft/autogen`; the MIT notice lives in
`jarvis/missions/conductor/ledger.py`. It is rolled out behind
`missions.conductor` (`off` | `shadow` | `on`), where `shadow` plans every
order and steers nothing.

Code has cited this decision as ADR-0043 since that rollout; this file records
it, and adds the part that was missing: **who** the conductor hands work to.

Until now the conductor conducted over raw executor names — `conductor`,
`codex`, `claude-code`. Two consequences. The model picked by the binary it
remembered rather than by what the step needed; and write access depended on
the conductor's phrasing, which on 2026-10-24 let a read-only order run with
write access because a step was worded "Run `wc -l` ...".

The owner's requirements handover asked for ten domain leads ("sergeants")
under a General. Ten resident models would spend budget without adding
judgement, and lead logic competing with the existing router is the largest
design risk in `docs/plans/2026-11-05-general-architecture.md`.

## Decision

1. **One conductor. The leads are roles inside it, never a second engine.**
   `jarvis/missions/conductor/leads.py` is a pure roster: names, remits, and
   which lane each works in. The loop that plans the order, names who acts
   next and bounds the rounds is unchanged, and the router stays the pure
   dispatcher of [ADR-0011](0011-router-pure-dispatcher.md). The team roster
   of named specialists is Magentic-One's own shape; the only change is that
   the names the model may say are roles with a remit.
2. **Four leads, not ten.** Planning, Coding, Quality, Memory. Testing and
   Security are gate functions, not always-on models; Research,
   Infrastructure, UI/UX and Communication are capabilities the Coding lead
   asks for when a task needs them.
3. **Write access follows the remit, not the phrasing.** Only Coding may
   change files. Every other lead's instruction carries the read-only marker
   whatever words the conductor chose, on top of the existing rule that the
   user's own words decide whether an order may write at all.
4. **The scale of the work still decides the seat.** Coding inherits the
   order's own deterministic lane (`change` or `deep`), so a typo fix stays
   cheap and a refactor reaches the capable worker. A lead does not re-route
   the order with a second judgement, and it can never add an executor.
5. **Independence is real or absent.** Quality resolves to a connected
   executor of another family than whoever built the work. With one family
   connected there is no reviewer: the lead is unavailable, and the conductor
   is told so plainly, rather than a reviewer in name only.
6. **A lead with nobody behind it is reported, not substituted.** Resolution
   returns nothing, the step does not run, and the conductor hears it in the
   thread so its next progress record routes the step somewhere that exists.
   Whether a resolved executor can really start is still decided at the spawn,
   which halts the order with `SETUP_FAILED` or `WAITING_CAPACITY`.
7. **A worker the user named overrides the roster.** "Let Codex look at
   `a.py`" is an instruction: that executor is conducted alone
   (`conductor/requested.py`), outside the leads, so the conductor cannot
   answer in its place from a seat with no file access.

## Consequences

- The plan and progress prompts read as roles, so a step is assigned by what
  it needs. Shadow mode logs both the lead and the executor it resolves to,
  so the two can still be compared against the worker that really ran.
- The lanes' cost-ordered chains (`pick_executor`, `pick_reviewer`) are now
  the live resolution path rather than unused helpers.
- A read-only lead cannot damage a workspace even if the conductor words its
  step as a change, which is the 2026-10-24 defect closed from the other side.
- Adding a fifth lead is a roster entry plus its lane; it needs no new engine,
  no new process and no change to the loop.
- Quality is unavailable on a single-family install. That is the honest
  outcome, and it is visible in the thread instead of silently degrading into
  self-review.

## Alternatives

- **Ten leads as configured models.** Rejected: resident models that add no
  judgement, against an explicit budget constraint.
- **A lead as its own orchestrator with its own workers.** Rejected: a second
  orchestration engine beside the router and the conductor, which the
  architecture plan names as the largest risk.
- **Keep raw executor names and teach the prompt to choose well.** Rejected:
  the 2026-10-24 write-access defect came from exactly that — behaviour that
  depends on wording rather than on a remit the code enforces.
