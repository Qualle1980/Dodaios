# 0002 — A destructive action is never approved by bypass

- Status: accepted
- Date: 2026-11-16
- Scope: `jarvis/safety/tool_executor.py`, `jarvis/safety/command_impact.py`,
  `jarvis/safety/explicit_intent.py`

## Context

Bypass permissions is the project default (`AGENTS.md` §2): fresh installs,
chats, agents and delegated work execute permitted work without asking. That
default is correct for the product and must stay.

It also means the only unbypassable class today is the blacklist. Reading the
executor:

- `ToolExecutor.execute` raises on a blacklist hit (`ActionBlocked`), and
  `jarvis/society/approvals.py:72` returns `Verdict.BLOCK` for tier `block`
  before any mode is consulted. Those two cannot be bypassed.
- Everything else at tier `ask` arms an approval ticket and then, at
  `tool_executor.py:456`, bypass publishes `ActionApproved(approved_by=
  "bypass")` the moment nothing else has answered. A destructive shell
  command classifies as `ask`
  (`run_shell.py:158-174` → `command_impact.classify_command`), so under the
  default it runs with no person involved.

This is not a theory about the code. The same hole was measured once already
on a different tool: with `approval_mode=bypass` the telephony tool
`call-contact` auto-approved, proven with a faked probe
(`place_call_count=1`). The lesson recorded then is the general rule — *a tool
whose action cannot be undone needs an explicit human yes even under a bypass
default, and a risk tier on the tool is not that guarantee by itself.*

Tier `block` is not the answer: a blocked delete can never run at all, not
even when the person wants it. What is needed is an action class that bypass
cannot answer for, while a person still can.

## Decision

**1. A destructive action is a class, not a tier.** `is_destructive(tool,
args)` is settled centrally in `jarvis/safety/` and reuses the classifiers
that already exist (`command_impact.classify_command(...)​.level ==
DESTRUCTIVE`, which already splits `|`, `&&`, `;` and newlines into segments,
unwraps `sudo`/`env`/`xargs`, and is subcommand-sensitive for `git`). An
unknown or nested command classifies as destructive, never as safe.

**2. Bypass may not answer for that class.** The single branch at
`tool_executor.py:456` gains that condition: bypass supplies its standing
approval only for a call that is not in the destructive class. Everything else
about bypass is unchanged.

**3. A person still can.** The class does not block. The approval ticket is
armed and `ActionApprovalRequired` is published exactly as today, so the chat
card, the voice question and the scoped authorizers all work. What changes is
only who may say yes.

**4. Nobody there means no.** On the `UNATTENDED` surface an undecided ticket
already returns `_approval_unavailable` instead of waiting out the clock
(audit GT-12). A destructive action therefore does not happen in a headless
run — that is the intended outcome, not a regression.

**5. `intent_confirms_args` keeps its waiver, but loses its silence.** The
hook is deliberately narrow: `explicit_intent.command_confirms_destruction`
`fullmatch`es one deletion against an utterance that names the exact absolute
path, rejects `..`, globs, redirection, compound programs and inferred
targets, and fails closed. That is a real human instruction in the same turn,
so it stays a valid authorization — the alternative would be asking the person
to repeat the sentence they just said.

What it may no longer do is skip the record. Today it sets `tier_confirm =
False` (`tool_executor.py:362`), so no ticket is armed, no
`ActionApprovalRequired` is published, and `ActionExecuted` carries no
`approved_by` field at all — the audit trail cannot tell "the user named this
path" from "nothing authorized this". Explicit intent becomes a pre-answer on
the armed ticket instead of a skip: the action is proposed, the approval is
recorded as `explicit-intent`, and a plausibility doubt or an explicit denial
from a scoped authorizer can still stop it.

**6. `ActionExecuted` carries `approved_by`.** Without it there is no way to
prove after the fact who authorized a deletion.

## Consequences

- A destructive command in an unattended run fails with an honest "no approver
  available" instead of deleting something. Mission workers that relied on
  bypass for cleanup must ask, or use a non-destructive path.
- `AGENTS.md` §2 keeps its bypass default; this is the one documented carve-out
  and it is stated there.
- The telephony hole above is the same shape and the same fix site. It is NOT
  closed here — this decision covers the destructive class only. Spending money
  per use is a separate class and gets its own change, so that one stays a
  known gap rather than being quietly folded in.
- `.agent-context/tools/guard-destructive.ps1` (#175) is unaffected and is not
  this gate: it is an optional per-machine `PreToolUse` hook for an agent's own
  shell, it is not versioned into a machine's settings, and it denies in a
  headless run. It protects agent sessions, not the app's tool path.

## Alternatives rejected

- **Make destructive commands tier `block`.** Unbypassable, but also
  unapprovable — the person could no longer delete anything through the app.
- **Leave it to the whitelist/blacklist.** A `run_shell *` whitelist entry
  already downgrades everything (`run_shell.py:164-166`); relying on the
  blacklist means enumerating every destructive spelling, and the classifier
  exists precisely because that enumeration never holds.
- **Drop `intent_confirms_args` entirely.** It would make the app ask "shall I
  delete C:\x\y?" one sentence after the person said "delete C:\x\y". That
  trains people to confirm without reading, which is worse than the hole.
