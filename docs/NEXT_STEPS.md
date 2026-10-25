# Next steps (agreed 2026-10-17)

Branch: `feat/mobile-gateway`. Start the next session from step 1.

## Done on 2026-10-17
- ChatGPT web bridge runs windowless (headless Chrome with a user-agent override).
- Prompt caps and the `chatgpt-web/instant` default (commits f33d5a103, a775bbc17).
- The bridge prewarms the next Temporary Chat tab (`prewarmNextPage` in
  `Dodaios-ChatGPT-Bridge/src-pin/src/adapters/chatgpt-web/browser-worker.ts`; that folder is not a git repo).
  A turn takes about 14 s, which is acceptable for now.
- Claude CLI test: 5 s with `--strict-mcp-config --setting-sources "" --tools ""`. Decision: keep the bridge for now.

## Open, in order
1. **Mobile app** (`mobile/android`, Kotlin): make chat and calls work reliably.
   First ask what currently fails on the phone (no USB, no adb).
2. ~~**Delegation**~~ **done 2026-10-19** (not committed yet) — see "Delegation wired" below.
3. **Token usage per task** (Codex, Claude) shown in Jarvis and the app.
   Usage is already counted in `jarvis/agent_runtimes/gateway.py`; the desktop UI display is not checked yet.
4. **Design, phone AND PC:** futuristic "Iron Man / JARVIS HUD" look
   (dark, glowing cyan, animated rings, holographic panels).
5. **DK:** second person in the app; DK uses an **iPhone**.
   - DK's own Claude and Codex run on DK's accounts and machine; never share credentials.
   - One shared chat that both people can see.
   - Needs second-user pairing and a shared chat room.
6. **TypeScript rewrite** of the app, one codebase for Android + iPhone (e.g. React Native/Expo),
   which also solves step 5 for the iPhone. Interim option for DK: an installable web app (PWA).

7. **Desktop window (later):**
   - Windows Snap does not work: the title bar drags via pywebview's JS `window.move`, not the native move loop.
     Fix: hand the drag to Windows (`ReleaseCapture` + `WM_NCLBUTTONDOWN`/`HTCAPTION`).
   - After minimize/restore the window stays grey: WebView2 is not re-sized/shown on restore
     (suspected; reproduce live first). Workaround for now: use the UI in Chrome.

## Constraints for distribution
- **No costs.** No App Store and no paid Apple developer account; the app is only for the user's and DK's phones.
  - iPhone (DK): use an installable web app (PWA via Safari "Add to Home Screen"), which is free.
    Free Xcode sideloading expires after 7 days and needs a Mac, so it is not an option.
  - Android: direct APK install, which is free.
- **Desktop app for DK:** at the end DK must install it from a single link without problems.
  - Current problem: DK's antivirus blocks the AK Matrix 2 install. Ask which antivirus it is and the exact message.
  - Likely cause: an unsigned PyInstaller-style executable.
  - Free countermeasures: avoid one-file packed executables (e.g. Tauri/Electron build with the TypeScript rewrite),
    submit false positives to the antivirus vendor, publish a checksum.
  - A paid code-signing certificate is excluded.
  - Update: DK's antivirus is TotalAV. A false-positive report was sent 5 days ago, with no result.
    The AK Matrix 2 app comes from GitHub (third party). A GitHub link does not sign anything.
  - Preferred free path: DK installs nothing. Jarvis runs on the user's PC and DK opens it as a web app (PWA)
    over a secure private link (e.g. free Tailscale).
  - Fallbacks: a source install script (clone + install + run) instead of a packed exe,
    or a TotalAV exclusion set by DK.

## Optional bridge speedups
- Skip effort selection when it is unchanged (about 2.5 s).
- Shorten the send-accepted wait (about 2.8 s).

## Research result 2026-10-19 — delegation for the text-only brain

Prior art checked: LangGraph supervisor (`Command(goto=..., payload=...)`, FAQ: without tool
calling you must convert the raw LLM string into a decision yourself), AutoGen/AG2 GroupChat
`select_speaker_prompt_template` (model replies with ONLY an agent name out of `{agentlist}`),
CrewAI hierarchical manager agent, OpenAI Agents SDK handoffs (destination agent + filtered
input payload), Anthropic orchestrator-workers. Common shape everywhere:
**destination (validated against a known list) + payload (the task text)**.

Our repo already has both halves:
- `parse_route_decision` (`jarvis/plugins/brain/chatgpt_web.py:81`) = "raw string -> decision",
  and `_DELEGATE_TARGETS = {codex, claude}` = the destination validation.
- `spawn_worker` (`jarvis/plugins/tool/spawn_worker.py`) = the payload transport:
  `manager.dispatch(prompt=...)` **then** `kontrollierer.run_mission(id)` (both steps, BUG-016).

**Key finding (new):** the tool-less-brain problem is ALREADY solved for the normal case by the
"intelligent router" in `jarvis/brain/manager.py:10949` — when the active brain's
`can_call_tools()` is False, a tool-capable provider LEADS the turn, picks the tool, and the
chain falls through to the chosen talker if no tool is picked.
BUT `_first_tool_capable_provider` (`manager.py:10828`) returns `None` when no tool-capable
provider is reachable (no API key / rate-limited / dead). Then the chain stays unchanged and
**nothing can delegate at all** — that is exactly the hole `parse_route_decision` was written for.

**Decision:** do not invent a new path. Use `parse_route_decision` as the *fallback* router that
only runs for a text-only brain when no tool-capable provider leads the turn. `delegate` ->
build the mission prompt the `spawn_worker` way (target goes in the prompt as
`Zielort/Kontext: <target>.`, like `_build_mission_prompt`), then dispatch + run_mission.
`clarify` -> ask the user back, `self` -> answer as-is. Never build a second `Kontrollierer`.

Environment check: the project venv is **Python 3.13.16** (`.venv`); the newly installed 3.15.0
is NOT what the project runs on, and nothing should be switched to it without a dependency check.

## Delegation wired (2026-10-19, uncommitted)

Changed files:
- `jarvis/brain/manager.py` — new `BrainManager._route_text_only_delegation()` plus a hook in
  `generate()` just before the history append. Gate order matters: the cheap envelope shape test
  first, only then `_active_can_call_tools()` (that call builds the brain instance — calling it on
  every turn constructed a provider no turn had asked for and broke
  `tests/unit/brain/test_manager_screen_context.py`).
- `jarvis/plugins/brain/chatgpt_web.py` — `ROUTING_INSTRUCTION`, appended to the system message
  AFTER the clip, so the 8k cap can never cut the envelope rule in half.
- `jarvis/voice/action_phrases.py` — new phrase `delegate_task_started` (de/en/es/pt).
- `tests/unit/brain/test_text_only_delegation.py` — 6 tests, all green.

Behaviour: `delegate` → background mission (`dispatch()` **and** `run_mission()`), spoken promise,
result announced later by the existing readback. `clarify` → the question is the turn.
`self` / broken JSON / unknown target → the reply stays untouched. No mission stack yet → answer
directly instead of promising work nobody runs. When the intelligent router led the turn, the
envelope is only unwrapped (no second mission), so raw JSON never reaches the user.

Verified: `py_compile` on all changed files; `ruff` clean on the new/changed code (the remaining
`action_phrases.py` / `manager.py` warnings are pre-existing); `pytest tests/unit/brain -k
"router or manager or delegat or chatgpt or fallback"` → 609 passed;
`tests/unit/brain/test_text_only_delegation_generate.py` → 2 passed (the hook inside the real
`generate()`, with a text-only brain and a stubbed mission stack).

**Live-verified 2026-10-19** against the real bridge (status `ready`, two single turns):
- "Bitte schreibe Unit-Tests fuer jarvis/missions/manager.py." → the model answered with a clean
  envelope, parsed to `kind='delegate' target='codex'` and a fully written-out task.
- Control turn "Was ist die Hauptstadt von Frankreich?" → plain prose, parsed to `kind='self'`.
  So the routing rule (512 chars) does NOT make ordinary questions delegate.

Not verified: mypy (not installed in the venv), and a full mission actually running end-to-end
through a real Codex worker.

Open follow-up: whether `claude` vs `codex` really reaches the intended worker — `_worker_factory` (`jarvis/missions/init.py`) still picks the worker from the
step model, so the target currently only steers the worker via the mission prompt.

## Live mission run 2026-10-19 (option a) — wiring proven, pre-existing critic bug found

Two real missions were driven through the real stack by the new helper
(`bootstrap_missions` in a side process, `JARVIS_ISOLATION_ROOT` on a scratch dir,
`recover_missions=False`, so nothing touched the shared outputs or live missions).

What worked, observed in `mission_events`: `MissionDispatched` -> `MissionPlanReady` ->
`WorkerSpawned` -> `WorkerProgress` -> `WorkerDraftReady` -> `CriticVerdictReady` -> iterate.
The prompt the worker received carried `Zielort/Kontext: codex.` and the user's own wording.
So chat -> envelope -> mission -> worker is proven end-to-end, including the mandatory
`dispatch()` + `run_mission()` pair.

Run 1 ("count the lines in docs/NEXT_STEPS.md") failed for a reason worth remembering:
**the worker runs in a git worktree, so untracked files do not exist for it.**
`docs/NEXT_STEPS.md` is untracked, the worker honestly reported a missing file, the critic
correctly said `revise`. Not a wiring bug — a badly chosen test task.

Run 2 ("name the first heading in README.md", a tracked file) exposed a **pre-existing
orchestrator bug**: the worker did real work (a PowerShell `Select-String` call) and answered
correctly ("## Install"), but every iteration was vetoed with
*"the mandatory empty-diff rule prevents approval of this read-only task"*, ending in
`critic_loop_exhausted` -> FAILED.
The read-only exception for exactly this case EXISTS (`jarvis/missions/critic/runner.py:894`,
BUG-LIVE-09) but keys off `_extract_tool_call_evidence(worker_log)`, which evidently did not
recognise this worker's log format — the same sibling-blindness class as the codex-worker note
from 2026-06-10 in that file. Consequence for delegation: a *research/answer* task that is
delegated can never be approved today, while a code-changing task can.

Decision pending: either narrow `ROUTING_INSTRUCTION` to code-changing work only, or fix the
read-only evidence extraction. Not touched yet — the user decides.
Also still open: a worktree path longer than 200 chars is rejected outright
(`jarvis/missions/isolation/worktree.py:290`), which is why the scratchpad dir could not be used.

## Fixed 2026-10-19 — a delegated question is now approved (option 2)

The earlier diagnosis was wrong in its blame: the critic is fine. Its read-only
auto-approve path (`CriticRunner`, keyed off `is_informational_request`) works — but it reads the
MISSION PROMPT, and the prompt built by `_route_text_only_delegation` led with an imperative
`Aufgabe: ...` line, which makes the request look like work with a file deliverable. Measured
directly against `jarvis.missions.stream_evidence.is_informational_request`:

    "was steht oben in der readme"                      -> True
    "Aufgabe: Nenne die erste Ueberschrift ..."         -> False
    "<question>
Zielort/Kontext: codex."               -> True   (target line is harmless)
    "<question>
<imperative task sentence>"            -> False  (an appended imperative kills it)

Fix in `jarvis/brain/manager.py`: an informational turn leads with the user's own wording plus
the target line, and the model's longer task text is appended as a `Details:` line ONLY while a
self-check confirms the prompt still reads as informational. Real work keeps the task shape.

Live proof, same question as the failing run, same stack:
- before: 3 critic iterations, every one `revise` on the empty-diff rule -> `critic_loop_exhausted`
  -> FAILED.
- after: `CriticVerdictReady: approve` — *"Informational request answered; the spoken answer is
  the deliverable."* -> `all_tasks_approved` -> **APPROVED on iteration 0**.

`ROUTING_INSTRUCTION` therefore keeps "longer research": a delegated research/question task now
completes. Tests: `tests/unit/brain/test_text_only_delegation.py` (8) and
`..._generate.py` (2) green; `tests/unit/brain -k "router or manager or delegat or chatgpt or
fallback"` -> 613 passed; `tests/unit/missions -k "informational or readonly or critic or
evidence"` -> 14 passed. Pre-existing and unrelated: `tests/unit/telephony/test_security.py`
fails to import (`No module named 'twilio'`).

Not verified live: a delegated CODE change (the task-shaped path) — covered by unit test only.

Leftover scratch dirs from the live runs, safe to delete by hand: `D:\jm-run4` (approved
mission's db + isolation) and the empty `D:\jm-live3`. The automated removal was refused by a
safety check, so they were left in place.

## 2026-10-19, the three open points worked through

**1. A delegated CODE change now proven live.** Mission APPROVED on iteration 0; the captured
`diff.patch` holds the new file (`delegation_demo.py`, a `greet(name: str) -> str` with
docstring), and the deliverable was mirrored to `~/Downloads/Jarvis-Outputs/`. Two earlier
attempts failed on MY test task, not on the code, and both are worth remembering:
  * a worker only sees TRACKED files — `docs/NEXT_STEPS.md` is untracked, so it does not exist
    in the worktree;
  * `.gitignore:280` has `/scratch_*`, so a file named `scratch_*.py` is skipped by the
    `git add -A` capture and the diff is legitimately empty. The critic was right both times.
  Also checked: a mission never writes into the shared checkout. `deliver_to_user_folder` COPIES
  deliverables out; there is no `git apply` back into `D:\Projects\Dodaios` (verified clean).

**2. `codex` vs `claude` cannot be routed per task today — by explicit mandate, not by a bug.**
`step.worker_cli` (`Literal["claude","codex"]`) exists and is set by the decomposer, but nothing
reads it for worker selection; its only reader is cost attribution
(`jarvis/costs/sources.py:491`). Worker choice is the pure function
`_select_subagent_worker_kind(provider, step_model)` (`jarvis/missions/init.py:405`), whose
docstring records the user mandate of 2026-05-29: *when a subagent provider IS configured it is
the single source of truth; a per-step `model` string can never divert heavy work.*
Measured on this install: `_live_subagent_provider(None)` -> `'openai-codex'`, so EVERY step
resolves to `'codex_direct'` regardless of the step model. A delegated `target: "claude"`
therefore cannot reach a Claude worker without overriding that mandate. Not changed — that is the
user's decision. If it should change, the clean shape is an optional, capability-gated
`preferred_cli` argument on `_select_subagent_worker_kind` plus making `step.worker_cli` the
carrier, honoured only when the requested family is actually viable (AP-21/22).

**3. Cleanup.** The ChatGPT bridge that the live tests started is stopped again (`status: offline`,
`/healthz` unreachable). No mission worktrees remain registered in git. Left for the user to
delete by hand, because the automated removal is refused by a safety check:
`D:\jm-run4`, `D:\jm-code1`, `D:\jm-code2`, `D:\jm-code3` (~4.3 MB total). Mission
deliverables in `~/Downloads/Jarvis-Outputs/` (`delegation_demo.py`,
`was-steht-oben-in-der-readme-zielort-kontext-cod.md`) are ordinary mission output and were left
in place.

## 2026-10-19 — point 2 implemented after all (user gave approval)

`Step.preferred_cli` (new, additive, default None) carries the family a request asked for; it is
set only from an exact `Zielort/Kontext: claude|codex` line, via the shared
`TARGET_CLI_MARKER` / `parse_target_cli` in `jarvis/missions/stream_evidence.py`.
`_select_subagent_worker_kind` takes it as an optional third argument and honours it ONLY between
the two subscription CLIs, and only when the configured provider is one of them. `worker_cli`
could not be the carrier: the heuristic single-step plan sets it to "claude" on every mission.

The 2026-05-29 mandate still holds — measured table:

    openai-codex + (none)   -> codex_direct      claude-api + (none)   -> claude_direct
    openai-codex + claude   -> claude_direct     claude-api + codex    -> codex_direct
    gemini      + claude    -> gemini            openai     + claude   -> api_agent
    (none)      + claude    -> subjarvis         openai-codex + gemini -> codex_direct

Live proof: the same delegated task with `"target": "claude"` ran on `claude-opus-4-8` (read from
the worker's own stream `system/init` frame) and was APPROVED, on an install whose
`_live_subagent_provider()` is `openai-codex`. The codex run before it had the Codex
`thread.started` stream format, so the family really switched.

Commits: `a9bf95228` (delegation), `01bed9d56` (worker pick), `ef88084d5` (i18n markers).
Tests: `tests/unit/missions/test_preferred_cli_routing.py` (20) plus the two delegation files (10);
`tests/missions` + `tests/unit/missions` -> 1842 passed, 5 skipped; the four mandatory guards ->
663 passed; `check_no_new_german.py` -> OK. Note: that gate reads the committed HEAD, not the
index, so German lines in NEW files only surface after the first commit that tracks them.

## How the chain actually works today (correcting an assumption)

For "erledige PR3" typed into the chat:
1. **Routing** happens in the ChatGPT web chat on the subscription — no API tokens. It decides
   self / clarify / delegate, the target, and writes the task text.
2. **Splitting does NOT happen in ChatGPT.** `server.py:3232` passes `brain_caller=None`, so the
   decomposer runs heuristic-only: one single step per mission. There is no LLM
   token-optimisation of the plan today. Giving the decomposer a brain is a separate decision.
3. **The work runs on Codex or Claude** — that is what delegation means. The chat only decides.
4. **Counter-check: yes, always.** Every iteration gets a critic verdict (`CriticVerdictReady`);
   up to three revise loops, then `critic_loop_exhausted` -> FAILED. The critic runs as its own
   CLI subprocess (like the worker), and both live runs reported `cost_usd 0.0`.

## 2026-10-19, evening: what landed, and where the boot time goes

Landed on main today: delegation from a text-only brain (#18), the native caption drag so Snap
works (#19 + #21), the user profile kept inside every prompt cap (#20), and the maximize monitor
resolved from the restore rectangle (#22 + #23) -- the grey window after a minimize is gone,
confirmed by hand on the 125% laptop screen. PR #24 (telephony mounted only when switched on,
19 checks green) is OPEN and ready to merge.

Twilio audit result: the package is NOT installed here, it is an optional extra, and the direct
PC-to-phone path (`jarvis/mobile_gateway/app.py`, a separate ASGI app) has no dependency on it.
Removing the feature would save about 100 ms of boot (3544 ms vs 3642 ms median over three runs
each, ranges overlapping) for ~5,100 lines of change, so the switch is the whole of it. The
secret-scanning alert was a synthetic test identifier and is resolved as `used_in_tests`.

**Boot numbers (2026-10-19, SUPERSEDED — measured with a broken harness):** backend ready 120 ms,
server bound 984 ms, voice ready 24.2 s, voice usable 29.8 s, app interactive 36.4 s. Do not use
these as a baseline; every anchor past the window included a spurious 12 s.

**~~The single biggest block is 12.12 s and it is a lost message~~ — WRONG, corrected 2026-10-19.**
The claim was that `server.py:1200` swallows the paint acknowledgement. It does not:
`data/jarvis_desktop.log` shows nearly every real boot acknowledging immediately, and only the
window-less measurement runs hit the timeout. The real cause was in the HARNESS —
`scripts/measure_desktop_boot.py` sent the ack with no `Origin` header, `surface_security.py:1197`
answers that route only for a same-origin browser-shaped request, so it was refused and the run
paid the full `wait_shell_painted` fallback. Fixed in `ffa9ab140`; the harness now aborts instead
of reporting numbers (`check_paint_ack`, `564098c59`).

**Boot numbers after the fix (3 serial cold boots, 1 discarded warmup, voice on):** window median
1174 ms, voice TTU median 6712 ms, app interactive median 9713 ms (samples 8952 / 9713 / 22786 —
the spread is reported, not hidden), post-voice work 3001 ms. `check_boot_budget.py` passes.
Phase medians: `db_wake_gate` 4191 ms (deliberate wait for the wake model), `db_feature_routes`
2870 ms, of which `fs_route_imports` 2680 ms and `fs_route_mounting` 187 ms.

**Open decision (Maßnahme 4, deferred on purpose):** the optimizable work after voice is route
IMPORT cost, ~2.7 s. PR #24's pattern (import inside the `enabled` branch) is the right lever,
but which routers deserve it is not decided — that needs per-router numbers, not a guess. Do not
start it without approval.

After that, the next two posts are a wake-CUDA probe that reports `cache MISS` every boot for an
answer that never changes (2.59 s) and the speech model (2.57 s); and the log itself raises
`UnicodeEncodeError` on an emoji under cp1252 (AP-1).

## Open items, recorded 2026-10-19 (no work started, approval required per item)

1. **"Wer bin ich?" takes 10-20 s for the first answer.** No cause measurement exists yet. Measure
   before touching anything: which phase of the first turn spends it (identity/memory read, model
   call, TTS), on the same clock.
2. **Memory architecture review** — Obsidian vault, SQLite, IdentityCard and the self-learning
   loop, how they interact and what the first turn actually reads. Still open.
3. **Twilio removal, prepared but not performed.** Telephony is not needed for the PC-to-phone
   path (`jarvis/mobile_gateway/app.py` is independent, pinned by a test). Remove only on proven
   benefit and without touching the mobile gateway. PR #24 (opt-in switch) is the reversible step
   in that direction; it is NOT merged yet.
4. **Maßnahme 4 (per-router import deferral)** — blocked on per-router numbers, see above.
5. **Maßnahme 5 (boot-timing diagnostics endpoint)** — deliberately NOT built. Check first whether
   the existing logs and `diagnostics_routes.py` already answer the question; no new surface
   unless they do not.

## Phase 8 acceptance + fixes (2026-10-23)

Install shape: primary `chatgpt-plan` (set 2026-10-22 via the UI switch: toml + config-soll.json + env),
worker `openai-codex` over the ChatGPT login, critic Claude CLI (Max subscription), warm-up off,
no Ollama installed. Real subscription turns used: 3 of 3. Paid API calls: 0.

**Merged to main and integrated into `feat/mobile-gateway`:**
- #37: flash tier (spawn announcer, readback composer) no longer falls back to a stored API key (was Gemini).
  loguru `exc_info=True` is ignored, so 17 handlers now use `logger.opt(exception=True)`.
- #38: chat-seat warm task read `self.config`, which does not exist (every boot failed). Now logs `Chat-seat warm: off`.
- #39: keyless flash fallback requires an installed runtime, so canned lines are used instead of refused localhost calls.
- #40: local-openai without a server URL no longer leads action turns (-1.3 s per turn).
  `llm_usage` records the resolved model (`gpt-6.1-sol`, was empty).
- #41: "Lass/Frag Codex …", "ask/have/let Codex/Claude Code" count as explicit delegation.
  An uninstalled Ollama no longer leads (-13.4 s per action turn).

**Live results:**
| Scenario | Result | Time |
|---|---|---|
| "danke" | one-shot, no model | 1.9 s |
| "wie spät ist es" | one-shot, no model | 0.5 s |
| "Was ist ein Git-Rebase?" | chatgpt-plan/gpt-6.1-sol, 11,706 tokens in / 36 out | 10.3 s |
| Mission, 1st try (before #41) | FAILED: no trigger, Ollama lead timeout, no mission | 22.9 s |
| Mission, 2nd try | dispatched deterministically in 1.0 s; Codex 197 s; critic Claude approve 0.95; APPROVED, empty diff, $0 | 288 s |
| Boot | no keyed flash provider, warm-up "off", no traceback | |

Offline: 576 scenario tests passed. All 44 ollama-related test files and the routing/spawn suites passed. CI gate green on #37-#41.

**Still open (not fixed):**
1. Mission answer quality (F6): `budget.py` has 0 TODOs, but the worker dumped the whole file and the
   critic approved it. The planner wraps every step in "Deliver a complete, polished, production-quality
   result…", which is wrong for a read-only question. The spoken announcement then tried to read a
   code block. Needs a live turn to verify a fix, so it needs approval first.
2. Codex worker and critic tokens are not recorded (`tokens_used 0`); `WorkerSpawned.model` is empty.
3. The critic still escalates to opus automatically (no approval card). Effort is not recorded anywhere.
4. A plain question costs about 11.7k input tokens on the plan seat (system prompt); about 5 s of the first
   turn goes to the OAuth refresh plus the model list (cached 900 s).
5. Progress and result announcements are dropped when no voice session is open. The result is only in
   `~/Downloads/Jarvis-Outputs`.

## F6 + critic model + tokens (2026-10-23, PR #42 merged)

Live check, 1 approved subscription turn, same order as the failure:
| | Before (#41) | After (#42) |
|---|---|---|
| Order the worker received | "/budget.py auflisten …" (path segment "missions" stripped as routing meta) | the full order |
| Brief | "production-quality result" + "complete default deliverable" | READ-ONLY brief |
| Answer | whole file dumped | "0 TODO-Kommentare in jarvis/missions/budget.py gefunden. Es wurden keine Dateien verändert." (correct) |
| Critic | LLM approve 0.95 on the dump | informational shortcut approve 1.0 (NOT independently verified) |
| Duration | 288 s | 158 s |
| Worker tokens | 109,911 (event showed 0) | 48,989 (event shows it) |
| Spoken result | code block | no code; dropped anyway (no voice session) |
Critic model: `choose_critic_model` always returns `sonnet`; no opus escalation, no retry on the CLI default (ADR-0009 amended).
Tests: missions 1905 passed / 5 skipped; plugins/tool 989; the 4 CI guards 663; CI gate green.

**Still open (negative results):**
1. A read order is approved by the informational shortcut without independent verification (confidence 1.0). Fix proposal: a read-only order skips the shortcut and goes to the LLM critic with the READ-TASK rule. That costs one critic call (Claude subscription) per read mission, so it needs a decision.
2. The critic log says "no tool calls" although the codex stream has a `command_execution`: two evidence extractors disagree on codex streams.
3. `MissionApproved.tokens_used` = 0 (not summed); critic tokens are not recorded.
4. The result is not spoken without a voice session; it is file only. See the priority-3 plan.
5. Appshot picker/recorder "preparation failed" with a traceback on every boot since 2026-10-20 (pre-existing, unrelated).
6. The task-kind classifier is a word-list heuristic. Unclear orders default to `change` (old behaviour).
Priority-3 plan (token capture, prompt breakdown, codex context, result in chat): see the PR #42 comment and the handover below.
### Priority 3: token capture, prompt size, result display (analysis + plan, not implemented)

## Measured facts (2026-10-23, no new model turns)
| Point | Finding | Source |
|---|---|---|
| Codex worker tokens | 107,310 input (88,576 cached) + 2,601 output for ONE read-only listing | stream.jsonl of mission 01a1696b |
| Worker tokens in the event | 0, because `ClaudeResult` had no field. **Fixed in #42** (`tokens_used`) | log `tokens=109911` vs `WorkerDraftReady.tokens_used=0` |
| `WorkerSpawned.model` | empty (`''`), so nobody knows which Codex model ran | mission_events |
| Critic tokens | not recorded at all (claude-direct; no usage row) | llm_usage.db |
| Orchestrator fallback | `tokens = tokens_used or total_tokens or num_turns`; `num_turns` is a turn count, not tokens, so the field becomes wrong instead of empty | orchestrator.py ~2519 |
| Plain chat question | 11,706 input / 36 output tokens on chatgpt-plan | llm_usage row 62 |
| First turn after a start | ~5 s OAuth refresh + `GET /v1/models` before the request (cached 900 s) | log 16:51:44 to 16:51:48 |
| Result display | progress and result announcements are dropped when no voice session is open ("no active listening session"); the result only lands as `.md` in Downloads/Jarvis-Outputs | log 17:40 to 17:44 |

## Plan (each step its own PR, in this order)
1. **Usage per mission step** (small): write `WorkerDraftReady.tokens_used`, model and provider into `llm_usage` (caller `mission-worker`). Read critic tokens from the claude `result.usage` frame. Drop the `num_turns` fallback.
   Test: fake streams. No model turn needed.
2. **Prompt breakdown before any trimming** (measure first): one debug log per turn with chars/tokens per block (system persona, identity, memory/wiki, tool hints, history). No change to the prompt yet. Then decide from the numbers which block shrinks.
   Needs 1 subscription turn for the measurement, so it needs approval.
3. **Codex worker context** (measure first): 88k cached input suggests Codex reads AGENTS.md and the repo context itself. Check whether the worker starts in the repo root (it reads the whole AGENTS.md chain) or whether a read-only order could run with a smaller context.
   No change before the measurement.
4. **Result display without a voice session** (UI): post the finished mission result as a chat message into the session that started it. Today it is only spoken (if a session is open) plus a file.
   Touches `missions/voice/announcer.py` + the agent-chat event path; frontend in light and dark mode. Larger, so it needs approval.
5. **First turn ~5 s** (optional): cache the plan seat's model list longer, or resolve it at boot without a model call (the `/v1/models` call is free and not a turn).

## 2026-10-23 evening: PR #43 (Codex stream semantics, read-only sandbox)
- Compared with openai/codex `exec_events.rs`: a no-match `rg` exits 1 -> Codex `status:"failed"`; our critic read that as "no tool calls". Read-order gate now counts a lookup that ran; side-effect evidence stays strict.
- Read order naming a file, answered without any read/search -> deterministic revise (no LLM).
- Read orders run `codex exec --sandbox read-only`.
- MissionApproved.tokens_used = sum of worker drafts; num_turns fallback removed.
- Tests: tests/missions + 4 guards 2502 passed / 5 skipped; CI gate green; merged; app restarted, health 200.
- NOT verified live: read-only sandbox on Windows (needs 1 subscription turn). Open: critic tokens, result display without voice.

## 2026-10-23 late: orchestrator reference research (points 1+2) — research stays OUT of the repo (AGENTS.md)
Proposal in repo: ADR-0043 draft PR #44 (Qualle1980/Dodaios) — subscription chat (chatgpt-plan) conducts every order.
References checked (license -> what we may take):
- microsoft/autogen Magentic-One (MIT, code license LICENSE-CODE): task ledger (facts/plan) + progress ledger JSON (is_request_satisfied, is_in_loop, is_progress_being_made, next_speaker, instruction), stall counter -> replan with "root cause + new plan". Prompts in python/packages/autogen-agentchat/src/autogen_agentchat/teams/_group_chat/_magentic_one/_prompts.py. PORTABLE with MIT notice.
- code-yeongyu/oh-my-openagent (custom license, NOT copyable): task categories (quick, deep-low, deep-high, ultrabrain, visual, artistry) each with an ordered fallback chain over subscriptions + effort variant. Take the IDEA only (packages/model-core/src/category-model-requirements.ts).
- OrchestratorInc/agent-orchestrator (Apache-2.0, Go+TS): orchestrator is itself an agent session using a CLI (spawn/send/status); daemon derives each worker's state from facts (activity, PR, CI) -> board "Working / Needs you / In review / Ready". Pattern for supervision + returning feedback to the same worker.
- musistudio/claude-code-router (MIT): rule-based route by request features (token count, background, thinking) + fallback. Less relevant (API proxy).
- lm-sys/RouteLLM (Apache-2.0): learned strong/weak router; overkill for now.
Our gaps: decomposer plans once (<200 chars never splits); Step has only worker_cli claude|codex; no "answer directly" lane; no progress/loop monitor or replan; global worker provider + fixed effort; 4 parallel orchestration paths (router spawn, Kontrollierer, team threads ADR-0042, subagent_fanout); result only voice/file.
Next (after user OK): implement ADR-0043 behind missions.conductor=off|shadow|on: (1) lanes + cheapest-first chains, (2) progress ledger ported from Magentic-One (MIT notice), (3) answer lane without worker, (4) final answer into chat.

## 2026-10-23 end of day (stop by user)
- User approved ADR-0043 (draft PR #44) and added rules: every error is documented and becomes a lesson so it never repeats (Jarvis + dev work); reuse public GitHub code, attribution only where the license requires it.
- PR #45 merged (+ merged into feat/mobile-gateway): `jarvis/missions/conductor/` lanes (answer/lookup/change/deep, cheapest connected subscription first), ledger (progress record + stall/replan, from Magentic-One, MIT notice), lessons (incidents + AGREE/EDIT/REMOVE/ADD rules with weights, from ExpeL, Apache-2.0 notice), incidents recorder wired at bootstrap -> `data/conductor/incidents.jsonl`. Tests: tests/missions + 4 guards 2516 passed; CI green (first run red: 3 uncommented except handlers, fixed).
- NOT done: app restart after #45 (do it first tomorrow, check health 200 and that incidents.jsonl appears on the next failure). No live turn sent.
- Tomorrow, in order (each its own PR, behind `missions.conductor` off|shadow|on): (2) lesson rule update on the conductor seat after a failure, rules injected into plans; (3) conductor loop using ledger + lanes, shadow first; (4) answer lane without worker; (5) final answer into the chat. Then live acceptance with approved subscription turns.
- Later backlog (user order): memory/answer retrieval, startup, voice + latency, app design, login + DK integration, in-app update button.
