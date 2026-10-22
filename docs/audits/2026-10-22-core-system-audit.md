# Core system audit — chat, orchestration, memory, budgets (2026-10-22)

Read-only audit of the Jarvis core on `feat/mobile-gateway` (HEAD `5c71fe097`).
No code was changed. Evidence is code locations plus read-only queries of the
local `data/*.db` stores. Starting point for the next implementation round.

## 1. ChatGPT integration

| Path | What it is | Can it orchestrate work? |
|---|---|---|
| `chatgpt-plan` (`jarvis/plugins/brain/chatgpt_plan.py`) | Responses API with the plan's access token, `store:false` | **No.** `supports_tools = False` (line 148). `reasoning_effort` is not read. |
| `chatgpt-web` (`chatgpt_web.py`, `chatgpt_web_bridge.py`) | Local bridge process driving the user's own ChatGPT web session in a dedicated browser profile | **No**, text only ("The bridge accepts no tool schemas"). |
| Text-only delegation | JSON envelope `{"route":"delegate",...}` parsed by `parse_route_decision()`; becomes a mission (`BrainManager._route_text_only_delegation`, `manager.py` ~11015) | Only as router of last resort when no tool-capable provider is reachable. Normally a tool-capable provider leads and calls `spawn-worker`. |
| Agent chat service (`jarvis/agent_chat/service.py`) | Sessions, events, approval cards, one running turn per session; picks runner per turn | Yes, but only through tool-capable providers. |

- Jarvis cannot read conversations from the external ChatGPT app. `chatgpt-web`
  creates its own conversations (`dodaios:<session>`).
- Authorized sync paths that exist: the OpenAI data export (imported into the
  external Obsidian vault on 2026-10-07) and the PC Control search/remember
  actions that ChatGPT can call.
- Runtime data: 44 agent-chat sessions (25 on `chatgpt-plan`), 66 turns, only 6
  tool calls. Current chat selection: `chatgpt-web/medium`.

## 2. Chat to worker and back (one real mission)

`missions.db` holds exactly one mission: `APPROVED`, 2026-10-22 15:58 UTC,
16 events, `cost_usd 0`.

| Step | Entry point | Hand-off / events | Error handling | Result stored |
|---|---|---|---|---|
| Chat turn | `AgentChatService` → `BrainManager.generate` | `spawn-worker` (ROUTER_TOOLS) or JSON envelope | one running turn per session; cancel at next boundary | `agent_chat_events` |
| Dispatch | `MissionManager.dispatch` | `MissionDispatched` (hauptjarvis) | PENDING without `run_mission` is marked crashed on next start (BUG-016) | `missions`, `mission_events` |
| Plan | `missions/kontrollierer/orchestrator.py` (decomposer, 1–5 steps) | `MissionPlanReady` | per-mission state lock | event log |
| Worker | worker factory → Codex/Claude direct worker in its own git worktree | `WorkerSpawned`, `WorkerProgress` x7, `WorkerDraftReady` | per-step try/except, semaphore on workers | worktree diff |
| Critic | `CriticRunner` | `CriticVerdictReady` | up to 3 critic loops, then escalation | event log |
| Finish | Kontrollierer | `MissionApproved` / `MissionFailed` | budget overrun → `budget_exceeded` | `missions.state` |

Not evidenced in the data: result written back to memory, crash resume,
multi-step missions, failure paths. `jarvis/orchestrator/` is an empty package;
the real orchestrator is `missions/kontrollierer/orchestrator.py`.

## 3. Model and token governance: enforced or not

| Rule | Status |
|---|---|
| Cheapest suitable model for reading/analysis | Not enforced. `background_policy` keeps background work on subscriptions/local models; no per-task-type selection. |
| Lowest suitable reasoning effort | Not enforced. Codex worker hardcodes `"medium"` (`codex_direct_worker.py:92`); `chatgpt-plan` ignores effort; chat effort is a UI hint. |
| No changes during analysis | Partial. Agent-chat plan mode sets tools read-only (`agent_chat/control.py:386`); nothing equivalent for missions. |
| Higher settings only after user approval | Violated. Critic escalates automatically to `claude-opus-4-8` (iteration 2, confidence < 0.4, or security tag) in `missions/critic/escalation.py`. |
| No uncontrolled retries | Enforced for `chatgpt-plan`/`chatgpt-web` (never re-sent) and by the 3-loop critic cap. |
| Token/time/cost budgets | USD only: $5 per mission, $50 per day (`missions/budget.py`), task/daily budget in the CostMeter. Plan seats price at $0, so no token limit applies. |
| Stop-loss on misbehaviour | Budget abort and stall watchdogs exist; no abort on repeated identical failures. |
| Proof of provider, model, effort | Provider/model per voice turn since `edd2cad6a`. `llm_usage`: 45 of 61 rows are `chatgpt-plan` with an empty model. Effort is never recorded as actually used. |

UI presets are mistaken for enforcement: `agent_chat_sessions.effort` and
`jarvis_chat_selection.effort` are hints that some providers ignore.

## 4. Shared memory

| Question | Current state |
|---|---|
| How is new information stored? | Claude/Codex: `obsidian_memory_sync.py remember` → external vault `60-Projekte/Agent-Learnings` (SHA-256 dedup). ChatGPT: PC Control remember. Jarvis: its own wiki (`wiki/obsidian-vault`, extractor + consolidator). Plus `EXPERIENCE_LOG.md` and `docs/BUGS.md`. |
| Confirmed vs. assumed | Free-text marker "not automatically verified" only; no evidence field (commit, test). |
| Errors and their fixes | `docs/BUGS.md` and the AP register in `AGENTS.md`; coding agents only, Jarvis does not search them. |
| Preventing re-work on solved bugs | Instruction to search first; nothing enforces it. |
| Stale entries | Jarvis wiki has `valid_until` / `superseded-by` (consolidator). External vault and agent learnings do not. |
| Duplicates, private data | Hash dedup exists. One indexed external source contains private conversation content that may reach model context automatically. Needs a privacy filter or source narrowing (unchecked, high priority). |
| GitHub history | No link. The four Jarvis commits from 2026-10-21 are not in any memory source. |

The AK-Matrix memory plan (8 layers, including "experiences: problem → cause →
fix → verification → prevention" and "failure memory") exists as a document in
the external vault; those two layers are not implemented.

## 5. Mechanism comparison (from prior knowledge, not re-checked against current docs)

| Gap | Mechanism elsewhere | What Jarvis has | Minimal extension |
|---|---|---|---|
| Only USD budgets | AutoGen: composable termination conditions (message count, token usage, timeout, external stop; combinable with OR/AND) | USD `BudgetTracker`, 3 critic loops | token + wall-clock limit per mission in the existing `BudgetTracker`, fed from `tokens_in/out` |
| Automatic escalation | AutoGen: selector function chooses the next speaker by rules | `choose_critic_model()` by iteration/confidence | turn escalation into an approval card (agent chat already parks turns on approval futures) |
| No crash resume | LangGraph: checkpointer per `thread_id`, `interrupt()` and resume command | `mission_events` with `seq`; crash → `crashed` | resume a mission from its last completed step in `mission_events` |
| Thin hand-off tracing | OpenAI Agents SDK: hand-offs as tools with input filters; nested trace spans (agent, generation, tool, hand-off) | `parent_event_id`, bus `trace_id`, `source_actor` | provider, model, effort and tokens as fields on `WorkerSpawned`, `WorkerDraftReady`, `CriticVerdictReady` |

## 6. UI (existing surfaces only)

Existing: `MissionsView`, `MissionDeckView`, `MissionControlPanel`, `WorkTrace`,
`AgentChatPanel`, `TeamLeadPanel`, `CostsView`, agent-chat approval cards.
Missions/steps/agent/progress can likely be fed from mission events (current
rendering not verified). Gaps: actual provider/effort (same data gap as §3),
model for `chatgpt-plan` in costs, approvals and stop for missions (open), memory
write-back (none found).

## A–G

- **A. Proven:** ChatGPT chat (`chatgpt-plan`, `chatgpt-web`); one complete
  mission dispatch → approval; agent-chat sessions with tool calls;
  provider/model per voice turn; wiki context from the external vault.
- **B. Implemented, not integrated or not evidenced:** text-only delegation
  envelope; team/plan mode (`agent_chat/team`); multi-step and failure paths;
  budget warnings; `valid_until` (Jarvis wiki only); `tasks/runner`; society /
  team-lead layer.
- **C. Missing:** token/time/effort budgets; escalation only after approval;
  mission resume; confirmed/assumed fields; failure memory Jarvis can search;
  GitHub-history link; privacy filter for external sources.
- **D. Duplicated:** two orchestrator locations (`jarvis/orchestrator` empty,
  `missions/kontrollierer`); three delegation paths (`spawn-worker`, JSON
  envelope, agent-chat team); three memory write paths (Jarvis wiki,
  `obsidian_memory_sync`, `EXPERIENCE_LOG`/`BUGS.md`); two cost meters
  (`CostMeter`, `BudgetTracker`).
- **E. Three highest-impact changes:**
  1. provider/model/effort/tokens on every mission event + token/time budget
     in `BudgetTracker`;
  2. critic escalation only via an approval card;
  3. every mission result as an agent learning with required fields (cause,
     change, test, commit, verified/assumed) via the existing `remember`, plus
     the privacy check of the external sources.
- **F. First end-to-end test:** a read-only mission ("list the TODOs in
  `jarvis/missions/budget.py`"); check plan → worker → critic → `APPROVED`,
  provider/model/tokens per event, and an empty worktree diff. Only free on a
  subscription or local model — verify that before running it.
- **G. Files for the next round:** `jarvis/missions/events.py`,
  `jarvis/missions/budget.py`, `jarvis/missions/critic/escalation.py`,
  `jarvis/missions/critic/runner.py`,
  `jarvis/missions/kontrollierer/orchestrator.py`,
  `jarvis/missions/workers/codex_direct_worker.py`,
  `jarvis/plugins/brain/chatgpt_plan.py`, the external-source configuration.
