# AGENTS.md

This file is the agent navigation and debugging guide for Swival, a CLI-based AI
coding agent. It is auto-generated cartography + traced lifecycles + hard
invariants. For project overview, commands, and module roles, see `CLAUDE.md`
in this same directory — they are complementary, not duplicates.

## Codebase Navigation (`.slim/`)

This repo has 53 source modules and 98 test files (153 .py files total).
Use the auto-generated structured data in `.slim/` to narrow your search
before reading source files:

1. **`.slim/symbols.json`** — every export: name, kind, line number, signature
   (2684 exports across 153 files)
2. **`.slim/imports.json`** — full bidirectional import graph
   (150 files with imports, 506 import edges)
3. **`.slim/cartography.json`** — file hashes for incremental updates

**`imports.json` schema** (this Python cartographer produces bidirectional edges):
- Top-level keys: `_metadata`, `files`. Per file: `imports[]` and `importedBy[]`.
- `imports[].from` — module path being imported
- `imports[].names` — symbols imported
- `importedBy[]` — reverse index (same shape)
  - "Who calls this function?" → search `importedBy` for the name
  - "Blast radius of changing X?" → trace `importedBy` transitively
  - "What does this file import?" → look up `files.{path}.imports`

To regenerate after code changes:
```bash
python3 ~/.hermes/skills/cartography/scripts/cartographer_py.py init \
  --root ./ --include '**/*.py' \
  --exclude '.venv/**' --exclude 'venv/**' --exclude 'node_modules/**' \
  --exclude '.swival/**' --exclude 'build/**' --exclude 'dist/**' \
  --exclude 'local-wheels/**'

python3 ~/.hermes/skills/cartography/scripts/cartographer_py.py extract --root ./
python3 ~/.hermes/skills/cartography/scripts/cartographer_py.py update --root ./
```

A daily cron job refreshes this automatically (see "Cron job" at the end).

## Request Lifecycles

### CLI entry: `swival "task"`

```
swival "task"
  → swival/agent.py::main()                       [line 5957]
    → build_parser()                              — argparse
    → load_config(base_dir)                        [config.py]
      → merges ~/.config/swival/config.toml + <base_dir>/swival.toml
    → resolve_profile_config(args, file_config)   [config.py]
    → apply_config_to_args(args, file_config)      [config.py]
    → _should_try_onboarding() → run_onboarding() [onboarding.py]
    → dispatches: REPL | one-shot | A2A | ACP | reviewer

REPL path: start_repl(args) → run_agent_loop_repl()    [agent.py]
One-shot:  Session.run(question)                         [session.py:638]
A2A:       uvicorn swival.a2a_server:app
ACP:       swival --acp → AcpServer.serve()             [acp_server.py:120]
```

### Single LLM turn (system prompt → user → tool calls → next turn)

```
run_agent_loop(messages, tools, **kwargs)         [agent.py:8180]
  while turns < max_turns:
    ├─ turns += 1                                  [agent.py:8517]
    ├─ snapshot_state.inject_into_prompt()         [agent.py:8565]
    ├─ _canonicalize_tool_calls(messages)          [agent.py:8578]
    ├─ estimate_tokens(messages, tools)            [agent.py:8598]
    ├─ call_llm()                                  [agent.py:4397]
    │   ├─ llm_filter.run_llm_filter()             [filter.py]  (optional)
    │   ├─ _escape_special_tokens_in_messages()    [agent.py:797]
    │   ├─ secret_shield.encrypt_messages()        [secrets.py] (if enabled)
    │   ├─ litellm.completion(...)                 — provider-specific kwargs
    │   │   (lmstudio | llamacpp | huggingface | openrouter
    │   │    | google | geap | chatgpt | bedrock | generic | command)
    │   ├─ _patch_chatgpt_responses_empty_output() [agent.py:709]
    │   └─ returns (msg, finish_reason, ...)
    ├─ _maybe_scavenge_tool_calls()                [agent.py:1200+]
    │   → tool_call_repair.scavenge_tool_calls()   [tool_call_repair.py:214]
    ├─ _raise_if_truncated_tool_call()             — repair_truncated_json()
    ├─ [ContextOverflowError handler]              [agent.py:8660]
    │   → graduated compaction: compact_messages()
    │   → drop_middle_turns() → aggressive_drop_turns()
    │   → drop all tools → emergency truncation (50%/25%/10%)
    ├─ messages.append(_msg_to_dict(msg))          [agent.py:9216]
    ├─ [text tool-call leak check]                 [agent.py:9222]
    │   → _classify_textual_tool_call_leak()       [agent.py:877]
    ├─ [if no tool_calls]: final answer            [agent.py:9387]
    └─ [tool calls present]: execute each
         ├─ storm_breaker.inspect()                 [agent.py:9417]  (repeat-loop guard)
         ├─ handle_tool_call(tool_call)             [agent.py:3289]
         │   ├─ repair_tool_args()                  [repair.py:100]
         │   └─ tools.dispatch(name, args, ...)     [tools.py:3393]
         ├─ messages.append(tool_msg)               [agent.py:9505]
         └─ _post_tool_bookkeeping()                [agent.py:9507]
            → updates thinking/todo/snapshot/goal state
```

### Audit mode (multi-phase security review)

```
/audit <args> command
  → run_audit_command(cmd_arg, ctx)               [audit.py:5037]
    → _run_audit_phases()                          [audit.py:5130]
      → _run_pipeline()                            [audit.py:5171]
        → _run_pipeline_body()                     [audit.py:5203]

Phase 1 — Inventory:
  _load_file_contents() + _build_context_indices() + _order_by_attack_surface()
  + _phase1_repo_profile()  (LLM call for repo profile)

Phase 2 — Triage (threaded, default 4 workers):
  _run_batch(_triage, pending, max_workers=workers) [audit.py:5332]
    → _phase2_triage_one(path, state, ctx)  — LLM call per file

Phase 3 — Deep Review (threaded):
  _run_batch(_deep_review, candidates, ...)
    → _phase3_deep_review_one(path, state, ctx)  — FindingRecord

Phase 4 — Verification (threaded):
  _phase4_verification_one(finding, state, ctx)  — verifier LLM call

Phase 5 — Patch Generation:
  _phase5_patch_one(finding, state, ctx)
    → run_agent_loop() in sandbox                  [agent.py:8180]  (max 50 turns)
```

**Phase ordering is strict** (init → triage → review → verify → artifacts →
adjudication). Skipping loses state later phases depend on. Workers
report via queue, not barrier — using a barrier would stall when workers
finish at different rates. This is the "audit workers saturated instead
of stalling" invariant.

### A2A server (HTTP request → agent invocation)

```
uvicorn swival.a2a_server:app                     — Starlette app
POST /message (METHOD_SEND_MESSAGE)
  → A2aServer._handle_message()                   [a2a_server.py:200+]
    → auth check (HMAC token)
    → create A2aTask record
    → Session.ask(question, parse_commands=True)  [session.py:822]
      → run_agent_loop()                          [agent.py:8180]
        → event_callback emits: TEXT_CHUNK, TOOL_START, TOOL_FINISH, TOOL_ERROR

POST /streaming_message (METHOD_SEND_STREAMING_MESSAGE)
  → StreamingResponse — yields SSE events

GET  /tasks/{id}        → task state
POST /tasks/{id}/cancel → sets cancel_flag Event
```

### ACP server (stdio JSON-RPC → agent)

```
Editor spawns: swival --acp
  → AcpServer.serve()                             [acp_server.py:120]
    → asyncio.run(_run())                         [acp_server.py:145]
    → StreamReader on stdin
    → _handle_line(line) → _dispatch(req)         [acp_server.py:182]

session/new   → create AcpSession + Session
session/prompt → Session.ask(question)
                 → run_agent_loop()
                 → event_callback → stdout JSON-RPC notification
session/cancel → AcpSession.cancel_flag.set()
                 → run_agent_loop checks cancel_flag.is_set() each turn
                 [agent.py:8519]
```

### Tool execution (`read_file`, `edit_file`, `write_file`, `run_command`)

```
handle_tool_call(tool_call)                       [agent.py:3289]
  ├─ repair_tool_args()                           [repair.py:100]
  └─ tools.dispatch(name, args, ...)              [tools.py:3393]
       │
       ├─ "read_file"   → _do_read_file()         [tools.py:2000+]
       │   → path resolution + sandbox check
       │   → checksum (sha256)
       │   → returns formatted output with [checksum=...] trailer
       │
       ├─ "edit_file"   → _do_edit()              [tools.py:2100+]
       │   → read current content
       │   → validate checksum if provided        — INVARIANT: see #13 below
       │   → patch(mode='replace', ...)          [edit.py]  (9 fuzzy strategies)
       │   → return fresh checksum trailer
       │
       ├─ "write_file"  → _do_write()
       │
       └─ "run_command" → _do_run_command()       [tools.py:2500+]
           → CommandPolicy.check()                [command_policy.py]
           → sandbox execution
             ├─ builtin: subprocess.run() w/ timeout
             ├─ agentfs: sandbox_agentfs.py
             └─ nono:   sandbox_nono.py  (network blocking)
           → TerminalSink streaming for long output
```

### Subagent invocation

```
spawn_subagent tool call
  → SubagentManager.spawn()                       [subagent.py:169]
    → Thread(target=_subagent_loop)
      → run_agent_loop()                          [agent.py:8180]
        — uses SA_TEMPLATE_EXCLUDE to strip per-run state
        — shares tools (minus spawn_subagent / check_subagents)
        — composite cancel flag (parent + own)
    → returns SubagentHandle with id

check_subagents tool call
  → SubagentManager.check()                       [subagent.py]
    → poll / collect (wait) / cancel
```

### Session persistence and memory

```
Session.run() / Session.ask()
  ├─ Session._setup()                             [session.py:261]  (one-shot)
  ├─ Session._system_with_memory()                [session.py:501]
  │   → load_memory()                             → injects .swival/memory/MEMORY.md
  │                                                 (BM25 if memory_full)
  ├─ append_history(base_dir, question, answer)
  │   → appends to .swival/history/<date>.jsonl
  └─ _write_trace(messages)                       → traces.write_trace_to_dir()
                                                   [traces.py]
```

## Hard Invariants

### Agent loop
1. **Messages list is mutated in-place** — `run_agent_loop()` appends to
   `messages` directly; caller shares the reference. Mutating outside the
   loop duplicates messages on retry. (`agent.py:8180`, `9216`)
2. **`_setup_done` is a one-time gate** — `Session._setup()` returns
   immediately if already run. Calling twice double-initializes
   MCP/A2A managers. (`session.py:261`)
3. **`cancel_flag` checked at start of each turn AND each tool_call** —
   missing a check causes the loop to not respect cancellation
   mid-turn. (`agent.py:8519`, `9399`)
4. **`_swival_synthetic` messages must be last** in the messages list —
   inserting them mid-list breaks context for downstream code paths.
   (`agent.py:9154`, `9213`)
5. **`turn_offset` for report/goal accounting** — when `run_agent_loop` is
   called with a non-zero offset (subagent path), all turn numbers in
   reports must be offset. (`agent.py:8209`, used at `8501`, `8648`)

### Audit
6. **Phase ordering is strict** — init → triage → deep_review →
   verification → artifacts → adjudication. Skipping phases loses
   state later phases depend on. (`audit.py:5218`, `5308`)
7. **`_run_batch` uses chunk-based queuing, not barrier** — workers
   report via queue, not a shared counter. A barrier stalls when
   workers produce at different rates. (`audit.py:5332`)
8. **Mandatory files are loaded in Phase 1 before triage** — triage
   workers expect file contents in `_content_cache`. Violating
   raises `KeyError` in workers. (`audit.py:5227`, `5231`)
9. **`force_review` must be re-resolved on resume** — `swival.toml` may
   have changed between runs. Stale promotion decisions get reused.
   (`audit.py:5271`)

### A2A / ACP
10. **A2A `cancel_flag` is a `threading.Event`** — set externally by
    cancel endpoint; checked in `run_agent_loop` at turn boundaries.
    Without this, tasks don't honor cancel. (`a2a_server.py:89`,
    `session.py:224`)
11. **ACP `AcpSession.in_flight` is an `asyncio.Task`** — must be
    awaited before cleanup. Otherwise the task never completes and
    the session leaks. (`acp_server.py:93`, `171`)
12. **A2A task state transitions are monotonic** — WORKING → COMPLETED /
    FAILED / CANCELED and never back. (`a2a_server.py:89`)

### Tools
13. **Checksum must be passed from read → edit** — `edit_file` accepts
    an optional checksum; without it, concurrent modifications go
    undetected and changes get silently clobbered. (`tools.py:202`,
    `edit.py`)
14. **`dispatch()` is the single routing function** — every tool
    implementation must be registered there. Otherwise `KeyError` or
    the wrong tool runs. (`tools.py:3393`)
15. **Sandbox mode set at Session creation, not runtime** — switching
    sandbox mid-session is not supported. (`session.py:81`)

### Config
16. **`_UNSET` sentinel is distinct from `None`** — `None` means
    "explicitly disabled", `_UNSET` means "not set by user." Mixing
    them returns wrong defaults. (`config.py:20`, `session.py:148-154`)
17. **Provider must be resolved before `_context_length` is known** —
    MCP tool token budgets depend on context length. Resolving the
    provider after MCP starts miscalculates the budget.
    (`session.py:291-298`, `406-413`)

### Session / memory
18. **`_transcript_rollback` only rolls back the `messages` list** —
    state objects (thinking, todo, snapshot, file_tracker) are NOT
    rolled back on failure. Result: partial progress preserved
    but transcript reset. (`session.py:766`, `836-840`)
19. **Lifecycle exit hook runs at most once** —
    `_lifecycle_exit_ran` flag prevents double-run. Repeated
    `close()` corrupts remote sync. (`session.py:930`)

### Subagent
20. **`SA_TEMPLATE_EXCLUDE` keys cannot appear in subagent loop
    kwargs** — these are per-run mutable state. Subagent sharing
    parent's thinking/todo state causes cross-contamination.
    (`subagent.py:25`)

## Debugging Guide

### 1. Check logs first
- REPL: rich-formatted output, errors in red
- A2A server: standard uvicorn logs, JSON request/response bodies
- ACP server: stdio JSON-RPC frames (use `--log-level debug` on the
  spawning editor)
- Audit: per-phase progress in `audit_ui.py` thread, findings in
  `.swival/audit/` per run

### 2. Trace by symptom

**Model returns malformed tool calls**
  → `litellm.completion()` response
  → `agent.py::_has_malformed_tool_args()` (`agent.py:9174`)
  → `_maybe_scavenge_tool_calls()` → `tool_call_repair.scavenge_tool_calls()`
    [tool_call_repair.py:214]
  → `repair_truncated_json()` [repair.py:100]
  → If still bad, repair prompt injected; check `agent.py:9222`

**Command times out / hangs**
  → `tools.py::_do_run_command()` [tools.py:2500+]
  → `subprocess.run()` with `timeout=` argument
  → For background processes: `keepawake.py`
  → For network: `sandbox_nono.py` blocking
  → Check `tools.py:2500+` for the specific timeout config

**Session doesn't resume**
  → `Session.ask()` → `_transcript_rollback` [session.py:766]
  → messages reverted but `_conv_state` reused → new `setup()` skipped
  → Check `continue_here.py::write_continue_file()` for the on-disk
    resume marker
  → `session.py:854` `_conv_state` cache lifetime

**Edit fails / file already changed**
  → `tools.py::_do_edit()` [tools.py:2100+]
  → checksum mismatch → `patch()` fails
  → Check tools.py:202 (checksum param), edit.py (patch modes)
  → 9 fuzzy-match strategies; pick the one matching your diff

**Provider auth fails**
  → `agent.py::call_llm()` [agent.py:4397]
  → litellm → provider auth flow
  → `_is_transient()` [agent.py:589] decides retry vs. surface
  → Provider-specific kwargs: `agent.py:4450+`

**Audit loop stalls**
  → `audit.py::_run_batch()` [audit.py:5332]
  → workers blocked on queue → chunk size too small relative to
    worker count
  → Phase logic: `audit.py:5309` (triage), `audit_ui.py` (UI
    thread signaling)
  → If you see repeated "queue empty, waiting" messages, increase
    chunk size

**Tool result too large**
  → `tools.py::dispatch()` [tools.py:3393] returns tool result
  → exceeds `max_output_lines` / `max_output_kb` → truncated
  → `fmt.py` truncation formatting
  → Token estimation: `agent.py:8598` `estimate_tokens()`

### 3. Hot files

| Symptom | Likely File | Function |
|---|---|---|
| Malformed tool calls | `tool_call_repair.py:214` | `scavenge_tool_calls` |
| Malformed tool calls | `agent.py:9174` | `_has_malformed_tool_args` |
| Malformed tool calls | `repair.py:100` | `repair_truncated_json` |
| Command timeout | `tools.py:2500+` | `_do_run_command` |
| Command timeout | `keepawake.py` | `keep_awake` |
| Session won't resume | `session.py:766` | `_transcript_rollback` |
| Session won't resume | `session.py:854` | `_conv_state` cache |
| Session won't resume | `continue_here.py:1` | `write_continue_file` |
| Edit checksum fail | `tools.py:202` | `checksum` param |
| Edit checksum fail | `tools.py:2100+` | `_do_edit` |
| Edit checksum fail | `edit.py` | `patch` |
| Provider auth | `agent.py:4397` | `call_llm` |
| Provider auth | `agent.py:589` | `_is_transient` |
| Audit stall | `audit.py:5332` | `_run_batch` |
| Audit stall | `audit.py:5309` | triage phase |
| Tool output too large | `tools.py:3393` | `dispatch` |
| Tool output too large | `fmt.py` | truncation |
| Tool output too large | `agent.py:8598` | `estimate_tokens` |

### 4. Key grep patterns

```bash
# Find tool implementations
grep -n "^def _do_" swival/tools.py

# Find session lifecycle hooks
grep -n "_setup_done\|_transcript_rollback\|cancel_flag" swival/session.py

# Find audit phase boundaries
grep -n "_phase[0-9]\|_run_batch\|_run_pipeline" swival/audit.py

# Find LLM call sites (every provider dispatch)
grep -n "litellm.completion\|call_llm(" swival/agent.py

# Find MCP tool registration
grep -n "register_mcp_tools\|mcp_manager" swival/agent.py swival/mcp_client.py

# Find subagent exclusions
grep -n "SA_TEMPLATE_EXCLUDE" swival/subagent.py

# Find provider-specific kwargs
grep -n "lmstudio\|llamacpp\|huggingface\|openrouter\|bedrock" swival/agent.py
```

## Cron job

Daily cartography refresh runs automatically:

```
schedule: "0 0 * * *"  (midnight local)
deliver:  "local"
workdir:  /home/frank/repos/swival
```

The job runs `cartographer_py.py changes` → `extract --changed-only` →
`update` (skipping if no changes). See the job in the Hermes scheduler
(`cronjob` action='list') to inspect or modify.
