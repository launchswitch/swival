# LSP-as-Context (automatic context injection, no LSP tools)

Status: **Phase A + B implemented and verified**
Date: 2026-06-12
Owner: frank

## Goal

Today Swival's LSP integration exposes `lsp_*` **tools** (definition, references,
hover, diagnostics, symbols…) that the model must choose to call. The vision is
the opposite: the system **automatically derives code context from the running
LSP servers and injects it into the model's context each turn**, with **no LSP
tools advertised**. Inspiration: Pi-Lens, which prepends post-edit findings as
`role:"user"` messages before each LLM call.

This is delivered in two phases. Phase A is the core, low-risk change. Phase B
adds an optional small local LLM that *plans* what code to pull in, since pure
heuristics for relevance are the hard part.

> Reference model for Phase B: `LFM2.5-8B-A1B-exl3-4.10bpw`, served by TabbyAPI
> (OpenAI-compatible, `api_servers: ["OAI"]`) on **`http://localhost:8084/v1`**.
> Confirmed live 2026-06-12. Swival already speaks this dialect via its
> `generic` / `openai/{model}` provider path — **no new provider plumbing**.

---

## Current state (what exists and is reused)

The LSP plumbing stays almost entirely intact. What changes is the *interface* to
the model — tools → injected text.

| Piece | Location | Fate |
|---|---|---|
| `LspManager` + `_LspConnection` (process, wire protocol, asyncio loop) | `lsp_client.py:705`, `:251` | **Keep** |
| Document sync: `on_file_read/write/delete` → `did_open/change/close` | `lsp_client.py:879-906` | **Keep** (this is how LSP learns about files) |
| Diagnostics buffer `_diagnostics: dict[uri, list]` from `publishDiagnostics` | `lsp_client.py:276`, `:641` | **Keep + read** |
| Internal query methods: `definition`, `references`, `hover`, `document_symbols`, `workspace_symbols` | `lsp_client.py:546-641` | **Keep**, reuse internally for Phase B |
| Formatters `_format_diagnostics` etc. | `lsp_client.py:1454+` | **Keep** |
| `_notify_lsp()` + file-tool hooks (read/write/edit/delete) | `tools.py:3448-3708` | **Keep** |
| Tool schemas `LSP_TOOLS` + `list_tools()` + `get_tool_info()` | `lsp_client.py:828, :844, :1231` | **Gated** by `lsp_mode` (off in context mode) |
| `lsp_*` dispatch branch in `dispatch()` | `tools.py:3524` | Dead in context mode (kept for tools mode) |
| LSP tool info appended to system prompt | `agent.py:7253`, `_format_lsp_tool_info:7299` | **Skipped** in context mode |

---

## Architecture after the change

```
file tools (read/write/edit/delete)
        │  _notify_lsp()  (unchanged)
        ▼
LspManager ── did_open/change/close ──► LSP servers
   │                                          │
   │  tracks "dirty" files written this turn  │ publishDiagnostics
   ▼                                          ▼
LspManager.collect_turn_context()  ──►  formatted text
        │   (Phase A: diagnostics for dirty files)
        │   (Phase B: + planner-selected code from small LLM)
        ▼
run_agent_loop: builds effective_messages = messages + [ephemeral context msg]
        │   (canonical `messages` list untouched)
        ▼
call_llm(... effective_messages ...)   ──► model sees context, NO lsp_* tools
```

Key mechanic: the context is **ephemeral and re-derived every iteration** from
live LSP state. It is never appended to the canonical `messages` history, so it
cannot bloat the transcript and **survives compaction automatically** (same
principle as `thinking`/`todo` state). This differs from Pi-Lens, which caches
findings into history once — we instead recompute per turn, which is simpler and
gives faster in-turn feedback (model sees diagnostics from its own just-made
edit on the next iteration).

---

## Phase A — diagnostics injection (core)

### A1. Config: `lsp_mode`  (`config.py`)

Add a tri-state selector so nothing is destroyed and the old path stays A/B-able.

- `CONFIG_KEYS`: add `"lsp_mode": str`  (after `lsp_config`, ~line 107).
- `_ARGPARSE_DEFAULTS`: add `"lsp_mode": "tools"`  (after `lsp_config`, ~line 220).
- Add a CLI flag `--lsp-mode {tools,context,off}` (alongside `--no-lsp`).
  - `off` ≡ today's `--no-lsp` (no manager at all).
  - `tools` = today's behavior (default; backward compatible).
  - `context` = new behavior.
- `--no-lsp` continues to force `off` for backward compat.

### A2. Dirty-file tracking + `collect_turn_context()`  (`lsp_client.py`)

On `LspManager`:

- Add `self._dirty_paths: list[str]` (ordered, deduped) in `__init__`.
- In `_send_notification_sync` (line 958), when `action == "write"`, append the
  resolved abs path to `_dirty_paths` (dedupe; cap length to e.g. 64).
- Add:

  ```python
  def collect_turn_context(self, *, clear: bool = True) -> str | None:
      """Return diagnostics text for files written since the last call.

      Phase A. Returns None when there is nothing to report. When clear=True
      (default) the dirty set is drained so each turn reports only new edits.
      """
  ```

  Implementation: for each dirty path → resolve server → `conn.get_diagnostics(uri)`
  → `_format_diagnostics(diags, rel_path)`. Skip files with no diagnostics. If the
  resulting text is empty, return None. (The diagnostics data is already maintained
  by `_handle_response` from `publishDiagnostics` — no new LSP traffic.)

  > Latency note: `publishDiagnostics` is async. Diagnostics for an edit may land
  > one iteration late. Acceptable for Phase A; Phase B can add an optional short
  > wait. Documented as a known lag.

### A3. Stop advertising tools in context mode  (`agent.py` `_run()` ~7654, `session.py` `_init_lsp` ~516)

In both init paths, branch on `lsp_mode`:

```python
lsp_mode = getattr(args, "lsp_mode", "tools")      # or self.lsp_mode in Session
...
if lsp_servers:
    lsp_manager = LspManager(...)
    lsp_manager.start()
    if lsp_mode == "tools":
        lsp_tools = lsp_manager.list_tools()
        if lsp_tools:
            tools.extend(lsp_tools)
        lsp_tool_info = lsp_manager.get_tool_info()
    # context mode: manager runs for diagnostics/sync only; no tools, no tool_info
```

And in `build_system_prompt` (line 7253): the existing
`if lsp_tool_info and not system_prompt:` guard already suppresses the section
when `lsp_tool_info` is empty — so simply *not populating* it in context mode is
enough. No edit needed there beyond ensuring we pass `{}`.

Session must also accept `lsp_mode` (`session.py` `__init__` + `_build_loop_kwargs`
pass-through) and read it from config.

### A4. Injection seam  (`agent.py` `run_agent_loop` ~8620-8645)

Single insertion point — every retry/repair/recovered call site
(8660, 8793, 8944, 9044) reuses the same `_llm_args`, so one build covers all.

Right before `token_est = estimate_tokens(...)` (line 8621):

```python
# --- LSP-as-context: ephemeral diagnostics for files edited this turn ---
_lsp_context_msg = None
if lsp_manager is not None:
    _lsp_ctx = lsp_manager.collect_turn_context()
    if _lsp_ctx:
        _lsp_context_msg = {
            "role": "user",
            "content": f"[lsp automated context]\n{_lsp_ctx}",
        }

effective_messages = (
    messages if _lsp_context_msg is None else [*messages, _lsp_context_msg]
)
```

Then replace `messages` with `effective_messages` in:
- `token_est = estimate_tokens(effective_messages, effective_tools)`
- `clamp_output_tokens(effective_messages, ...)` (line 8627)
- `_llm_args` element index 2 (line 8638)

`messages` (the canonical history) is never mutated, so compaction, history
viewing, `/restore`, and subagent handoff are unaffected. The context message is
recomputed fresh every iteration from current LSP state.

Role-ordering: we append a `role:"user"` message. Most OpenAI-compatible
backends (TabbyAPI, llama.cpp, LM Studio) accept consecutive same-role messages;
this mirrors Pi-Lens, which injects user-role messages. Flagged as a risk below.

### A5. What becomes dead (kept, not deleted)

- `tools.py:3524` `if name.startswith("lsp_"):` branch — unreachable in context
  mode (no `lsp_*` tools advertised → model can't call them). Left intact so
  `tools` mode keeps working.

### A6. Tests

- `tests/test_lsp_client.py` (or new file):
  - `_dirty_paths` populated on simulated `write` notification; dedup + cap.
  - `collect_turn_context()` formats stored diagnostics, returns None when clean,
    drains dirty set on `clear=True`, preserves on `clear=False`.
- `tests/test_*_lsp_mode`:
  - context mode → `list_tools()` still works but the loop advertises zero
    `lsp_*` tools; `_format_lsp_tool_info` section absent from system prompt.
  - tools mode unchanged (regression).
- A loop-level test (mocked LLM) asserting the injected `[lsp automated context]`
  message is present in the call to `call_llm` and **absent** from the persisted
  `messages` after the turn.

### A7. Manual verification on :8084

Drive `swival` against the live LFM server to confirm injection end-to-end (see
"Manual test plan" below). `make test && make check` must pass.

---

## Phase B — small-LLM relevance planner (optional)

The hard part of "automatically append code" is deciding *what* code. Instead of
pure heuristics, a small, fast local model (LFM2.5-8B-A1B, ~1B active MoE) acts as
a **retrieval planner**, run **once per user turn** (not per inner iteration — too
slow). Four stages, all additive to Phase A's `collect_turn_context`:

### B1. Candidate gathering (no LLM, cheap)

- Extract identifier-like tokens from the user query (CamelCase / snake_case /
  dotted) + `documentSymbol` outlines of files touched this session.
- Resolve each via `workspace/symbol` (reuses `conn.workspace_symbols`, line 631).
- Build a bounded candidate menu `{name, kind, file, line}` — cap ~40, deduped.
- This is what keeps the planner prompt small (never feed it the whole workspace).

### B2. Planner call (the small LLM)

- `litellm.completion(model="openai/LFM2.5-8B-A1B-exl3-4.10bpw",
  api_base="http://localhost:8084/v1", messages=[...])` — no new provider code.
- Prompt: task + candidate menu + hard token budget. Output: JSON
  `[{symbol, file, action}]`, ≤K items.
- **Quirk to handle**: this model emits `reasoning_content` and can return
  `content: null` when starved — set a generous `max_tokens` and parse
  `.message.content`, falling back to scanning `reasoning_content` for a JSON
  block. Reuse Swival's existing JSON-extraction/repair helpers
  (`repair.py` / `tool_call_repair.py`).
- **Timeout**: hard ~4s. On timeout / error / server down → **skip B silently**,
  keep A only.

### B3. Fetch (LSP, not tools)

For each planner pick, call the existing connection methods **internally** (these
are the same methods `_dispatch_tool` wraps, line 1078+): `definition`,
`references`, `hover`. Read the actual source lines for definition bodies. Format
as fenced code blocks with `file:line` headers via existing formatters.

### B4. Merge + inject

Combine B's code blocks with A's diagnostics into the single per-turn
`[lsp context]` message, under a hard token cap (`budget_tokens`, default ~1500).
The planner is told the cap so it prioritizes.

### B config surface (nested)

```toml
lsp_mode = "context"
[lsp_context_planner]
enabled = false                       # opt-in
base_url = "http://localhost:8084/v1"
model = "LFM2.5-8B-A1B-exl3-4.10bpw"
budget_tokens = 1500
max_symbols = 6
timeout = 4.0
```

- Add `lsp_context_planner` to `_NESTED_KEYS` (`config.py:1340`).
- Run async / non-blocking where possible (mirror the notification-queue pattern,
  `lsp_client.py:816`) so a slow planner never stalls the main loop.

### B tests

- Candidate extraction + dedup + cap.
- JSON parse with `reasoning_content` fallback; malformed → empty selection
  (graceful).
- Timeout / connection-refused → Phase A output unchanged.
- Budget enforcement trims to cap.

---

## Manual test plan (Phase A, against :8084)

1. Config in a scratch project with an LSP server (e.g. pyright) + `lsp_mode =
   "context"`.
2. Ask the model to edit a file so it introduces a type error.
3. Confirm on the next iteration the `[lsp automated context]` message appears in
   the outgoing `call_llm` payload (verbose log) with the diagnostic, **and** that
   `tools` advertised to the model contains no `lsp_*` entry.
4. Confirm the model's next response references the injected diagnostic.
5. Repeat with `lsp_mode = "tools"` — regression: `lsp_*` tools present, no
   auto-context message.

---

## Risks / open questions

- **Role ordering**: appending a user message after a user message (first turn).
  Mitigation: the `[lsp automated context]` prefix; most backends tolerate it. If
  a provider rejects it, fallback = fold context into the system prompt for that
  call.
- **Diagnostic lag**: edits surface one iteration late. Acceptable; documented.
- **Token cost on small models**: mitigated by caps in both phases.
- **Phase B quality**: planner may pick irrelevant symbols. Mitigated by budget
  cap, `[lsp context]` framing, and the fact that B is opt-in and degrades to A.
- **`generic` provider model string**: confirm Swival resolves
  `openai/<id>` against a custom `api_base` for the planner (matches the
  `lmstudio` path at `agent.py:3672/4527`). Verify during B implementation.

## Rollout

Default `lsp_mode = "tools"` (no behavior change). Users opt into `context`.
Phase A ships first and is independently useful; Phase B is purely additive
behind `[lsp_context_planner] enabled = false`.

---

## Implementation status (2026-06-12)

Both phases are implemented, lint/format clean, and verified.

**Files added**
- `swival/lsp_context_planner.py` — the small-LLM planner (LLM call injectable
  via `completion_fn`).
- `tests/test_lsp_context_mode.py`, `tests/test_lsp_context_planner.py` — unit
  tests.

**Files modified**
- `swival/config.py` — `lsp_mode` key + default; `--no-lsp`↔`lsp_mode=off`
  reconciliation; `[lsp_context_planner]` added to `_NESTED_KEYS`.
- `swival/lsp_client.py` — `context_mode` flag; dirty + open file tracking;
  `collect_turn_context()`, `dirty_paths()`, `open_paths()`; structured
  `gather_symbol_candidates()` / `fetch_symbol_body()`; `_symbol_range()` /
  `_walk_doc_symbols()` helpers.
- `swival/agent.py` — `--lsp-mode` flag; tool advertisement gated on mode;
  `_build_lsp_context_planner()`; the injection seam (ephemeral
  `effective_messages`, planner runs once per user turn); param threaded through
  `run_agent_loop` + `repl_loop`.
- `swival/session.py` — `lsp_mode` plumbed through constructor/`_setup`/`_init_lsp`.

**Verification**
- Unit tests: 37 new (planner parse/fetch/budget/degradation; `_symbol_range`;
  `fetch_symbol_body` scoping; dirty/open tracking; config reconciliation;
  context-mode gating). Relevant suite: 321 passed.
  - One pre-existing failure (`test_config::…api_key_without_git…`) is
    environmental (stray `.git` from other tests' `git init`) and fails
    identically on clean master — not introduced by this change.
- End-to-end against live services (script at `/tmp/test_lsp_context_e2e.py`):
  real **pyright** + a live OpenAI-compatible model. Confirmed:
  - Phase A: undefined-variable diagnostic injected for an edited file.
  - Phase B: planner gathered document-symbol candidates, the model picked
    `Calculator`/`add`/`subtract`, and scoped source was fetched per symbol.

**Findings that shaped the implementation**
- pyright's `workspace/symbol` returns nothing useful, so the planner's
  candidate source is **document symbols of files opened this session**
  (tracked via `open_paths()`), not workspace search. The planner still works at
  turn-start (before any edit) because reads populate `open_paths`.
- pyright returns document symbols in **SymbolInformation** form (range nested
  under `location`), not DocumentSymbol — `_symbol_range()` handles both.
- Reasoning models (LFM2.5 EXL3, Qwopus) may emit the answer in
  `reasoning_content` or wrap it in `<think>`; the planner reads both fields and
  regex-extracts the JSON, and degrades to `None` (Phase A only) on any
  server/parse failure.

**Not yet wired (deferred)**
- `repl_loop` / subagent paths default `lsp_context_planner=None` (Phase A
  diagnostics still inject there; the planner runs only in the one-shot
  `_run` → `run_agent_loop` path). Forward it into `repl_loop`/subagents if the
  REPL should also get planner-selected code.
- No `lsp_mode` enum validation for TOML values (CLI `--lsp-mode` is
  choice-restricted); an unknown value behaves like `context` minus planner.

