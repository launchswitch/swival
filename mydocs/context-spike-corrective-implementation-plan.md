# Context Spike Corrective Implementation Plan

Date: 2026-06-12
Baseline: current worktree after the LSP-as-context spike
Primary test server: `http://localhost:8083/v1`

## Goal

Reduce context-window bloat without breaking the current LSP work. The immediate
priority is correctness and containment:

1. Make `lsp_mode` behavior explicit and testable.
2. Stop losing late LSP diagnostics.
3. Add symbol-aware reads to both single and batched file reads.
4. Add targeted reduction for large recent file-read tool results.
5. Make project-map context adaptive, query-relevant, and cheap by default.
6. Use the local small model only as a bounded retrieval planner.

The work should be delivered in small phases. Phase A correctness must land
before broader Phase B/project-map changes.

## Current State To Preserve

The current worktree already contains a working LSP-as-context spike:

- `swival/config.py` has `lsp_mode = "tools" | "context" | "off"`.
- `swival/agent.py` creates an `LspManager(context_mode=(lsp_mode == "context"))`.
- `swival/agent.py::run_agent_loop()` injects ephemeral LSP context only when
  `getattr(lsp_manager, "context_mode", False)` is true.
- `swival/lsp_client.py::collect_turn_context()` formats diagnostics for dirty
  files.
- `swival/lsp_context_planner.py` implements a best-effort small-LLM symbol
  selector.

Those are useful foundations. The corrective work below hardens the boundaries
and fixes the bloat failure modes that remain.

## Phase 0: Lock The Baseline

### Implementation

- Add focused regression tests before changing behavior:
  - `lsp_mode="tools"` never injects `[lsp automated context]`.
  - `lsp_mode="context"` does not advertise `lsp_*` tools.
  - `lsp_mode="off"` does not create an LSP manager.
  - Existing `read_file` and `read_multiple_files` behavior is unchanged when no
    new symbol parameter is used.
- Keep the existing `mydocs/lsp-as-context.md` as the spike note. This document
  is the implementation plan for the corrective pass.

### Verification

Run:

```bash
python -m pytest tests/test_lsp_context_mode.py tests/test_lsp_context_planner.py tests/test_session_lsp.py
python -m pytest tests/test_run_command.py tests/test_read_multiple_files.py
```

## Phase 1: Make LSP Context Mode Explicit

### Problem

`run_agent_loop()` currently decides whether to inject context by looking at a
mutable property on `lsp_manager`:

```python
if lsp_manager is not None and getattr(lsp_manager, "context_mode", False):
```

That is safer than checking only `lsp_manager is not None`, but it still couples
loop behavior to manager internals. The loop should receive explicit intent from
the caller.

### Implementation

- Add `lsp_context_enabled: bool = False` to `run_agent_loop(...)`.
- Thread the flag from the two loop call sites that already know `lsp_mode`:
  - CLI path around `swival/agent.py` LSP setup.
  - `Session` loop kwargs in `swival/session.py`.
- Keep `LspManager.context_mode` for manager-local behavior and tests, but do
  not use it as the loop's source of truth.
- Build the planner only when `lsp_context_enabled` is true.
- Gate both diagnostics and planner injection on `lsp_context_enabled`.

Expected loop condition:

```python
if lsp_context_enabled and lsp_manager is not None:
    ...
```

### Tests

- Mock `lsp_manager.context_mode = True` while passing
  `lsp_context_enabled=False`; assert no context injection.
- Mock `lsp_manager.context_mode = False` while passing
  `lsp_context_enabled=True`; assert the loop still attempts injection using the
  explicit flag.
- Assert `lsp_mode="tools"` can still notify LSP on read/write without automatic
  context injection.

## Phase 2: Retain Dirty Paths Until Diagnostics Are Observed

### Problem

`LspManager.collect_turn_context(clear=True)` currently drains `_dirty_paths`
before diagnostics are guaranteed to have arrived. Since
`publishDiagnostics` is asynchronous, diagnostics can appear one iteration late
and be missed permanently.

### Implementation

Replace the drain-on-read list with pending dirty entries that survive for a
small turn TTL.

Recommended shape in `swival/lsp_client.py`:

```python
@dataclass
class _DirtyPath:
    path: str
    turns_remaining: int = 3
    observed_diagnostics: bool = False
```

Use an ordered dict keyed by absolute path rather than a list:

- On write:
  - Insert or refresh the path.
  - Set `turns_remaining = dirty_ttl` (default 3).
  - Preserve insertion order and cap to `_dirty_cap`.
- On `dirty_paths()`:
  - Return the pending path keys.
- On `collect_turn_context()`:
  - Query diagnostics for each pending path.
  - If diagnostics exist, include them and remove the path after formatting.
  - If no diagnostics exist, decrement `turns_remaining`.
  - Remove only when `turns_remaining <= 0`.
- Keep `clear=False` as a snapshot mode for tests and planner candidate
  gathering; it must not mutate pending state.

This intentionally waits for either diagnostics or TTL expiry. It avoids both
permanent stale dirty state and lost one-turn-late diagnostics.

### Tests

- Dirty path with no diagnostics remains after first `collect_turn_context()`.
- Diagnostics arriving on the second collection are included.
- Dirty path expires after 3 empty collections.
- `clear=False` does not decrement TTL.
- Deduping a repeated write refreshes TTL and does not duplicate the path.

## Phase 3: Add Symbol-Aware `read_file`

### Problem

The model still often reads whole files. `outline.py::symbol_spans()` already
knows symbol ranges, but `read_file` cannot use it.

### Implementation

- Extend `read_file` schema in `swival/tools.py`:

```json
"symbol": {
  "type": "string",
  "description": "Optional top-level symbol name to read. When provided, Swival reads only that symbol's definition span."
}
```

- Add an internal helper:

```python
def _resolve_symbol_read_range(text: str, file_path: str, symbol: str) -> tuple[int, int] | str:
    ...
```

- Use `swival.outline.symbol_spans(text, file_path)` after UTF-8 decode.
- If a symbol resolves:
  - Override `offset` and `limit`.
  - Ignore `tail_lines`.
  - Use `span.start` and `span.render_end`.
- If no symbol resolves:
  - Return a helpful error that names the file and suggests `outline`.
- Include the selected symbol in the continuation/status text when practical.

Do not make fuzzy matching part of the first pass. Exact symbol names are easier
to reason about and test.

### Tests

- Python function returns only its body span.
- Python class returns class span.
- Missing symbol returns an error.
- `symbol` plus `tail_lines` returns an error or deterministically ignores tail;
  prefer an error because the request is ambiguous.
- Checksums are still appended from the full file bytes.

## Phase 4: Add Symbol-Aware `read_multiple_files`

### Problem

Agents often batch reads. If only `read_file` supports symbol spans, the model
can still dump large files through `read_multiple_files`.

### Implementation

- Extend each `read_multiple_files.files[]` item with `symbol`.
- Parse `symbol = spec.get("symbol")`.
- Pass `symbol` through to `_read_file(...)`.
- Update `_format_read_request(...)` to include symbol:

```text
symbol=MyClass
offset=10 limit=40
tail=80
```

- Treat `symbol` as mutually exclusive with `offset`, `limit`, and `tail_lines`
  unless a clear use case emerges. This prevents confusing partial-symbol reads.
- Preserve existing per-file inline errors.
- Ensure LSP `didOpen` notifications still get useful content. If a symbol read
  is partial, do not pretend it is a full-file read in `full_contents`.

### Tests

- Batch with two symbol reads returns two bounded sections.
- Batch with one missing symbol and one valid symbol reports one error and one
  success.
- Existing string-only batch entries still read as before.
- Batch output respects `MAX_OUTPUT_BYTES`.

## Phase 5: Target Recent Large Tool Results

### Problem

General compaction preserves recent turns. If the most recent tool result is a
giant `read_file` or `read_multiple_files`, it can survive untouched and keep
the request bloated.

### Implementation

Add a targeted reducer in `swival/agent.py` that runs before estimating tokens
for the next LLM call.

Suggested helper:

```python
def reduce_recent_large_file_tool_results(messages: list, *, keep_tail_turns: int = 2, threshold_chars: int = 6000) -> int:
    ...
```

Behavior:

- Group messages into turns with `group_into_turns(messages)`.
- Inspect only the last `keep_tail_turns` turns.
- For tool messages whose tool name is `read_file` or `read_multiple_files` and
  content exceeds `threshold_chars`, replace content with
  `compact_tool_result(name, args, content)`.
- Preserve checksums where possible. For file reads, include:
  - path(s)
  - original line count / char count
  - checksum if present
  - instruction to re-read with `symbol`, `offset/limit`, or `outline`
- Do not compact `edit_file`, `write_file`, or command output in this targeted
  pass.

Call it after tool dispatch bookkeeping and before the next token estimate. Add
a verbose log line when it reduces at least one result.

### Tests

- Recent large `read_file` result is compacted before the next LLM call.
- Small reads remain unchanged.
- Non-file tool results remain unchanged.
- Tool-call/result pairing stays valid after replacement.

## Phase 6: Adaptive Project Map

### Problem

A fixed 4K project map in every request is a permanent tax. The map should be
cheap by default and expand only when the context window and repo size justify
it.

### Implementation

Add a project-map builder that prefers existing cartography data:

1. Read `.slim/symbols.json` and `.slim/imports.json` when present.
2. Fall back to `outline.py` only when `.slim` is missing or stale enough to be
   useless.
3. Produce a query-relevant map when a user question is available:
   - Extract identifiers and path-like tokens from the latest user task.
   - Rank symbols/files by direct token match, imported-by proximity, entrypoint
     hints, and recently opened/dirty paths.
4. Budget adaptively:
   - Default: 1500-2500 tokens.
   - Expand only for large context windows, e.g. up to 4000 tokens when
     `context_length >= 64k`.
   - Shrink or skip for very small windows.
5. Inject as ephemeral context, not permanent transcript history.

Config:

```toml
project_map = "auto"       # auto | off | always
project_map_min_tokens = 1500
project_map_max_tokens = 4000
```

Recommended first implementation: keep this independent from LSP. It should
work even when `lsp_mode="off"`.

### Tests

- Uses `.slim` data when present.
- Falls back cleanly when `.slim` is absent.
- Does not exceed configured budget.
- Query mentioning a symbol ranks that symbol's file above unrelated files.
- `project_map="off"` injects nothing.

## Phase 7: Small-LLM Planner Hardening On `localhost:8083`

### Problem

The small planner is useful as a symbol selector, but only if failure is cheap.
The earlier `localhost:8084` test showed `content: null` and partial JSON in
`reasoning_content` under tight `max_tokens`. For this pass, test against
`localhost:8083` with the server running `parallel=2`, using it for both the
main agent and the planner.

### Implementation

- Update local test config examples to use:

```toml
provider = "generic"
base_url = "http://localhost:8083/v1"
model = "Qwopus3.6-27B-Coder-MTP-Q4_K_M.gguf"
lsp_mode = "context"

[lsp_context_planner]
enabled = true
base_url = "http://localhost:8083/v1"
# The 8083 server serves a single 27B Coder model; use it for the planner too.
model = "Qwopus3.6-27B-Coder-MTP-Q4_K_M.gguf"
budget_tokens = 1500
max_symbols = 6
candidate_cap = 40
timeout = 4
response_tokens = 512   # headroom for reasoning models to emit complete JSON
```

- Increase planner response allowance from `max_tokens=256` to at least `512`,
  preferably configurable as `response_tokens`.
- Extract JSON from both `content` and `reasoning_content`; current
  `_default_completion()` concatenates both, so add tests for malformed leading
  prose and partial reasoning blocks.
- Keep planner cadence once per user turn, never once per inner loop iteration.
- On timeout, malformed JSON, empty picks, or server error: return `None`
  silently unless verbose logging is enabled.

### Tests

- Planner parses `{"pick":[0,2]}` from content.
- Planner parses the same JSON from reasoning text.
- Planner ignores malformed output.
- Planner timeout returns `None`.
- Planner response token config is threaded from `[lsp_context_planner]`.

### Manual Test

With the 8083 server already running:

```bash
swival --provider generic \
  --base-url http://localhost:8083/v1 \
  --model 'Qwopus3.6-27B-Coder-MTP-Q4_K_M.gguf' \
  --lsp-mode context \
  'make a small edit that triggers a Python diagnostic, then fix it'
```

Expected:

- No `lsp_*` tools are advertised in context mode.
- Diagnostics from the bad edit appear in `[lsp automated context]` within 1-3
  loop iterations.
- Planner context, if configured, contains only a few selected symbols.
- Failure of planner output does not interrupt the main agent.

## Implementation Order

1. Phase 0: baseline tests.
2. Phase 1: explicit `lsp_context_enabled`.
3. Phase 2: dirty-path TTL.
4. Phase 3: symbol-aware `read_file`.
5. Phase 4: symbol-aware `read_multiple_files`.
6. Phase 5: recent large file-result reducer.
7. Phase 7: planner hardening on `localhost:8083`.
8. Phase 6: adaptive project map.

The project map is intentionally last. It is valuable, but the first five phases
remove correctness hazards and the biggest accidental dumps without adding a
permanent context tax.

## Acceptance Criteria

- `lsp_mode="tools"` and `lsp_mode="context"` have distinct, tested behavior.
- Late diagnostics are not lost after a write.
- The model can read a symbol span through both single and batched file reads.
- Recent large file reads are reduced before the next LLM call.
- The planner is best-effort and bounded; bad output is a no-op.
- Project-map injection is adaptive and uses `.slim` data when available.
- Existing tests plus new focused tests pass.

