# Phase 6 (Adaptive Project Map) — Deferred

Date: 2026-06-13
Status: Deferred from the corrective implementation pass (Phases 0–5 + 7 landed).

## Decision

Phase 6 is intentionally **not implemented** in this pass. The corrective pass
landed Phases 0–5 and 7, which remove the correctness hazards and the biggest
accidental context dumps. Phase 6 is *additive value* (a cheaper, query-relevant
project map), not a correctness fix, and it is the largest single piece of new
design surface in the plan.

## Why defer

- No `project_map` module, config key, or `CONFIG_KEYS` entry exists today — it
  is a fresh design surface (~250–400 lines: cartography adapter, query-token
  extraction, adaptive budget logic, and tests).
- The plan's acceptance criteria are met without it.
- The plan's own Implementation Order sequences it last, noting the first five
  phases already remove the correctness hazards.

## Scope when picked up

See Phase 6 in `context-spike-corrective-implementation-plan.md`. In summary:

1. Prefer existing cartography data (`.slim/symbols.json`, `.slim/imports.json`)
   when present; fall back to `outline.py` only when `.slim` is missing/stale.
2. Produce a query-relevant map: extract identifiers/path-like tokens from the
   latest user task; rank symbols/files by token match, imported-by proximity,
   entrypoint hints, and recently opened/dirty paths.
3. Budget adaptively: 1500–2500 tokens by default; expand to ~4000 when
   `context_length >= 64k`; shrink/skip for very small windows.
4. Inject as ephemeral context (like the LSP-as-context seam), not as permanent
   transcript history.
5. Must work with `lsp_mode="off"` (independent of LSP).

Suggested config:

```toml
project_map = "auto"        # auto | off | always
project_map_min_tokens = 1500
project_map_max_tokens = 4000
```

## Reuse from this pass

- `swival.outline.symbol_spans` (now also driving the Phase 3 `read_file`
  `symbol=` path) is the natural fallback symbol source.
- The ephemeral-injection seam (`lsp_context_enabled` + the `[lsp automated
  context]` message pattern in `run_agent_loop`) is the model to copy for a
  `[project map]` injection that survives compaction without bloating history.
