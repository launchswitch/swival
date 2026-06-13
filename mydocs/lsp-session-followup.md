# Add LSP support to the Session library API

The LSP integration (`swival/lsp_client.py`) is fully wired through the CLI path but the library API (`swival.session.Session`) has no LSP support. Users of `Session(...)` / `swival.run(...)` cannot use any `lsp_*` tools.

## What exists

- `LspManager` in `swival/lsp_client.py` — fully functional, takes `server_configs` dict + `workspace_root`, runs on a background asyncio loop
- `auto_detect_lsp(base_dir)` — scans for `pyproject.toml`/`package.json`/`Cargo.toml`/`go.mod` and checks if matching language servers are on PATH
- `load_lsp_config(path)` in `swival/config.py` — loads `[lsp_servers.*]` from a TOML file
- `_validate_lsp_server_configs()` — validates config structure
- CLI path in `swival/agent.py:_run_main()` (around line 7640) shows the full initialization pattern: resolve configs → `LspManager(servers, workspace_root=base_dir)` → `.start()` → `.list_tools()` → `.get_tool_info()` → thread `lsp_manager` through `loop_kwargs` → cleanup on exit

## What needs to happen

1. **Add `lsp_servers` parameter to `Session.__init__()`** — optional dict of server configs (same shape as what `LspManager` accepts), plus `lsp_config: str | Path | None` for a TOML config file path. If neither is provided and the workspace has project markers, auto-detect.

2. **Create `LspManager` in `Session._setup()`** — mirror how `mcp_manager` and `a2a_manager` are created there. Call `auto_detect_lsp()` when no explicit config is given. Start the manager.

3. **Thread `lsp_manager` through `_build_loop_kwargs()`** — add it alongside `mcp_manager` and `a2a_manager` in the kwargs dict that gets passed to `run_agent_loop()`.

4. **Thread `lsp_manager` through `_make_input_context()`** — add `lsp_manager=lsp_manager` to the `InputContext` construction so `/tools` and other REPL commands in `ask(parse_commands=True)` can list LSP tools.

5. **Cleanup in `Session.close()`** — call `lsp_manager.close()` alongside the existing `mcp_manager.close()` and `a2a_manager.close()`.

6. **Add `lsp_servers`/`lsp_config` to `swival.run()` convenience function** — pass through to `Session`.

## Design decisions already made

- **Auto-detect by default** — if no explicit `lsp_servers` or `lsp_config` is provided, call `auto_detect_lsp(base_dir)` to find language servers based on project files. This matches the CLI behavior.
- **Opt-out via `no_lsp=True`** — consistent with `no_mcp`/`no_a2a` pattern already in `Session.__init__`.
- **LSP tools are added to the tool list** via `lsp_manager.list_tools()` — same pattern as MCP.
- **LSP tool info is injected into the system prompt** via `lsp_tool_info` parameter on `build_system_prompt` — already implemented.

## Files to modify

- `swival/session.py` — main changes (`__init__`, `_setup`, `_build_loop_kwargs`, `_make_input_context`, `close`)
- `swival/agent.py` — may need to expose `_resolve_lsp_servers` or move it to a shared location

## Tests

- Add unit tests in `tests/test_session.py` (or a new `test_session_lsp.py`) that mock `LspManager` and verify the lifecycle: creation → start → tools registered → cleanup
- Test auto-detection with a mock project directory containing `pyproject.toml`
- Test that `Session(lsp_servers={...})` passes config through correctly

## Reference

Look at how `mcp_manager` is handled in `Session` — that's the exact pattern to follow. The MCP and A2A integrations were added incrementally; LSP should follow the same approach.
