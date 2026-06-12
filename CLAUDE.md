# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Swival is a CLI-based AI coding agent that connects to many LLM providers (LM Studio, llama.cpp, HuggingFace, OpenRouter, Gemini, Bedrock, ChatGPT, any OpenAI-compatible server, or external commands) and runs an autonomous tool-calling loop. It is designed to work well with small and local models that have tight context windows.

## Commands

```sh
make install          # uv sync — install dependencies
make test             # uv run python -m pytest tests/ -v
make lint             # uv run ruff check swival/ tests/
make format           # uv run ruff format swival/ tests/
make check            # lint + format check (CI gate)
make website          # build docs/ from docs.md/ sources
make dist             # clean + build sdist/wheel + generate Homebrew formula
```

Run a single test file or test:
```sh
uv run python -m pytest tests/test_tools.py -v
uv run python -m pytest tests/test_tools.py::test_something -v
```

## Architecture

### Entry Points

- **CLI:** `swival.agent:main()` (registered as `swival` console script). Parses args, loads config, dispatches to one-shot, REPL, A2A server, ACP server, or reviewer mode.
- **Library API:** `swival.session.Session` class with `.run()` / `.ask()` methods. Also `swival.run(question, ...)` convenience function.

### Core Loop

`run_agent_loop()` in `agent.py` is the heart of the system:
1. Build system prompt from template (`system_prompt.txt`) + instructions + skills + memory
2. Call LLM via `litellm.completion()`
3. Route tool calls through `handle_tool_call()`
4. Handle compaction (context window management), retries, error recovery
5. Loop until answer produced or max turns reached

`handle_tool_call()` dispatches to implementations in `tools.py` or delegates to MCP/A2A/LSP managers.

### Key Modules

| Module | Role |
|---|---|
| `agent.py` (~12K LOC) | CLI parser, agent loop, REPL, most orchestration — the monolithic core |
| `tools.py` (~3.8K LOC) | Tool definitions (`TOOLS` list) and implementations (read/write/edit/grep/list files, run commands) |
| `config.py` | TOML config loading, validation, profile resolution. Merges global (`~/.config/swival/config.toml`) + project (`swival.toml`) |
| `session.py` | Public `Session`/`Result` API wrapping the agent loop |
| `snapshot.py` | Context collapse/restore for long sessions |
| `thinking.py` / `todo.py` / `goal.py` | Mutable state objects that survive compaction via prompt injection |
| `memory.py` | BM25-ranked cross-session memory |
| `skills.py` / `metaskills.py` | SKILL.md discovery/activation + Starlark workflow programs |
| `mcp_client.py` | Model Context Protocol client (external tool servers) |
| `lsp_client.py` | LSP client integration |
| `a2a_client.py` / `a2a_server.py` | Agent-to-Agent protocol |
| `acp_server.py` | Agent Client Protocol server (JSON-RPC on stdio, editor integration) |
| `audit.py` / `audit_ui.py` | Security audit pipeline (triage → review → verify) |
| `subagent.py` | Parallel subagent support |
| `secrets.py` | Transparent secret encryption/decryption |
| `cache.py` | SQLite-backed LLM response cache |
| `repair.py` / `tool_call_repair.py` | Response repair for malformed/truncated tool calls |
| `_msg.py` | Message normalization helpers used throughout |
| `_env.py` | Subprocess environment (PATH sanitization) |

### Configuration

Two TOML config files merged with precedence: CLI args > project `swival.toml` > global `~/.config/swival/config.toml` > defaults. Supports 130+ keys, nested sections for MCP/A2A/LSP servers, profiles, and audit settings. All keys defined in `CONFIG_KEYS` dict in `config.py`.

`AGENTS.md` / `CLAUDE.md` files are automatically loaded from the project root as instructions.

### Tools

Built-in tools are defined as JSON schemas in the `TOOLS` list in `tools.py` (plus `RUN_COMMAND_TOOL`, `RUN_SHELL_COMMAND_TOOL`, `COMPLETE_GOAL_TOOL`). Each tool has a matching handler function (e.g., `_read_file`, `_write_file`, `_edit_file`, `_list_files`, `_grep`). Additional tools come from MCP servers, LSP servers, and A2A agents.

### State Objects

`ThinkingState`, `TodoState`, `SnapshotState`, `GoalState` are passed through the agent loop as mutable state. They serialize their content into the system prompt to survive context compaction.

## Build & Dependencies

- **Python >=3.13**, built with **hatchling**, managed by **uv** (`uv.lock`)
- **LLM calls:** `litellm` (multi-provider abstraction)
- **Token counting:** `tiktoken` (cl100k_base)
- **REPL:** `prompt-toolkit` with custom completer
- **Terminal output:** `rich`
- **Protocols:** `mcp` SDK, `lsprotocol`, `starlette`+`uvicorn` (A2A server)
- **Starlark:** `starlark-go` for metaskill workflows
- **Dev:** `pytest`, `ruff`

## Testing

~90 test files in `tests/`. Tests use autouse fixtures in `conftest.py` that isolate global skills and AGENTS.md paths. No external LLM calls in unit tests — the LLM layer is mocked.
