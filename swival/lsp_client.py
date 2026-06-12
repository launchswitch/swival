"""LSP (Language Server Protocol) client integration for swival.

Connects to language servers (pyright, typescript-language-server,
rust-analyzer, etc.) and exposes LSP capabilities as agent tools:
definition lookup, references, hover, diagnostics, code actions,
rename, workspace symbols, and document symbols.

Follows the same architecture as McpManager: a background asyncio event
loop in a daemon thread, with synchronous public methods that block on
async futures.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from . import fmt
from ._env import child_env

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LSP_PROTOCOL_VERSION = 3

# Default file-extension -> language mapping used for auto-routing.
# Maps a file extension (with leading dot) to an LSP languageId.
EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".js": "javascript",
    ".jsx": "javascriptreact",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".vue": "vue",
    ".svelte": "svelte",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".less": "less",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sh": "shellscript",
    ".bash": "shellscript",
    ".zsh": "shellscript",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".zig": "zig",
    ".nim": "nim",
    ".v": "vlang",
    ".tf": "terraform",
    ".sql": "sql",
}

# Map server name keywords to the languageIds they handle.
# Used for auto-routing when the config doesn't specify file_extensions.
SERVER_LANGUAGE_HINTS: dict[str, list[str]] = {
    "pyright": ["python"],
    "pylsp": ["python"],
    "python": ["python"],
    "typescript": ["typescript", "typescriptreact", "javascript", "javascriptreact"],
    "ts": ["typescript", "typescriptreact", "javascript", "javascriptreact"],
    "vtsls": ["typescript", "typescriptreact", "javascript", "javascriptreact"],
    "rust": ["rust"],
    "rust-analyzer": ["rust"],
    "ra": ["rust"],
    "gopls": ["go"],
    "go": ["go"],
    "jdtls": ["java"],
    "java": ["java"],
    "clangd": ["c", "cpp"],
    "ccls": ["c", "cpp"],
    "csharp": ["csharp"],
    "omnisharp": ["csharp"],
    "ruby": ["ruby"],
    "solargraph": ["ruby"],
    "php": ["php"],
    "intelephense": ["php"],
    "swift": ["swift"],
    "kotlin": ["kotlin"],
    "kotlin-language-server": ["kotlin"],
    "scala": ["scala"],
    "metals": ["scala"],
    "vue": ["vue"],
    "html": ["html"],
    "css": ["css", "scss", "less"],
    "json": ["json"],
    "yaml": ["yaml"],
    "toml": ["toml"],
    "shellcheck": ["shellscript"],
    "bash": ["shellscript"],
    "lua": ["lua"],
    "r": ["r"],
    "dart": ["dart"],
    "dartls": ["dart"],
    "elixir": ["elixir"],
    "erlang": ["erlang"],
    "haskell": ["haskell"],
    "ocaml": ["ocaml"],
    "zig": ["zig"],
    "nim": ["nim"],
    "terraform": ["terraform"],
    "sql": ["sql"],
}

# Built-in auto-detection: project marker files -> suggested LSP server config.
AUTO_DETECT_RULES: list[tuple[list[str], dict[str, Any]]] = [
    # Python
    (
        ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile"],
        {
            "name": "pyright",
            "command": "pyright-langserver",
            "args": ["--stdio"],
        },
    ),
    # TypeScript / JavaScript
    (
        ["tsconfig.json", "package.json"],
        {
            "name": "typescript",
            "command": "npx",
            "args": ["-y", "typescript-language-server", "--stdio"],
        },
    ),
    # Rust
    (
        ["Cargo.toml"],
        {
            "name": "rust-analyzer",
            "command": "rust-analyzer",
        },
    ),
    # Go
    (
        ["go.mod"],
        {
            "name": "gopls",
            "command": "gopls",
        },
    ),
]

# LSP diagnostic severity -> human label
DIAGNOSTIC_SEVERITY = {1: "error", 2: "warning", 3: "info", 4: "hint"}

# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------


def path_to_uri(path: Path) -> str:
    """Convert an absolute filesystem path to an LSP file URI.

    Per RFC 8089 the path component is percent-encoded so that reserved
    characters (spaces, '#', '?', '%', non-ASCII) round-trip correctly
    through LSP servers that strictly validate URIs.
    """
    return "file://" + quote(path.as_posix(), safe="/")


def uri_to_path(uri: str) -> Path | None:
    """Convert an LSP file URI to an absolute Path, or None on failure."""
    if not uri.startswith("file://"):
        return None
    path_str = uri[7:]
    # Handle Windows URIs like file:///C:/path
    if os.name == "nt" and len(path_str) >= 3 and path_str[1:2] == ":":
        path_str = path_str[1:]
    return Path(unquote(path_str))


# ---------------------------------------------------------------------------
# LSP wire protocol helpers
# ---------------------------------------------------------------------------


def _encode_message(obj: Any) -> bytes:
    """Encode a Python object as an LSP wire message with Content-Length header."""
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
    return header + payload


async def _read_message(stdout) -> dict | None:
    """Read one LSP message from stdout. Returns None on EOF."""
    # Read header lines until we find Content-Length
    headers = {}
    while True:
        line = await stdout.readline()
        if line == b"":
            return None  # EOF
        line_str = line.decode("utf-8").strip()
        if line_str == "":
            break  # blank line = end of headers
        if ":" in line_str:
            key, value = line_str.split(":", 1)
            headers[key.strip().lower()] = value.strip()

    content_length = int(headers.get("content-length", 0))
    if content_length == 0:
        return None

    body = await stdout.readexactly(content_length)
    return json.loads(body.decode("utf-8"))


# ---------------------------------------------------------------------------
# LSP request/response correlation
# ---------------------------------------------------------------------------


class _LspConnection:
    """Manages a single LSP server process with request/response correlation."""

    def __init__(
        self,
        name: str,
        config: dict,
        workspace_root: Path,
        verbose: bool = False,
    ):
        self.name = name
        self.config = config
        self.workspace_root = workspace_root
        self.verbose = verbose

        self._process: asyncio.subprocess.Process | None = None
        self._next_id: int = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._capabilities: dict = {}
        self._initialized: bool = False
        self._closing: bool = False

        # Document state: uri -> {"content": str, "version": int}
        self._documents: dict[str, dict] = {}
        # Diagnostics: uri -> list of diagnostic dicts
        self._diagnostics: dict[str, list] = {}

    @property
    def languages(self) -> list[str]:
        """Return the set of languageIds this server handles."""
        # Check explicit config first
        if "languages" in self.config:
            return list(self.config["languages"])
        if "file_extensions" in self.config:
            exts = self.config["file_extensions"]
            return [EXT_TO_LANGUAGE.get(e, e.lstrip(".")) for e in exts]
        # Auto-detect from server name
        name_lower = self.name.lower()
        for hint, langs in SERVER_LANGUAGE_HINTS.items():
            if hint in name_lower:
                return langs
        return []

    def supports_language(self, language_id: str) -> bool:
        """Check if this server handles the given languageId.

        Defaults to False when no languages are configured. The caller
        is expected to validate non-empty ``languages`` at startup (see
        ``LspManager._build_routing``); returning True here would let
        a misconfigured server silently claim every file.
        """
        langs = self.languages
        if not langs:
            return False
        return language_id in langs

    def supports_file(self, file_path: Path) -> bool:
        """Check if this server handles the given file based on extension."""
        ext = file_path.suffix.lower()
        language_id = EXT_TO_LANGUAGE.get(ext)
        if not language_id:
            return False
        return self.supports_language(language_id)

    async def start(self, timeout: float = 30) -> None:
        """Start the LSP server process and complete initialization."""
        command = self.config["command"]
        args = list(self.config.get("args", []))

        if self.verbose:
            full_cmd = [command] + args
            print(f"  LSP: starting {self.name!r}: {' '.join(full_cmd)}", flush=True)

        # Use child_env() to strip the agent's venv from PATH, matching
        # mcp_client.py and the run_command sandbox. Without this, an LSP
        # server (e.g. pyright, pylsp) launched via `uv run` may pick up
        # the wrong Python interpreter or libraries from the agent's env.
        env = child_env(self.config.get("env"))
        self._process = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # Drain stderr in a background task to avoid the OS pipe buffer
            # filling (typically 64KB on Linux) and blocking the server.
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Start reading responses in background
        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"lsp-{self.name}-reader"
        )
        # Drain stderr to logger to keep the pipe from filling.
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(), name=f"lsp-{self.name}-stderr"
        )

        # Send initialize request
        init_params = {
            "processId": os.getpid(),
            "rootUri": path_to_uri(self.workspace_root),
            "capabilities": {
                "textDocument": {
                    "definition": {"dynamicRegistration": False},
                    "references": {"dynamicRegistration": False},
                    "hover": {"dynamicRegistration": False},
                    "codeAction": {"dynamicRegistration": False},
                    "rename": {"dynamicRegistration": False},
                    "documentSymbol": {"dynamicRegistration": False},
                    "publishDiagnostics": {"relatedInformation": True},
                },
                "workspace": {
                    "symbol": {"dynamicRegistration": False},
                    "workspaceFolders": True,
                },
            },
        }
        init_options = self.config.get("init_options")
        if init_options:
            init_params["initializationOptions"] = init_options

        capabilities = await self._request("initialize", init_params, timeout=timeout)

        self._capabilities = capabilities.get("capabilities", {})

        # Send initialized notification
        await self._notify("initialized", {})
        self._initialized = True

        if self.verbose:
            print(f"  LSP: {self.name!r} initialized", flush=True)

    async def _request(self, method: str, params: dict, timeout: float = 30) -> dict:
        """Send an LSP request and wait for the response."""
        if self._closing:
            raise RuntimeError(f"LSP server {self.name!r} is shutting down")

        msg_id = self._next_id
        self._next_id += 1

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = future

        message = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        await self._write(message)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(msg_id, None)
            raise TimeoutError(
                f"LSP request {method} to {self.name!r} timed out after {timeout}s"
            )

    async def _notify(self, method: str, params: dict) -> None:
        """Send an LSP notification (no response expected)."""
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        await self._write(message)

    async def _write(self, message: dict) -> None:
        """Write a message to the server's stdin."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError(f"LSP server {self.name!r} has no process")
        self._process.stdin.write(_encode_message(message))
        await self._process.stdin.drain()

    async def _read_loop(self) -> None:
        """Background task: read responses from stdout."""
        try:
            while not self._closing:
                if self._process is None or self._process.stdout is None:
                    break
                response = await _read_message(self._process.stdout)
                if response is None:
                    break
                self._handle_response(response)
        except Exception as e:
            if self.verbose:
                print(f"  LSP: {self.name!r} read loop error: {e}", flush=True)
        finally:
            # Resolve any pending requests with an error
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(
                        RuntimeError(f"LSP server {self.name!r} disconnected")
                    )
            self._pending.clear()

    async def _drain_stderr(self) -> None:
        """Drain the LSP server's stderr to the logger to prevent the
        OS pipe buffer from filling and blocking the server process.

        Most language servers write progress, warnings, and debug logs
        to stderr on every request. A 64KB buffer fills within seconds
        for verbose servers like rust-analyzer or gopls; once full the
        server blocks on its next stderr write and the read_loop starves.
        """
        if self._process is None or self._process.stderr is None:
            return
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug("LSP %s stderr: %s", self.name, text)
        except Exception as e:
            if self.verbose:
                print(f"  LSP: {self.name!r} stderr drain error: {e}", flush=True)

    def _handle_response(self, response: dict) -> None:
        """Process an incoming LSP message."""
        msg_id = response.get("id")

        # Response to a request
        if msg_id is not None and msg_id in self._pending:
            future = self._pending.pop(msg_id)
            if not future.done():
                if "result" in response:
                    future.set_result(response["result"])
                elif "error" in response:
                    error = response["error"]
                    future.set_exception(
                        RuntimeError(
                            f"LSP error {error.get('code', -1)}: {error.get('message', 'unknown')}"
                        )
                    )
            return

        # Notification
        method = response.get("method", "")
        if method == "textDocument/publishDiagnostics":
            params = response.get("params", {})
            uri = params.get("uri", "")
            self._diagnostics[uri] = params.get("diagnostics", [])
        elif method == "$/progress" or method.startswith("window/"):
            pass  # Ignore progress and window notifications
        # Silently ignore other notifications

    async def did_open(self, uri: str, language_id: str, content: str) -> None:
        """Notify the server that a document was opened.

        Idempotent: if the URI is already open, sends ``didChange`` with
        the new content and a bumped version. The LSP spec requires that
        ``didOpen`` is sent only once per document; sending it twice for
        the same URI is a protocol violation that strict servers
        (rust-analyzer, gopls) may reject.
        """
        if uri in self._documents:
            # Already open: re-route to didChange to keep state monotonic.
            await self.did_change(uri, content, language_id)
            return
        self._documents[uri] = {"content": content, "version": 1}
        await self._notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": content,
                }
            },
        )

    async def did_change(
        self, uri: str, content: str, language_id: str = "plaintext"
    ) -> None:
        """Notify the server that a document changed."""
        doc = self._documents.get(uri)
        if doc is None:
            # File wasn't opened yet; open it first
            await self.did_open(uri, language_id, content)
            return
        new_version = doc["version"] + 1
        doc["content"] = content
        doc["version"] = new_version
        await self._notify(
            "textDocument/didChange",
            {
                "textDocument": {"uri": uri, "version": new_version},
                "contentChanges": [{"text": content}],
            },
        )

    async def did_close(self, uri: str) -> None:
        """Notify the server that a document was closed."""
        self._documents.pop(uri, None)
        await self._notify(
            "textDocument/didClose",
            {"textDocument": {"uri": uri}},
        )

    async def definition(self, uri: str, line: int, column: int) -> list[dict] | None:
        """Find definition(s) of the symbol at the given position."""
        if not self._capabilities.get("definitionProvider"):
            return None
        result = await self._request(
            "textDocument/definition",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": column},
            },
        )
        return self._locations_to_list(result)

    async def references(self, uri: str, line: int, column: int) -> list[dict] | None:
        """Find all references to the symbol at the given position."""
        if not self._capabilities.get("referencesProvider"):
            return None
        result = await self._request(
            "textDocument/references",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": column},
                "context": {"includeDeclaration": True},
            },
        )
        return self._locations_to_list(result)

    async def hover(self, uri: str, line: int, column: int) -> dict | None:
        """Get hover information at the given position."""
        if not self._capabilities.get("hoverProvider"):
            return None
        result = await self._request(
            "textDocument/hover",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": column},
            },
        )
        return result

    async def code_actions(
        self, uri: str, start_line: int, start_col: int, end_line: int, end_col: int
    ) -> list[dict] | None:
        """Get code actions (quick fixes, refactorings) for a range."""
        if not self._capabilities.get("codeActionProvider"):
            return None
        result = await self._request(
            "textDocument/codeAction",
            {
                "textDocument": {"uri": uri},
                "range": {
                    "start": {"line": start_line, "character": start_col},
                    "end": {"line": end_line, "character": end_col},
                },
                "context": {"diagnostics": []},
            },
        )
        return result if isinstance(result, list) else None

    async def rename(
        self, uri: str, line: int, column: int, new_name: str
    ) -> dict | None:
        """Rename a symbol across the workspace."""
        if not self._capabilities.get("renameProvider"):
            return None
        result = await self._request(
            "textDocument/rename",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": column},
                "newName": new_name,
            },
        )
        return result

    async def document_symbols(self, uri: str) -> list[dict] | None:
        """Get document symbols for a file."""
        if not self._capabilities.get("documentSymbolProvider"):
            return None
        result = await self._request(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": uri}},
        )
        return result if isinstance(result, list) else None

    async def workspace_symbols(self, query: str) -> list[dict] | None:
        """Search workspace symbols."""
        if not self._capabilities.get("workspace", {}).get("symbolProvider"):
            return None
        result = await self._request(
            "workspace/symbol",
            {"query": query},
        )
        return result if isinstance(result, list) else None

    def get_diagnostics(self, uri: str) -> list[dict]:
        """Get stored diagnostics for a file URI."""
        return self._diagnostics.get(uri, [])

    def _locations_to_list(self, result: Any) -> list[dict] | None:
        """Convert LSP Location or LocationLink to a list of location dicts."""
        if result is None:
            return None
        if isinstance(result, dict):
            # Single Location
            return [self._location_to_dict(result)]
        if isinstance(result, list):
            return [self._location_to_dict(item) for item in result]
        return None

    def _location_to_dict(self, loc: dict) -> dict:
        """Convert an LSP Location/LocationLink to a simple dict."""
        if "targetUri" in loc:
            # LocationLink
            target_range = loc.get("targetSelectionRange", loc.get("targetRange", {}))
            return {
                "uri": loc.get("targetUri", ""),
                "range": target_range,
            }
        return loc

    async def shutdown(self) -> None:
        """Shutdown the LSP server."""
        try:
            await self._request("shutdown", {}, timeout=5)
            await self._notify("exit", {})
        except Exception:
            pass
        finally:
            self._closing = True
            # Cancel reader task
            if hasattr(self, "_reader_task") and not self._reader_task.done():
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except (asyncio.CancelledError, Exception):
                    pass

            # Kill the process
            if self._process is not None:
                try:
                    self._process.terminate()
                    await asyncio.wait_for(self._process.wait(), timeout=5)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# LspManager — public API
# ---------------------------------------------------------------------------


class LspShutdownError(Exception):
    """Raised when call_tool() is invoked during or after shutdown."""


class LspManager:
    """Manages connections to one or more LSP servers.

    Runs an asyncio event loop in a background daemon thread.
    All public methods are synchronous.
    """

    def __init__(
        self,
        server_configs: dict[str, dict],
        workspace_root: str,
        verbose: bool = False,
    ):
        """
        server_configs: {
            "server-name": {
                "command": "pyright-langserver",
                "args": ["--stdio"],
                # optional:
                "languages": ["python"],
                "init_options": {...},
            }
        }
        workspace_root: absolute path to the project root.
        """
        self._server_configs = server_configs
        self._workspace_root = Path(workspace_root).resolve()
        self._verbose = verbose

        # Background event loop
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

        # Connections (populated by start())
        self._connections: dict[str, _LspConnection] = {}
        # Language -> server name routing
        self._language_to_server: dict[str, str] = {}
        # Extension -> server name routing
        self._ext_to_server: dict[str, str] = {}

        # Asynchronous notification queue. File-tool hooks (read/write/
        # edit/delete) enqueue didOpen/didChange/didClose events here
        # and return immediately; a dedicated worker thread drains
        # the queue and posts to the LSP loop. The agent thread is
        # never blocked on a slow LSP server. The queue is bounded
        # so a wedged LSP cannot OOM the agent; full-queue drops the
        # oldest item (newer state is strictly more useful).
        self._notification_queue: queue.Queue | None = None
        self._notification_worker: threading.Thread | None = None
        self._worker_stop = threading.Event()

        # Lifecycle flags
        self._closing = False
        self._closed = False
        self._started = False

    # --- Public API ---

    def start(self, timeout: float = 30) -> None:
        """Start background event loop and connect to all servers."""
        if self._closed:
            raise LspShutdownError("manager is already closed")
        if self._started:
            return

        loop_ready = threading.Event()
        self._loop = asyncio.new_event_loop()

        def _run_loop():
            self._loop.call_soon(lambda: loop_ready.set())
            self._loop.run_forever()

        self._thread = threading.Thread(
            target=_run_loop,
            name="swival-lsp-loop",
            daemon=True,
        )
        self._thread.start()
        if not loop_ready.wait(timeout=10):
            raise LspShutdownError("LSP event loop failed to start")

        # Start each server
        for name, config in self._server_configs.items():
            try:
                self._start_server(name, config, timeout=timeout)
                # Surface a tool-count notice when the server exposes tools.
                conn = self._connections.get(name)
                if conn is not None:
                    tool_count = sum(
                        1
                        for cap_name in (
                            "definitionProvider",
                            "referencesProvider",
                            "hoverProvider",
                            "codeActionProvider",
                            "renameProvider",
                            "documentSymbolProvider",
                        )
                        if conn._capabilities.get(cap_name)
                    )
                    if tool_count:
                        fmt.lsp_server_start(name, tool_count)
            except Exception as e:
                # Always surface startup failures (not just in verbose)
                # so the user knows LSP isn't available.
                fmt.lsp_server_error(name, str(e))

        # Build routing tables
        self._build_routing()

        # Start notification worker (drains queue → LSP loop)
        self._notification_queue = queue.Queue(maxsize=1000)
        self._worker_stop.clear()
        self._notification_worker = threading.Thread(
            target=self._notification_loop,
            name="swival-lsp-notify",
            daemon=True,
        )
        self._notification_worker.start()

        self._started = True
        atexit.register(self.close)

    def list_tools(self) -> list[dict]:
        """Return LSP tools in OpenAI function-calling format.

        Filters ``LSP_TOOLS`` to only those whose required capability is
        advertised by at least one active server. ``lsp_diagnostics``
        and ``lsp_workspace_symbols`` are always included: the former
        uses stored ``publishDiagnostics`` (no provider capability is
        required to read them), and the latter picks the first server.
        """
        if not self._started or not self._connections:
            return []
        available = _tools_for_capabilities(
            [c._capabilities for c in self._connections.values()]
        )
        return [t for t in LSP_TOOLS if t["function"]["name"] in available]

    def get_tool_info(self) -> dict[str, list[tuple[str, str]]]:
        """Return {server_name: [(tool_name, description), ...]}."""
        info: dict[str, list[tuple[str, str]]] = {}
        if not self._connections:
            return info
        # Mirror list_tools() filtering so the system-prompt listing and
        # the OpenAI tool schema stay in sync.
        available = _tools_for_capabilities(
            [c._capabilities for c in self._connections.values()]
        )
        tool_descs = {
            t["function"]["name"]: t["function"].get("description", "")
            for t in LSP_TOOLS
            if t["function"]["name"] in available
        }
        # Group under a virtual "lsp" key
        info["lsp"] = list(tool_descs.items())
        return info

    def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        """Dispatch an LSP tool call. Returns (result_text, is_error)."""
        if self._closing or self._closed:
            raise LspShutdownError("manager is shutting down")

        if not self._connections:
            return ("error: no LSP servers are active", True)

        try:
            result = self._run_sync(self._dispatch_tool(name, arguments), timeout=30)
            return (result, False)
        except LspShutdownError:
            raise
        except Exception as e:
            return (f"error: LSP tool {name!r} failed: {e}", True)

    def on_file_read(self, abs_path: Path, content: str) -> None:
        """Notify LSP servers that a file was read (didOpen).

        Non-blocking: enqueues the notification and returns immediately.
        The background worker thread drains the queue and posts to the
        LSP event loop.
        """
        if not self._started or not self._connections:
            return
        self._enqueue_notification("read", abs_path, content)

    def on_file_write(self, abs_path: Path, content: str) -> None:
        """Notify LSP servers that a file was written (didChange).

        Non-blocking: enqueues the notification and returns immediately.
        """
        if not self._started or not self._connections:
            return
        self._enqueue_notification("write", abs_path, content)

    def on_file_delete(self, abs_path: Path) -> None:
        """Notify LSP servers that a file was deleted (didClose).

        Non-blocking: enqueues the notification and returns immediately.
        """
        if not self._started or not self._connections:
            return
        self._enqueue_notification("delete", abs_path, None)

    def _enqueue_notification(self, action: str, abs_path: Path, content: str | None) -> None:
        """Add a notification to the async queue, dropping the oldest if full."""
        q = self._notification_queue
        if q is None:
            return
        try:
            q.put_nowait((action, abs_path, content))
        except queue.Full:
            # Drop the oldest item — newer state is strictly more useful.
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait((action, abs_path, content))
            except queue.Full:
                pass  # Give up silently; best-effort delivery.

    def _notification_loop(self) -> None:
        """Worker thread: drain the notification queue and post to LSP.

        Runs in a daemon thread started by ``start()``. Each item is a
        ``(action, abs_path, content)`` tuple. The worker resolves the
        target server, calls the appropriate ``did_*`` method on the
        connection via ``_run_sync``, and continues. Errors are logged
        but never propagate to the caller — notifications are fire-and-forget.
        """
        q = self._notification_queue
        assert q is not None  # set before thread start
        while not self._worker_stop.is_set():
            try:
                item = q.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:  # shutdown sentinel
                break
            action, abs_path, content = item
            try:
                self._send_notification_sync(action, abs_path, content)
            except Exception as exc:
                if self._verbose:
                    print(
                        f"  LSP: notification {action} {abs_path} failed: {exc}",
                        flush=True,
                    )
            finally:
                q.task_done()

    def _send_notification_sync(self, action: str, abs_path: Path, content: str | None) -> None:
        """Send a single notification synchronously. Called from the worker."""
        if not self._connections or self._closing:
            return
        uri = path_to_uri(abs_path)
        server_name = self._resolve_server(abs_path)
        if server_name is None:
            return
        conn = self._connections[server_name]
        ext = abs_path.suffix.lower()
        language_id = EXT_TO_LANGUAGE.get(ext, "plaintext")
        if action == "read":
            if content is not None:
                self._run_sync(conn.did_open(uri, language_id, content), timeout=5)
        elif action == "write":
            if content is not None:
                self._run_sync(conn.did_change(uri, content, language_id), timeout=5)
        elif action == "delete":
            self._run_sync(conn.did_close(uri), timeout=5)

    def close(self) -> None:
        """Idempotent shutdown."""
        if self._closed:
            return
        self._closing = True

        # Stop notification worker first (no new events to servers).
        self._worker_stop.set()
        if self._notification_queue is not None:
            # Unblock the worker if it's waiting on get().
            try:
                self._notification_queue.put_nowait(None)
            except queue.Full:
                pass
        if self._notification_worker is not None and self._notification_worker.is_alive():
            self._notification_worker.join(timeout=5)

        if self._loop is not None and self._loop.is_running():
            try:
                self._run_sync(self._shutdown_all(), timeout=10)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)

        self._closed = True
        self._closing = False

    # --- Internal ---

    def _run_sync(self, coro, timeout: float = 30):
        """Submit a coroutine to the background loop and wait for result."""
        if self._loop is None or not self._loop.is_running():
            raise LspShutdownError("event loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except asyncio.CancelledError:
            raise LspShutdownError("operation cancelled during shutdown")
        except TimeoutError:
            future.cancel()
            raise

    def _start_server(self, name: str, config: dict, timeout: float = 30) -> None:
        """Start one LSP server connection."""
        conn = _LspConnection(name, config, self._workspace_root, self._verbose)
        self._run_sync(conn.start(timeout=timeout))
        self._connections[name] = conn

    def _build_routing(self) -> None:
        """Build language/extension -> server name routing tables.

        Validates that every server declares a non-empty language set,
        either explicitly (``languages`` or ``file_extensions`` in the
        config) or implicitly via a recognized name in
        ``SERVER_LANGUAGE_HINTS``. A server with no resolvable languages
        would otherwise be a silent catch-all (issue #12).
        """
        from .report import ConfigError

        for name, conn in self._connections.items():
            langs = conn.languages
            if not langs:
                raise ConfigError(
                    f"LSP server {name!r} has no languages configured. "
                    f"Specify 'languages' or 'file_extensions' in the "
                    f"config, or use a recognized server name (one of: "
                    f"{', '.join(sorted(SERVER_LANGUAGE_HINTS.keys()))})."
                )
            for lang in langs:
                if lang not in self._language_to_server:
                    self._language_to_server[lang] = name
            # Also build extension routing
            for ext, lang in EXT_TO_LANGUAGE.items():
                if conn.supports_language(lang):
                    if ext not in self._ext_to_server:
                        self._ext_to_server[ext] = name

    def _resolve_server(self, file_path: Path) -> str | None:
        """Find the best server for a file path."""
        ext = file_path.suffix.lower()
        # Direct extension match
        if ext in self._ext_to_server:
            return self._ext_to_server[ext]
        # Language match
        language_id = EXT_TO_LANGUAGE.get(ext)
        if language_id and language_id in self._language_to_server:
            return self._language_to_server[language_id]
        # Fallback: first server
        if self._connections:
            return next(iter(self._connections))
        return None

    async def _dispatch_tool(self, name: str, arguments: dict) -> str:
        """Dispatch a tool call to the appropriate LSP server."""
        file_path = arguments.get("file_path", "")
        if not file_path:
            return "error: file_path is required"

        # Resolve absolute path
        abs_path = self._workspace_root / file_path
        if not abs_path.is_absolute():
            abs_path = Path(file_path).resolve()

        uri = path_to_uri(abs_path)

        # Position: convert 1-based to 0-based
        line = arguments.get("line", 1) - 1
        column = arguments.get("column", 1) - 1

        if name == "lsp_definition":
            server_name = self._resolve_server(abs_path)
            if not server_name:
                return f"error: no LSP server found for {abs_path.suffix or 'unknown extension'}"
            conn = self._connections[server_name]
            result = await conn.definition(uri, line, column)
            return _format_locations(result)

        elif name == "lsp_references":
            server_name = self._resolve_server(abs_path)
            if not server_name:
                return f"error: no LSP server found for {abs_path.suffix or 'unknown extension'}"
            conn = self._connections[server_name]
            result = await conn.references(uri, line, column)
            return _format_locations(result)

        elif name == "lsp_hover":
            server_name = self._resolve_server(abs_path)
            if not server_name:
                return f"error: no LSP server found for {abs_path.suffix or 'unknown extension'}"
            conn = self._connections[server_name]
            result = await conn.hover(uri, line, column)
            return _format_hover(result)

        elif name == "lsp_code_actions":
            server_name = self._resolve_server(abs_path)
            if not server_name:
                return f"error: no LSP server found for {abs_path.suffix or 'unknown extension'}"
            conn = self._connections[server_name]
            end_line = arguments.get("end_line", line)
            end_col = arguments.get("end_column", column)
            result = await conn.code_actions(uri, line, column, end_line, end_col)
            return _format_code_actions(result)

        elif name == "lsp_rename":
            new_name = arguments.get("new_name", "")
            if not new_name:
                return "error: new_name is required for rename"
            server_name = self._resolve_server(abs_path)
            if not server_name:
                return f"error: no LSP server found for {abs_path.suffix or 'unknown extension'}"
            conn = self._connections[server_name]
            result = await conn.rename(uri, line, column, new_name)
            return _format_workspace_edit(result, self._workspace_root)

        elif name == "lsp_diagnostics":
            server_name = self._resolve_server(abs_path)
            if not server_name:
                return f"error: no LSP server found for {abs_path.suffix or 'unknown extension'}"
            conn = self._connections[server_name]
            diagnostics = conn.get_diagnostics(uri)
            return _format_diagnostics(diagnostics, file_path)

        elif name == "lsp_document_symbols":
            server_name = self._resolve_server(abs_path)
            if not server_name:
                return f"error: no LSP server found for {abs_path.suffix or 'unknown extension'}"
            conn = self._connections[server_name]
            result = await conn.document_symbols(uri)
            return _format_document_symbols(result, file_path)

        elif name == "lsp_workspace_symbols":
            query = arguments.get("query", "")
            if not query:
                return "error: query is required for workspace symbol search"
            # Use first available server for workspace-wide search
            if not self._connections:
                return "error: no LSP servers are active"
            conn = next(iter(self._connections.values()))
            result = await conn.workspace_symbols(query)
            return _format_workspace_symbols(result, self._workspace_root)

        else:
            return f"error: unknown LSP tool: {name}"

    async def _shutdown_all(self) -> None:
        """Shutdown all LSP server connections."""
        tasks = []
        for conn in self._connections.values():
            tasks.append(conn.shutdown())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# LSP tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------


# Mapping from LSP tool name -> capability key(s) required to expose it.
# ``lsp_diagnostics`` is unconditionally available because it reads the
# stored ``textDocument/publishDiagnostics`` buffer (no provider
# capability is required to read what the server already sent).
# ``lsp_workspace_symbols`` always uses the first active server.
_LSP_TOOL_REQUIRED_CAPS: dict[str, tuple[str, ...]] = {
    "lsp_definition": ("definitionProvider",),
    "lsp_references": ("referencesProvider",),
    "lsp_hover": ("hoverProvider",),
    "lsp_code_actions": ("codeActionProvider",),
    "lsp_rename": ("renameProvider",),
    "lsp_document_symbols": ("documentSymbolProvider",),
    "lsp_workspace_symbols": ("workspace.symbolProvider",),
}


def _tools_for_capabilities(
    capabilities_list: list[dict],
) -> set[str]:
    """Return the set of LSP tool names supported by the union of these
    server capabilities.

    ``lsp_diagnostics`` is always included (no provider capability
    required). ``lsp_workspace_symbols`` is included if *any* server
    advertises ``workspace.symbolProvider`` since the manager delegates
    to the first active server.
    """
    available: set[str] = {"lsp_diagnostics"}
    for caps in capabilities_list:
        for tool, required in _LSP_TOOL_REQUIRED_CAPS.items():
            if any(_capability_present(caps, key) for key in required):
                available.add(tool)
    return available


def _capability_present(caps: dict, dotted_key: str) -> bool:
    """Check a (possibly nested) capability key, e.g.
    ``"workspace.symbolProvider"``.
    """
    node: Any = caps
    for part in dotted_key.split("."):
        if not isinstance(node, dict):
            return False
        node = node.get(part)
    return bool(node)


LSP_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "lsp_definition",
            "description": (
                "Find the definition of a symbol at the given position using "
                "the Language Server Protocol. Returns file path and line number "
                "of the definition location(s)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file containing the symbol.",
                    },
                    "line": {
                        "type": "integer",
                        "description": "1-based line number of the symbol.",
                    },
                    "column": {
                        "type": "integer",
                        "description": "1-based column number of the symbol.",
                    },
                },
                "required": ["file_path", "line", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_references",
            "description": (
                "Find all references to a symbol at the given position using "
                "the Language Server Protocol. Returns file paths and line "
                "numbers of all reference locations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file containing the symbol.",
                    },
                    "line": {
                        "type": "integer",
                        "description": "1-based line number of the symbol.",
                    },
                    "column": {
                        "type": "integer",
                        "description": "1-based column number of the symbol.",
                    },
                },
                "required": ["file_path", "line", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_hover",
            "description": (
                "Get hover information (type signature, docstring, etc.) for "
                "the symbol at the given position using the Language Server "
                "Protocol."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                    "line": {
                        "type": "integer",
                        "description": "1-based line number.",
                    },
                    "column": {
                        "type": "integer",
                        "description": "1-based column number.",
                    },
                },
                "required": ["file_path", "line", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_code_actions",
            "description": (
                "Get available code actions (quick fixes, refactorings) for "
                "a range in a file using the Language Server Protocol."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                    "line": {
                        "type": "integer",
                        "description": "1-based start line number.",
                    },
                    "column": {
                        "type": "integer",
                        "description": "1-based start column number.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-based end line number (defaults to line).",
                    },
                    "end_column": {
                        "type": "integer",
                        "description": "1-based end column number (defaults to column).",
                    },
                },
                "required": ["file_path", "line", "column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_rename",
            "description": (
                "Rename a symbol across the codebase using the Language Server "
                "Protocol. Returns the proposed changes without applying them. "
                "Use edit_file to apply the changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file containing the symbol.",
                    },
                    "line": {
                        "type": "integer",
                        "description": "1-based line number of the symbol.",
                    },
                    "column": {
                        "type": "integer",
                        "description": "1-based column number of the symbol.",
                    },
                    "new_name": {
                        "type": "string",
                        "description": "The new name for the symbol.",
                    },
                },
                "required": ["file_path", "line", "column", "new_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_diagnostics",
            "description": (
                "Get compiler/linter diagnostics (errors, warnings) for a file "
                "from the Language Server. Requires the file to have been read "
                "first so the LSP server can analyze it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to check.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_document_symbols",
            "description": (
                "Get a structured outline of symbols (functions, classes, "
                "variables, etc.) in a file using the Language Server Protocol. "
                "More accurate than the built-in outline tool for typed languages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lsp_workspace_symbols",
            "description": (
                "Search for symbols across the entire workspace by name using "
                "the Language Server Protocol. Useful for finding functions, "
                "classes, or types by name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Symbol name to search for.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Response formatters
# ---------------------------------------------------------------------------


def _format_locations(locations: list[dict] | None) -> str:
    """Format LSP Location results for the agent."""
    if not locations:
        return "No definitions found."

    lines = []
    for loc in locations:
        uri = loc.get("uri", "")
        path = uri_to_path(uri)
        path_str = str(path) if path else uri
        # Try to make relative
        try:
            path_str = str(path.relative_to(Path.cwd()))
        except (ValueError, TypeError):
            pass
        range_info = loc.get("range", {})
        start = range_info.get("start", {})
        line = start.get("line", 0) + 1  # 0-based -> 1-based
        character = start.get("character", 0) + 1
        lines.append(f"{path_str}:{line}:{character}")

    return "\n".join(lines)


def _format_hover(hover: dict | None) -> str:
    """Format LSP hover result for the agent."""
    if not hover:
        return "No hover information available."

    contents = hover.get("contents", {})
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        value = contents.get("value", "")
        kind = contents.get("kind", "")
        if kind == "markdown":
            return value
        return value
    if isinstance(contents, list):
        parts = []
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("value", ""))
        return "\n".join(parts)
    return str(contents)


def _format_code_actions(actions: list[dict] | None) -> str:
    """Format LSP code actions for the agent."""
    if not actions:
        return "No code actions available."

    lines = []
    for i, action in enumerate(actions, 1):
        title = action.get("title", "unnamed")
        kind = action.get("kind", "")
        lines.append(f"{i}. {title}")
        if kind:
            lines.append(f"   Kind: {kind}")
        edit = action.get("edit")
        if edit:
            changes = edit.get("changes", {})
            if changes:
                lines.append(f"   Files affected: {len(changes)}")
                for uri, file_changes in list(changes.items())[:5]:
                    path = uri_to_path(uri)
                    path_str = str(path) if path else uri
                    try:
                        path_str = str(path.relative_to(Path.cwd()))
                    except (ValueError, TypeError):
                        pass
                    lines.append(f"   - {path_str}: {len(file_changes)} change(s)")
    return "\n".join(lines)


def _format_workspace_edit(edit: dict | None, workspace_root: Path) -> str:
    """Format LSP WorkspaceEdit result for the agent."""
    if not edit:
        return "No changes proposed."

    lines = []
    changes = edit.get("changes", {})
    if not changes:
        # Try documentChanges
        doc_changes = edit.get("documentChanges", [])
        for dc in doc_changes:
            if isinstance(dc, dict) and "textDocument" in dc:
                uri = dc["textDocument"].get("uri", "")
                path = uri_to_path(uri)
                path_str = str(path) if path else uri
                try:
                    path_str = str(path.relative_to(workspace_root))
                except (ValueError, TypeError):
                    pass
                edits = dc.get("edits", [])
                for e in edits:
                    if isinstance(e, dict):
                        text = e.get("newText", "")
                        lines.append(f"{path_str}: {text[:200]}")
        if not doc_changes:
            return "No changes proposed."
        return "\n".join(lines)

    for uri, file_changes in changes.items():
        path = uri_to_path(uri)
        path_str = str(path) if path else uri
        try:
            path_str = str(path.relative_to(workspace_root))
        except (ValueError, TypeError):
            pass
        lines.append(f"File: {path_str}")
        for change in file_changes:
            text = change.get("newText", "")
            range_info = change.get("range", {})
            start = range_info.get("start", {})
            end = range_info.get("end", {})
            start_line = start.get("line", 0) + 1
            end_line = end.get("line", 0) + 1
            preview = text[:200].replace("\n", "\\n")
            lines.append(f"  Lines {start_line}-{end_line}: {preview}")
    return "\n".join(lines)


def _format_diagnostics(diagnostics: list[dict], file_path: str) -> str:
    """Format LSP diagnostics for the agent."""
    if not diagnostics:
        return f"No diagnostics for {file_path}."

    lines = [f"Diagnostics for {file_path}:"]
    for diag in diagnostics:
        severity = diag.get("severity", 0)
        severity_label = DIAGNOSTIC_SEVERITY.get(severity, f"unknown({severity})")
        message = diag.get("message", "")
        range_info = diag.get("range", {})
        start = range_info.get("start", {})
        line = start.get("line", 0) + 1
        source = diag.get("source", "")
        code = diag.get("code", "")
        parts = [f"  {severity_label} (line {line}): {message}"]
        if source:
            parts.append(f"source={source}")
        if code:
            parts.append(f"code={code}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _format_document_symbols(symbols: list[dict] | None, file_path: str) -> str:
    """Format LSP document symbols for the agent."""
    if not symbols:
        return f"No symbols found in {file_path}."

    lines = [f"Symbols in {file_path}:"]

    def _render_symbol(sym: dict, indent: int = 0) -> None:
        prefix = "  " * indent
        name = sym.get("name", "<unnamed>")
        kind_num = sym.get("kind", 0)
        kind_name = _symbol_kind_name(kind_num)
        range_info = sym.get("range", {})
        start = range_info.get("start", {})
        line = start.get("line", 0) + 1
        lines.append(f"{prefix}{name} ({kind_name}) — line {line}")
        children = sym.get("children", [])
        if children:
            for child in children:
                _render_symbol(child, indent + 1)

    for sym in symbols:
        _render_symbol(sym)

    return "\n".join(lines)


def _format_workspace_symbols(symbols: list[dict] | None, workspace_root: Path) -> str:
    """Format LSP workspace symbols for the agent."""
    if not symbols:
        return "No workspace symbols found."

    lines = []
    for sym in symbols:
        name = sym.get("name", "<unnamed>")
        kind_num = sym.get("kind", 0)
        kind_name = _symbol_kind_name(kind_num)
        location = sym.get("location", {})
        uri = location.get("uri", "")
        path = uri_to_path(uri)
        path_str = str(path) if path else uri
        try:
            path_str = str(path.relative_to(workspace_root))
        except (ValueError, TypeError):
            pass
        range_info = location.get("range", {})
        start = range_info.get("start", {})
        line = start.get("line", 0) + 1
        lines.append(f"{name} ({kind_name}) — {path_str}:{line}")

    return "\n".join(lines)


# LSP SymbolKind values (from the spec)
_SYMBOL_KIND_NAMES = {
    1: "file",
    2: "module",
    3: "namespace",
    4: "package",
    5: "class",
    6: "method",
    7: "property",
    8: "field",
    9: "constructor",
    10: "enum",
    11: "interface",
    12: "function",
    13: "variable",
    14: "constant",
    15: "string",
    16: "number",
    17: "boolean",
    18: "array",
    19: "object",
    20: "key",
    21: "null",
    22: "enum_member",
    23: "struct",
    24: "event",
    25: "operator",
    26: "type_parameter",
}


def _symbol_kind_name(kind: int) -> str:
    """Convert an LSP SymbolKind number to a human-readable name."""
    return _SYMBOL_KIND_NAMES.get(kind, f"unknown({kind})")


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def auto_detect_lsp(base_dir: str) -> dict[str, dict] | None:
    """Auto-detect LSP servers based on project files and available binaries.

    Returns a server_configs dict suitable for LspManager, or None if no
    servers could be detected.
    """
    base = Path(base_dir)
    detected: dict[str, dict] = {}

    for marker_files, server_config in AUTO_DETECT_RULES:
        # Check if any marker file exists
        has_marker = any((base / f).exists() for f in marker_files)
        if not has_marker:
            continue

        # Check if the command is available
        command = server_config["command"]
        if _command_exists(command):
            name = server_config["name"]
            if name not in detected:
                detected[name] = {
                    "command": command,
                    "args": server_config.get("args", []),
                }

    return detected if detected else None


def _command_exists(command: str) -> bool:
    """Check if a command is available on PATH."""
    import shutil

    return shutil.which(command) is not None
