"""Unit tests for swival.lsp_client core (no real server required).

These tests cover the wire protocol, document-state tracking, URI
encoding, and request correlation. Real-server end-to-end coverage
lives in the manual smoke test in /tmp/test_lsp_fixes.py.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from swival.lsp_client import (
    LSP_TOOLS,
    _LspConnection,
    _encode_message,
    _read_message,
    auto_detect_lsp,
    path_to_uri,
    uri_to_path,
)


# ---------------------------------------------------------------------------
# URI encoding (fix for: paths with spaces, #, %, etc.)
# ---------------------------------------------------------------------------


class TestPathToUri:
    def test_simple_path(self):
        p = Path("/tmp/foo.py")
        assert path_to_uri(p) == "file:///tmp/foo.py"
        assert uri_to_path(path_to_uri(p)) == p

    def test_path_with_space(self):
        p = Path("/tmp/has space.py")
        uri = path_to_uri(p)
        # Space must be percent-encoded for valid LSP URIs
        assert "%20" in uri
        assert " " not in uri
        assert uri_to_path(uri) == p

    def test_path_with_hash(self):
        p = Path("/tmp/has#hash.py")
        uri = path_to_uri(p)
        # '#' must be encoded — it delimits the URI fragment
        assert "%23" in uri
        assert "#" not in uri
        assert uri_to_path(uri) == p

    def test_path_with_percent(self):
        p = Path("/tmp/percent%file.py")
        uri = path_to_uri(p)
        # '%' must be encoded to avoid ambiguity
        assert "%25" in uri
        assert uri_to_path(uri) == p

    def test_path_with_unicode(self):
        p = Path("/tmp/unicode-é.py")
        uri = path_to_uri(p)
        assert uri_to_path(uri) == p

    def test_uri_to_path_rejects_non_file_uri(self):
        assert uri_to_path("http://example.com/foo") is None
        assert uri_to_path("not a uri") is None


# ---------------------------------------------------------------------------
# Wire protocol encode/decode roundtrip
# ---------------------------------------------------------------------------


class TestWireProtocol:
    def test_encode_format(self):
        msg = {"jsonrpc": "2.0", "id": 1, "method": "test", "params": {"x": 1}}
        encoded = _encode_message(msg)
        # Header must end with \r\n\r\n before the body
        assert b"\r\n\r\n" in encoded
        header, body = encoded.split(b"\r\n\r\n", 1)
        # Content-Length must equal body byte length
        assert b"Content-Length: " in header
        cl = int(header.split(b"Content-Length: ")[1].split(b"\r\n")[0])
        assert cl == len(body)
        # Body must be the JSON
        assert json.loads(body) == msg

    def test_read_message_roundtrip(self):
        """_encode_message -> _read_message returns the same dict."""

        async def go():
            msg = {"jsonrpc": "2.0", "id": 42, "method": "ping", "params": {"k": "v"}}
            encoded = _encode_message(msg)

            class FakeStdout:
                def __init__(self, data):
                    self._data = data
                    self._pos = 0

                async def readline(self):
                    while self._pos < len(self._data):
                        nl = self._data.index(b"\n", self._pos)
                        line = self._data[self._pos : nl + 1]
                        self._pos = nl + 1
                        return line
                    return b""

                async def readexactly(self, n):
                    chunk = self._data[self._pos : self._pos + n]
                    self._pos += n
                    return chunk

            out = FakeStdout(encoded)
            return await _read_message(out)

        assert asyncio.run(go()) == {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "ping",
            "params": {"k": "v"},
        }

    def test_read_message_eof(self):
        async def go():
            class FakeStdout:
                async def readline(self):
                    return b""

            out = FakeStdout()
            return await _read_message(out)

        assert asyncio.run(go()) is None


# ---------------------------------------------------------------------------
# didOpen / didChange state tracking (fix for: duplicate didOpen violations)
# ---------------------------------------------------------------------------


class TestDocumentStateTracking:
    @pytest.fixture
    def conn(self):
        """Build an _LspConnection without starting a process."""
        c = _LspConnection(
            name="test",
            config={"command": "echo"},
            workspace_root=Path("/tmp"),
            verbose=False,
        )
        c._initialized = True
        c._capabilities = {}
        c._notify = AsyncMock()
        return c

    def _methods(self, conn):
        return [c[0][0] for c in conn._notify.call_args_list]

    def _versions(self, conn):
        return [c[0][1]["textDocument"]["version"] for c in conn._notify.call_args_list]

    def test_first_did_open_sends_didopen(self, conn):
        asyncio.run(conn.did_open("file:///x.py", "python", "x = 1"))
        assert self._methods(conn) == ["textDocument/didOpen"]
        assert self._versions(conn) == [1]

    def test_second_did_open_sends_didchange(self, conn):
        """Idempotency: second did_open for same URI sends didChange, not didOpen."""
        asyncio.run(conn.did_open("file:///x.py", "python", "x = 1"))
        asyncio.run(conn.did_open("file:///x.py", "python", "x = 2"))
        assert self._methods(conn) == ["textDocument/didOpen", "textDocument/didChange"]
        assert self._versions(conn) == [1, 2]

    def test_three_did_opens_produce_one_didopen_two_didchange(self, conn):
        asyncio.run(conn.did_open("file:///x.py", "python", "a"))
        asyncio.run(conn.did_open("file:///x.py", "python", "b"))
        asyncio.run(conn.did_open("file:///x.py", "python", "c"))
        methods = self._methods(conn)
        assert methods.count("textDocument/didOpen") == 1
        assert methods.count("textDocument/didChange") == 2

    def test_did_change_for_new_uri_opens_first(self, conn):
        """didChange on a not-yet-opened URI should open it first."""
        asyncio.run(conn.did_change("file:///y.py", "y = 1", "python"))
        assert self._methods(conn) == ["textDocument/didOpen"]

    def test_did_close_removes_document(self, conn):
        asyncio.run(conn.did_open("file:///z.py", "python", "z = 1"))
        assert "file:///z.py" in conn._documents
        asyncio.run(conn.did_close("file:///z.py"))
        assert "file:///z.py" not in conn._documents


# ---------------------------------------------------------------------------
# Response correlation
# ---------------------------------------------------------------------------


class TestResponseCorrelation:
    def test_response_with_id_resolves_future(self):
        """A response with a known id resolves the pending future."""
        c = _LspConnection("t", {"command": "x"}, Path("/tmp"), verbose=False)
        loop = asyncio.new_event_loop()
        try:
            fut = loop.create_future()
            c._pending[1] = fut
            c._handle_response({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
            assert fut.result() == {"ok": True}
            assert 1 not in c._pending
        finally:
            loop.close()

    def test_response_with_error_raises(self):
        c = _LspConnection("t", {"command": "x"}, Path("/tmp"), verbose=False)
        loop = asyncio.new_event_loop()
        try:
            fut = loop.create_future()
            c._pending[2] = fut
            c._handle_response(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "error": {"code": -32601, "message": "Method not found"},
                }
            )
            with pytest.raises(RuntimeError, match="Method not found"):
                fut.result()
        finally:
            loop.close()

    def test_publish_diagnostics_stored(self):
        c = _LspConnection("t", {"command": "x"}, Path("/tmp"), verbose=False)
        c._handle_response(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {
                    "uri": "file:///a.py",
                    "diagnostics": [{"range": {}, "message": "oops", "severity": 1}],
                },
            }
        )
        assert c.get_diagnostics("file:///a.py") == [
            {"range": {}, "message": "oops", "severity": 1}
        ]


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------


class TestToolSchemas:
    def test_all_lsp_tools_have_required_fields(self):
        for tool in LSP_TOOLS:
            assert tool["type"] == "function"
            fn = tool["function"]
            assert fn["name"].startswith("lsp_")
            assert "description" in fn
            assert "parameters" in fn
            params = fn["parameters"]
            assert params["type"] == "object"
            assert "required" in params
            for req in params["required"]:
                assert req in params["properties"], (
                    f"{fn['name']}: required {req!r} not in properties"
                )

    def test_no_duplicate_tool_names(self):
        names = [t["function"]["name"] for t in LSP_TOOLS]
        assert len(names) == len(set(names)), f"duplicates: {names}"


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


class TestAutoDetect:
    def test_detects_python_project(self, tmp_path, monkeypatch):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        monkeypatch.setattr(
            "swival.lsp_client._command_exists", lambda c: c == "pyright-langserver"
        )
        result = auto_detect_lsp(str(tmp_path))
        assert result is not None
        assert "pyright" in result

    def test_returns_none_for_empty_project(self, tmp_path, monkeypatch):
        monkeypatch.setattr("swival.lsp_client._command_exists", lambda c: True)
        result = auto_detect_lsp(str(tmp_path))
        assert result is None

    def test_skips_server_when_command_missing(self, tmp_path, monkeypatch):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        monkeypatch.setattr("swival.lsp_client._command_exists", lambda c: False)
        result = auto_detect_lsp(str(tmp_path))
        assert result is None
