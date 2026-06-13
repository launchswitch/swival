"""Tests for LSP-as-context (``lsp_mode="context"``).

Phase A: dirty-file tracking + ``LspManager.collect_turn_context()``, the
``lsp_mode`` config selector, and the gating that stops ``lsp_*`` tools from
being advertised in context mode while the manager still runs for diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch


from swival import Session, agent
from swival.config import apply_config_to_args
from swival.lsp_client import LspManager, _LspConnection, path_to_uri


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(tmp_path) -> tuple[LspManager, _LspConnection]:
    """Build a started LspManager with one fake python connection.

    The connection is real (so routing + get_diagnostics work) but never talks
    to a server: we seed its diagnostics buffer directly.
    """
    mgr = LspManager(
        {"pyright": {"command": "x", "languages": ["python"]}},
        workspace_root=str(tmp_path),
    )
    conn = _LspConnection(
        "pyright",
        {"command": "x", "languages": ["python"]},
        Path(tmp_path),
    )
    mgr._connections["pyright"] = conn
    mgr._build_routing()
    mgr._started = True
    return mgr, conn


def _diag(message: str, line: int = 1, severity: int = 1) -> dict:
    return {
        "message": message,
        "severity": severity,
        "range": {"start": {"line": line - 1, "character": 0}},
        "source": "pyright",
        "code": "err-1",
    }


# ---------------------------------------------------------------------------
# collect_turn_context()
# ---------------------------------------------------------------------------


class TestCollectTurnContext:
    def test_returns_none_when_nothing_dirty(self, tmp_path):
        mgr, _ = _make_manager(tmp_path)
        assert mgr.collect_turn_context() is None

    def test_returns_none_for_dirty_file_with_no_diagnostics(self, tmp_path):
        mgr, _ = _make_manager(tmp_path)
        f = tmp_path / "src" / "a.py"
        f.parent.mkdir(parents=True)
        f.write_text("x = 1\n")
        mgr._mark_dirty(f)
        assert mgr.collect_turn_context() is None

    def test_formats_diagnostics_and_drains(self, tmp_path):
        mgr, conn = _make_manager(tmp_path)
        f = tmp_path / "src" / "a.py"
        f.parent.mkdir(parents=True)
        f.write_text("x = 1\n")
        conn._diagnostics[path_to_uri(f)] = [_diag("undefined name 'y'", line=3)]
        mgr._mark_dirty(f)

        out = mgr.collect_turn_context()
        assert out is not None
        # The "[lsp automated context]" prefix is added by the loop, not here.
        assert "Diagnostics for files edited this turn:" in out
        assert "src/a.py" in out  # rendered relative to workspace root
        assert "undefined name 'y'" in out
        # Dirty set is drained...
        assert mgr._dirty_paths == []
        # ...so a follow-up call reports nothing new.
        assert mgr.collect_turn_context() is None

    def test_preserves_dirty_when_clear_false(self, tmp_path):
        mgr, conn = _make_manager(tmp_path)
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        conn._diagnostics[path_to_uri(f)] = [_diag("boom")]
        mgr._mark_dirty(f)

        out = mgr.collect_turn_context(clear=False)
        assert out is not None
        assert mgr._dirty_paths  # not drained
        assert mgr.collect_turn_context() is not None  # still reportable

    def test_clears_dirty_even_when_not_started(self, tmp_path):
        mgr, _ = _make_manager(tmp_path)
        mgr._started = False
        mgr._mark_dirty(tmp_path / "a.py")
        assert mgr.collect_turn_context() is None
        assert mgr._dirty_paths == []

    def test_skips_files_with_no_resolvable_server(self, tmp_path):
        mgr, conn = _make_manager(tmp_path)
        py = tmp_path / "a.py"
        py.write_text("x = 1\n")
        conn._diagnostics[path_to_uri(py)] = [_diag("py err")]
        # A .rs file has no server in this manager.
        rs = tmp_path / "b.rs"
        rs.write_text("fn main(){}")
        mgr._mark_dirty(py)
        mgr._mark_dirty(rs)

        out = mgr.collect_turn_context()
        assert out is not None
        assert "a.py" in out
        assert "b.rs" not in out


# ---------------------------------------------------------------------------
# _mark_dirty()
# ---------------------------------------------------------------------------


class TestMarkDirty:
    def test_dedupes(self, tmp_path):
        mgr, _ = _make_manager(tmp_path)
        f = tmp_path / "a.py"
        mgr._mark_dirty(f)
        mgr._mark_dirty(f)
        assert mgr._dirty_paths == [str(f)]

    def test_caps_size_dropping_oldest(self, tmp_path):
        mgr, _ = _make_manager(tmp_path)
        mgr._dirty_cap = 3
        for i in range(6):
            mgr._mark_dirty(tmp_path / f"f{i}.py")
        assert len(mgr._dirty_paths) == 3
        # Oldest entries dropped, most recent retained.
        assert str(tmp_path / "f5.py") in mgr._dirty_paths
        assert str(tmp_path / "f0.py") not in mgr._dirty_paths


# ---------------------------------------------------------------------------
# lsp_mode config selector + reconciliation
# ---------------------------------------------------------------------------


class TestLspModeConfig:
    def test_default_is_tools(self):
        args = argparse.Namespace()
        apply_config_to_args(args, {})
        assert args.lsp_mode == "tools"

    def test_config_lsp_mode_context_passes_through(self):
        args = argparse.Namespace()
        apply_config_to_args(args, {"lsp_mode": "context"})
        assert args.lsp_mode == "context"

    def test_config_no_lsp_forces_off(self):
        args = argparse.Namespace()
        apply_config_to_args(args, {"no_lsp": True})
        assert args.lsp_mode == "off"

    def test_cli_no_lsp_flag_forces_off(self):
        # Simulate --no-lsp on the CLI (no_lsp set, lsp_mode still default).
        args = argparse.Namespace(no_lsp=True, lsp_mode="tools")
        apply_config_to_args(args, {})
        assert args.lsp_mode == "off"


# ---------------------------------------------------------------------------
# Session gating: context mode runs the manager but advertises no lsp_* tools
# ---------------------------------------------------------------------------


def _simple_llm(*args, **kwargs):
    msg = MagicMock()
    msg.content = "the answer"
    msg.tool_calls = None
    return msg, "stop"


class TestSessionContextModeGating:
    def test_context_mode_starts_manager_but_no_lsp_tools(self, tmp_path, monkeypatch):
        """lsp_mode='context': manager starts; no lsp_* tools in _tools."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        lsp_tool = {
            "type": "function",
            "function": {"name": "lsp_definition", "parameters": {}},
        }
        mock_mgr = MagicMock()
        mock_mgr.list_tools.return_value = [lsp_tool]
        mock_mgr.get_tool_info.return_value = {
            "lsp": [("lsp_definition", "Find definitions")]
        }

        with patch("swival.lsp_client.LspManager", return_value=mock_mgr):
            s = Session(
                base_dir=str(tmp_path),
                history=False,
                lsp_servers={"s": {"command": "x"}},
                lsp_mode="context",
            )
            s._setup()

        # Manager is running (for diagnostics/sync)...
        mock_mgr.start.assert_called_once()
        assert s._lsp_manager is mock_mgr
        # ...but no lsp_* tools are advertised to the model.
        tool_names = [t["function"]["name"] for t in s._tools]
        assert "lsp_definition" not in tool_names

    def test_context_mode_omits_lsp_tool_info_from_prompt(self, tmp_path, monkeypatch):
        """In context mode, lsp_tool_info is None (no 'LSP Tools' prompt section)."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        mock_mgr = MagicMock()
        mock_mgr.list_tools.return_value = []
        mock_mgr.get_tool_info.return_value = {"lsp": [("lsp_definition", "x")]}

        captured = {}

        def fake_build(*a, **kw):
            captured["lsp_tool_info"] = kw.get("lsp_tool_info")
            return ("system prompt", [])

        monkeypatch.setattr(agent, "build_system_prompt", fake_build)

        with patch("swival.lsp_client.LspManager", return_value=mock_mgr):
            s = Session(
                base_dir=str(tmp_path),
                history=False,
                lsp_servers={"s": {"command": "x"}},
                lsp_mode="context",
            )
            s._setup()

        assert captured["lsp_tool_info"] is None

    def test_tools_mode_still_advertises_lsp_tools(self, tmp_path, monkeypatch):
        """Regression: tools mode (default) still adds lsp_* tools."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        lsp_tool = {
            "type": "function",
            "function": {"name": "lsp_definition", "parameters": {}},
        }
        mock_mgr = MagicMock()
        mock_mgr.list_tools.return_value = [lsp_tool]
        mock_mgr.get_tool_info.return_value = {"lsp": [("lsp_definition", "x")]}

        with patch("swival.lsp_client.LspManager", return_value=mock_mgr):
            s = Session(
                base_dir=str(tmp_path),
                history=False,
                lsp_servers={"s": {"command": "x"}},
                lsp_mode="tools",
            )
            s._setup()

        tool_names = [t["function"]["name"] for t in s._tools]
        assert "lsp_definition" in tool_names


# ---------------------------------------------------------------------------
# _symbol_range + fetch_symbol_body (Phase B helpers, no server needed)
# ---------------------------------------------------------------------------


class TestSymbolRange:
    def test_document_symbol_form(self):
        from swival.lsp_client import _symbol_range

        assert _symbol_range({"range": {"start": {"line": 1}}}) == {
            "start": {"line": 1}
        }

    def test_symbol_information_form(self):
        # pyright returns document symbols in this flat, location-nested form.
        from swival.lsp_client import _symbol_range

        sym = {"name": "add", "location": {"uri": "f", "range": {"start": {"line": 3}}}}
        assert _symbol_range(sym) == {"start": {"line": 3}}

    def test_missing(self):
        from swival.lsp_client import _symbol_range

        assert _symbol_range({}) == {}


class TestFetchSymbolBody:
    def test_scoped_range_reads_exact_lines(self, tmp_path):
        mgr, _ = _make_manager(tmp_path)
        f = tmp_path / "m.py"
        f.write_text("\n".join(f"line{i}" for i in range(20)))
        # Lines 3-5 (1-based inclusive) -> 0-based slice [2:5].
        assert mgr.fetch_symbol_body(f, 3, 5) == "line2\nline3\nline4"

    def test_tiny_range_expands_to_window(self, tmp_path):
        mgr, _ = _make_manager(tmp_path)
        f = tmp_path / "m.py"
        f.write_text("\n".join(f"line{i}" for i in range(40)))
        body = mgr.fetch_symbol_body(f, 4, 4)  # 1-line selection range
        lines = body.splitlines()
        assert lines[0] == "line3"
        assert len(lines) > 5  # grown, not just one line

    def test_caps_at_max_lines(self, tmp_path):
        mgr, _ = _make_manager(tmp_path)
        f = tmp_path / "m.py"
        f.write_text("\n".join(f"line{i}" for i in range(200)))
        body = mgr.fetch_symbol_body(f, 1, 200, max_lines=10)
        assert len(body.splitlines()) <= 10

    def test_missing_file_returns_empty(self, tmp_path):
        mgr, _ = _make_manager(tmp_path)
        assert mgr.fetch_symbol_body(tmp_path / "nope.py", 1, 5) == ""
