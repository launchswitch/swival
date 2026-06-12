"""Tests for LSP integration in the Session library API."""

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from swival import Session, ConfigError
from swival import agent


def _make_message(content=None, tool_calls=None):
    msg = types.SimpleNamespace()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.role = "assistant"
    return msg


def _simple_llm(*args, **kwargs):
    return _make_message(content="the answer"), "stop"


class TestSessionLspExplicitConfig:
    """Passing explicit lsp_servers dict to Session."""

    def test_lsp_manager_created(self, tmp_path, monkeypatch):
        """Explicit lsp_servers creates an LspManager."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        mock_mgr = MagicMock()
        mock_mgr.list_tools.return_value = []
        mock_mgr.get_tool_info.return_value = {}

        with patch("swival.lsp_client.LspManager", return_value=mock_mgr) as MockLspMgr:
            servers = {"pyright": {"command": "pyright-langserver", "args": ["--stdio"]}}
            s = Session(base_dir=str(tmp_path), history=False, lsp_servers=servers)
            s._setup()

            MockLspMgr.assert_called_once_with(
                servers,
                workspace_root=str(tmp_path),
                verbose=False,
            )
            mock_mgr.start.assert_called_once()
            assert s._lsp_manager is mock_mgr

    def test_lsp_tools_added_to_tool_list(self, tmp_path, monkeypatch):
        """LSP tools from list_tools() are added to the session tool list."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        lsp_tool = {"type": "function", "function": {"name": "lsp_definition", "parameters": {}}}
        mock_mgr = MagicMock()
        mock_mgr.list_tools.return_value = [lsp_tool]
        mock_mgr.get_tool_info.return_value = {"lsp": [("lsp_definition", "Find definitions")]}

        with patch("swival.lsp_client.LspManager", return_value=mock_mgr):
            s = Session(base_dir=str(tmp_path), history=False, lsp_servers={"s": {"command": "x"}})
            s._setup()

            tool_names = [t["function"]["name"] for t in s._tools]
            assert "lsp_definition" in tool_names

    def test_lsp_tool_info_in_system_prompt(self, tmp_path, monkeypatch):
        """LSP tool info is passed to build_system_prompt."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        tool_info = {"lsp": [("lsp_definition", "Find definitions")]}
        mock_mgr = MagicMock()
        mock_mgr.list_tools.return_value = []
        mock_mgr.get_tool_info.return_value = tool_info

        with patch("swival.lsp_client.LspManager", return_value=mock_mgr):
            s = Session(base_dir=str(tmp_path), history=False, lsp_servers={"s": {"command": "x"}})
            s._setup()

            # Verify get_tool_info was called (used in build_system_prompt)
            mock_mgr.get_tool_info.assert_called_once()

    def test_lsp_manager_in_loop_kwargs(self, tmp_path, monkeypatch):
        """lsp_manager is passed through _build_loop_kwargs."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        mock_mgr = MagicMock()
        mock_mgr.list_tools.return_value = []
        mock_mgr.get_tool_info.return_value = {}

        with patch("swival.lsp_client.LspManager", return_value=mock_mgr):
            s = Session(base_dir=str(tmp_path), history=False, lsp_servers={"s": {"command": "x"}})
            s._setup()

            state = s._make_per_run_state(system_content=None)
            kwargs = s._build_loop_kwargs(state)
            assert kwargs["lsp_manager"] is mock_mgr

    def test_lsp_manager_in_input_context(self, tmp_path, monkeypatch):
        """lsp_manager is passed through _make_input_context."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        mock_mgr = MagicMock()
        mock_mgr.list_tools.return_value = []
        mock_mgr.get_tool_info.return_value = {}

        with patch("swival.lsp_client.LspManager", return_value=mock_mgr):
            s = Session(base_dir=str(tmp_path), history=False, lsp_servers={"s": {"command": "x"}})
            s._setup()

            state = s._make_per_run_state(system_content=None)
            ctx = s._make_input_context(state)
            assert ctx.lsp_manager is mock_mgr

    def test_lsp_cleanup_on_close(self, tmp_path, monkeypatch):
        """LspManager.close() is called during Session cleanup."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        mock_mgr = MagicMock()
        mock_mgr.list_tools.return_value = []
        mock_mgr.get_tool_info.return_value = {}

        with patch("swival.lsp_client.LspManager", return_value=mock_mgr):
            s = Session(base_dir=str(tmp_path), history=False, lsp_servers={"s": {"command": "x"}})
            s._setup()

            assert s._lsp_manager is mock_mgr
            s._cleanup()
            mock_mgr.close.assert_called_once()

    def test_lsp_cleanup_via_context_manager(self, tmp_path, monkeypatch):
        """LspManager is cleaned up when Session is used as context manager."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        mock_mgr = MagicMock()
        mock_mgr.list_tools.return_value = []
        mock_mgr.get_tool_info.return_value = {}

        with patch("swival.lsp_client.LspManager", return_value=mock_mgr):
            with Session(base_dir=str(tmp_path), history=False, lsp_servers={"s": {"command": "x"}}) as s:
                s._setup()
                assert s._lsp_manager is mock_mgr

            mock_mgr.close.assert_called_once()


class TestSessionLspConfigFile:
    """Loading LSP config from a TOML file."""

    def test_lsp_config_file_loaded(self, tmp_path, monkeypatch):
        """lsp_config parameter loads servers from TOML file."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        config_file = tmp_path / "lsp.toml"
        config_file.write_text(
            '[lsp_servers.pyright]\ncommand = "pyright-langserver"\nargs = ["--stdio"]\n'
        )

        mock_mgr = MagicMock()
        mock_mgr.list_tools.return_value = []
        mock_mgr.get_tool_info.return_value = {}

        with patch("swival.lsp_client.LspManager", return_value=mock_mgr) as MockLspMgr:
            s = Session(
                base_dir=str(tmp_path), history=False, lsp_config=str(config_file)
            )
            s._setup()

            MockLspMgr.assert_called_once()
            call_args = MockLspMgr.call_args
            servers = call_args[0][0]
            assert "pyright" in servers
            assert servers["pyright"]["command"] == "pyright-langserver"

    def test_lsp_config_file_not_found(self, tmp_path, monkeypatch):
        """Missing lsp_config file raises ConfigError during setup."""
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        s = Session(
            base_dir=str(tmp_path),
            history=False,
            lsp_config="/nonexistent/lsp.toml",
        )
        with pytest.raises(ConfigError, match="lsp_config file not found"):
            s._setup()

    def test_explicit_servers_override_config_file(self, tmp_path, monkeypatch):
        """lsp_servers dict takes precedence over lsp_config file."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        config_file = tmp_path / "lsp.toml"
        config_file.write_text(
            '[lsp_servers.pyright]\ncommand = "pyright-langserver"\n'
        )

        mock_mgr = MagicMock()
        mock_mgr.list_tools.return_value = []
        mock_mgr.get_tool_info.return_value = {}

        explicit = {"gopls": {"command": "gopls"}}
        with patch("swival.lsp_client.LspManager", return_value=mock_mgr) as MockLspMgr:
            s = Session(
                base_dir=str(tmp_path),
                history=False,
                lsp_servers=explicit,
                lsp_config=str(config_file),
            )
            s._setup()

            servers = MockLspMgr.call_args[0][0]
            # Explicit lsp_servers wins; config file is not loaded
            assert "gopls" in servers


class TestSessionLspAutoDetect:
    """Auto-detection of LSP servers when no explicit config is given."""

    def test_auto_detect_called_when_no_config(self, tmp_path, monkeypatch):
        """auto_detect_lsp is called when no lsp_servers or lsp_config."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        detected = {"pyright": {"command": "pyright-langserver", "args": ["--stdio"]}}
        mock_mgr = MagicMock()
        mock_mgr.list_tools.return_value = []
        mock_mgr.get_tool_info.return_value = {}

        with (
            patch("swival.lsp_client.auto_detect_lsp", return_value=detected) as mock_detect,
            patch("swival.lsp_client.LspManager", return_value=mock_mgr),
        ):
            s = Session(base_dir=str(tmp_path), history=False)
            s._setup()

            mock_detect.assert_called_once_with(str(tmp_path))
            assert s._lsp_manager is mock_mgr

    def test_auto_detect_none_no_manager(self, tmp_path, monkeypatch):
        """No LspManager created when auto_detect_lsp returns None."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        with (
            patch("swival.lsp_client.auto_detect_lsp", return_value=None),
        ):
            s = Session(base_dir=str(tmp_path), history=False)
            s._setup()

            assert s._lsp_manager is None

    def test_auto_detect_with_project_marker(self, tmp_path, monkeypatch):
        """auto_detect_lsp finds servers when project markers exist."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        # Create a pyproject.toml to trigger auto-detection
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

        # Mock the command check to simulate pyright being available.
        from swival.lsp_client import LspManager

        def fake_start(self):
            # Skip actually starting server processes
            self._started = True

        monkeypatch.setattr(LspManager, "start", fake_start)

        with patch("swival.lsp_client._command_exists", return_value=True):
            s = Session(base_dir=str(tmp_path), history=False)
            s._setup()

            assert s._lsp_manager is not None
            assert s._lsp_manager._started is True


class TestSessionLspNoLsp:
    """no_lsp=True skips all LSP initialization."""

    def test_no_lsp_skips_init(self, tmp_path, monkeypatch):
        """no_lsp=True prevents any LspManager creation."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        with (
            patch("swival.lsp_client.auto_detect_lsp") as mock_detect,
            patch("swival.lsp_client.LspManager") as MockLspMgr,
        ):
            s = Session(
                base_dir=str(tmp_path),
                history=False,
                lsp_servers={"s": {"command": "x"}},
                no_lsp=True,
            )
            s._setup()

            mock_detect.assert_not_called()
            MockLspMgr.assert_not_called()
            assert s._lsp_manager is None

    def test_no_lsp_no_loop_kwargs(self, tmp_path, monkeypatch):
        """With no_lsp=True, lsp_manager is absent from loop kwargs."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        with (
            patch("swival.lsp_client.auto_detect_lsp"),
            patch("swival.lsp_client.LspManager"),
        ):
            s = Session(base_dir=str(tmp_path), history=False, no_lsp=True)
            s._setup()

            state = s._make_per_run_state(system_content=None)
            kwargs = s._build_loop_kwargs(state)
            assert "lsp_manager" not in kwargs


class TestSessionLspNoManager:
    """When no LSP is active, no lsp_manager is threaded through."""

    def test_no_lsp_manager_in_loop_kwargs(self, tmp_path, monkeypatch):
        """Without LSP config, lsp_manager is absent from loop kwargs."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        with (
            patch("swival.lsp_client.auto_detect_lsp", return_value=None),
        ):
            s = Session(base_dir=str(tmp_path), history=False)
            s._setup()

            state = s._make_per_run_state(system_content=None)
            kwargs = s._build_loop_kwargs(state)
            assert "lsp_manager" not in kwargs

    def test_no_lsp_manager_in_input_context(self, tmp_path, monkeypatch):
        """Without LSP config, lsp_manager in InputContext is None."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        with (
            patch("swival.lsp_client.auto_detect_lsp", return_value=None),
        ):
            s = Session(base_dir=str(tmp_path), history=False)
            s._setup()

            state = s._make_per_run_state(system_content=None)
            ctx = s._make_input_context(state)
            assert ctx.lsp_manager is None


class TestSessionLspRunIntegration:
    """Full run() cycle with LSP mocked end-to-end."""

    def test_run_with_lsp(self, tmp_path, monkeypatch):
        """A full run() works with LSP manager active."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        mock_mgr = MagicMock()
        mock_mgr.list_tools.return_value = []
        mock_mgr.get_tool_info.return_value = {}

        with (
            patch("swival.lsp_client.auto_detect_lsp", return_value=None),
            patch("swival.lsp_client.LspManager", return_value=mock_mgr),
        ):
            s = Session(
                base_dir=str(tmp_path),
                history=False,
                lsp_servers={"s": {"command": "x"}},
            )
            result = s.run("hello")

            assert result.answer == "the answer"
            mock_mgr.start.assert_called_once()

    def test_close_after_run_cleans_up_lsp(self, tmp_path, monkeypatch):
        """Calling close() after run() cleans up the LSP manager."""
        monkeypatch.setattr(agent, "call_llm", _simple_llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        mock_mgr = MagicMock()
        mock_mgr.list_tools.return_value = []
        mock_mgr.get_tool_info.return_value = {}

        with (
            patch("swival.lsp_client.auto_detect_lsp", return_value=None),
            patch("swival.lsp_client.LspManager", return_value=mock_mgr),
        ):
            s = Session(
                base_dir=str(tmp_path),
                history=False,
                lsp_servers={"s": {"command": "x"}},
            )
            s.run("hello")
            s.close()

            mock_mgr.close.assert_called_once()
