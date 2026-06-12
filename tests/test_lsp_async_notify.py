"""Tests for the async LSP notification queue (PR-3).

Verify that on_file_read/write/delete return immediately (non-blocking),
that the worker eventually drains the queue, and that close() is clean.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from swival.lsp_client import LspManager, _LspConnection


class TestAsyncNotificationQueue:
    """Verify non-blocking enqueue and worker drain behavior."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a started LspManager with a mock connection."""
        mgr = LspManager(
            {"mock-server": {"command": "echo", "languages": ["python"]}},
            workspace_root=str(tmp_path),
            verbose=False,
        )
        # Stub connection to avoid spawning a real server process.
        conn = _LspConnection(
            "mock-server",
            {"command": "echo", "languages": ["python"]},
            Path(tmp_path),
        )
        conn._initialized = True
        conn._capabilities = {
            "definitionProvider": True,
            "referencesProvider": True,
            "hoverProvider": True,
            "codeActionProvider": True,
            "renameProvider": True,
            "documentSymbolProvider": True,
        }
        conn._notify = MagicMock()

        # Start the manager's event loop and worker, then inject our stub.
        mgr._loop = __import__("asyncio").new_event_loop()
        mgr._thread = threading.Thread(target=mgr._loop.run_forever, daemon=True)
        mgr._thread.start()
        mgr._connections["mock-server"] = conn
        mgr._build_routing()

        # Start the notification worker.
        import queue

        mgr._notification_queue = queue.Queue(maxsize=1000)
        mgr._worker_stop.clear()
        mgr._notification_worker = threading.Thread(
            target=mgr._notification_loop, daemon=True
        )
        mgr._notification_worker.start()
        mgr._started = True

        yield mgr

        mgr.close()

    def test_on_file_read_returns_immediately(self, manager, tmp_path):
        """on_file_read should return in microseconds, not milliseconds."""
        f = tmp_path / "a.py"
        f.write_text("x = 1")
        t0 = time.monotonic()
        for _ in range(200):
            manager.on_file_read(f, "x = 1")
        elapsed = time.monotonic() - t0
        # 200 queue puts should take <100ms (pure enqueue, no LSP blocking).
        assert elapsed < 0.1, f"Expected <100ms for 200 enqueues, got {elapsed:.3f}s"

    def test_on_file_write_returns_immediately(self, manager, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x = 1")
        t0 = time.monotonic()
        for _ in range(200):
            manager.on_file_write(f, "x = 2")
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1

    def test_worker_drains_queue(self, manager, tmp_path):
        """Items enqueued by on_file_read are eventually processed."""
        f = tmp_path / "a.py"
        f.write_text("x = 1")
        manager.on_file_read(f, "x = 1")
        # The worker should drain within a few seconds.

        q = manager._notification_queue
        # Wait for the queue to empty (worker processed the item).
        for _ in range(50):
            if q.empty():
                break
            time.sleep(0.1)
        assert q.empty(), "Worker did not drain the notification queue"

    def test_close_stops_worker(self, tmp_path):
        """close() should stop the notification worker thread."""
        mgr = LspManager(
            {"mock-server": {"command": "echo", "languages": ["python"]}},
            workspace_root=str(tmp_path),
            verbose=False,
        )
        import asyncio
        import queue as q_mod

        mgr._loop = asyncio.new_event_loop()
        mgr._thread = threading.Thread(target=mgr._loop.run_forever, daemon=True)
        mgr._thread.start()
        conn = _LspConnection(
            "mock-server",
            {"command": "echo", "languages": ["python"]},
            Path(tmp_path),
        )
        conn._initialized = True
        conn._capabilities = {}
        conn._notify = MagicMock()
        mgr._connections["mock-server"] = conn
        mgr._notification_queue = q_mod.Queue(maxsize=10)
        mgr._notification_worker = threading.Thread(
            target=mgr._notification_loop, daemon=True
        )
        mgr._notification_worker.start()
        mgr._started = True

        assert mgr._notification_worker.is_alive()
        mgr.close()
        assert not mgr._notification_worker.is_alive()

    def test_enqueue_drops_oldest_when_full(self, tmp_path):
        """A full queue drops the oldest item to make room."""
        # Use a manager with a small queue and NO active worker so the
        # test can fill it without racing the drain thread.
        mgr = LspManager(
            {"mock-server": {"command": "echo", "languages": ["python"]}},
            workspace_root=str(tmp_path),
            verbose=False,
        )
        import queue

        q = queue.Queue(maxsize=10)
        mgr._notification_queue = q
        mgr._started = True

        dummy = Path("/tmp/dummy.py")
        for i in range(10):
            q.put_nowait(("read", dummy, f"content-{i}"))

        assert q.full()
        # Enqueue one more — should drop the oldest.
        mgr._enqueue_notification("read", dummy, "overflow")
        # The queue should still have 10 items (dropped oldest, added new).
        assert q.qsize() == 10
        # Drain to check — the last item should be "overflow".
        last = None
        while not q.empty():
            try:
                last = q.get_nowait()
            except queue.Empty:
                break
        assert last is not None
        assert last[2] == "overflow"
