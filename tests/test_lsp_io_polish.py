"""Tests for PR-4 I/O polish: _read_files and _edit_file refactors.

These tests verify the new tuple return shapes and confirm that the
LSP notification hooks consume the already-read content rather than
re-reading from disk.
"""

from __future__ import annotations



from swival import tools as t


# ---------------------------------------------------------------------------
# _read_files return shape
# ---------------------------------------------------------------------------


class TestReadFilesReturnShape:
    def test_returns_formatted_result_and_content_dict(self, tmp_path):
        f1 = tmp_path / "a.py"
        f1.write_text("alpha\nbeta\n")
        f2 = tmp_path / "b.py"
        f2.write_text("gamma\ndelta\n")

        result, contents = t._read_files(
            files=[str(f1), str(f2)],
            base_dir=str(tmp_path),
        )
        # The formatted result is a non-empty string
        assert isinstance(result, str)
        assert "alpha" in result
        assert "gamma" in result
        # The content dict is populated for full reads
        assert isinstance(contents, dict)
        assert str(f1) in contents
        assert str(f2) in contents
        assert contents[str(f1)] == "alpha\nbeta\n"
        assert contents[str(f2)] == "gamma\ndelta\n"

    def test_offset_read_does_not_populate_content_dict(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("a\nb\nc\nd\ne\n")
        result, contents = t._read_files(
            files=[{"file_path": str(f), "offset": 2, "limit": 2}],
            base_dir=str(tmp_path),
        )
        # Partial read: content dict should not include this file
        # (downstream hooks fall back to a disk read for LSP didOpen).
        # The formatted result includes line-number prefixes.
        assert "2: b" in result
        assert "3: c" in result
        assert contents == {}

    def test_empty_files_returns_error_and_empty_dict(self, tmp_path):
        result, contents = t._read_files(files=[], base_dir=str(tmp_path))
        assert result.startswith("error:")
        assert contents == {}

    def test_too_many_files_returns_error_and_empty_dict(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("x")
        # MAX is 64 in tools.py; pass 65 entries
        many = [{"file_path": str(f)} for _ in range(65)]
        result, contents = t._read_files(files=many, base_dir=str(tmp_path))
        assert result.startswith("error:")
        assert contents == {}

    def test_partial_failure_still_populates_successful_reads(self, tmp_path):
        f1 = tmp_path / "a.py"
        f1.write_text("alpha")
        # f2 doesn't exist
        f2 = tmp_path / "missing.py"
        result, contents = t._read_files(
            files=[str(f1), str(f2)], base_dir=str(tmp_path)
        )
        # Result should mention both files
        assert "alpha" in result
        # Content dict should have only the successful read
        assert str(f1) in contents
        assert str(f2) not in contents


# ---------------------------------------------------------------------------
# _apply_edit return shape
# ---------------------------------------------------------------------------


class TestApplyEditReturnShape:
    def test_returns_result_and_new_content(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("alpha\nbeta\ngamma\n")
        result, new_content = t._apply_edit(
            file_path=str(f),
            old_string="beta",
            new_string="BETA",
            base_dir=str(tmp_path),
        )
        assert isinstance(result, str)
        assert "Edited" in result
        assert new_content == "alpha\nBETA\ngamma\n"

    def test_error_path_returns_none_content(self, tmp_path):
        f = tmp_path / "missing.py"
        result, new_content = t._apply_edit(
            file_path=str(f),
            old_string="x",
            new_string="y",
            base_dir=str(tmp_path),
        )
        assert result.startswith("error:")
        assert new_content is None

    def test_replace_all_returns_full_new_content(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("foo\nfoo\nfoo\n")
        result, new_content = t._apply_edit(
            file_path=str(f),
            old_string="foo",
            new_string="bar",
            base_dir=str(tmp_path),
            replace_all=True,
        )
        assert "Edited" in result
        assert new_content == "bar\nbar\nbar\n"


# ---------------------------------------------------------------------------
# _edit_file still returns just a string (backward compat for tests)
# ---------------------------------------------------------------------------


class TestEditFileBackwardCompat:
    def test_returns_only_string(self, tmp_path):
        f = tmp_path / "a.py"
        f.write_text("alpha\n")
        result = t._edit_file(
            file_path=str(f),
            old_string="alpha",
            new_string="beta",
            base_dir=str(tmp_path),
        )
        assert isinstance(result, str)
        assert "Edited" in result


# ---------------------------------------------------------------------------
# LSP hooks use already-read content (no redundant disk reads)
# ---------------------------------------------------------------------------


class TestLspHooksUseAlreadyReadContent:
    """Verify the dispatch() LSP hooks don't re-read from disk for
    read_multiple_files or edit_file when the content was already
    available from the tool function."""

    def test_read_multiple_files_hook_does_not_reread(self, tmp_path, monkeypatch):
        from swival.tools import dispatch

        f = tmp_path / "a.py"
        f.write_text("alpha\nbeta\n")

        # Track disk reads of the file
        from pathlib import Path as _P

        original_read_text = _P.read_text
        read_count = {"n": 0}

        def counting_read_text(self, *a, **kw):
            if self == f:
                read_count["n"] += 1
            return original_read_text(self, *a, **kw)

        monkeypatch.setattr(_P, "read_text", counting_read_text)

        # Patch the LSP manager with a no-op recorder
        notifications = []

        class FakeLspManager:
            def on_file_read(self, abs_path, content):
                notifications.append((str(abs_path), content))

        # Call dispatch with an LSP manager
        result = dispatch(
            "read_multiple_files",
            {"files": [str(f)]},
            base_dir=str(tmp_path),
            lsp_manager=FakeLspManager(),
        )
        assert "alpha" in result
        # The hook should have used the already-read content, not re-read
        # (read_text was called once by _read_files, not a second time
        # by the hook)
        assert read_count["n"] == 1, (
            f"expected 1 disk read (from _read_files), got {read_count['n']}"
        )
        assert len(notifications) == 1
        assert notifications[0][1] == "alpha\nbeta\n"

    def test_edit_file_hook_does_not_reread(self, tmp_path, monkeypatch):
        from swival.tools import dispatch

        f = tmp_path / "a.py"
        f.write_text("alpha\n")

        from pathlib import Path as _P

        original_read_text = _P.read_text
        read_count = {"n": 0}

        def counting_read_text(self, *a, **kw):
            if self == f:
                read_count["n"] += 1
            return original_read_text(self, *a, **kw)

        monkeypatch.setattr(_P, "read_text", counting_read_text)

        notifications = []

        class FakeLspManager:
            def on_file_write(self, abs_path, content):
                notifications.append((str(abs_path), content))

        result = dispatch(
            "edit_file",
            {
                "file_path": str(f),
                "old_string": "alpha",
                "new_string": "beta",
            },
            base_dir=str(tmp_path),
            lsp_manager=FakeLspManager(),
        )
        assert "Edited" in result
        # Exactly one disk read (by _apply_edit to apply the change).
        # The LSP hook uses the new content from _apply_edit instead
        # of re-reading.
        assert read_count["n"] == 1, (
            f"expected 1 disk read (by _apply_edit), got {read_count['n']}"
        )
        assert len(notifications) == 1
        assert notifications[0][1] == "beta\n"
