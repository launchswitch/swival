"""Tests for symbol-aware ``read_file`` / ``read_multiple_files`` (Phases 3-4).

``symbol`` reads only a top-level definition span instead of the whole file,
saving model context. It is mutually exclusive with offset/limit/tail_lines.
"""

from __future__ import annotations

from swival.tools import _hash_bytes, _read_file, _read_files, dispatch


# ---------------------------------------------------------------------------
# Phase 3: single read_file symbol support
# ---------------------------------------------------------------------------


class TestReadFileSymbol:
    def test_function_returns_only_its_span(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("def my_func():\n    return 1\n\ndef other():\n    return 2\n")
        result = _read_file("m.py", str(tmp_path), symbol="my_func")
        assert "1: def my_func" in result
        assert "2:     return 1" in result
        # other() is outside the symbol span.
        assert "other" not in result
        # No "[N more lines]" continuation hint for a symbol read.
        assert "more lines" not in result
        # Checksum is still computed over the FULL file bytes.
        assert f"[checksum={_hash_bytes(f.read_bytes())}]" in result

    def test_class_span_includes_methods(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text(
            "class Foo:\n"
            "    def __init__(self):\n"
            "        pass\n"
            "\n"
            "def bar():\n"
            "    pass\n"
        )
        result = _read_file("m.py", str(tmp_path), symbol="Foo")
        assert "1: class Foo:" in result
        # The class span covers its body, including the method.
        assert "__init__" in result
        # bar() is a separate top-level symbol, not included.
        assert "bar" not in result

    def test_missing_symbol_returns_error_with_suggestions(self, tmp_path):
        f = tmp_path / "m.py"
        f.write_text("def my_func():\n    return 1\n")
        result = _read_file("m.py", str(tmp_path), symbol="nope")
        assert result.startswith("error:")
        assert "not found" in result
        # The available top-level symbol is listed to guide the model.
        assert "my_func" in result

    def test_no_symbol_reads_whole_file_as_before(self, tmp_path):
        # Regression: omitting symbol preserves the default full read.
        f = tmp_path / "m.py"
        f.write_text("def my_func():\n    return 1\n\ndef other():\n    return 2\n")
        result = _read_file("m.py", str(tmp_path))
        assert "my_func" in result
        assert "other" in result


class TestReadFileSymbolDispatchConflicts:
    """symbol is mutually exclusive with offset/limit/tail_lines (explicit)."""

    def test_symbol_with_offset_errors(self, tmp_path):
        (tmp_path / "m.py").write_text("def my_func():\n    return 1\n")
        result = dispatch(
            "read_file",
            {"file_path": "m.py", "symbol": "my_func", "offset": 1},
            str(tmp_path),
        )
        assert result.startswith("error:")
        assert "mutually exclusive" in result

    def test_symbol_with_limit_errors(self, tmp_path):
        (tmp_path / "m.py").write_text("def my_func():\n    return 1\n")
        result = dispatch(
            "read_file",
            {"file_path": "m.py", "symbol": "my_func", "limit": 2000},
            str(tmp_path),
        )
        assert "mutually exclusive" in result

    def test_symbol_with_tail_lines_errors(self, tmp_path):
        (tmp_path / "m.py").write_text("def my_func():\n    return 1\n")
        result = dispatch(
            "read_file",
            {"file_path": "m.py", "symbol": "my_func", "tail_lines": 5},
            str(tmp_path),
        )
        assert "mutually exclusive" in result

    def test_symbol_dispatch_reads_span(self, tmp_path):
        (tmp_path / "m.py").write_text(
            "def my_func():\n    return 1\n\ndef other():\n    return 2\n"
        )
        result = dispatch(
            "read_file", {"file_path": "m.py", "symbol": "my_func"}, str(tmp_path)
        )
        assert "my_func" in result
        assert "other" not in result


# ---------------------------------------------------------------------------
# Phase 4: read_multiple_files symbol support
# ---------------------------------------------------------------------------


class TestReadMultipleFilesSymbol:
    def test_two_symbol_reads_return_bounded_sections(self, tmp_path):
        (tmp_path / "a.py").write_text(
            "def alpha():\n    return 1\n\ndef beta():\n    return 2\n"
        )
        (tmp_path / "b.py").write_text("class Gamma:\n    pass\n")
        result, full = _read_files(
            [
                {"file_path": "a.py", "symbol": "alpha"},
                {"file_path": "b.py", "symbol": "Gamma"},
            ],
            str(tmp_path),
        )
        assert "symbol=alpha" in result
        assert "symbol=Gamma" in result
        assert "alpha" in result
        assert "Gamma" in result
        # beta() was not selected.
        assert "beta" not in result
        # Symbol reads are partial — omitted from full_contents so didOpen
        # falls back to a real full read rather than a synthetic partial one.
        assert full == {}

    def test_one_missing_one_valid_symbol(self, tmp_path):
        (tmp_path / "a.py").write_text("def alpha():\n    return 1\n")
        result, _ = _read_files(
            [
                {"file_path": "a.py", "symbol": "alpha"},
                {"file_path": "a.py", "symbol": "nope"},
            ],
            str(tmp_path),
        )
        assert "files_succeeded: 1" in result
        assert "files_with_errors: 1" in result
        assert "alpha" in result
        assert "not found" in result

    def test_symbol_conflict_is_inline_error(self, tmp_path):
        (tmp_path / "a.py").write_text("def alpha():\n    return 1\n")
        result, _ = _read_files(
            [{"file_path": "a.py", "symbol": "alpha", "limit": 5}],
            str(tmp_path),
        )
        assert "files_with_errors: 1" in result
        assert "mutually exclusive" in result

    def test_string_entries_still_read_full(self, tmp_path):
        # Regression: plain string entries ignore symbol machinery.
        (tmp_path / "a.py").write_text(
            "def alpha():\n    return 1\n\ndef beta():\n    x=1\n"
        )
        result, full = _read_files(["a.py"], str(tmp_path))
        assert "alpha" in result
        assert "beta" in result
        # Full read populates full_contents (drives didOpen without re-read).
        assert str(tmp_path / "a.py") in full
