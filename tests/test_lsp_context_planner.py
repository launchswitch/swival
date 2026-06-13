"""Tests for the Phase B LSP-context planner (swival.lsp_context_planner).

The LLM call is injectable (completion_fn) and the LSP manager is faked, so
these exercise the selection / fetch / budget / degradation logic with no
server and no model.
"""

from __future__ import annotations


from swival.lsp_context_planner import LspContextPlanner, _parse_picks


class FakeLsp:
    """Stand-in for LspManager with just the methods the planner calls."""

    def __init__(self, candidates, bodies):
        self._candidates = candidates
        self._bodies = bodies  # {(rel_file, start_line): body_text}
        self.dirty = []

    def gather_symbol_candidates(self, tokens, dirty, *, max_candidates=40):
        self.dirty = list(dirty)
        return list(self._candidates)[:max_candidates]

    def fetch_symbol_body(self, abs_path, start_line, end_line, **kw):
        return self._bodies.get((str(abs_path), start_line), "")

    def dirty_paths(self):
        return list(self.dirty)


def _candidate(name, file, line, end_line=None, kind="function"):
    return {
        "name": name,
        "kind": kind,
        "file": file,
        "line": line,
        "end_line": end_line or line,
        "source": "doc",
    }


def _make_planner(completion_fn, **kw):
    defaults = dict(
        base_url="http://localhost:8084/v1",
        model="LFM2.5-8B-A1B-exl3-4.10bpw",
        budget_tokens=1500,
        max_symbols=6,
        completion_fn=completion_fn,
    )
    defaults.update(kw)
    return LspContextPlanner(**defaults)


# ---------------------------------------------------------------------------
# _extract_tokens
# ---------------------------------------------------------------------------


class TestExtractTokens:
    def test_keeps_identifiers_drops_stopwords_and_short(self):
        toks = LspContextPlanner._extract_tokens(
            "Add a cache to the Foo class and update bar_baz"
        )
        assert "Foo" in toks
        assert "bar_baz" in toks
        assert "cache" in toks
        # stopwords / too-short dropped
        assert "Add" not in toks  # 'add' is a stopword
        assert "a" not in toks
        assert "the" not in toks

    def test_dotted_name_uses_head(self):
        toks = LspContextPlanner._extract_tokens("Call os.path.join here")
        assert "os" in toks or "join" in toks

    def test_caps_at_twelve(self):
        toks = LspContextPlanner._extract_tokens(
            " ".join(f"thing{i}" for i in range(30))
        )
        assert len(toks) == 12


# ---------------------------------------------------------------------------
# _parse_picks
# ---------------------------------------------------------------------------


class TestParsePicks:
    def test_pick_object(self):
        assert _parse_picks('{"pick":[0,2,4]}', 5, 6) == [0, 2, 4]

    def test_bare_array(self):
        assert _parse_picks("[1, 3]", 5, 6) == [1, 3]

    def test_reasoning_then_json(self):
        text = 'Let me think... the relevant ones are {"pick":[0,1]}.'
        assert _parse_picks(text, 5, 6) == [0, 1]

    def test_drops_out_of_range_and_dedupes(self):
        assert _parse_picks('{"pick":[0,9,0,2]}', 3, 6) == [0, 2]

    def test_caps_at_limit(self):
        assert _parse_picks('{"pick":[0,1,2,3,4]}', 5, 2) == [0, 1]

    def test_empty_or_garbage(self):
        assert _parse_picks("", 5, 6) == []
        assert _parse_picks("I cannot decide", 5, 6) == []


# ---------------------------------------------------------------------------
# collect_code_context orchestration
# ---------------------------------------------------------------------------


class TestCollectCodeContext:
    def test_fetches_selected_symbols(self, tmp_path):
        candidates = [
            _candidate("alpha", "a.py", 10),
            _candidate("beta", "b.py", 3),
            _candidate("gamma", "c.py", 50),
        ]
        bodies = {
            (str(tmp_path / "a.py"), 10): "def alpha():\n    return 1",
            (str(tmp_path / "c.py"), 50): "def gamma():\n    pass",
        }
        lsp = FakeLsp(candidates, bodies)
        planner = _make_planner(lambda **kw: '{"pick":[0,2]}')

        out = planner.collect_code_context(
            "use alpha and gamma", lsp, [], str(tmp_path)
        )
        assert out is not None
        assert "a.py:10  alpha (function)" in out
        assert "def alpha" in out
        assert "gamma" in out
        # beta was not picked
        assert "beta" not in out

    def test_no_candidates_returns_none(self, tmp_path):
        lsp = FakeLsp([], {})
        planner = _make_planner(lambda **kw: '{"pick":[0]}')
        assert (
            planner.collect_code_context("do something", lsp, [], str(tmp_path)) is None
        )

    def test_completion_failure_returns_none(self, tmp_path):
        def boom(**kw):
            raise RuntimeError("server down")

        lsp = FakeLsp([_candidate("alpha", "a.py", 1)], {})
        planner = _make_planner(boom)
        assert planner.collect_code_context("use alpha", lsp, [], str(tmp_path)) is None

    def test_unparseable_response_returns_none(self, tmp_path):
        lsp = FakeLsp([_candidate("alpha", "a.py", 1)], {})
        planner = _make_planner(lambda **kw: "I am not sure what to pick")
        assert planner.collect_code_context("use alpha", lsp, [], str(tmp_path)) is None

    def test_budget_trims_output(self, tmp_path):
        candidates = [_candidate(f"f{i}", "a.py", i) for i in range(6)]
        bodies = {(str(tmp_path / "a.py"), i): "x" * 400 for i in range(6)}
        lsp = FakeLsp(candidates, bodies)
        # Tiny budget (~50 tokens => ~200 chars). Only the first block should fit.
        planner = _make_planner(lambda **kw: '{"pick":[0,1,2,3,4,5]}', budget_tokens=50)

        out = planner.collect_code_context("use them", lsp, [], str(tmp_path))
        assert out is not None
        assert "f0" in out
        # Not all six made it in under the budget.
        assert "f5" not in out

    def test_empty_query_returns_none(self, tmp_path):
        lsp = FakeLsp([_candidate("alpha", "a.py", 1)], {})
        planner = _make_planner(lambda **kw: '{"pick":[0]}')
        assert planner.collect_code_context("", lsp, [], str(tmp_path)) is None


# ---------------------------------------------------------------------------
# response_tokens config (Phase 7)
# ---------------------------------------------------------------------------


class TestResponseTokens:
    def test_default_is_512(self):
        planner = _make_planner(lambda **kw: '{"pick":[0]}')
        assert planner._response_tokens == 512

    def test_custom_value_stored(self):
        planner = _make_planner(lambda **kw: "{}", response_tokens=1024)
        assert planner._response_tokens == 1024

    def test_plan_passes_response_tokens_as_max_tokens(self, tmp_path):
        captured = {}

        def cap(**kw):
            captured.update(kw)
            return '{"pick":[0]}'

        lsp = FakeLsp([_candidate("alpha", "a.py", 1)], {})
        planner = _make_planner(cap, response_tokens=700)
        planner.collect_code_context("use alpha", lsp, [], str(tmp_path))
        assert captured.get("max_tokens") == 700


class TestBuildPlannerFromConfig:
    """_build_lsp_context_planner threads response_tokens from the TOML table."""

    def test_response_tokens_threaded_from_toml(self):
        import argparse

        from swival.agent import _build_lsp_context_planner

        args = argparse.Namespace()
        args._lsp_context_planner_toml = {
            "enabled": True,
            "base_url": "http://localhost:8083/v1",
            "model": "Qwopus3.6-27B-Coder-MTP-Q4_K_M.gguf",
            "response_tokens": 900,
        }
        args.verbose = False
        planner = _build_lsp_context_planner(args, "/tmp")
        assert planner is not None
        assert planner._response_tokens == 900

    def test_response_tokens_defaults_to_512_when_absent(self):
        import argparse

        from swival.agent import _build_lsp_context_planner

        args = argparse.Namespace()
        args._lsp_context_planner_toml = {
            "enabled": True,
            "base_url": "http://localhost:8083/v1",
            "model": "x",
        }
        args.verbose = False
        planner = _build_lsp_context_planner(args, "/tmp")
        assert planner is not None
        assert planner._response_tokens == 512
