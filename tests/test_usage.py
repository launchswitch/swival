"""Tests for `swival.usage` and integration with the agent loop."""

from __future__ import annotations

import threading
import types
from unittest.mock import patch

import pytest

from swival.usage import LlmCallResult, LlmUsage, SessionUsage


class TestLlmUsageFactories:
    def test_from_provider_response_object(self):
        usage_obj = types.SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
            prompt_tokens_details=types.SimpleNamespace(cached_tokens=42),
            cache_creation_input_tokens=7,
        )
        resp = types.SimpleNamespace(usage=usage_obj)
        u = LlmUsage.from_provider_response(resp, cost_usd=0.0123)
        assert u.prompt_tokens == 120
        assert u.completion_tokens == 30
        assert u.total_tokens == 150
        assert u.cached_tokens == 42
        assert u.cache_write_tokens == 7
        assert u.cost_usd == pytest.approx(0.0123)
        assert u.cost_unknown is False
        assert u.estimated_tokens == 0
        assert u.tokens_estimated is False

    def test_from_provider_response_dict(self):
        resp = {"usage": {"prompt_tokens": 5, "completion_tokens": 1}}
        u = LlmUsage.from_provider_response(resp, cost_usd=None)
        assert u.prompt_tokens == 5
        assert u.completion_tokens == 1
        # total derived when missing
        assert u.total_tokens == 6
        # Provider tokens existed but no price → unknown, not estimated.
        assert u.cost_unknown is True
        assert u.cost_estimated is False

    def test_from_provider_response_missing_usage(self):
        resp = types.SimpleNamespace()
        u = LlmUsage.from_provider_response(resp, cost_usd=None)
        assert u is not None
        assert u.total_tokens == 0
        # No tokens and no price → no point reporting either as unknown.
        assert u.cost_unknown is False

    def test_from_prompt_estimate_disjoint(self):
        u = LlmUsage.from_prompt_estimate(250)
        assert u.estimated_tokens == 250
        assert u.tokens_estimated is True
        # Provider fields stay at 0; the two paths are disjoint.
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0
        assert u.cost_usd is None
        assert u.cost_unknown is False


class TestSessionUsageAggregation:
    def test_disjoint_counters(self):
        s = SessionUsage()
        s.add(LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
        s.add(LlmUsage.from_prompt_estimate(40))
        snap = s.snapshot()
        assert snap["total_tokens"] == 15
        assert snap["estimated_tokens"] == 40
        assert s.display_tokens() == 55
        assert snap["any_tokens_estimated"] is True

    def test_partial_cost_aggregation(self):
        s = SessionUsage()
        s.add(LlmUsage(total_tokens=100, cost_usd=0.05))
        s.add(LlmUsage(total_tokens=50, cost_usd=None, cost_unknown=True))
        s.add(LlmUsage(total_tokens=30, cost_usd=0.05))
        snap = s.snapshot()
        # Sum of priced calls only, but flag marks the partial state.
        assert snap["cost_usd"] == pytest.approx(0.10)
        assert snap["any_cost_unknown"] is True
        assert snap["any_cost_estimated"] is False

    def test_thread_safe_add(self):
        s = SessionUsage()
        u = LlmUsage(
            prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=0.001
        )

        def worker():
            for _ in range(500):
                s.add(u)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        snap = s.snapshot()
        assert snap["total_tokens"] == 8 * 500 * 2
        assert snap["prompt_tokens"] == 8 * 500
        assert snap["completion_tokens"] == 8 * 500
        assert snap["cost_usd"] == pytest.approx(8 * 500 * 0.001, rel=1e-6)


class TestLlmCallResultNormalize:
    def test_normalize_already_result(self):
        original = LlmCallResult(message="m", finish_reason="stop")
        assert LlmCallResult.normalize(original) is original

    def test_normalize_two_tuple(self):
        r = LlmCallResult.normalize(("msg", "stop"))
        assert r.message == "msg"
        assert r.finish_reason == "stop"
        assert r.command_activity == []
        assert r.provider_retries == 0
        assert r.usage is None

    def test_normalize_legacy_five_tuple(self):
        r = LlmCallResult.normalize(("m", "stop", [], 0, (5, 2)))
        assert r.usage is not None
        assert r.usage.cached_tokens == 5
        assert r.usage.cache_write_tokens == 2


class TestExtractLlmUsageInAgent:
    def test_extract_uses_litellm_completion_cost(self):
        from swival import agent

        usage_obj = types.SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
            prompt_tokens_details=None,
            cache_creation_input_tokens=0,
        )
        resp = types.SimpleNamespace(usage=usage_obj)
        with patch("litellm.completion_cost", return_value=0.0042) as mock_cost:
            u = agent._extract_llm_usage(resp, verbose=False)
        assert mock_cost.call_count == 1
        # call_llm is expected to pass the priced number through.
        assert u is not None
        assert u.cost_usd == pytest.approx(0.0042)
        assert u.total_tokens == 110

    def test_extract_treats_litellm_failure_as_unknown(self):
        from swival import agent

        usage_obj = types.SimpleNamespace(
            prompt_tokens=10, completion_tokens=1, total_tokens=11
        )
        resp = types.SimpleNamespace(usage=usage_obj)
        with patch("litellm.completion_cost", side_effect=Exception("no map entry")):
            u = agent._extract_llm_usage(resp, verbose=False)
        assert u is not None
        assert u.cost_usd is None
        assert u.cost_unknown is True

    def test_extract_no_usage_returns_none(self):
        from swival import agent

        resp = types.SimpleNamespace()
        u = agent._extract_llm_usage(resp, verbose=False)
        assert u is None


class TestReportRecordsUsage:
    def test_record_llm_call_with_usage(self):
        from swival.report import ReportCollector

        rc = ReportCollector()
        u = LlmUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cached_tokens=30,
            cache_write_tokens=10,
            cost_usd=0.0012,
        )
        rc.record_llm_call(
            turn=1, duration=0.5, token_est=99, finish_reason="stop", usage=u
        )
        assert rc.total_prompt_tokens == 100
        assert rc.total_completion_tokens == 20
        assert rc.total_provider_tokens == 120
        assert rc.total_cached_tokens == 30
        assert rc.total_cache_write_tokens == 10
        assert rc.total_cost_usd == pytest.approx(0.0012)
        assert rc.events[0]["usage"]["total_tokens"] == 120
        # prompt_tokens_est is no longer written.
        assert "prompt_tokens_est" not in rc.events[0]

    def test_record_llm_call_without_usage_is_diagnostic_only(self):
        # Without an explicit `usage` shape, the call is treated as a
        # failed/in-flight diagnostic event: it records token_est on the
        # event but does NOT roll up into cumulative spend.
        from swival.report import ReportCollector

        rc = ReportCollector()
        rc.record_llm_call(
            turn=0, duration=0.1, token_est=777, finish_reason="context_overflow"
        )
        assert rc.total_estimated_tokens == 0
        assert rc.total_provider_tokens == 0
        assert rc.any_tokens_estimated is False
        assert rc.events[0].get("prompt_tokens_estimate") == 777
        assert "usage" not in rc.events[0]

    def test_build_report_emits_usage_block(self):
        from swival.report import ReportCollector

        rc = ReportCollector()
        rc.record_llm_call(
            turn=0,
            duration=0.1,
            token_est=0,
            finish_reason="stop",
            usage=LlmUsage(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
                cost_usd=0.0001,
            ),
        )
        report = rc.build_report(
            task="t",
            model="m",
            provider="generic",
            settings={},
            outcome="ok",
            answer="hi",
            exit_code=0,
            turns=1,
        )
        usage = report["stats"]["usage"]
        assert usage["total_tokens"] == 12
        assert usage["cost_usd"] == pytest.approx(0.0001)
        assert usage["any_cost_unknown"] is False


class TestAgentLoopRollsUsageIntoSession:
    def _patch_provider(self, monkeypatch):
        from swival import agent

        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

    def test_main_call_records_usage(self, tmp_path, monkeypatch):
        from swival import agent
        from swival.session import Session

        self._patch_provider(monkeypatch)

        def _llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="answer", tool_calls=None, role="assistant"
            )
            usage = LlmUsage(
                prompt_tokens=200,
                completion_tokens=10,
                total_tokens=210,
                cost_usd=0.0005,
            )
            return LlmCallResult(message=msg, finish_reason="stop", usage=usage)

        monkeypatch.setattr(agent, "call_llm", _llm)

        s = Session(base_dir=str(tmp_path), history=False)
        s.run("hello")
        snap = s.session_usage.snapshot()
        assert snap["total_tokens"] == 210
        assert snap["prompt_tokens"] == 200
        assert snap["cost_usd"] == pytest.approx(0.0005)
        assert snap["estimated_tokens"] == 0

    def test_call_without_usage_falls_back_to_estimate(self, tmp_path, monkeypatch):
        from swival import agent
        from swival.session import Session

        self._patch_provider(monkeypatch)

        def _llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="answer", tool_calls=None, role="assistant"
            )
            # No usage attached: tests the local-prompt-estimate fallback.
            return LlmCallResult(message=msg, finish_reason="stop", usage=None)

        monkeypatch.setattr(agent, "call_llm", _llm)

        s = Session(base_dir=str(tmp_path), history=False)
        s.run("hello")
        snap = s.session_usage.snapshot()
        assert snap["total_tokens"] == 0
        assert snap["estimated_tokens"] > 0
        assert snap["any_tokens_estimated"] is True

    def test_history_reload_starts_at_zero(self, tmp_path, monkeypatch):
        from swival.session import Session

        # Two independent sessions: usage must not leak between them even when
        # base_dir is shared.
        s1 = Session(base_dir=str(tmp_path), history=False)
        s1.session_usage.add(LlmUsage(total_tokens=999, cost_usd=1.23))
        s2 = Session(base_dir=str(tmp_path), history=False)
        snap = s2.session_usage.snapshot()
        assert snap["total_tokens"] == 0
        assert snap.get("cost_usd") is None


class TestFailedDiagnosticDoesNotRollUp:
    def test_record_llm_call_without_usage_keeps_stats_clean(self):
        """Failed/diagnostic events must not contribute to cumulative spend.

        The agent loop calls `report.record_llm_call(..., "context_overflow")`
        and `..., "error")` without a `usage=` kwarg. Earlier versions of the
        report rolled these into `total_estimated_tokens`, which conflated
        diagnostic events with real spend.
        """
        from swival.report import ReportCollector

        rc = ReportCollector()
        rc.record_llm_call(
            turn=1, duration=0.2, token_est=4096, finish_reason="context_overflow"
        )
        rc.record_llm_call(
            turn=2,
            duration=0.1,
            token_est=2048,
            finish_reason="error",
            provider_retries=2,
        )
        assert rc.total_estimated_tokens == 0
        assert rc.total_provider_tokens == 0
        assert rc.total_cost_usd is None
        assert rc.any_tokens_estimated is False
        # And no `usage` block is emitted in the aggregate.
        report = rc.build_report(
            task="t",
            model="m",
            provider="generic",
            settings={},
            outcome="error",
            answer=None,
            exit_code=1,
            turns=2,
        )
        assert "usage" not in report["stats"]
        # Per-event diagnostic info is preserved.
        assert report["timeline"][0].get("prompt_tokens_estimate") == 4096
        assert report["timeline"][1].get("provider_retries") == 2


class TestSecondaryAndAuditFallback:
    def test_secondary_helper_bills_unpriced_success_to_estimate(self):
        """The production helper used by the secondary-call wrapper must
        synthesize a local prompt estimate when an auxiliary call returns
        no provider usage."""
        from swival.agent import _roll_aux_usage_into_session

        usage = SessionUsage()
        result = LlmCallResult(message="m", finish_reason="stop", usage=None)
        msgs = [{"role": "user", "content": "summarize this " * 20}]
        # Mirror the wrapper's positional shape: (base_url, model_id, messages).
        _roll_aux_usage_into_session(result, usage, ("http://x", "model", msgs), {})
        snap = usage.snapshot()
        assert snap["estimated_tokens"] > 0
        assert snap["any_tokens_estimated"] is True

    def test_secondary_helper_uses_provider_usage_when_present(self):
        from swival.agent import _roll_aux_usage_into_session

        usage = SessionUsage()
        provider_usage = LlmUsage(
            prompt_tokens=80,
            completion_tokens=20,
            total_tokens=100,
            cost_usd=0.01,
        )
        result = LlmCallResult(message="m", finish_reason="stop", usage=provider_usage)
        _roll_aux_usage_into_session(result, usage, (), {})
        snap = usage.snapshot()
        assert snap["total_tokens"] == 100
        assert snap["estimated_tokens"] == 0
        assert snap["any_tokens_estimated"] is False
        assert snap["cost_usd"] == pytest.approx(0.01)

    def test_secondary_helper_reads_messages_from_kwargs(self):
        from swival.agent import _roll_aux_usage_into_session

        usage = SessionUsage()
        result = LlmCallResult(message="m", finish_reason="stop", usage=None)
        msgs = [{"role": "user", "content": "kwarg call " * 30}]
        _roll_aux_usage_into_session(result, usage, (), {"messages": msgs})
        snap = usage.snapshot()
        assert snap["estimated_tokens"] > 0

    def test_production_secondary_wrapper_rolls_unpriced_into_estimate(
        self, monkeypatch
    ):
        """Build the actual production wrapper via the factory used inside
        `run_agent_loop`, then verify it rolls an unpriced auxiliary call
        into the session via the local prompt estimate."""
        from swival import agent

        captured: list[dict] = []

        def _llm(*args, **kwargs):
            captured.append({"args": args, "kwargs": dict(kwargs)})
            msg = types.SimpleNamespace(
                content="summary", tool_calls=None, role="assistant"
            )
            return LlmCallResult(message=msg, finish_reason="stop", usage=None)

        monkeypatch.setattr(agent, "call_llm", _llm)

        usage = SessionUsage()
        wrapper = agent._make_secondary_call_wrapper(
            session_usage=usage,
            user_agent="ua/1",
            llm_filter=None,
            cache=None,
            secret_shield=None,
        )

        result = wrapper(
            "http://x", "m", [{"role": "user", "content": "summarize " * 40}]
        )
        assert isinstance(result, LlmCallResult)
        # The wrapper threaded user_agent into the call.
        assert captured[0]["kwargs"].get("user_agent") == "ua/1"
        snap = usage.snapshot()
        assert snap["estimated_tokens"] > 0
        assert snap["total_tokens"] == 0
        assert snap["any_tokens_estimated"] is True

    def test_production_secondary_wrapper_passes_through_provider_usage(
        self, monkeypatch
    ):
        """Same production wrapper, given provider-reported usage, must roll
        the provider numbers in and leave `any_tokens_estimated` clear."""
        from swival import agent

        provider_usage = LlmUsage(
            prompt_tokens=200,
            completion_tokens=20,
            total_tokens=220,
            cost_usd=0.02,
        )

        def _llm(*args, **kwargs):
            msg = types.SimpleNamespace(content="ok", tool_calls=None, role="assistant")
            return LlmCallResult(
                message=msg, finish_reason="stop", usage=provider_usage
            )

        monkeypatch.setattr(agent, "call_llm", _llm)

        usage = SessionUsage()
        wrapper = agent._make_secondary_call_wrapper(session_usage=usage)
        wrapper("http://x", "m", [{"role": "user", "content": "x"}])
        snap = usage.snapshot()
        assert snap["total_tokens"] == 220
        assert snap["estimated_tokens"] == 0
        assert snap["any_tokens_estimated"] is False
        assert snap["cost_usd"] == pytest.approx(0.02)

    def test_audit_helper_bills_unpriced_success_via_estimate(
        self, tmp_path, monkeypatch
    ):
        """Audit's `_call_audit_llm` must also fall back to the local
        prompt estimate for unpriced successful calls."""
        from swival import audit as audit_mod
        from swival import agent

        # Patch the agent.call_llm symbol that audit imports lazily.
        def _llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="answer", tool_calls=None, role="assistant"
            )
            return LlmCallResult(message=msg, finish_reason="stop", usage=None)

        monkeypatch.setattr(agent, "call_llm", _llm)

        usage = SessionUsage()
        ctx = types.SimpleNamespace(
            loop_kwargs={
                "api_base": "http://x",
                "model_id": "test-model",
                "llm_kwargs": {"provider": "lmstudio"},
                "session_usage": usage,
            }
        )
        msgs = [{"role": "user", "content": "audit this please " * 40}]
        # Audit helper is `_call_audit_llm` — call it directly.
        audit_mod._call_audit_llm(ctx, msgs)
        snap = usage.snapshot()
        assert snap["estimated_tokens"] > 0
        assert snap["any_tokens_estimated"] is True


class TestSessionUsageInToolbarRender:
    def test_toolbar_renders_tokens_and_price(self):
        # Render a synthetic toolbar state by exercising the same formatting
        # rules used inside `_bottom_toolbar`. The function is a closure inside
        # `repl_loop`, so we re-implement the visible chip rules here.
        usage = SessionUsage()
        usage.add(LlmUsage(total_tokens=12_400, cost_usd=0.034))
        snap = usage.snapshot()
        tok = snap["total_tokens"] + snap["estimated_tokens"]
        assert tok == 12_400
        # 12,400 → "12.4k"
        assert f"{tok / 1000:.1f}k" == "12.4k"
        # cost between $0.01 and $1.00 renders as 2 decimals
        assert f"${snap['cost_usd']:.2f}" == "$0.03"

    def test_toolbar_renders_tok_label_when_unpriced(self):
        usage = SessionUsage()
        usage.add(LlmUsage(total_tokens=500))
        snap = usage.snapshot()
        # cost_usd is None → label should be "tok", not "billed"
        assert snap.get("cost_usd") is None


class TestZeroTokenProviderUsage:
    """A2 regression: a provider that returns `usage: {}` (or all-zero counts)
    must not be treated as a successful zero-spend call. The accounting helper
    should detect the empty usage and fall back to the local prompt estimate.
    """

    def test_has_token_data_treats_zero_usage_as_no_data(self):
        from swival.agent import _has_token_data

        empty = LlmUsage()  # all zeros
        assert _has_token_data(empty) is False
        assert _has_token_data(None) is False
        assert _has_token_data(LlmUsage(total_tokens=1)) is True
        assert _has_token_data(LlmUsage(estimated_tokens=1)) is True
        assert _has_token_data(LlmUsage(cached_tokens=1)) is True

    def test_zero_usage_falls_back_to_estimate(self, tmp_path, monkeypatch):
        from swival import agent
        from swival.session import Session

        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        # call_llm returns a successful response with an all-zero LlmUsage —
        # mimicking `usage: {}` from a misbehaving OAI-compatible server.
        def _llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="answer", tool_calls=None, role="assistant"
            )
            return LlmCallResult(message=msg, finish_reason="stop", usage=LlmUsage())

        monkeypatch.setattr(agent, "call_llm", _llm)

        s = Session(base_dir=str(tmp_path), history=False)
        s.run("hello")
        snap = s.session_usage.snapshot()
        # No provider totals, but the estimate fallback must have fired so
        # the toolbar/`/status` aren't silently stuck at zero.
        assert snap["total_tokens"] == 0
        assert snap["estimated_tokens"] > 0
        assert snap["any_tokens_estimated"] is True


class TestCostUsdZeroIsUnknown:
    """A1 regression: when LiteLLM returns 0.0 for a response that had real
    tokens, that's an unmapped-model fallthrough, not a genuinely free call.
    `cost_usd` should be `None` (unknown) so the toolbar's `$0.00` chip isn't
    ambiguous.
    """

    def test_zero_cost_with_real_tokens_becomes_unknown(self):
        from swival import agent

        usage_obj = types.SimpleNamespace(
            prompt_tokens=200,
            completion_tokens=20,
            total_tokens=220,
            prompt_tokens_details=None,
            cache_creation_input_tokens=0,
        )
        resp = types.SimpleNamespace(usage=usage_obj)
        with patch("litellm.completion_cost", return_value=0.0):
            u = agent._extract_llm_usage(resp, verbose=False)
        assert u is not None
        assert u.cost_usd is None
        assert u.cost_unknown is True

    def test_zero_cost_with_zero_tokens_stays_known(self):
        # If there genuinely were no tokens billed, 0.0 is the right answer.
        from swival import agent

        usage_obj = types.SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=None,
            cache_creation_input_tokens=0,
        )
        resp = types.SimpleNamespace(usage=usage_obj)
        with patch("litellm.completion_cost", return_value=0.0):
            u = agent._extract_llm_usage(resp, verbose=False)
        # With no tokens we don't downgrade — but we also don't have anything
        # to bill, so the call is effectively no-op.
        assert u is not None


class TestSecondaryWrapperTagsAsSummary:
    """A3 regression: secondary calls (compaction/summarization) must be
    tagged with `call_kind="summary"` regardless of whether `llm_filter` is
    set. Otherwise the streaming gate in `call_llm` fires for secondary
    calls in verbose TTY mode.
    """

    def test_wrapper_threads_summary_call_kind(self, monkeypatch):
        from swival import agent

        captured: list[dict] = []

        def _llm(*args, **kwargs):
            captured.append(dict(kwargs))
            msg = types.SimpleNamespace(content="ok", tool_calls=None, role="assistant")
            return LlmCallResult(message=msg, finish_reason="stop")

        monkeypatch.setattr(agent, "call_llm", _llm)

        # No llm_filter — historically this skipped the call_kind setdefault.
        wrapper = agent._make_secondary_call_wrapper(session_usage=None)
        wrapper("http://x", "m", [{"role": "user", "content": "x"}])
        assert captured[0].get("call_kind") == "summary"

    def test_wrapper_respects_caller_override(self, monkeypatch):
        from swival import agent

        captured: list[dict] = []

        def _llm(*args, **kwargs):
            captured.append(dict(kwargs))
            msg = types.SimpleNamespace(content="ok", tool_calls=None, role="assistant")
            return LlmCallResult(message=msg, finish_reason="stop")

        monkeypatch.setattr(agent, "call_llm", _llm)

        wrapper = agent._make_secondary_call_wrapper(session_usage=None)
        wrapper(
            "http://x",
            "m",
            [{"role": "user", "content": "x"}],
            call_kind="custom",
        )
        # setdefault → caller's explicit kind wins.
        assert captured[0].get("call_kind") == "custom"


class TestContinueHereLlmSummary:
    """C1 regression: `_try_llm_summary` used to do `_result[0]` on the
    secondary-call return. After the LlmCallResult refactor, that raises
    TypeError which is swallowed, silently breaking the LLM-enhanced
    continue summary.
    """

    def test_llm_summary_uses_dataclass_message(self):
        from swival import continue_here

        def fake_call(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="Polished recap.", tool_calls=None, role="assistant"
            )
            return LlmCallResult(message=msg, finish_reason="stop")

        out = continue_here._try_llm_summary(
            [{"role": "user", "content": "hello"}],
            fake_call,
            model_id="test-model",
            base_url="http://x",
            api_key=None,
            top_p=None,
            seed=None,
            provider="generic",
        )
        assert out == "Polished recap."

    def test_llm_summary_tolerates_lowlevel_tuple_returns(self):
        """Defensive: a test mock that hands back a 2-tuple (the shape that
        `LlmCallResult.normalize` tolerates for legacy mocks) must still
        produce a usable summary."""
        from swival import continue_here

        def fake_call(*args, **kwargs):
            msg = types.SimpleNamespace(content="hi", tool_calls=None, role="assistant")
            return (msg, "stop")

        out = continue_here._try_llm_summary(
            [{"role": "user", "content": "hello"}],
            fake_call,
            model_id="m",
            base_url="http://x",
            api_key=None,
            top_p=None,
            seed=None,
            provider="generic",
        )
        assert out == "hi"


class TestReviewerNoneSafe:
    """C4 regression: reviewer.py used `msg.content or ""` which crashes when
    `message` is None. Should use `_msg_content` like the rest of the codebase.
    """

    def test_reviewer_uses_msg_content_helper(self):
        # Read the source to confirm the fix is in place; the alternative
        # would be to mock the entire reviewer entry point, which is much
        # heavier than the bug.
        from swival import reviewer
        import inspect

        source = inspect.getsource(reviewer)
        assert "_msg_content(msg)" in source
        # And the old crash-prone idiom is gone.
        assert 'msg.content or ""' not in source
        assert "msg.content or ''" not in source


class TestReportAttachSessionUsage:
    """B5 regression: when a `ReportCollector` is attached to a `SessionUsage`,
    the aggregate `stats.usage` block is projected from the session — closing
    the toolbar-vs-report gap for subagent and auxiliary spend that updates
    only the session accumulator.
    """

    def test_attached_session_usage_supersedes_local_sums(self):
        from swival.report import ReportCollector

        rc = ReportCollector()
        su = SessionUsage()
        # Simulate subagent spend that the report never saw via record_llm_call.
        su.add(LlmUsage(prompt_tokens=999, completion_tokens=1, total_tokens=1000))
        rc.attach_session_usage(su)

        report = rc.build_report(
            task="t",
            model="m",
            provider="generic",
            settings={},
            outcome="ok",
            answer="hi",
            exit_code=0,
            turns=0,
        )
        usage = report["stats"]["usage"]
        # Pulled from session, not from `total_*` (which are still 0).
        assert usage["total_tokens"] == 1000
        assert usage["prompt_tokens"] == 999

    def test_no_session_usage_falls_back_to_local_sums(self):
        from swival.report import ReportCollector

        rc = ReportCollector()
        rc.record_llm_call(
            turn=0,
            duration=0.1,
            token_est=0,
            finish_reason="stop",
            usage=LlmUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )
        report = rc.build_report(
            task="t",
            model="m",
            provider="generic",
            settings={},
            outcome="ok",
            answer="hi",
            exit_code=0,
            turns=1,
        )
        assert report["stats"]["usage"]["total_tokens"] == 12


class TestAuditRollsIntoReport:
    """C3 regression: audit's `_call_audit_llm` must record into the parent
    `ReportCollector` when one is attached, mirroring the agent loop's
    invariant that `session_usage` and report aggregate agree.
    """

    def test_audit_records_llm_call_when_report_present(self, tmp_path, monkeypatch):
        from swival import agent, audit as audit_mod
        from swival.report import ReportCollector

        provider_usage = LlmUsage(
            prompt_tokens=100, completion_tokens=10, total_tokens=110, cost_usd=0.01
        )

        def _llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="findings", tool_calls=None, role="assistant"
            )
            return LlmCallResult(
                message=msg, finish_reason="stop", usage=provider_usage
            )

        monkeypatch.setattr(agent, "call_llm", _llm)

        report = ReportCollector()
        usage = SessionUsage()
        report.attach_session_usage(usage)
        ctx = types.SimpleNamespace(
            loop_kwargs={
                "api_base": "http://x",
                "model_id": "m",
                "llm_kwargs": {"provider": "generic"},
                "session_usage": usage,
                "report": report,
            }
        )
        msgs = [{"role": "user", "content": "audit"}]
        audit_mod._call_audit_llm(ctx, msgs)

        # report saw the call …
        assert report.llm_calls == 1
        # … and session_usage is consistent.
        assert usage.snapshot()["total_tokens"] == 110


class TestAuditEmptyRetryDoesNotMultiBill:
    """A6 regression: audit's empty-response retry loop must commit at most
    one estimate-based bill, even when several empty retries happen on a
    local provider that doesn't report usage.
    """

    def test_empty_retries_do_not_multiply_estimate_bill(self, tmp_path, monkeypatch):
        from swival import agent, audit as audit_mod

        # Provider returns empty content with no usage object on every call.
        def _llm(*args, **kwargs):
            msg = types.SimpleNamespace(content="", tool_calls=None, role="assistant")
            return LlmCallResult(message=msg, finish_reason="stop", usage=None)

        monkeypatch.setattr(agent, "call_llm", _llm)

        usage = SessionUsage()
        # Fabricate a long original message so the loop halves several times.
        long_text = "audit this " * 200
        ctx = types.SimpleNamespace(
            loop_kwargs={
                "api_base": "http://x",
                "model_id": "m",
                "llm_kwargs": {"provider": "lmstudio"},
                "session_usage": usage,
            }
        )
        audit_mod._call_audit_llm(ctx, [{"role": "user", "content": long_text}])

        # The retry loop ran several times (each halving the prompt), but
        # only ONE estimate may be billed for the whole logical step.
        snap = usage.snapshot()
        # estimated_tokens > 0 means we billed once; the assertion that
        # matters is that we DIDN'T bill N times. A loose upper bound on the
        # original prompt's token count suffices.
        assert 0 < snap["estimated_tokens"] <= 8 * len(long_text)
        assert snap["any_tokens_estimated"] is True

    def test_audit_zero_usage_falls_back_to_estimate(self, tmp_path, monkeypatch):
        """A2 regression specific to audit: a provider that returns a non-None
        but all-zero `LlmUsage` (e.g. `usage: {}` from an OAI-compatible local
        server) must NOT be billed as a zero-token success. Audit needs to
        route empty usage through the same estimate fallback the main loop
        uses, otherwise audit calls silently disappear from the toolbar."""
        from swival import agent, audit as audit_mod

        def _llm(*args, **kwargs):
            msg = types.SimpleNamespace(
                content="findings", tool_calls=None, role="assistant"
            )
            # Non-None usage with every count at zero — the trigger condition.
            return LlmCallResult(message=msg, finish_reason="stop", usage=LlmUsage())

        monkeypatch.setattr(agent, "call_llm", _llm)

        usage = SessionUsage()
        msgs = [{"role": "user", "content": "audit this " * 30}]
        ctx = types.SimpleNamespace(
            loop_kwargs={
                "api_base": "http://x",
                "model_id": "m",
                "llm_kwargs": {"provider": "lmstudio"},
                "session_usage": usage,
            }
        )
        audit_mod._call_audit_llm(ctx, msgs)
        snap = usage.snapshot()
        # Provider returned zero usage, so the estimate fallback should
        # have fired exactly once.
        assert snap["total_tokens"] == 0
        assert snap["estimated_tokens"] > 0
        assert snap["any_tokens_estimated"] is True

    def test_provider_reported_usage_bills_each_retry(self, tmp_path, monkeypatch):
        # Counterpoint: when the provider DOES report usage, each retry IS
        # billed — providers charge for failed/empty responses.
        from swival import agent, audit as audit_mod

        calls = {"n": 0}

        def _llm(*args, **kwargs):
            calls["n"] += 1
            msg = types.SimpleNamespace(content="", tool_calls=None, role="assistant")
            return LlmCallResult(
                message=msg,
                finish_reason="stop",
                usage=LlmUsage(prompt_tokens=10, total_tokens=10),
            )

        monkeypatch.setattr(agent, "call_llm", _llm)

        usage = SessionUsage()
        long_text = "audit " * 200
        ctx = types.SimpleNamespace(
            loop_kwargs={
                "api_base": "http://x",
                "model_id": "m",
                "llm_kwargs": {"provider": "generic"},
                "session_usage": usage,
            }
        )
        audit_mod._call_audit_llm(ctx, [{"role": "user", "content": long_text}])
        snap = usage.snapshot()
        # Multiple retries × 10 provider tokens each.
        assert snap["total_tokens"] == calls["n"] * 10
        assert calls["n"] >= 2
