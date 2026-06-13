"""Tests for reduce_recent_large_file_tool_results (Phase 5).

Targeted compaction of oversized recent file-read results that survive general
compaction (which preserves the last turns verbatim).
"""

from __future__ import annotations

import json
import types

from swival import agent
from swival.agent import reduce_recent_large_file_tool_results
from swival.tools import TOOLS


def _ns_msg(content, tool_calls=None):
    """Build a provider-like message (SimpleNamespace) the loop accepts."""
    msg = types.SimpleNamespace(
        content=content, tool_calls=tool_calls, role="assistant"
    )
    msg.get = lambda key, default=None: getattr(msg, key, default)
    return msg


def _ns_tc(tc_id, name, args):
    return types.SimpleNamespace(
        id=tc_id,
        function=types.SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _tc(tc_id: str, name: str, args: dict | str) -> dict:
    if not isinstance(args, str):
        args = json.dumps(args)
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ],
    }


def _tool(tc_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


# ---------------------------------------------------------------------------
# Unit tests for the helper
# ---------------------------------------------------------------------------


class TestReduceRecentLargeFileResults:
    def test_large_recent_read_file_compacted(self):
        big = "line\n" * 2000  # well over the 6000-char threshold
        messages = [_tc("t1", "read_file", {"file_path": "big.py"}), _tool("t1", big)]
        n = reduce_recent_large_file_tool_results(messages)
        assert n == 1
        tool_msg = messages[1]
        assert tool_msg["content"].startswith("[read_file:")
        assert len(tool_msg["content"]) < len(big)
        # Pairing preserved: only content changed, not the tool_call_id link.
        assert tool_msg["tool_call_id"] == "t1"

    def test_large_read_multiple_files_compacted(self):
        big = "y" * 8000
        messages = [
            _tc("t1", "read_multiple_files", {"files": ["a.py", "b.py"]}),
            _tool("t1", big),
        ]
        assert reduce_recent_large_file_tool_results(messages) == 1
        assert messages[1]["content"].startswith("[read_multiple_files:")

    def test_small_read_unchanged(self):
        small = "only a few lines\n"
        messages = [_tc("t1", "read_file", {"file_path": "s.py"}), _tool("t1", small)]
        assert reduce_recent_large_file_tool_results(messages) == 0
        assert messages[1]["content"] == small

    def test_at_threshold_not_reduced(self):
        content = "a" * 6000  # exactly the threshold: not reduced
        messages = [_tc("t1", "read_file", {"file_path": "f.py"}), _tool("t1", content)]
        assert reduce_recent_large_file_tool_results(messages) == 0
        assert messages[1]["content"] == content

    def test_non_file_tool_result_untouched(self):
        big = "x" * 10000
        messages = [_tc("t1", "run_command", {"command": "ls"}), _tool("t1", big)]
        assert reduce_recent_large_file_tool_results(messages) == 0
        assert messages[1]["content"] == big

    def test_only_recent_turns_targeted(self):
        big = "z" * 8000
        # Three consecutive tool turns; default keep_tail_turns=2 inspects only
        # the last two. The first is left for general compaction to handle.
        messages = []
        for i in range(3):
            messages.append(_tc(f"t{i}", "read_file", {"file_path": "f.py"}))
            messages.append(_tool(f"t{i}", big))
        assert reduce_recent_large_file_tool_results(messages) == 2
        # First turn's result untouched by this pass.
        assert messages[1]["content"] == big

    def test_empty_messages(self):
        assert reduce_recent_large_file_tool_results([]) == 0


# ---------------------------------------------------------------------------
# Loop integration: the reducer fires before the next LLM call
# ---------------------------------------------------------------------------


def _loop_kwargs(tmp_path, **overrides):
    from swival.thinking import ThinkingState
    from swival.todo import TodoState

    defaults = dict(
        api_base="http://localhost",
        model_id="test-model",
        max_turns=4,
        max_output_tokens=1024,
        temperature=0.0,
        top_p=None,
        seed=None,
        context_length=None,
        base_dir=str(tmp_path),
        thinking_state=ThinkingState(),
        todo_state=TodoState(),
        resolved_commands={},
        skills_catalog={},
        skill_read_roots=[],
        extra_write_roots=[],
        files_mode="all",
        verbose=False,
        llm_kwargs={},
    )
    defaults.update(overrides)
    return defaults


class TestLoopIntegration:
    def test_large_read_compacted_before_next_call(self, tmp_path, monkeypatch):
        big = tmp_path / "big.py"
        # Many short lines -> read_file returns >6000 chars even after the
        # default line limit, so the reducer targets it.
        big.write_text("line\n" * 4000)

        state = {"calls": 0, "captured": None}

        def llm(*args, **kwargs):
            state["calls"] += 1
            msgs = args[2] if len(args) > 2 else kwargs.get("messages") or []
            if state["calls"] == 1:
                tc = _ns_tc("c1", "read_file", {"file_path": str(big)})
                return _ns_msg(None, [tc]), "tool_calls"
            state["captured"] = list(msgs)
            return _ns_msg("done", None), "stop"

        monkeypatch.setattr(agent, "call_llm", llm)
        monkeypatch.setattr(agent, "discover_model", lambda *a: ("test-model", None))

        agent.run_agent_loop(
            [{"role": "user", "content": "read big.py"}],
            TOOLS,
            **_loop_kwargs(tmp_path),
        )

        assert state["captured"] is not None, "LLM was never called a second time"
        tool_msgs = [
            m
            for m in state["captured"]
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        assert tool_msgs, "tool result was missing from the captured messages"
        # The oversized read was compacted before this (second) LLM call.
        assert tool_msgs[-1]["content"].startswith("[read_file:")
