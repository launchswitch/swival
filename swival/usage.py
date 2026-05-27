"""Runtime session usage and pricing accounting.

`LlmUsage` captures one successful LLM call. `SessionUsage` accumulates calls
across an interactive session and is thread-safe so subagents running in
worker threads can roll their spend into the parent.

The accumulator owns a `threading.Lock`; never serialize `SessionUsage`
directly (e.g. with `dataclasses.asdict`). Project fields explicitly when
emitting them to JSON or the report block.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class LlmUsage:
    """Usage for one successful LLM call.

    Provider-reported usage and fallback estimates are disjoint. Provider
    responses populate prompt/completion/total tokens and leave
    estimated_tokens at zero. Fallback estimates populate estimated_tokens
    only and leave provider token fields at zero. Do not populate both for
    the same call; SessionUsage relies on the disjoint invariant to display
    a single additive total.

    cost_usd holds the priced portion of this call when known. cost_unknown
    means the call had token usage but no LiteLLM price. cost_estimated is
    reserved for an explicit future price-estimation fallback; unknown is
    not estimated.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    estimated_tokens: int = 0
    tokens_estimated: bool = False
    cost_usd: float | None = None
    cost_unknown: bool = False
    cost_estimated: bool = False

    @classmethod
    def from_provider_response(cls, response, *, cost_usd: float | None) -> "LlmUsage":
        usage = _get(response, "usage")
        if usage is None:
            # No provider usage at all: don't mark cost_unknown — that flag
            # is for "had tokens but no price". With nothing to spend on,
            # the silent state is the honest one.
            return cls(cost_usd=cost_usd)
        prompt = int(_get(usage, "prompt_tokens", 0) or 0)
        completion = int(_get(usage, "completion_tokens", 0) or 0)
        total = int(_get(usage, "total_tokens", 0) or 0)
        if total == 0 and (prompt or completion):
            total = prompt + completion
        details = _get(usage, "prompt_tokens_details")
        cached = int(_get(details, "cached_tokens", 0) or 0) if details else 0
        cache_write = int(_get(usage, "cache_creation_input_tokens", 0) or 0)
        has_tokens = total > 0 or prompt > 0 or completion > 0
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cached_tokens=cached,
            cache_write_tokens=cache_write,
            cost_usd=cost_usd,
            cost_unknown=has_tokens and cost_usd is None,
        )

    @classmethod
    def from_prompt_estimate(cls, prompt_tokens: int) -> "LlmUsage":
        n = max(0, int(prompt_tokens or 0))
        return cls(
            estimated_tokens=n,
            tokens_estimated=n > 0,
            cost_unknown=False,
        )


@dataclass
class SessionUsage:
    """Thread-safe cumulative session usage.

    total_tokens is the sum of provider-reported totals only.
    estimated_tokens is the sum of fallback prompt estimates only. They are
    disjoint by construction in `LlmUsage`, so the display token count is
    their sum.

    Never serialize this object directly; it owns a `threading.Lock` that
    is not JSON-serializable. Project fields explicitly when writing
    reports.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    any_tokens_estimated: bool = False
    any_cost_unknown: bool = False
    any_cost_estimated: bool = False
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def add(self, usage: LlmUsage | None) -> None:
        if usage is None:
            return
        with self._lock:
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens
            self.estimated_tokens += usage.estimated_tokens
            self.cached_tokens += usage.cached_tokens
            self.cache_write_tokens += usage.cache_write_tokens
            if usage.cost_usd is not None:
                self.cost_usd = (self.cost_usd or 0.0) + float(usage.cost_usd)
            if usage.tokens_estimated:
                self.any_tokens_estimated = True
            if usage.cost_unknown:
                self.any_cost_unknown = True
            if usage.cost_estimated:
                self.any_cost_estimated = True

    def display_tokens(self) -> int:
        with self._lock:
            return self.total_tokens + self.estimated_tokens

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.estimated_tokens = 0
            self.cached_tokens = 0
            self.cache_write_tokens = 0
            self.cost_usd = None
            self.any_tokens_estimated = False
            self.any_cost_unknown = False
            self.any_cost_estimated = False

    def snapshot(self) -> dict:
        """Project current state to a plain dict suitable for JSON."""
        with self._lock:
            out = {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "estimated_tokens": self.estimated_tokens,
                "cached_tokens": self.cached_tokens,
                "cache_write_tokens": self.cache_write_tokens,
                "any_tokens_estimated": self.any_tokens_estimated,
                "any_cost_unknown": self.any_cost_unknown,
                "any_cost_estimated": self.any_cost_estimated,
            }
            if self.cost_usd is not None:
                out["cost_usd"] = round(float(self.cost_usd), 6)
            return out


@dataclass
class LlmCallResult:
    """Named result for a successful LLM call.

    Replaces the historical 5-tuple `(message, finish_reason, command_activity,
    provider_retries, cache_stats)`. Every successful return path from
    `call_llm()` produces an `LlmCallResult`; `usage` is `None` only when the
    caller cannot supply token usage (e.g. command provider with no metering).
    """

    message: Any
    finish_reason: str
    command_activity: list[dict] = field(default_factory=list)
    provider_retries: int = 0
    usage: LlmUsage | None = None

    def __iter__(self):
        # Yields the four positional fields, mirroring the historical tuple
        # order. Lets tests still write `msg, *_ = call_llm(...)` without
        # forcing every test to switch to attribute access. The usage field
        # is intentionally not yielded — callers that need it must read
        # `.usage` so they don't confuse it with the old cache-stats tuple.
        yield self.message
        yield self.finish_reason
        yield self.command_activity
        yield self.provider_retries

    @classmethod
    def normalize(cls, value) -> "LlmCallResult":
        """Coerce a return value into an `LlmCallResult`.

        Accepts `LlmCallResult`, 2/3/4/5-tuples (legacy/test shapes), or
        falls through for the dataclass case. Test mocks frequently return
        `(message, finish_reason)`; tolerating those keeps the call-site
        contract simple without polluting production code with tuple checks.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, tuple):
            msg = value[0] if len(value) > 0 else None
            finish_reason = value[1] if len(value) > 1 else "stop"
            command_activity = value[2] if len(value) > 2 else []
            provider_retries = value[3] if len(value) > 3 else 0
            usage: LlmUsage | None = None
            if len(value) > 4:
                cs = value[4]
                if isinstance(cs, LlmUsage):
                    usage = cs
                elif isinstance(cs, tuple) and len(cs) >= 2:
                    cached, written = int(cs[0] or 0), int(cs[1] or 0)
                    if cached or written:
                        usage = LlmUsage(
                            cached_tokens=cached,
                            cache_write_tokens=written,
                        )
            return cls(
                message=msg,
                finish_reason=finish_reason,
                command_activity=list(command_activity or []),
                provider_retries=int(provider_retries or 0),
                usage=usage,
            )
        # Unknown shape (e.g. a bare message namespace from a sloppy mock).
        return cls(message=value, finish_reason="stop")
