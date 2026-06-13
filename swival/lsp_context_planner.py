"""Phase B: small-LLM relevance planner for LSP-as-context.

When enabled via the ``[lsp_context_planner]`` config table, a small, fast
local LLM (served OpenAI-compatible — e.g. an EXL3 MoE on TabbyAPI) acts as a
*retrieval planner*. Given the user's task and a bounded menu of code symbols
gathered from the LSP servers, it selects the few symbols whose source the
agent most needs. That source is then read directly (no ``lsp_*`` tools) and
folded into the per-turn ``[lsp context]`` message alongside the Phase A
diagnostics.

Guarantees:

- The caller runs it **once per user turn** (not per inner loop iteration) —
  see the ``run_agent_loop`` seam.
- Best-effort throughout: server down / timeout / unparseable JSON → returns
  ``None``, leaving Phase A diagnostics untouched.
- A hard token budget caps the injected code; the planner is told the budget
  so it prioritizes.

The LLM call is injectable (``completion_fn``) so the selection logic is
unit-testable without a server.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .lsp_client import LspManager

# Identifier-like tokens (incl. dotted names) we may look up via workspace/symbol.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")

# Common words not worth an LSP workspace query. Kept small and conservative.
_STOPWORDS = frozenset(
    """
    a an the this that these those it its is are was were be been being
    do does did doing have has had having to of in on for with without
    and or not no if then else when while as at by from into over under
    you your we our my me i they them their he she his her
    add make run use get set put fix update change create delete remove
    file code test function method class variable name new now can will
    should would could please need want like just also some any all each
    what which who how why where there here about up out so than too very
    """.split()
)

_PLANNER_SYSTEM = (
    "You are a code-context retrieval planner. You are given a coding task and "
    "a numbered menu of code symbols found by a language server. Pick the few "
    "symbols whose source an engineer most needs to see to do the task. Prefer "
    "the definitions of things the task names, uses, or modifies. Skip symbols "
    "unlikely to matter. Respond with ONLY compact JSON."
)

CompletionFn = Callable[..., str]


def _default_completion(
    *,
    base_url: str,
    model: str,
    api_key: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    """Call an OpenAI-compatible server via litellm and return assistant text.

    Some local reasoning models (e.g. LFM2.5 MoE on EXL3) emit their answer in
    ``reasoning_content`` and leave ``content`` empty when token-starved, so we
    return both concatenated and let the caller regex out the JSON.
    """
    import litellm

    resp = litellm.completion(
        model=f"openai/{model}",
        api_base=base_url,
        api_key=api_key or "dummy",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    msg = resp.choices[0].message
    content = getattr(msg, "content", None) or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    return f"{content}\n{reasoning}"


def _parse_picks(text: str, n: int, cap: int) -> list[int]:
    """Extract a capped list of valid 0-based indices from planner output."""
    if not text:
        return []
    nums: list[int] = []
    # Prefer {"pick":[..]}; fall back to any bare integer array.
    obj = re.search(r'\{\s*"pick"\s*:\s*\[([0-9,\s]*)\]', text, re.S)
    if obj:
        nums = [int(x) for x in re.findall(r"\d+", obj.group(1))]
    else:
        arr = re.search(r"\[\s*([0-9,\s]+)\]", text, re.S)
        if arr:
            nums = [int(x) for x in re.findall(r"\d+", arr.group(1))]
    seen: set[int] = set()
    out: list[int] = []
    for x in nums:
        if 0 <= x < n and x not in seen:
            seen.add(x)
            out.append(x)
        if len(out) >= cap:
            break
    return out


class LspContextPlanner:
    """Selects relevant code to inject each turn using a small local LLM."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        budget_tokens: int = 1500,
        max_symbols: int = 6,
        timeout: float = 8.0,
        candidate_cap: int = 40,
        api_key: str | None = None,
        completion_fn: CompletionFn | None = None,
        verbose: bool = False,
    ):
        self._base_url = base_url
        self._model = model
        self._budget_tokens = budget_tokens
        self._max_symbols = max_symbols
        self._timeout = timeout
        self._candidate_cap = candidate_cap
        self._api_key = api_key
        self._completion = completion_fn or _default_completion
        self._verbose = verbose

    def collect_code_context(
        self,
        query: str,
        lsp_manager: "LspManager",
        dirty_paths: list[str],
        workspace_root: str,
    ) -> str | None:
        """Gather candidates, plan, fetch. Returns formatted code text or None."""
        if not query or lsp_manager is None:
            return None
        tokens = self._extract_tokens(query)
        candidates = lsp_manager.gather_symbol_candidates(
            tokens, dirty_paths, max_candidates=self._candidate_cap
        )
        if not candidates:
            return None
        picks = self._plan(query, candidates)
        if not picks:
            return None
        return self._fetch_and_format(picks, candidates, lsp_manager, workspace_root)

    @staticmethod
    def _extract_tokens(query: str) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for m in _TOKEN_RE.findall(query or ""):
            # Consider each segment of a dotted name (e.g. "os.path.join" ->
            # "path", "join"); the leading module name is usually too generic.
            for seg in m.split("."):
                key = seg.lower()
                if len(seg) < 3 or key in seen or key in _STOPWORDS:
                    continue
                seen.add(key)
                out.append(seg)
        return out[:12]

    def _plan(self, query: str, candidates: list[dict]) -> list[int]:
        menu = "\n".join(
            f"{i}. {c['name']} ({c['kind']}) — {c['file']}:{c['line']}"
            for i, c in enumerate(candidates)
        )
        user = (
            f"TASK:\n{query}\n\n"
            f"SYMBOL MENU:\n{menu}\n\n"
            f"Pick up to {self._max_symbols} indices (0-based) whose source the "
            f'agent most needs. Respond with ONLY JSON: {{"pick":[..]}}.'
        )
        try:
            text = self._completion(
                base_url=self._base_url,
                model=self._model,
                api_key=self._api_key or "dummy",
                system=_PLANNER_SYSTEM,
                user=user,
                max_tokens=256,
                temperature=0.0,
                timeout=self._timeout,
            )
        except Exception as e:
            if self._verbose:
                print(f"  LSP planner: call failed: {e}", flush=True)
            return []
        return _parse_picks(text, len(candidates), self._max_symbols)

    def _fetch_and_format(
        self,
        picks: list[int],
        candidates: list[dict],
        lsp_manager: "LspManager",
        workspace_root: str,
    ) -> str | None:
        root = Path(workspace_root)
        budget_chars = max(1, self._budget_tokens) * 4  # rough chars/token
        chunks: list[str] = []
        used = 0
        for idx in picks:
            c = candidates[idx]
            body = lsp_manager.fetch_symbol_body(
                root / c["file"], c["line"], c["end_line"]
            )
            if not body:
                continue
            block = f"// {c['file']}:{c['line']}  {c['name']} ({c['kind']})\n{body}"
            if chunks and used + len(block) > budget_chars:
                break
            chunks.append(block)
            used += len(block)
        if not chunks:
            return None
        return "Relevant code (selected by the LSP-context planner):\n\n" + "\n\n".join(
            chunks
        )
