"""Make Jnana's ProtoGnosis usable against ARGO (and other fenced-JSON backends).

Why this exists
---------------
Jnana's ``OpenAILLM.generate_with_json_output`` does a raw ``json.loads`` on the
model response. ARGO's OpenAI-compatible endpoint — even with
``response_format={"type": "json_object"}`` — returns the JSON wrapped in a
markdown code fence (```` ```json ... ``` ````). The raw parse then fails with
"Expecting value: line 1 column 1", the SupervisorAgent can't parse the research
goal, no generation tasks are queued, and ``wait_for_completion`` hangs forever.

This module fixes that WITHOUT editing Jnana. ``FencedJSONLLM`` wraps a real
Jnana ``LLMInterface`` and:
  * strips ``` / ```json fences (and common prose preambles) before JSON parsing;
  * repairs the ``(dict, prompt_tokens, completion_tokens)`` contract that the
    specialized agents expect, even when the underlying call raised on a fence.

``build_argo_coscientist`` constructs a CoScientist pointed at ARGO and swaps
every registered agent's ``.llm`` for the wrapper at runtime — configuration from
the adapter side, not a source patch.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

# Jnana imports are resolved at call time so importing this module never requires
# a Jnana checkout to be present.

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def strip_json_fence(text: str) -> str:
    """Return the most likely raw-JSON substring from a possibly-fenced reply."""
    if not text:
        return text
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # No fence: trim to the outermost JSON object/array if there's prose around it.
    start = min(
        [i for i in (text.find("{"), text.find("[")) if i != -1],
        default=-1,
    )
    if start > 0:
        return text[start:].strip()
    return text.strip()


def _coerce_json(text: str) -> Dict[str, Any]:
    """Parse JSON from a fenced/prose-wrapped string, raising ValueError on failure."""
    cleaned = strip_json_fence(text)

    def _try(s: str):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    # 1) straight parse
    out = _try(cleaned)
    if out is not None:
        return out

    # 2) some backends emit Python-dict-style single quotes / True/False/None.
    #    Try ast.literal_eval, which handles both quote styles and py literals.
    import ast
    try:
        val = ast.literal_eval(cleaned)
        if isinstance(val, dict):
            return val
    except (ValueError, SyntaxError):
        pass

    # 3) grab the first balanced {...} block and retry both parsers
    depth, s = 0, None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                s = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and s is not None:
                block = cleaned[s : i + 1]
                out = _try(block)
                if out is not None:
                    return out
                try:
                    val = ast.literal_eval(block)
                    if isinstance(val, dict):
                        return val
                except (ValueError, SyntaxError):
                    pass
                break

    raise ValueError(f"could not parse JSON from response: {text[:200]!r}")


class _DictTuple(dict):
    """A dict that also unpacks as (dict, prompt_tokens, completion_tokens).

    Jnana is internally inconsistent about generate_with_json_output's return
    shape: parse_research_goal treats it as a bare dict, while the specialized
    agents unpack it as a 3-tuple. This type satisfies BOTH without patching
    Jnana — indexing/attr access behave like the dict, iteration yields the
    3-tuple (dict, pt, ct).
    """

    def __new__(cls, data, prompt_tokens=0, completion_tokens=0):
        obj = super().__new__(cls, data)
        return obj

    def __init__(self, data, prompt_tokens=0, completion_tokens=0):
        super().__init__(data)
        self._triple = (dict(data), prompt_tokens, completion_tokens)

    def __iter__(self):
        # unpacking `a, b, c = result` uses iteration -> yield the triple
        return iter(self._triple)

    def __len__(self):
        return 3


def wrap_llm(inner: Any) -> Any:
    """Wrap an existing Jnana LLMInterface instance with fence-tolerant parsing."""
    cls = _get_fenced_llm_class()
    return cls(inner)


_FENCED_CLS = None


def _get_fenced_llm_class():
    """Build (once) FencedJSONLLM subclassing Jnana's LLMInterface (resolved lazily)."""
    global _FENCED_CLS
    if _FENCED_CLS is not None:
        return _FENCED_CLS
    from jnana.protognosis.core.llm_interface import LLMInterface  # type: ignore

    class _FencedJSONLLM(LLMInterface):
        """Delegates to a real Jnana LLM but tolerates markdown-fenced JSON."""

        _is_fenced_wrapper = True

        def __init__(self, inner: "LLMInterface"):
            # Mirror the inner LLM's identity so token accounting/logging still work.
            super().__init__(getattr(inner, "model", "wrapped"), getattr(inner, "model_adapter", None))
            self._inner = inner

        # plain text passes straight through
        def generate(self, prompt: str, system_prompt: Optional[str] = None,
                     temperature: float = 0.7, max_tokens: int = 1024) -> str:
            return self._inner.generate(
                prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens
            )

        def generate_with_json_output(self, prompt: str, json_schema: Dict,
                                      system_prompt: Optional[str] = None,
                                      temperature: float = 0.7,
                                      max_tokens: int = 1024) -> Tuple[Dict, int, int]:
            # Try the inner implementation first; if it trips on a fence, fall back
            # to a plain-text generate + tolerant parse. Always return a _DictTuple
            # so BOTH Jnana call-site shapes work (bare dict vs 3-tuple unpack).
            try:
                result = self._inner.generate_with_json_output(
                    prompt, json_schema, system_prompt=system_prompt,
                    temperature=temperature, max_tokens=max_tokens,
                )
                if isinstance(result, tuple):
                    data = result[0]
                    pt = result[1] if len(result) > 1 else 0
                    ct = result[2] if len(result) > 2 else 0
                    return _DictTuple(data, pt, ct)
                return _DictTuple(result, 0, 0)
            except Exception:
                schema_hint = (
                    "Return ONLY a JSON object matching this schema (no markdown, no code fences):\n"
                    f"{json_schema}"
                )
                sys_prompt = system_prompt or "You output only valid JSON."
                raw = self._inner.generate(
                    f"{prompt}\n\n{schema_hint}", system_prompt=sys_prompt,
                    temperature=temperature, max_tokens=max_tokens,
                )
                parsed = _coerce_json(raw)
                self.total_calls += 1
                return _DictTuple(parsed, 0, 0)

    _FENCED_CLS = _FencedJSONLLM
    return _FENCED_CLS


def _is_wrapped(obj: Any) -> bool:
    return getattr(obj, "_is_fenced_wrapper", False) is True


def build_argo_coscientist(
    model: str = "argo:gpt-4o",
    base_url: str = "http://localhost:63375/v1",
    api_key: str = "dummy",
    max_workers: int = 2,
    storage_path: Optional[str] = None,
):
    """Construct a CoScientist wired to ARGO with fence-tolerant LLM wrappers.

    Sets OPENAI_BASE_URL/OPENAI_API_KEY so Jnana's OpenAILLM (which builds an
    ``openai.OpenAI`` client) targets ARGO, constructs CoScientist, then swaps
    every registered agent's ``.llm`` for a FencedJSONLLM. Returns the ready
    CoScientist.
    """
    import os

    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ.setdefault("OPENAI_API_KEY", api_key)

    from jnana.protognosis.core.coscientist import CoScientist  # type: ignore
    from jnana.protognosis.core.multi_llm_config import LLMConfig  # type: ignore

    cfg = LLMConfig(provider="openai", model=model, api_key=api_key)
    cs = CoScientist(llm_config=cfg, max_workers=max_workers, storage_path=storage_path)

    _wrap_all_agent_llms(cs)
    return cs


def _wrap_all_agent_llms(cs: Any) -> int:
    """Replace each registered agent's .llm with a FencedJSONLLM. Returns count."""
    wrapped = 0
    supervisor = getattr(cs, "supervisor", None)
    agents = getattr(supervisor, "agents", None) if supervisor else None
    # SupervisorAgent stores agents in a dict {agent_id: agent} in this checkout.
    if isinstance(agents, dict):
        iterable = agents.values()
    elif agents is not None:
        iterable = agents
    else:
        iterable = []
    for agent in iterable:
        inner = getattr(agent, "llm", None)
        if inner is not None and not _is_wrapped(inner):
            agent.llm = wrap_llm(inner)
            wrapped += 1
    # also wrap the supervisor's own llm if present
    sup_llm = getattr(supervisor, "llm", None)
    if sup_llm is not None:
        supervisor.llm = wrap_llm(sup_llm)
        wrapped += 1
    return wrapped
