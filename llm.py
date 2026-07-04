"""
llm.py — thin, deterministic wrapper around the Anthropic (Claude) API.

Used by BOTH generation.py (answering) and evaluate.py (LLM-as-judge for
groundedness/hallucination). Temperature is pinned to 0 so judging is
reproducible.

If no credentials are available (no ANTHROPIC_API_KEY), `LLM.available` is False
and callers fall back to their own deterministic OFFLINE heuristics — so
`python run.py` still works end-to-end without a key. Set a key to get real
Claude generation + judging.
"""
from __future__ import annotations

import os

import config

_FORCE_OFFLINE = os.environ.get("LLM_BACKEND", "").lower() == "offline"


class LLM:
    def __init__(self, model: str = config.ANTHROPIC_MODEL):
        self.model = model
        self.available = False
        self._client = None
        if _FORCE_OFFLINE or not os.environ.get("ANTHROPIC_API_KEY"):
            return
        try:
            import anthropic
            self._client = anthropic.Anthropic()
            self.available = True
        except Exception as e:  # pragma: no cover - defensive
            print(f"[llm] Anthropic client unavailable ({e}); using offline fallback.")

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        """Single deterministic completion. Only valid when self.available."""
        if not self.available:
            raise RuntimeError("LLM.complete called with no backend available")
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,  # deterministic; Haiku 4.5 accepts sampling params
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()


# Shared singleton so we don't reconstruct the client per call.
_LLM: LLM | None = None


def get_llm() -> LLM:
    global _LLM
    if _LLM is None:
        _LLM = LLM()
        print(f"[llm] backend = {'Anthropic:' + _LLM.model if _LLM.available else 'OFFLINE (heuristic)'}")
    return _LLM
