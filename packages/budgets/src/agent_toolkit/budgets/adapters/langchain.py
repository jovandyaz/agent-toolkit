"""LangChain callback handler that charges a run budget and stops it when a limit is exceeded.

Install with the `langchain` extra. Attach it per run, for example
`model.invoke(prompt, config={"callbacks": [BudgetCallbackHandler(limits, pricing)]})`; one
handler instance holds the state of one run.

Accounting:
- one iteration per LLM call, charged in `on_llm_end`, with tokens summed from the
  `usage_metadata` of every generated message (none reported means no tokens);
- one tool call per tool START, so a call that would exceed `max_tool_calls` raises before the
  tool runs; a tool error counts a failure and a successful end resets the streak.
After each update, exceeded limits raise `BudgetExceededError`; `raise_error = True` makes
LangChain propagate it instead of logging it. Failed LLM calls (`on_llm_error`) are not charged.
"""

from threading import Lock
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import ChatGeneration, LLMResult

from agent_toolkit.budgets.core import (
    BudgetExceededError,
    BudgetLimits,
    BudgetState,
    Pricing,
    Usage,
    charge,
    check,
)

__all__ = ["BudgetCallbackHandler", "BudgetExceededError"]


class BudgetCallbackHandler(BaseCallbackHandler):
    """Charges LLM and tool activity to a `BudgetState`; thread-safe."""

    raise_error = True

    def __init__(self, limits: BudgetLimits, pricing: Pricing | None) -> None:
        self._limits = limits
        self._pricing = pricing
        self._state = BudgetState.empty()
        self._lock = Lock()

    @property
    def state(self) -> BudgetState:
        """Totals consumed so far."""
        with self._lock:
            return self._state

    def _charge(self, usage: Usage) -> None:
        with self._lock:
            self._state = charge(self._state, usage, self._pricing)
            exceeded = check(self._state, self._limits)
        if exceeded:
            raise BudgetExceededError(exceeded)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        input_tokens = output_tokens = 0
        for generations in response.generations:
            for generation in generations:
                if isinstance(generation, ChatGeneration):
                    meta = getattr(generation.message, "usage_metadata", None)
                    if meta:
                        input_tokens += meta["input_tokens"]
                        output_tokens += meta["output_tokens"]
        self._charge(Usage(iterations=1, input_tokens=input_tokens, output_tokens=output_tokens))

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._charge(Usage(tool_calls=1))

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._charge(Usage(failed=False))

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._charge(Usage(failed=True))
