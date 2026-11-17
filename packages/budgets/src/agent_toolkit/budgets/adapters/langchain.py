"""LangChain callback handler that charges a run budget and stops it when a limit is exceeded.

Install with the `langchain` extra. Attach it per run, for example
`model.invoke(prompt, config={"callbacks": [BudgetCallbackHandler(limits, pricing)]})`; one
handler instance holds the state of one run.

Accounting:
- one iteration per LLM call, charged at call START (`on_chat_model_start` / `on_llm_start`),
  so the call that would exceed `max_iterations` raises before the model is invoked;
- tokens and cost are charged in `on_llm_end`, summed from the `usage_metadata` of every
  generated message. A model that reports none (for example streaming without `stream_usage`)
  cannot trip the token or cost limits;
- one tool call per tool START, so a call over `max_tool_calls` raises before the tool runs.
  A tool failure is a raised error (`on_tool_error`) or a handled one, which LangChain reports
  as `on_tool_end` with a `ToolMessage(status="error")` (only when the tool is invoked with a
  tool call, as agent tool nodes do; a bare input yields plain content). Both count a
  failure; any other successful end resets the streak.
After each update, exceeded limits raise `BudgetExceededError`; `raise_error = True` makes
LangChain propagate it instead of logging it. The handler is sticky: once a limit is exceeded
it raises again on every later callback (LLM, chain and tool starts and ends), so a caller that
swallows the first error cannot keep running the agent. Failed LLM calls (`on_llm_error`) are
not charged. With a `max_cost_usd` limit, `pricing` is required (ValueError otherwise).
"""

from threading import Lock
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agent_toolkit.budgets.core import (
    BudgetExceededError,
    BudgetLimits,
    BudgetState,
    ExceededKind,
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
        if limits.max_cost_usd is not None and pricing is None:
            raise ValueError("max_cost_usd needs pricing: without it cost never accumulates")
        self._limits = limits
        self._tripped: tuple[ExceededKind, ...] = ()
        self._pricing = pricing
        self._state = BudgetState.empty()
        self._lock = Lock()

    @property
    def state(self) -> BudgetState:
        """Totals consumed so far."""
        with self._lock:
            return self._state

    def _charge(self, usage: Usage, cause: BaseException | None = None) -> None:
        with self._lock:
            self._state = charge(self._state, usage, self._pricing)
            exceeded = check(self._state, self._limits)
            if exceeded:
                self._tripped = exceeded
            tripped = self._tripped
        if tripped:
            raise BudgetExceededError(tripped) from cause

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **kwargs: Any
    ) -> None:
        self._charge(Usage(iterations=1))

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        self._charge(Usage(iterations=1))

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        self._charge(Usage())

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        input_tokens = output_tokens = 0
        for generations in response.generations:
            for generation in generations:
                if isinstance(generation, ChatGeneration):
                    meta = getattr(generation.message, "usage_metadata", None)
                    if meta:
                        input_tokens += meta["input_tokens"]
                        output_tokens += meta["output_tokens"]
        self._charge(Usage(input_tokens=input_tokens, output_tokens=output_tokens))

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        self._charge(Usage(tool_calls=1))

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        self._charge(Usage(failed=getattr(output, "status", None) == "error"))

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self._charge(Usage(failed=True), cause=error)
