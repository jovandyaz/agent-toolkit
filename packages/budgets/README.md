# agent-toolkit-budgets

Run budgets for agents: caps on iterations, tool calls, tokens, USD cost and consecutive failures,
as pure functions over immutable state.

**Why.** OWASP Top 10 for Agentic Applications 2026 (ASI08) asks for cost, rate and token budgets
and circuit breakers against runaway agents. OpenAI's practical guide to building agents (p. 31)
sets human intervention when failures pass a threshold; `consecutive_failures` is that trigger.

```
pip install "agent-toolkit-budgets[langchain]"   # the extra adds the LangChain callback
```

## Example

```python
from decimal import Decimal

from agent_toolkit.budgets import BudgetLimits, BudgetState, Pricing, Usage, charge, check

limits = BudgetLimits(max_iterations=3, max_cost_usd=Decimal("0.02"))
pricing = Pricing(input_per_mtok_usd=Decimal(3), output_per_mtok_usd=Decimal(15))

state = BudgetState.empty()
state = charge(state, Usage(iterations=1, input_tokens=1_000, output_tokens=500), pricing)
assert check(state, limits) == ()  # within budget

state = charge(state, Usage(iterations=1, input_tokens=2_000, output_tokens=1_000), pricing)
assert check(state, limits) == ("cost",)  # 0.0315 USD > 0.02
```

With LangChain (needs a real chat model, so not run here):

```python notest
from agent_toolkit.budgets.adapters.langchain import BudgetCallbackHandler

handler = BudgetCallbackHandler(limits, pricing)  # one handler per run
model.invoke(prompt, config={"callbacks": [handler]})  # raises BudgetExceededError
```

## Contracts

- `charge(state, usage, pricing) -> BudgetState` and `check(state, limits) -> tuple[ExceededKind, ...]`.
  A limit is exceeded only when the value is strictly greater; `None` means unlimited.
- Cost is `Decimal` arithmetic. The caller injects the price per million tokens; there is no model
  catalog. With `pricing=None` cost never accumulates, so a cost limit needs pricing (the
  callback raises `ValueError` otherwise).
- The callback charges an iteration at LLM-call start and a tool call at tool start, so the call
  that would exceed a limit raises before it runs. It is sticky: once tripped it raises on every
  later callback. A model that reports no `usage_metadata` cannot trip token or cost limits.

## Caller duties

- Attach one handler per run; it holds that run's state.
- `BudgetExceededError` must reach your code. The stock LangGraph `ToolNode` with
  `handle_tool_errors=True` (or any handler covering `Exception`) turns it into an error message
  to the model instead of stopping the run; the sticky handler then stops it at the next callback.
- On `exceeded(consecutive_failures)` hand the run to a person.

## Limits

Complements, never replaces, LangGraph's native `recursion_limit`. No price catalog, no
persistence; token counts come from the model's own reports.
