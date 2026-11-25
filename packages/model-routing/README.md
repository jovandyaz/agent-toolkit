# agent-toolkit-model-routing

Model routing for agents. Today: a circuit breaker per model provider, so a fallback chain skips a
provider that keeps failing instead of waiting out its timeout on every request.

**Why.** LangChain's `with_fallbacks` (or any loop over SDK clients) has no memory: a dead primary
is retried on every request. The circuit breaker pattern (Azure Architecture Center, "Circuit
Breaker pattern"; Martin Fowler, "CircuitBreaker") opens after consecutive failures, skips the
provider for a window, then lets exactly one trial call through.

```
agent-toolkit-model-routing[langchain] @ git+https://github.com/jovandyaz/agent-toolkit@v0.1.0#subdirectory=packages/model-routing
```

Not on PyPI: see the root README, "Install from a git tag".

## Example

```python
from agent_toolkit.model_routing import ProviderCooldown, ProviderCoolingDown

now = [0.0]
cooldown = ProviderCooldown(failures_to_open=3, cooldown_s=60, clock=lambda: now[0])
chain = ("anthropic:claude-haiku-4-5", "openrouter:deepseek/deepseek-v3.2")
primary, fallback = chain


def dead() -> str:
    raise TimeoutError("provider timed out")


for _ in range(3):  # three retryable failures in a row open the breaker
    try:
        cooldown.call(primary, chain, dead)
    except TimeoutError:
        pass

try:
    cooldown.call(primary, chain, lambda: "never called")
except ProviderCoolingDown as skipped:
    assert skipped.key == primary  # skipped without calling it: go to the fallback
assert cooldown.call(fallback, chain, lambda: "answer") == "answer"

now[0] += 60  # the window ends: one trial call, and its success closes the breaker
assert cooldown.call(primary, chain, lambda: "back") == "back"
assert cooldown.admit(primary, chain) == "call"
```

With LangChain (needs real chat models, so not run here):

```python notest
from functools import partial

from agent_toolkit.budgets import BudgetExceededError
from agent_toolkit.model_routing import ProviderCooldown
from agent_toolkit.model_routing.adapters.langchain import (
    langchain_is_retryable,
    with_cooldown_fallbacks,
)

# One tracker per process: a tracker built per request remembers nothing.
COOLDOWN = ProviderCooldown(classify=partial(langchain_is_retryable, never=(BudgetExceededError,)))

chain = with_cooldown_fallbacks(
    [
        ("anthropic:claude-haiku-4-5", haiku.bind_tools(tools)),
        ("openrouter:deepseek/deepseek-v3.2", deepseek.bind_tools(tools)),
    ],
    COOLDOWN,
)
chain.invoke(messages)  # a cooling entry is skipped without calling its provider
```

## Contracts

- `ProviderCooldown(failures_to_open=3, cooldown_s=60.0, clock=time.monotonic, classify=is_retryable, on_transition=None)`,
  thread-safe, keyed by any string (`provider` or `provider:model`).
- `admit(key, chain) -> "call" | "trial" | None`; report the outcome with `record_success`,
  `record_failure`, `record_other` or `record_error(key, error)`, passing what `admit` returned.
  `call(key, chain, fn)` / `acall(...)` do all of it and raise `ProviderCoolingDown(key)` when
  refused.
- Closed: every call admitted; `failures_to_open` consecutive retryable failures open the key.
  A success resets the count; an outcome that says nothing about the provider (`record_other`)
  neither counts nor resets.
- Open: refused for `cooldown_s`, then half-open: exactly one trial; its success closes the key,
  its failure reopens it for another window, any other outcome frees the next trial.
- All keys of the chain cooling: the one that reopens soonest (ties: earliest in the chain) gets
  a forced trial, one at a time; requests during it are refused by every key (load shedding).
- A failure from a call admitted before the key opened neither extends the window nor ends a
  trial it was not. A success from such a call closes the key: the provider answered.
- A trial whose outcome never comes back (a hung call, a dropped admission) is abandoned after
  `cooldown_s` and the next trial is admitted. Through `call`/`acall`, the late outcome of an
  abandoned trial counts as a plain call's and leaves its replacement in flight; with `admit` and
  `record_*` by hand it ends the current trial.
- `is_retryable(error, retryable=(), never=(), decide=None)` walks `__cause__`/`__context__`:
  a `never` type anywhere means no; else the first link `decide` answers for decides; else a
  `TimeoutError`, `ConnectionError`, a `retryable` type or an `int` `status_code` of 408, 409,
  429 or 5xx means yes.
- `on_transition(key, "open" | "half_open" | "closed")` runs outside the lock, so events from two
  threads can arrive out of order; an exception it raises is logged and dropped. By default
  transitions go to the `agent_toolkit.model_routing` logger with `provider` and `state` in
  `extra`.
- `with_cooldown_fallbacks(entries, cooldown, *, exceptions_to_handle=(Exception,))` returns a
  `CooldownFallbacks` that tries entries in order. A refused entry is skipped; a handled error
  moves on; any other error is raised at once. When nothing answers, the first real error is raised, and `ProviderCoolingDown` of the
  first entry only when every entry was refused (unlike `with_fallbacks`, a skip never hides the
  fallback's error). Schemas are the first entry's; config specs are every entry's. A raw
  `with_fallbacks` over `CooldownRunnable`s also skips cooling entries but raises the first error,
  which may be a skip.

## Caller duties

- Keep one tracker per process (or per worker) and share it across requests.
- With LangChain, build it with `classify=langchain_is_retryable` (plus your budget stop in
  `never`, and your HTTP client's transport errors in `retryable` if they reach you unwrapped),
  and bind tools before wrapping: `CooldownRunnable` is a plain Runnable.
- Don't count your own congestion (a full local call pool) as the provider's: keep it out of
  `classify`.
- Bound every call with the SDK's own timeout, shorter than `cooldown_s`, so a hang counts as a
  failure instead of an abandoned trial. A caller-side
  `asyncio.timeout` around `acall` cancels it, and a cancellation ends a trial without counting.

## Limits

State lives in memory, per process; no persistence or cross-worker sharing. No sliding failure
window (consecutive failures only). Follow-ups planned in this package: a priced model catalog,
a token estimator and an OpenRouter provider.
