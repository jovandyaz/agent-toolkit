"""Model routing for agents: which provider serves a call.

`ProviderCooldown` is a thread-safe circuit breaker per provider key, so a fallback chain skips a
provider that keeps failing; `is_retryable` tells a provider's failure from the request's.
Frameworks plug in through `agent_toolkit.model_routing.adapters`.
"""

from agent_toolkit.model_routing.cooldown import (
    COOLDOWN_S,
    FAILURES_TO_OPEN,
    RETRYABLE_STATUSES,
    Admission,
    Classifier,
    Decider,
    ProviderCooldown,
    ProviderCoolingDown,
    Transition,
    TransitionHook,
    is_retryable,
)

__all__ = [
    "COOLDOWN_S",
    "FAILURES_TO_OPEN",
    "RETRYABLE_STATUSES",
    "Admission",
    "Classifier",
    "Decider",
    "ProviderCooldown",
    "ProviderCoolingDown",
    "Transition",
    "TransitionHook",
    "is_retryable",
]
