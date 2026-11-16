"""Request-to-pay payment domain without I/O: money, intents, attempts and payment policies.

The policies are pure functions, so sync and async code call them alike. The ports are
synchronous Protocols; an async application can wrap them (for example with
`asyncio.to_thread`) or write async adapters of its own. Test doubles for the ports live in
`agent_toolkit.payments_core.testing`.
"""

from agent_toolkit.payments_core.intent import (
    Attempt,
    AttemptStatus,
    IntentStatus,
    OutcomeKind,
    PaymentIntent,
    PaymentOutcome,
    ProviderPayment,
    ProviderStatus,
    ProviderStatusKind,
    RefundFlag,
    RefundReason,
)
from agent_toolkit.payments_core.money import CurrencyMismatchError, Money
from agent_toolkit.payments_core.policies import (
    RequestOutcome,
    apply_provider_payment,
    may_retry_request,
    reconcile_attempt,
)
from agent_toolkit.payments_core.ports import IntentRepository, RailConnector, UnitOfWork

__all__ = [
    "Attempt",
    "AttemptStatus",
    "CurrencyMismatchError",
    "IntentRepository",
    "IntentStatus",
    "Money",
    "OutcomeKind",
    "PaymentIntent",
    "PaymentOutcome",
    "ProviderPayment",
    "ProviderStatus",
    "ProviderStatusKind",
    "RailConnector",
    "RefundFlag",
    "RefundReason",
    "RequestOutcome",
    "UnitOfWork",
    "apply_provider_payment",
    "may_retry_request",
    "reconcile_attempt",
]
