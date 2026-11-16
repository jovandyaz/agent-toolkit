"""Ports the application implements around the pure policies (rail connector, storage).

Policies never call these: the application reads with them, runs a policy, and writes back.
The application must do that read and write in one `UnitOfWork` that locks the intent, and must
deduplicate `orphan` refund flags by `provider_event_id`, since an orphan has no intent to
remember its event.

The ports are synchronous. An async application can wrap them (for example with
`asyncio.to_thread`) or keep async adapters of its own around the pure policies.
"""

from datetime import datetime
from types import TracebackType
from typing import Protocol, Self

from agent_toolkit.payments_core.intent import Attempt, PaymentIntent, ProviderStatus


class RailConnector(Protocol):
    """A payment rail provider, for example a CoDi connector."""

    def request(self, intent: PaymentIntent, rail: str, expires_at: datetime) -> Attempt:
        """Ask the provider to collect what `intent` still owes until `expires_at`.

        Return the attempt as `requested`, or as `ambiguous` when the provider may have
        processed it without a conclusive answer: reconcile that attempt, never send it again.
        Raise when the request was not sent or was rejected; `may_retry_request` says whether
        sending it again is safe.
        """
        ...

    def query(self, provider_ref: str) -> ProviderStatus:
        """Report what the provider knows about the request `provider_ref`."""
        ...


class IntentRepository(Protocol):
    """Storage for intents.

    Inside a unit of work, reads must lock the intent they return (or `save` must reject a
    concurrent change), so two payments applied at once cannot overwrite each other's
    `paid_total` or `applied_event_ids`.
    """

    def get(self, intent_id: str) -> PaymentIntent | None: ...

    def get_by_provider_ref(self, provider_ref: str) -> PaymentIntent | None:
        """The intent owning the attempt with this `provider_ref`, if any."""
        ...

    def save(self, intent: PaymentIntent) -> None: ...


class UnitOfWork(Protocol):
    """One transaction: leaving the block without `commit()` rolls it back.

    Read the intent, run the policy and save its outcome inside the same unit of work.
    """

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...
