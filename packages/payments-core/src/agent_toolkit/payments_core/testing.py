"""In-memory test doubles for the ports. For tests only: single-threaded, nothing persisted."""

from collections.abc import Callable
from datetime import datetime

from agent_toolkit.payments_core.intent import Attempt, PaymentIntent, ProviderStatus


class InMemoryIntentRepository:
    """`IntentRepository` backed by a dict."""

    def __init__(self) -> None:
        self._intents: dict[str, PaymentIntent] = {}

    def get(self, intent_id: str) -> PaymentIntent | None:
        return self._intents.get(intent_id)

    def get_by_provider_ref(self, provider_ref: str) -> PaymentIntent | None:
        for intent in self._intents.values():
            if any(attempt.provider_ref == provider_ref for attempt in intent.attempts):
                return intent
        return None

    def save(self, intent: PaymentIntent) -> None:
        self._intents[intent.id] = intent


class FakeRailConnector:
    """`RailConnector` that opens attempts locally and answers what the test set.

    Attempts get ids `att_<n>` and provider refs `fake_ref_<n>`. A query for a ref with no
    status set answers `not_found`. `requests` and `queries` record every call.
    """

    def __init__(self, clock: Callable[[], datetime]) -> None:
        self._clock = clock
        self._statuses: dict[str, ProviderStatus] = {}
        self.requests: list[Attempt] = []
        self.queries: list[str] = []

    def request(self, intent: PaymentIntent, rail: str, expires_at: datetime) -> Attempt:
        owed = intent.amount - intent.paid_total
        if owed.amount_minor == 0:
            raise ValueError(f"intent {intent.id!r} owes nothing")
        number = len(self.requests) + 1
        attempt = Attempt(
            id=f"att_{number}",
            rail=rail,
            provider_ref=f"fake_ref_{number}",
            amount=owed,
            created_at=self._clock(),
            expires_at=expires_at,
            status="requested",
        )
        self.requests.append(attempt)
        return attempt

    def set_status(self, provider_ref: str, status: ProviderStatus) -> None:
        self._statuses[provider_ref] = status

    def query(self, provider_ref: str) -> ProviderStatus:
        self.queries.append(provider_ref)
        return self._statuses.get(provider_ref, ProviderStatus(kind="not_found", payment=None))
