from datetime import UTC, datetime, timedelta

import pytest

from agent_toolkit.payments_core import (
    IntentRepository,
    Money,
    PaymentIntent,
    ProviderPayment,
    ProviderStatus,
    RailConnector,
    apply_provider_payment,
)
from agent_toolkit.payments_core.testing import FakeRailConnector, InMemoryIntentRepository

T = datetime(2026, 11, 16, 18, 0, tzinfo=UTC)
EXPIRES_AT = T + timedelta(minutes=10)


def mxn(amount_minor: int) -> Money:
    return Money(amount_minor=amount_minor, currency="MXN")


def a_new_intent(intent_id: str = "pi_1") -> PaymentIntent:
    return PaymentIntent(
        id=intent_id,
        principal_id="merchant_1",
        reference="R-1001",
        amount=mxn(35000),
        idempotency_key=f"idem_{intent_id}",
        status="pending",
        attempts=(),
        paid_total=mxn(0),
        late=False,
        applied_event_ids=frozenset(),
    )


def with_attempt(intent: PaymentIntent, rail: RailConnector) -> PaymentIntent:
    attempt = rail.request(intent, "codi_push", EXPIRES_AT)
    return PaymentIntent.model_validate({**dict(intent), "attempts": (*intent.attempts, attempt)})


def test_fake_rail_opens_requested_attempts_for_what_is_still_owed() -> None:
    rail = FakeRailConnector(clock=lambda: T)

    first = rail.request(a_new_intent(), "codi_push", EXPIRES_AT)
    second = rail.request(a_new_intent(), "codi_qr", EXPIRES_AT)

    assert first.status == "requested"
    assert first.amount == mxn(35000)
    assert first.created_at == T
    assert first.expires_at == EXPIRES_AT
    assert first.rail == "codi_push"
    assert first.id != second.id
    assert first.provider_ref != second.provider_ref
    assert rail.requests == [first, second]


def test_fake_rail_refuses_to_request_a_paid_intent() -> None:
    rail = FakeRailConnector(clock=lambda: T)
    intent = with_attempt(a_new_intent(), rail)
    assert intent.attempts[0].provider_ref is not None
    payment = ProviderPayment(
        provider_event_id="evt_1",
        provider_ref=intent.attempts[0].provider_ref,
        amount=mxn(35000),
        received_at=T,
        rail="codi_push",
    )
    paid = apply_provider_payment(intent, payment).intent
    assert paid is not None

    with pytest.raises(ValueError, match="owes nothing"):
        rail.request(paid, "codi_push", EXPIRES_AT)


def test_fake_rail_answers_queries_with_the_status_the_test_set() -> None:
    rail = FakeRailConnector(clock=lambda: T)
    pending = ProviderStatus(kind="pending", payment=None)

    rail.set_status("ref_known", pending)

    assert rail.query("ref_known") == pending
    assert rail.query("ref_unknown") == ProviderStatus(kind="not_found", payment=None)
    assert rail.queries == ["ref_known", "ref_unknown"]


def test_repository_finds_intents_by_id_and_by_provider_ref() -> None:
    repository = InMemoryIntentRepository()
    intent = with_attempt(a_new_intent(), FakeRailConnector(clock=lambda: T))
    provider_ref = intent.attempts[0].provider_ref
    assert provider_ref is not None

    repository.save(intent)

    assert repository.get("pi_1") == intent
    assert repository.get_by_provider_ref(provider_ref) == intent
    assert repository.get("pi_missing") is None
    assert repository.get_by_provider_ref("ref_missing") is None


def test_repository_save_replaces_the_stored_intent() -> None:
    repository = InMemoryIntentRepository()
    repository.save(a_new_intent())
    updated = with_attempt(a_new_intent(), FakeRailConnector(clock=lambda: T))

    repository.save(updated)

    assert repository.get("pi_1") == updated


def test_doubles_drive_a_webhook_flow_through_the_ports() -> None:
    rail: RailConnector = FakeRailConnector(clock=lambda: T)
    repository: IntentRepository = InMemoryIntentRepository()
    intent = with_attempt(a_new_intent(), rail)
    repository.save(intent)
    provider_ref = intent.attempts[0].provider_ref
    assert provider_ref is not None
    payment = ProviderPayment(
        provider_event_id="evt_1",
        provider_ref=provider_ref,
        amount=mxn(35000),
        received_at=T + timedelta(minutes=2),
        rail="codi_push",
    )

    outcome = apply_provider_payment(repository.get_by_provider_ref(provider_ref), payment)
    assert outcome.intent is not None
    repository.save(outcome.intent)

    stored = repository.get("pi_1")
    assert stored is not None
    assert stored.status == "paid"
    assert outcome.kind == "confirmed"
