"""Content hash of a closed mandate, HMAC signing behind the Signer/Verifier ports, and receipts."""

import hashlib
import json
import secrets
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent_toolkit.mandates import (
    Amount,
    ClosedMandate,
    HmacSha256Signer,
    Receipt,
    Signer,
    Verifier,
)

T = datetime(2026, 12, 1, 18, 0, tzinfo=UTC)


def charge(minor: int | None = 35000, /, **overrides: str) -> ClosedMandate:
    fields: dict[str, object] = {
        "open_mandate_id": "om_1",
        "agent_id": "agent_1",
        "kind": "new_charge",
        "charge_ref": "ch_1",
        "payer": "cust-juan",
        "receiving_account": "acct-own",
        "rail": "codi_push",
        "amount": None if minor is None else Amount(minor=minor, currency="MXN"),
        "previous_amount": None,
        "reference": "R-1001",
    }
    return ClosedMandate.model_validate(fields | overrides)


def test_content_hash_is_stable_across_equal_instances() -> None:
    assert charge().content_hash == charge().content_hash


def test_content_hash_changes_with_the_amount() -> None:
    assert charge(35000).content_hash != charge(35001).content_hash
    assert charge(35000).content_hash != charge(None).content_hash


@pytest.mark.parametrize(("field", "value"), [("agent_id", "agent_2"), ("charge_ref", "ch_2")])
def test_content_hash_covers_the_agent_and_the_ledger_ref(field: str, value: str) -> None:
    assert charge(**{field: value}).content_hash != charge().content_hash


def test_content_hash_is_sha256_of_canonical_json() -> None:
    closed = charge()
    canonical = json.dumps(closed.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    assert closed.content_hash == hashlib.sha256(canonical.encode()).hexdigest()
    assert canonical.startswith(
        '{"agent_id":"agent_1","amount":{"currency":"MXN","minor":35000},"charge_ref":"ch_1",'
    )


def test_hmac_signer_verifies_its_own_signature() -> None:
    key = secrets.token_bytes(32)
    signer: Signer = HmacSha256Signer(key)
    verifier: Verifier = HmacSha256Signer(key)
    payload = charge().content_hash.encode()

    signature = signer.sign(payload)

    assert verifier.verify(payload, signature)


def test_hmac_signer_rejects_another_key_or_payload() -> None:
    signer = HmacSha256Signer(secrets.token_bytes(32))
    payload = charge().content_hash.encode()
    signature = signer.sign(payload)

    assert not HmacSha256Signer(secrets.token_bytes(32)).verify(payload, signature)
    assert not signer.verify(charge(35001).content_hash.encode(), signature)
    assert not signer.verify(payload, signature[:-1])
    assert not signer.verify(payload, b"")


@pytest.mark.parametrize("length", [0, 31])
def test_hmac_signer_rejects_short_keys(length: int) -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        HmacSha256Signer(secrets.token_bytes(length))


def test_hmac_signer_repr_hides_the_key() -> None:
    key = secrets.token_bytes(32)

    assert key.hex() not in repr(HmacSha256Signer(key))
    assert repr(key) not in repr(HmacSha256Signer(key))


def test_receipt_records_the_mandate_hash() -> None:
    closed = charge()

    receipt = Receipt(
        status="verified",
        verifier="policy-gate",
        reference=closed.reference,
        mandate_hash=closed.content_hash,
        at=T,
    )

    assert receipt.mandate_hash == closed.content_hash


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mandate_hash": "abc"}, "pattern"),
        ({"mandate_hash": "A" * 64}, "pattern"),
        ({"status": "approved"}, "status"),
        ({"at": datetime(2026, 12, 1, 18, 0)}, "timezone"),
        ({"verifier": ""}, "at least 1 character"),
    ],
)
def test_receipt_rejects_invalid_fields(overrides: dict[str, object], message: str) -> None:
    fields: dict[str, object] = {
        "status": "executed",
        "verifier": "policy-gate",
        "reference": "R-1001",
        "mandate_hash": charge().content_hash,
        "at": T,
    }

    with pytest.raises(ValidationError, match=message):
        Receipt.model_validate(fields | overrides)
