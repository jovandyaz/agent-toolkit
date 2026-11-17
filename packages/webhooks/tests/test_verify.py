import base64
import binascii
import os
from datetime import UTC, datetime, timedelta

import pytest
from standardwebhooks.webhooks import EmptyWebhookSecretError

from agent_toolkit.webhooks import InvalidSignatureError, VerifiedWebhook, verify
from agent_toolkit.webhooks.testing import sign

SECRET = "whsec_" + base64.b64encode(os.urandom(24)).decode()
BODY = b'{"event":"payment.paid","amount_minor":1000}'


def _now() -> datetime:
    return datetime.now(UTC)


def test_valid_signature_returns_webhook() -> None:
    ts = _now().replace(microsecond=0)
    headers = sign(BODY, SECRET, "msg_1", ts)
    webhook = verify(BODY, headers, SECRET)
    assert webhook == VerifiedWebhook(id="msg_1", timestamp=ts, body=BODY)


def test_header_names_are_case_insensitive() -> None:
    headers = {k.title(): v for k, v in sign(BODY, SECRET, "msg_1", _now()).items()}
    assert verify(BODY, headers, SECRET).id == "msg_1"


def test_tampered_body_raises() -> None:
    headers = sign(BODY, SECRET, "msg_1", _now())
    tampered = BODY[:-2] + b"9}"
    with pytest.raises(InvalidSignatureError):
        verify(tampered, headers, SECRET)


def test_wrong_secret_raises() -> None:
    headers = sign(BODY, SECRET, "msg_1", _now())
    other = "whsec_" + base64.b64encode(os.urandom(24)).decode()
    with pytest.raises(InvalidSignatureError):
        verify(BODY, headers, other)


def test_old_timestamp_raises() -> None:
    headers = sign(BODY, SECRET, "msg_1", _now() - timedelta(minutes=10))
    with pytest.raises(InvalidSignatureError):
        verify(BODY, headers, SECRET)


@pytest.mark.parametrize("missing", ["webhook-id", "webhook-timestamp", "webhook-signature"])
def test_missing_header_raises(missing: str) -> None:
    headers = sign(BODY, SECRET, "msg_1", _now())
    del headers[missing]
    with pytest.raises(InvalidSignatureError):
        verify(BODY, headers, SECRET)


def test_non_utf8_body_raises_invalid_signature() -> None:
    headers = sign(BODY, SECRET, "msg_1", _now())
    with pytest.raises(InvalidSignatureError):
        verify(b"\xff\xfe\x00", headers, SECRET)


def test_library_error_is_chained() -> None:
    headers = sign(BODY, SECRET, "msg_1", _now())
    with pytest.raises(InvalidSignatureError) as info:
        verify(BODY + b" ", headers, SECRET)
    assert info.value.__cause__ is not None


@pytest.mark.parametrize("signature", ["garbage", "v1,abc", "v1,!!!notbase64", "v1,a,b", " "])
def test_malformed_signature_header_raises_invalid_signature(signature: str) -> None:
    headers = sign(BODY, SECRET, "msg_1", _now()) | {"webhook-signature": signature}
    with pytest.raises(InvalidSignatureError):
        verify(BODY, headers, SECRET)


@pytest.mark.parametrize(
    ("secret", "error"),
    [("whsec_", EmptyWebhookSecretError), ("whsec_a", binascii.Error)],
)
def test_bad_secret_propagates_as_config_error(secret: str, error: type[Exception]) -> None:
    headers = sign(BODY, SECRET, "msg_1", _now())
    with pytest.raises(error) as info:
        verify(BODY, headers, secret)
    assert not isinstance(info.value, InvalidSignatureError)
