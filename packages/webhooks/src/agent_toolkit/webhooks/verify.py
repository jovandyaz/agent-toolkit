"""Standard Webhooks signature verification."""

from collections.abc import Mapping
from datetime import UTC, datetime

from pydantic import AwareDatetime, BaseModel, ConfigDict
from standardwebhooks.webhooks import Webhook, WebhookVerificationError


class InvalidSignatureError(Exception):
    """The webhook failed verification: bad or missing headers, stale timestamp or bad signature."""


class VerifiedWebhook(BaseModel):
    """A webhook whose signature and timestamp were verified."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    id: str
    timestamp: AwareDatetime
    body: bytes


def verify(body: bytes, headers: Mapping[str, str], secret: str) -> VerifiedWebhook:
    """Verify `body` against the Standard Webhooks headers and return the verified webhook.

    `body` must be the raw request body exactly as received, never re-serialized JSON. Header
    names are matched case-insensitively. `secret` may carry the `whsec_` prefix. The library's
    timestamp tolerance (5 minutes) applies. Bodies must be UTF-8 because standardwebhooks 1.1
    decodes them; byte-exact support exists only on the library's unreleased main branch.

    Raises `InvalidSignatureError` for any verification failure, with the library error chained.
    """
    lowered = {name.lower(): value for name, value in headers.items()}
    webhook = Webhook(secret)  # a malformed secret is a config error: let it propagate
    try:
        # The signature header is attacker-controlled: the library does not guard its split and
        # base64 decode, so malformed values surface as ValueError (binascii.Error is one).
        webhook.verify(body, lowered, json_parse=False)
    except (WebhookVerificationError, ValueError) as error:
        raise InvalidSignatureError(str(error) or type(error).__name__) from error
    # verify() guarantees both headers are present and the timestamp is a valid number.
    timestamp = datetime.fromtimestamp(float(lowered["webhook-timestamp"]), tz=UTC)
    return VerifiedWebhook(id=lowered["webhook-id"], timestamp=timestamp, body=body)
