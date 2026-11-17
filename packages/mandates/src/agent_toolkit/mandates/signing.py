"""Optional signing behind ports, and receipts of verifying or executing a closed mandate.

The MVP signs with one server key (`HmacSha256Signer`). AP2's SD-JWT compatibility is out of
scope; another scheme only needs to implement `Signer` and `Verifier`.
"""

import hashlib
import hmac
from typing import Annotated, Literal, Protocol

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from agent_toolkit.mandates.models import Identifier

ReceiptStatus = Literal["verified", "executed", "rejected"]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

MIN_HMAC_KEY_BYTES = 32


class Signer(Protocol):
    def sign(self, payload: bytes) -> bytes: ...


class Verifier(Protocol):
    def verify(self, payload: bytes, signature: bytes) -> bool: ...


class HmacSha256Signer:
    """Signs and verifies with HMAC-SHA256 under one secret key of at least 32 bytes."""

    __slots__ = ("_key",)

    def __init__(self, key: bytes) -> None:
        if len(key) < MIN_HMAC_KEY_BYTES:
            raise ValueError(f"an HMAC-SHA256 key needs at least {MIN_HMAC_KEY_BYTES} bytes")
        self._key = key

    def sign(self, payload: bytes) -> bytes:
        return hmac.new(self._key, payload, hashlib.sha256).digest()

    def verify(self, payload: bytes, signature: bytes) -> bool:
        """Constant-time comparison of `signature` with this key's signature of `payload`."""
        return hmac.compare_digest(self.sign(payload), signature)


class Receipt(BaseModel):
    """Outcome of verifying or executing a closed mandate, tied to it by `mandate_hash`."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    status: ReceiptStatus
    verifier: Identifier
    reference: Identifier
    mandate_hash: Sha256Hex
    at: AwareDatetime
