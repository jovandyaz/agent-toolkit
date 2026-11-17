"""Open and closed mandates for agents that issue payment requests on an owner's behalf.

`check` judges one charge (closed mandate) against what the owner authorized (open mandate);
`consumed_in_period` gives it the budget already used, from a ledger of reserves and releases.
`ClosedMandate.content_hash` identifies a charge for signatures (`Signer`, `Verifier`) and
receipts.
"""

from agent_toolkit.mandates.check import (
    ApprovalReason,
    Verdict,
    Violation,
    check,
    reserve_amount,
)
from agent_toolkit.mandates.consumption import (
    ConsumptionEntry,
    EntryKind,
    RefState,
    RefStatus,
    consumed_in_period,
    ref_state,
)
from agent_toolkit.mandates.models import (
    Amount,
    AmountRange,
    Budget,
    BudgetPeriod,
    ChargeKind,
    ClosedMandate,
    DailyWindow,
    OpenMandate,
)
from agent_toolkit.mandates.signing import (
    HmacSha256Signer,
    Receipt,
    ReceiptStatus,
    Signer,
    Verifier,
)

__all__ = [
    "Amount",
    "AmountRange",
    "ApprovalReason",
    "Budget",
    "BudgetPeriod",
    "ChargeKind",
    "ClosedMandate",
    "ConsumptionEntry",
    "DailyWindow",
    "EntryKind",
    "HmacSha256Signer",
    "OpenMandate",
    "Receipt",
    "ReceiptStatus",
    "RefState",
    "RefStatus",
    "Signer",
    "Verdict",
    "Verifier",
    "Violation",
    "check",
    "consumed_in_period",
    "ref_state",
    "reserve_amount",
]
