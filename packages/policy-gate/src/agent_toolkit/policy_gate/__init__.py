"""Deterministic gate for the tool calls an agent proposes (OWASP Agentic ASI02).

`decide` turns one `ToolCall` into `Allow`, `RequireApproval` or `Deny` before it runs, from
the `ToolSpec`, the session's Rule of Two flags, the owner's mandate and the run budget. Model
output is untrusted: arguments are validated strictly first. Reasons are stable,
machine-readable strings, listed in the docstring of `decide`.
"""

from agent_toolkit.policy_gate.decide import decide
from agent_toolkit.policy_gate.models import (
    Allow,
    BuildContext,
    Decision,
    Deny,
    Effects,
    GateConfig,
    GateContext,
    OnMandateViolation,
    RequireApproval,
    Risk,
    SessionProfile,
    ToolCall,
    ToolSpec,
)

__all__ = [
    "Allow",
    "BuildContext",
    "Decision",
    "Deny",
    "Effects",
    "GateConfig",
    "GateContext",
    "OnMandateViolation",
    "RequireApproval",
    "Risk",
    "SessionProfile",
    "ToolCall",
    "ToolSpec",
    "decide",
]
