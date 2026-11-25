"""Heuristic prompt-injection guard: `scan` gives a verdict with rule ids, `neutralize` wraps
untrusted text as data (OWASP LLM01). A heuristic, not a security boundary; the corpus in
`agent_toolkit.prompt_guard.corpus` is its CI gate.
"""

from agent_toolkit.prompt_guard.guard import (
    DEFAULT_PREFACE,
    DEFAULT_TAG,
    INJECTION_THRESHOLD,
    MAX_SCAN_CHARS,
    RULES,
    TOO_LARGE_RULE_ID,
    InjectionVerdict,
    Rule,
    neutralize,
    normalize,
    scan,
)

__all__ = [
    "DEFAULT_PREFACE",
    "DEFAULT_TAG",
    "INJECTION_THRESHOLD",
    "MAX_SCAN_CHARS",
    "RULES",
    "TOO_LARGE_RULE_ID",
    "InjectionVerdict",
    "Rule",
    "neutralize",
    "normalize",
    "scan",
]
