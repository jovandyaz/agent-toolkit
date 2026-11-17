# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). All packages share one version
(PEP 440) and release together under the tag `v{version}`.

## [Unreleased]

First release, `0.1.0`, of seven packages.

### Added

- `agent-toolkit-mandates`: open and closed mandates (AP2 adapted to payment requests), a pure
  `check` of a charge against its mandate, a reserve/release consumption ledger with per-period
  budgets in the mandate's time zone, optional HMAC signing behind `Signer`/`Verifier`, receipts.
- `agent-toolkit-budgets`: caps on iterations, tool calls, tokens, USD cost and consecutive
  failures as pure `charge`/`check` functions; LangChain callback adapter (extra `langchain`).
- `agent-toolkit-policy-gate`: deterministic `decide` returning allow, require approval or deny
  from tool risk, strict argument validation, the owner's mandate, the run budget and the Rule of
  Two; LangGraph `GatedToolNode` with hash-bound human approvals (extra `langgraph`).
- `agent-toolkit-audit-log`: append-only hash-chained events with keyed payload hashes,
  `verify_chain`, an `AuditStore` port and a Postgres store (extra `postgres`).
- `agent-toolkit-evals`: eval runner with k isolated attempts, deterministic final-state graders,
  exact pass@k and pass^k, JSONL transcripts and a pytest plugin (extra `pytest`).
- `agent-toolkit-payments-core`: money in minor units, payment intents and attempts, idempotent
  `apply_provider_payment`, `reconcile_attempt`, retry rules, ports and in-memory test doubles.
- `agent-toolkit-webhooks`: Standard Webhooks `verify` over the raw body and an idempotent
  `process_once` inbox.
- A README per package, executed as tests; `scripts/check_versions.py` and a CI job tying release
  tags to package versions.

[Unreleased]: https://github.com/jovandyaz/agent-toolkit/compare/v0.1.0a1...HEAD
