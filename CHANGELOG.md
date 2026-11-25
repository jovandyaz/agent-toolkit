# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). All packages share one version
(PEP 440) and release together under the tag `v{version}`.

## [Unreleased]

### Added

- `agent-toolkit-model-routing`: `ProviderCooldown`, a thread-safe circuit breaker per provider
  key (consecutive failures to open, cooldown window, one half-open trial, a forced trial for the
  soonest key when the whole chain is cooling, injectable clock, classification and transition
  hooks), `is_retryable`, `call`/`acall`, and a LangChain adapter (extra `langchain`):
  `CooldownRunnable`, `CooldownFallbacks` (a fallback chain that never hides a real error behind
  a skip, with `exceptions_to_handle`), `with_cooldown_fallbacks` and `langchain_is_retryable`.

## [0.1.0rc1] - 2026-11-17

Release candidate for `0.1.0`, the first release of seven packages.

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

### Changed

Since `v0.1.0a1`. That tag held mandates, payments-core and webhooks only; these changes break
code written against earlier commits of `agent-toolkit-policy-gate`.

- **Breaking:** a financial tool's builder is `to_closed_mandate(args, ctx)` with `ctx` a
  `BuildContext(thread_id, message_id, call_id)`, not `(args, call_id)`. Mint a new charge's
  `charge_ref` from all three ids: some providers repeat call ids such as `call_0` in every
  message. `decide` takes `thread_id=` and `message_id=`, both required for a financial call under
  a mandate.
- **Breaking:** `GatedToolNode` needs `configurable["thread_id"]` on every turn with tool calls,
  not only when a call needs approval, and an id on the `AIMessage` that proposes them
  (`add_messages`, as in `MessagesState`, gives one); either missing raises ValueError before
  anything is decided.
- **Breaking:** the interrupt value is plain JSON, a list of `ApprovalRequest.model_dump(mode=
  "json")`, so it survives LangGraph's strict msgpack checkpoints; read it back with
  `model_validate_json`. Resume with `{call_id: ApprovalDecision.model_dump(mode="json")}` (an
  `ApprovalDecision` object is still accepted). `ApprovalRequest` gains `closed`, the charge as
  it will run, bound by `closed_mandate_hash`; the `request_hash` also binds the message id.
- **Breaking:** `on_decision` reports an `Allow` or approved `RequireApproval` right before its
  tool runs, not once the tool ran or raised, so an audit log that fails blocks the call.
- Each call the node runs gets, in its config, an idempotency key and a fingerprint under
  `IDEMPOTENCY_KEY` and `FINGERPRINT_KEY`, read with `idempotency_key_from_config` (send it to the
  payment provider) and `call_fingerprint_from_config`.
- `GatedToolNode(..., replay=hook)`: `hook(key)` returns the `RecordedReply(fingerprint,
  content)` the tool stored when the call ran. On a re-run a call with a matching record gets its
  stored reply and is not run again; a record with another fingerprint denies it
  (`call_id_reused`).
- A call that may already have run (`mandate:charge_ref_reused`,
  `mandate:previous_amount_mismatch`, `call_id_reused`) gets a reply that tells the model to check
  its status before retrying, not `Not executed`.

[Unreleased]: https://github.com/jovandyaz/agent-toolkit/compare/v0.1.0rc1...HEAD
[0.1.0rc1]: https://github.com/jovandyaz/agent-toolkit/compare/v0.1.0a1...v0.1.0rc1
