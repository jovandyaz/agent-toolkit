# model-routing and prompt-guard: implementation plan

**Goal:** move the provider cooldown and the prompt-injection guard that pialy keeps as private
copies (ported from llm-router) into two reusable packages, so pialy and later consumers
install them instead.

**Packages (decided 2026-11-24):**

- `model-routing` (`agent_toolkit.model_routing`): everything about choosing which model serves a
  call. Starts with the provider cooldown (circuit breaker); the follow-ups (priced model catalog,
  token estimator, OpenRouter) belong here too. Core: stdlib only. Extra `langchain`.
- `prompt-guard` (`agent_toolkit.prompt_guard`): heuristic injection verdicts plus `neutralize`.
  Separate package because its consumers differ (a permit pre-review agent needs the guard, not a
  model chain) and it has no extra. Depends on `pydantic` only.

Neither depends on another package of this repo (AGENTS.md: only `policy-gate` does). The budget
stop that pialy treats as non-retryable is passed in by the caller (`never=`).

## Tasks

- [x] 1. `model-routing` core: `ProviderCooldown` (thread-safe, keyed by provider id, consecutive
  failures to open, cooldown window, one half-open trial, all-cooling soonest trial, injectable
  clock and transition hook, classification hook), `is_retryable`, `ProviderCoolingDown`,
  `call`/`acall` helpers. Tests: pialy's state-machine, transition and classification tests,
  including its two review fixes (stale failure, pre-open call vs trial).
- [x] 2. `model-routing` LangChain adapter: `CooldownRunnable` (`invoke`/`ainvoke`),
  `with_cooldown_fallbacks`, `langchain_is_retryable` (a LangChain `ModelError` decides by its
  own `is_retryable`). Tests through a real `with_fallbacks` chain with fake chat models
  (superseded in step 5 by `CooldownFallbacks`).
- [x] 3. `prompt-guard`: rules (llm-router heuristics + pialy fixes: case-sensitive `DAN … mode`,
  narrowed Spanish extraction, Spanish role hijack), `scan` -> `InjectionVerdict` with rule ids,
  size cap, `neutralize`. Corpus (es + en) as a CI gate.
- [x] 4. Repo wiring: import-linter roots, README table, package READMEs with runnable examples,
  CHANGELOG Unreleased, lockstep version `0.1.0rc1`, `uv.lock`.
- [x] 5. Review round 1 fixes: transition hooks isolated and a trial lease (`cooldown_s`) so a
  lost trial cannot wedge a key; `CooldownFallbacks` replaces `with_fallbacks` so a skip never
  hides the fallback's real error, with `exceptions_to_handle` and delegated schemas; guard strips
  every Cf character, `encoded.execute` needs whole words, `hijack.you_are_now` a word end, and
  `neutralize` escapes NFKC look-alikes of `<`, `>` and `&`.
- [x] 6. Review round 2 fixes: trial tokens so a late outcome of an abandoned trial leaves its
  replacement alone; attempts run in their own config context and the chain merges every entry's
  config specs; `encoded.execute` again catches `execute:<payload>` and `/`-led payloads; the
  guard also strips invisible non-Cf characters and skips the per-character pass for ASCII.
- [x] 7. Review round 3 fixes: `InputType`/`OutputType` typed `type[Input]`/`type[Output]` (no
  `Any`); `normalize` idempotent (strip, fold, strip, fold); README and CHANGELOG name
  `CooldownFallbacks` and warn about a raw `with_fallbacks`.

## Verification

`uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run scripts/typecheck.sh && uv run lint-imports`

## Follow-ups (out of scope)

Priced model catalog, token estimator, OpenRouter provider (all `model-routing`); windowed scans
for texts over the guard's size cap (llm-router `detectAiInput`) and span location for redaction.
