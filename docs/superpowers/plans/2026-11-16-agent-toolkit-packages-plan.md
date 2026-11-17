# agent-toolkit (paquetes v0.1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir los siete paquetes Python del primer corte (`payments-core`, `webhooks`, `mandates`, `budgets`, `policy-gate`, `audit-log`, `evals`) en un repo nuevo `agent-toolkit`, con versiones etiquetadas que la app del hackatón pueda consumir desde la semana 1.

**Architecture:** uv workspace con un paquete por directorio bajo `packages/`, todos en el namespace `agent_toolkit.*`. Núcleos puros (Pydantic v2, stdlib); adaptadores como extras opcionales en un subpaquete `adapters`. Un contrato de import-linter en CI impide que los núcleos importen frameworks, drivers o proveedores.

**Tech Stack:** Python 3.12, uv ≥ 0.13 (`uv_build`), Pydantic 2.14, pytest 9, ruff, mypy (strict), import-linter 2.16, standardwebhooks 1.1, langgraph 1.2 / langchain-core (solo adaptadores), psycopg 3.3 (solo adaptador), GitHub Actions con servicio Postgres 17.

**Spec:** `designs/agent-toolkit/2026-11-15-agent-toolkit-design.md` (aprobado 2026-11-16). Secciones 6 y 7 son la fuente de los contratos.

**Alcance de este plan:** solo el toolkit (lo construye el usuario). La app del hackatón tendrá su propio plan, con tareas asignables al equipo de 3–4 personas; consume las etiquetas que publica este plan (ver "Entregas a la app").

## Global Constraints

- Python `>=3.12`. uv `>=0.13` (la máquina tiene 0.8.12: `uv self update` antes de la Tarea 1).
- Distribuciones `agent-toolkit-<paquete>`, imports `agent_toolkit.<paquete>`; nombres provisionales, verificar PyPI antes de publicar (no se publica en este plan).
- El núcleo de cada paquete no importa `langgraph`, `langchain*`, `psycopg`, ningún SDK de pey ni de proveedores de modelos. Los adaptadores viven en `agent_toolkit.<paquete>.adapters.<nombre>` y se instalan como extra.
- Dependencias entre paquetes: solo `policy-gate` → `mandates` y `budgets`. `payments-core` no depende de ninguno.
- Dinero: enteros en unidades menores + código ISO 4217 de 3 letras mayúsculas. Nunca `float`.
- Tipos públicos: modelos Pydantic v2 `frozen=True, strict=True`; `mypy --strict` sin `Any` en APIs públicas.
- `audit-log` no guarda PII en claro: solo hashes o referencias.
- La salida del modelo es no confiable: todo argumento de herramienta se valida contra su esquema antes de decidir.
- Commits: Conventional Commits, una línea, inglés, sin trailers ni co-autoría. Se hacen solo cuando el usuario autorice la ejecución del plan; ramas por paquete, PR por paquete, nunca push directo a `main`.
- Sin secretos en el repo. Tests con secretos y llaves generadas en el propio test.

## Review Focus

1. **Reanudar después de un `interrupt()` re-ejecuta el nodo de LangGraph desde el inicio.** Si el nodo ya ejecutó una herramienta permitida antes de interrumpir, la ejecutaría dos veces (dos cobros). Se espera que cada herramienta se ejecute exactamente una vez. Test en Tarea 6: `test_resume_executes_each_tool_once`.
2. **Aprobación vieja o falsificada:** se reanuda con una decisión cuyo `request_hash` no coincide con la llamada pendiente (porque cambiaron los argumentos). Se espera que la llamada no se ejecute. Test en Tarea 6: `test_resume_with_mismatched_hash_rejects`.
3. **Evento del proveedor reenviado** (mismo `webhook-id` o mismo `provider_event_id`). Se espera que el pago no se cuente dos veces. Tests en Tareas 1 y 2.
4. **Presupuesto diario cerca de la medianoche local:** un cobro a las 23:30 y otro a las 00:30 en America/Mexico_City caen en días distintos aunque en UTC sea el mismo día. Test en Tarea 3: `test_daily_period_uses_mandate_timezone`.
5. **Dos `append` concurrentes al audit-log con el mismo `prev_hash`.** Se espera que el segundo falle con `ChainConflict` y que la cadena no se bifurque. Test en Tarea 7: `test_concurrent_append_conflicts`.

---

### Task 0: Repo y workspace

Decidido por el usuario: repo **público**, licencia **Apache-2.0**. Hecho el 2026-11-16 (rama `chore/scaffold-workspace`).

**Files:**
- Create: `~/Developer/Personal/Github/agent-toolkit/` con `pyproject.toml` (raíz del workspace, sin paquete propio), `.python-version` (`3.12`), `.gitignore`, `README.md`, `LICENSE`, `.importlinter` (o `[tool.importlinter]` en la raíz), `.github/workflows/ci.yml`
- Copy: este plan y el spec a `docs/superpowers/plans/` y `docs/superpowers/specs/`

- [x] **Step 1:** `gh auth status` muestra `jovandyaz` activo; `uv self update` y `uv --version` ≥ 0.13.
- [x] **Step 2:** `gh repo create jovandyaz/agent-toolkit --<visibilidad> --license apache-2.0 --clone` dentro de `~/Developer/Personal/Github/` (identidad personal por `includeIf`; verificar con `git config user.email` dentro del repo).
- [x] **Step 3:** Raíz `pyproject.toml`: `[tool.uv.workspace] members = ["packages/*"]`; grupo `dev` con `pytest`, `ruff`, `mypy`, `import-linter`; config de ruff (`select = ["E","F","I","UP","B","SIM","RUF"]`), mypy `strict = true`, pytest `--strict-markers` con marcador `postgres`.
- [x] **Step 4:** Contrato de import-linter `core-is-framework-free`: `root_packages = ["agent_toolkit"]`, `include_external_packages = true`, tipo `forbidden`, `source_modules = agent_toolkit.*`, `forbidden_modules = langgraph, langchain_core, langchain, psycopg`, `ignore_imports = agent_toolkit.**.adapters.** -> *` (en `pyproject.toml`). Por ser namespace, `root_packages` lista cada porción (`agent_toolkit.<pkg>`): **cada tarea de paquete añade la suya**. Como `ignore_imports` borra la arista adaptador→framework, el contrato no ve cadenas núcleo→adaptador→framework; las cubre `tests/test_core_isolation.py`, que importa cada módulo núcleo en un subproceso con esos paquetes bloqueados.
- [x] **Step 5:** CI (`ci.yml`): en push y PR → `uv sync --all-packages --locked`, `ruff check`, `ruff format --check`, `mypy packages`, `lint-imports`, `pytest` con servicio `postgres:17` y `DATABASE_URL` hacia él.
- [x] **Step 6: Probar que el contrato falla.** Hecho con un paquete sonda desechable: import en `adapters/` → KEPT; import directo en núcleo → BROKEN; núcleo→adaptador→`psycopg` → KEPT por el linter pero FAIL en `test_core_isolation.py`.
- [ ] **Step 7:** `main` ya traía el commit de LICENSE de GitHub, así que el scaffold va por PR desde `chore/scaffold-workspace`: `chore: scaffold uv workspace with lint, type and import contracts`. A partir de aquí, rama por paquete. En CI, `mypy packages` y `lint-imports` se activan cuando exista el primer `packages/*/pyproject.toml`.

### Task 1: `payments-core`

Fuente de los casos de aceptación: `cursor_hackathon_nov_2026` (pialy), `packages/payments/test/` y su README. Se reescriben en Python; los fixtures JSON se copian a `packages/payments-core/tests/fixtures/` declarando el origen.

**Files:**
- Modify: `pyproject.toml` raíz: añade `agent_toolkit.payments_core` a `[tool.importlinter] root_packages` (cada paquete hace lo mismo con el suyo).
- Create: `packages/payments-core/pyproject.toml` (`uv_build`, `module-name = "agent_toolkit.payments_core"`, dep `pydantic>=2.14,<3`)
- Create: `src/agent_toolkit/payments_core/{__init__,money,intent,policies,ports,testing}.py` (sin `__init__.py` en `agent_toolkit/`)
- Test: `packages/payments-core/tests/test_money.py`, `test_apply_payment.py`, `test_reconcile.py`, `test_retry.py`

**Estado:** hecho (PR feat/payments-core). Decisiones R4–R9 en el ledger de SDD: `duplicate` separado de `already_applied`; tardío = después del vencimiento del intento pagado; el conector asigna `provider_ref` antes de enviar; mismo `provider_event_id` en webhook y consulta. Type-check por paquete con `scripts/typecheck.sh`.

**Interfaces:**
- Produces:
  - `Money(amount_minor: int, currency: str)`; `+`/`-` con misma moneda; `CurrencyMismatchError`; resultado negativo → `ValueError`.
  - `IntentStatus = Literal["pending", "partially_paid", "paid", "expired"]`; `AttemptStatus = Literal["requested", "ambiguous", "paid", "expired"]`.
  - `Attempt(id: str, rail: str, provider_ref: str | None, amount: Money | None, created_at: datetime, expires_at: datetime, status: AttemptStatus)` (`amount=None` = monto abierto).
  - `PaymentIntent(id, principal_id, reference, amount: Money, idempotency_key, status, attempts: tuple[Attempt, ...], paid_total: Money, late: bool, applied_event_ids: frozenset[str])`.
  - `ProviderPayment(provider_event_id: str, provider_ref: str, amount: Money, received_at: datetime, rail: str)`.
  - `RefundFlag(reason: Literal["duplicate", "orphan", "excess"], amount: Money, provider_event_id: str)`.
  - `PaymentOutcome(kind: Literal["confirmed", "confirmed_late", "partial", "already_applied", "orphan"], intent: PaymentIntent | None, refund_flags: tuple[RefundFlag, ...])`.
  - `apply_provider_payment(intent: PaymentIntent | None, payment: ProviderPayment) -> PaymentOutcome`
  - `ProviderStatus(kind: Literal["paid", "pending", "expired_unpaid", "not_found"], payment: ProviderPayment | None)`
  - `reconcile_attempt(intent: PaymentIntent, attempt_id: str, status: ProviderStatus, now: datetime) -> PaymentOutcome`
  - `RequestOutcome = Literal["not_sent", "rejected", "ambiguous"]`; `may_retry_request(outcome: RequestOutcome) -> bool`
  - Protocols `RailConnector` (`request(intent, rail, expires_at) -> Attempt`, `query(provider_ref) -> ProviderStatus`), `IntentRepository` (`get`, `get_by_provider_ref`, `save`), `UnitOfWork` (context manager, `commit`, `rollback`); `testing.InMemoryIntentRepository`, `testing.FakeRailConnector`.

- [x] **Step 1: Tests de `Money`**
```python
def test_money_rejects_float_and_bad_currency():
    with pytest.raises(ValidationError): Money(amount_minor=1.5, currency="MXN")
    with pytest.raises(ValidationError): Money(amount_minor=100, currency="mxn")
def test_money_add_requires_same_currency():
    assert Money(amount_minor=100, currency="MXN") + Money(amount_minor=1, currency="MXN") == Money(amount_minor=101, currency="MXN")
    with pytest.raises(CurrencyMismatchError): Money(amount_minor=1, currency="MXN") + Money(amount_minor=1, currency="USD")
```
- [x] **Step 2:** `uv run pytest packages/payments-core -k money` → FAIL (import error).
- [x] **Step 3:** Implementa `money.py`.
- [x] **Step 4: Tests de `apply_provider_payment`** (uno por caso; `intent` con monto 35000 MXN, `expires_at` = T+10 min)
  - `test_payment_within_validity_confirms`: kind `confirmed`, status `paid`, `late is False`.
  - `test_late_payment_confirms_and_flags_late`: `received_at` > `expires_at` del último attempt → `confirmed_late`, `late is True`.
  - `test_replayed_event_is_already_applied`: mismo `provider_event_id` dos veces → segunda `already_applied`, `paid_total` sin cambio. **(Review Focus 3)**
  - `test_second_distinct_payment_on_paid_intent_flags_duplicate`: otro evento sobre intent `paid` → `RefundFlag(reason="duplicate", amount=pago)`.
  - `test_payment_without_intent_is_orphan`: `intent=None` → kind `orphan`, `RefundFlag(reason="orphan")`.
  - `test_partial_then_complete`: 20000 → `partial`/`partially_paid`; luego 15000 → `confirmed`/`paid`.
  - `test_overpayment_flags_excess`: 40000 → `confirmed` + `RefundFlag(reason="excess", amount=5000)`.
  - `test_currency_mismatch_raises`.
- [x] **Step 5:** Run → FAIL. **Step 6:** Implementa `intent.py` + `apply_provider_payment` en `policies.py` (función pura; devuelve un intent nuevo con `model_copy`). **Step 7:** Run → PASS.
- [x] **Step 8: Tests de `reconcile_attempt` y `may_retry_request`**
  - `paid` no aplicado → mismo resultado que `apply_provider_payment`; `pending` → intent sin cambios; `expired_unpaid` → attempt `expired` y, si ninguno sigue vivo y no hay pago, intent `expired`; `not_found` sobre attempt `ambiguous` → attempt `expired` (nunca llegó).
  - `may_retry_request("not_sent") is True`; `"rejected"` y `"ambiguous"` → `False`.
- [x] **Step 9:** Run → FAIL; implementa; Run → PASS.
- [x] **Step 10:** Port de los casos de pialy restantes que no cubran los tests de arriba (cada uno con comentario del test de origen). Run → PASS.
- [x] **Step 11:** `ruff`, `mypy`, `lint-imports` limpios. Commit `feat(payments-core): add money, intents and payment policies`.

### Task 2: `webhooks`

**Files:**
- Create: `packages/webhooks/pyproject.toml` (deps `standardwebhooks>=1.1,<2`, `pydantic`), `src/agent_toolkit/webhooks/{__init__,verify,inbox,testing}.py`
- Test: `packages/webhooks/tests/test_verify.py`, `test_inbox.py`

**Interfaces:**
- Produces:
  - `VerifiedWebhook(id: str, timestamp: datetime, body: bytes)`
  - `verify(body: bytes, headers: Mapping[str, str], secret: str) -> VerifiedWebhook`; lanza `InvalidSignatureError` (envuelve `WebhookVerificationError`). Acepta secretos con prefijo `whsec_` (la librería lo quita).
  - Protocol `Inbox`: `record(webhook_id: str, received_at: datetime) -> bool` (False si ya existía; la implementación real usa una clave única en la misma transacción que el efecto).
  - `process_once(inbox: Inbox, webhook: VerifiedWebhook, handler: Callable[[VerifiedWebhook], T]) -> T | None` (registra primero y llama al handler solo si es nuevo; el llamador envuelve ambos en su transacción).
  - `testing.InMemoryInbox`, `testing.sign(body, secret, msg_id, ts) -> dict[str, str]` (firma con la propia librería para tests).

- [ ] **Step 1: Tests**
  - `test_valid_signature_returns_webhook` (headers de `testing.sign`).
  - `test_tampered_body_raises` (un byte cambiado).
  - `test_old_timestamp_raises` (fuera de la tolerancia de la librería).
  - `test_non_utf8_body_verifies_as_bytes`.
  - `test_process_once_skips_replay`: mismo `id` dos veces → handler llamado una vez, segunda llamada devuelve `None`. **(Review Focus 3)**
- [x] **Step 2:** Run → FAIL. **Step 3:** Implementa. **Step 4:** Run → PASS.
- [ ] **Step 5:** Checks limpios. Commit `feat(webhooks): add standard webhooks verification and idempotent inbox`.

### Task 3: `mandates`

**Files:**
- Create: `packages/mandates/pyproject.toml`, `src/agent_toolkit/mandates/{__init__,models,check,consumption,signing}.py`
- Test: `packages/mandates/tests/test_check.py`, `test_consumption.py`, `test_signing.py`

**Interfaces:**
- Produces:
  - `Amount(minor: int, currency: str)` (propio; no depende de `payments-core`).
  - `AmountRange(min: Amount, max: Amount)`; `Budget(limit: Amount, period: Literal["day", "week", "month", "total"])`; `DailyWindow(start: time, end: time)`.
  - `OpenMandate(id, principal_id, agent_id, purpose, amount_range, budget, receiving_accounts: frozenset[str], allowed_payers: frozenset[str], allowed_rails: frozenset[str], daily_window: DailyWindow | None, tz: str, expires_at: datetime, revoked_at: datetime | None)`; validación: `receiving_accounts` no vacío; monedas iguales en rango y presupuesto; `tz` válida por pertenencia exacta a `zoneinfo.available_timezones()` (dep `tzdata`; sin `localtime`, `Factory` ni `posixrules`; `america/mexico_city` se rechaza en todo SO).
  - `ClosedMandate(open_mandate_id, agent_id: str, kind: Literal["new_charge", "amount_increase"], charge_ref: str, payer: str, receiving_account: str, rail: str, amount: Amount | None, previous_amount: Amount | None, reference: str)`; `charge_ref` es la clave del cobro en `ConsumptionEntry`; propiedad `content_hash: str` (sha256 de JSON canónico, `sort_keys`, sin espacios; cubre `agent_id` y `charge_ref`).
  - `Violation = Literal["expired", "revoked", "mandate_mismatch", "agent_mismatch", "account_not_allowed", "rail_not_allowed", "currency_mismatch", "charge_ref_reused", "previous_amount_mismatch", "amount_out_of_range", "budget_exceeded", "outside_window"]`; `ApprovalReason = Literal["payer_not_allowed", "open_amount"]`.
  - `Verdict(violations: tuple[Violation, ...], approval_reasons: tuple[ApprovalReason, ...])`, propiedad `ok` (ambas vacías).
  - `reserve_amount(closed: ClosedMandate) -> Amount | None` (nuevo → monto; aumento → `amount - previous_amount`; abierto → `None`).
  - `check(closed: ClosedMandate, open: OpenMandate, consumed: Amount, now: datetime, *, ref_state: RefState) -> Verdict` (R14/R18: `consumed` y `ref_state` salen del ledger de este mandato, nunca de argumentos de herramienta; `agent_mismatch` si `closed.agent_id != open.agent_id`; `charge_ref_reused` si un `new_charge` tiene `ref_state.status != "unknown"` (reservada o liberada); `previous_amount_mismatch` si un `amount_increase` tiene `status != "reserved"` o `reserved != closed.previous_amount`. El delta `amount - previous_amount` es confiable solo por esta comparación. El llamador bloquea el mandato desde leer el ledger hasta registrar la reserva: dos `check` sobre el mismo `consumed` pueden pasar ambos. La app genera `charge_ref` del lado del servidor).
  - `ConsumptionEntry(charge_ref: str, kind: Literal["reserve", "release"], amount: Amount, at: datetime)`
  - `consumed_in_period(entries: Iterable[ConsumptionEntry], mandate: OpenMandate, now: datetime) -> Amount`
  - `RefState(status: Literal["unknown", "reserved", "released"], reserved: Amount | None)` (`reserved` es el total sin liberar cuando `status == "reserved"`, `None` en otro caso) y `ref_state(entries: Iterable[ConsumptionEntry], charge_ref: str) -> RefState` (R18; `ValueError` ante un ledger inconsistente, igual que `consumed_in_period`; ambos reciben solo el ledger de este mandato).
  - `Receipt(status: Literal["verified", "executed", "rejected"], verifier: str, reference: str, mandate_hash: str, at: datetime)`
  - Protocols `Signer.sign(payload: bytes) -> bytes`, `Verifier.verify(payload: bytes, signature: bytes) -> bool`; `HmacSha256Signer(key: bytes)` implementa ambos (`hmac.compare_digest`).

- [x] **Step 1: Tests de `check`** (mandato base: rango 100–500000 MXN, presupuesto diario 1000000, cuentas `{"acct-own"}`, pagadores `{"cust-juan"}`, rieles `{"codi_push","codi_link"}`, tz `America/Mexico_City`)
  - `test_in_mandate_charge_is_ok`.
  - `test_foreign_receiving_account_is_violation` → `"account_not_allowed"`.
  - `test_unknown_payer_needs_approval` → `approval_reasons == ("payer_not_allowed",)`, `violations == ()`.
  - `test_open_amount_always_needs_approval` → `"open_amount"` aunque todo lo demás esté bien.
  - `test_increase_consumes_delta_and_total_must_fit_range`: previo 40000, nuevo 50000 → reserva 10000, ok; nuevo 600000 → `"amount_out_of_range"`.
  - `test_budget_counts_requested_amount`: `consumed=990000` + 20000 → `"budget_exceeded"`.
  - `test_expired_revoked_and_mismatch`.
  - `test_outside_daily_window` (21:59 vs 22:01 local con ventana 07:00–22:00).
- [x] **Step 2:** Run → FAIL. **Step 3:** Implementa `models.py` y `check.py`; reúne todas las violaciones (no se detiene en la primera). **Step 4:** Run → PASS.
- [x] **Step 5: Tests de `consumed_in_period`**
  - `test_release_only_subtracts_matching_reserve`: reserve 30000 + release del mismo `charge_ref` → 0; release de un `charge_ref` sin reserve → `ValueError`.
  - `test_daily_period_uses_mandate_timezone`: reserve a las 23:30 del 1 dic local y `now` 00:30 del 2 dic local → 0. **(Review Focus 4)**
  - `test_total_period_ignores_time`.
- [x] **Step 6:** Run → FAIL; implementa (la semana inicia el lunes, el mes en el día 1, ambos en `tz`); Run → PASS.
- [x] **Step 7: Tests de firma**: `content_hash` estable entre instancias iguales y distinto si cambia `amount`; `HmacSha256Signer` verifica su firma y rechaza otra llave. Implementa; Run → PASS.
- [x] **Step 8:** Checks limpios. Commit `feat(mandates): add open/closed mandates, check and consumption`.
- [ ] **Step 9: Entrega a la app.** Merge de las Tareas 1–3 a `main`; tag `v0.1.0a1`. Avisar al equipo de la app (ver "Entregas a la app").

### Task 4: `budgets`

**Files:**
- Create: `packages/budgets/pyproject.toml` (extra `langchain = ["langchain-core>=1,<2"]`), `src/agent_toolkit/budgets/{__init__,core}.py`, `src/agent_toolkit/budgets/adapters/langchain.py`
- Test: `packages/budgets/tests/test_core.py`, `test_langchain_adapter.py`

**Interfaces:**
- Produces:
  - `BudgetLimits(max_iterations: int | None, max_tool_calls: int | None, max_tokens: int | None, max_cost_usd: Decimal | None, max_consecutive_failures: int | None)`
  - `Pricing(input_per_mtok_usd: Decimal, output_per_mtok_usd: Decimal)`
  - `Usage(iterations: int = 0, tool_calls: int = 0, input_tokens: int = 0, output_tokens: int = 0, failed: bool | None = None)`
  - `BudgetState(iterations, tool_calls, input_tokens, output_tokens, cost_usd: Decimal, consecutive_failures)`, `BudgetState.empty()`
  - `ExceededKind = Literal["iterations", "tool_calls", "tokens", "cost", "consecutive_failures"]`
  - `charge(state: BudgetState, usage: Usage, pricing: Pricing | None) -> BudgetState`
  - `check(state: BudgetState, limits: BudgetLimits) -> tuple[ExceededKind, ...]` (vacía = ok)
  - `adapters.langchain.BudgetCallbackHandler(limits, pricing)`: `.state`; cuenta tokens desde `usage_metadata` de la respuesta y llamadas/fallas de herramientas; con `raise_error = True` lanza `BudgetExceededError(kinds)` al exceder y sigue lanzando en todo callback posterior (sticky); la iteración se cobra al iniciar la llamada al LLM; `ValueError` si `max_cost_usd` sin `pricing`.

- [x] **Step 1: Tests de núcleo**: costo = tokens/1e6 × precio, con `Decimal` exacto; `failed=True` incrementa y `failed=False` reinicia `consecutive_failures`; `failed=None` no lo toca; límite alcanzado exactamente (`==`) no excede, `+1` sí; varios tipos excedidos se reportan juntos.
- [x] **Step 2:** Run → FAIL. **Step 3:** Implementa. **Step 4:** Run → PASS.
- [x] **Step 5: Test del adaptador** con `GenericFakeChatModel` (o el fake vigente de `langchain_core`) que reporta `usage_metadata`: tras dos invocaciones el estado suma tokens; con `max_tokens` bajo, la tercera lanza `BudgetExceededError`. Verificar en la doc vigente de `langchain_core` el nombre del hook que recibe `usage_metadata` y que `raise_error` propaga.
- [x] **Step 6:** Run → FAIL; implementa; Run → PASS. Checks limpios (el import de `langchain_core` solo en `adapters/`).
- [x] **Step 7:** Commit `feat(budgets): add run budgets and langchain usage callback`.

### Task 5: `policy-gate` (núcleo)

**Files:**
- Create: `packages/policy-gate/pyproject.toml` (deps workspace `agent-toolkit-mandates`, `agent-toolkit-budgets` vía `[tool.uv.sources] … = { workspace = true }`), `src/agent_toolkit/policy_gate/{__init__,models,decide}.py`
- Test: `packages/policy-gate/tests/test_decide.py`

**Interfaces:**
- Consumes: `mandates.check`, `ClosedMandate`, `OpenMandate`, `Amount`, `Violation`; `budgets.check`, `BudgetState`, `BudgetLimits`.
- Produces:
  - `Risk = Literal["low", "medium", "high"]`; `Effects(reads_sensitive: bool, changes_state: bool, external_comm: bool)`
  - `ToolSpec(name, risk, effects, reversible: bool, financial: bool, args_model: type[BaseModel], to_closed_mandate: Callable[[BaseModel], ClosedMandate] | None)`; validación: `financial=True` exige `to_closed_mandate`.
  - `SessionProfile(agent_id, principal_id, untrusted_input: bool, sensitive_access: bool, state_change: bool)`
  - `ToolCall(id: str, name: str, args: Mapping[str, object])`
  - `GateConfig(on_mandate_violation: Literal["deny", "require_approval"] = "require_approval")`
  - `Allow(kind="allow", args: BaseModel, closed: ClosedMandate | None)`, `RequireApproval(kind="require_approval", reasons: tuple[str, ...], args, closed)`, `Deny(kind="deny", reasons: tuple[str, ...])`; `Decision = Allow | RequireApproval | Deny` (unión discriminada por `kind`).
  - `GateContext(session: SessionProfile, mandate: OpenMandate | None, consumed: Amount | None, budget_state: BudgetState, budget_limits: BudgetLimits, now: datetime, config: GateConfig)`
  - `decide(call: ToolCall, tool: ToolSpec, ctx: GateContext) -> Decision`
- Reglas, en orden (la primera que aplica decide `Deny`; las de aprobación se acumulan):
  1. `call.name != tool.name` → `Deny("tool_mismatch")`.
  2. Args inválidos según `args_model` → `Deny("invalid_args: …")`.
  3. `budgets.check` no vacío → `Deny("budget_exceeded:<kind>")`.
  4. `financial`: sin mandato → `RequireApproval("no_mandate")`. Con mandato: `account_not_allowed`, `mandate_mismatch`, `currency_mismatch`, `expired` o `revoked` → `Deny` siempre (no configurable). Otras violaciones → según `config`. `approval_reasons` → `RequireApproval`.
  5. No financiero con `risk="high"` → `RequireApproval("high_risk")`.
  6. Sesión con las tres banderas Rule of Two y `effects.changes_state`, salvo herramienta financiera con `Verdict.ok` → `RequireApproval("rule_of_two")`.
  7. Si no → `Allow`.

- [ ] **Step 1: Tests** (una por regla, más):
  - `test_foreign_account_is_denied_even_if_config_says_approval` (7.3(c)).
  - `test_open_amount_qr_requires_approval`.
  - `test_violation_follows_config` (`amount_out_of_range` con ambas configuraciones).
  - `test_rule_of_two_allows_verified_mandate_but_not_unmandated_state_change`.
  - `test_allow_returns_parsed_args_and_closed_mandate`.
  - `test_low_risk_read_is_allowed_without_mandate`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implementa `decide` como función pura. **Step 4:** Run → PASS.
- [ ] **Step 5:** Checks limpios; `lint-imports` confirma que el núcleo no importa `langgraph`. Commit `feat(policy-gate): add deterministic tool-call decisions`.

### Task 6: `policy-gate` (adaptador LangGraph)

**Files:**
- Create: `src/agent_toolkit/policy_gate/adapters/langgraph.py`; extra `langgraph = ["langgraph>=1.2,<2"]`
- Test: `packages/policy-gate/tests/test_langgraph_adapter.py`

**Interfaces:**
- Consumes: `decide`, `Decision`, `ToolSpec`, `ToolCall`, `GateContext` (Tarea 5).
- Produces:
  - `ApprovalRequest(call_id: str, tool: str, args: dict[str, object], reasons: tuple[str, ...], closed_mandate_hash: str | None, request_hash: str)` (`request_hash` = sha256 de `call_id`, `tool`, args canónicos y `closed_mandate_hash`).
  - `ApprovalDecision(request_hash: str, approved: bool, approver_id: str, approval_id: str, at: datetime)`
  - `GatedToolNode(tools: Sequence[BaseTool], specs: Mapping[str, ToolSpec], context: Callable[[StateT], GateContext], on_decision: Callable[[ToolCall, Decision, ApprovalDecision | None], None] | None = None)`, invocable como nodo; lee los `tool_calls` del último `AIMessage` y devuelve `ToolMessage`s.
- Algoritmo (fija el comportamiento de Review Focus 1 y 2):
  1. Decide **todas** las llamadas antes de ejecutar cualquiera.
  2. Si hay `RequireApproval`, un solo `interrupt([ApprovalRequest, …])` antes de ejecutar nada. El valor de reanudación es `dict[call_id, ApprovalDecision]`.
  3. Al reanudar, el nodo se re-ejecuta desde el inicio y `decide` se repite (es puro): una decisión se acepta solo si su `request_hash` coincide con el recalculado. Si falta o no coincide, esa llamada se rechaza.
  4. Ejecuta las `Allow` y las aprobadas; `Deny` y las rechazadas producen `ToolMessage(status="error", content=<razón accionable>)`.
  5. `on_decision` se llama una vez por llamada con su resultado final (la app lo conecta a `audit-log`).

- [ ] **Step 1: Tests** (grafo mínimo `StateGraph` + `InMemorySaver`, herramientas fake que cuentan invocaciones):
  - `test_allowed_call_executes`.
  - `test_denied_call_returns_error_message_and_does_not_execute`.
  - `test_approval_interrupts_then_executes_on_approved_resume` (`Command(resume={call_id: decision})`).
  - `test_resume_executes_each_tool_once`: una llamada `Allow` + una `RequireApproval` en el mismo turno; tras reanudar aprobada, cada contador == 1. **(Review Focus 1)**
  - `test_resume_with_mismatched_hash_rejects`. **(Review Focus 2)**
  - `test_on_decision_called_once_per_call`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implementa siguiendo la doc vigente de `interrupt`/`Command` (en langgraph 1.2, `interrupt(value, *, response_schema=…)` puede validar la reanudación). **Step 4:** Run → PASS.
- [ ] **Step 5:** Checks limpios. Commit `feat(policy-gate): add langgraph gated tool node with interrupt approvals`.

### Task 7: `audit-log`

**Files:**
- Create: `packages/audit-log/pyproject.toml` (extra `postgres = ["psycopg[binary]>=3.3,<4"]`), `src/agent_toolkit/audit_log/{__init__,core,testing}.py`, `src/agent_toolkit/audit_log/adapters/postgres.py`, `src/agent_toolkit/audit_log/adapters/schema.sql`
- Test: `packages/audit-log/tests/test_core.py`, `test_postgres.py` (marcador `postgres`)

**Interfaces:**
- Produces:
  - `ActorKind = Literal["agent", "human", "system"]`
  - `AuditEvent(seq: int, actor_id, actor_kind, action, subject, payload_hash: str, decision: str | None, at: datetime, prev_hash: str, hash: str)`; génesis `prev_hash = "0" * 64`; `hash` = sha256 del JSON canónico de los demás campos.
  - `hash_payload(payload: bytes) -> str`
  - `next_event(prev: AuditEvent | None, *, actor_id, actor_kind, action, subject, payload_hash, decision, at) -> AuditEvent`
  - Protocol `AuditStore`: `last() -> AuditEvent | None`, `append(event) -> None` (lanza `ChainConflict` si `event.prev_hash` ≠ hash del último), `iter_all() -> Iterator[AuditEvent]`
  - `append(store: AuditStore, **fields) -> AuditEvent`
  - `ChainReport(ok: bool, first_bad_seq: int | None)`; `verify_chain(events: Iterable[AuditEvent]) -> ChainReport`
  - `testing.InMemoryAuditStore`; `adapters.postgres.PostgresAuditStore(conn)`; `schema.sql` con tabla `audit_events` (`seq bigint primary key`, `prev_hash text unique not null`, `hash text unique not null`, …) y comentario con el `REVOKE UPDATE, DELETE, TRUNCATE ON audit_events FROM <rol_app>` que aplica la app.

- [ ] **Step 1: Tests de núcleo**: cadena de 3 eventos verifica; alterar `decision` del 2.º → `first_bad_seq == 2`; reordenar → falla; `payload_hash` de un payload con teléfono no contiene el teléfono.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implementa. **Step 4:** Run → PASS.
- [ ] **Step 5: Tests Postgres** (usan `DATABASE_URL`; se saltan con mensaje si falta):
  - `test_append_and_verify_roundtrip`.
  - `test_concurrent_append_conflicts`: dos conexiones leen el mismo `last()` y anexan; la segunda lanza `ChainConflict`. **(Review Focus 5)**
  - `test_app_role_cannot_update_or_delete`: crea un rol, aplica el `REVOKE`, `UPDATE` falla con `InsufficientPrivilege`.
- [ ] **Step 6:** Run (`docker run -e POSTGRES_PASSWORD=… -p 5432:5432 postgres:17` en local; servicio en CI) → FAIL; implementa; Run → PASS.
- [ ] **Step 7:** Checks limpios. Commit `feat(audit-log): add hash-chained audit events with postgres store`.

### Task 8: `evals`

Herramienta base [decisión del plan, dentro del criterio de 7.5]: runner propio y mínimo en Python puro con un punto de entrada para pytest. Inspect AI y DeepEval se descartan para este corte porque están pensados para evaluar modelos o métricas LLM-judge, no estados finales de un agente con un entorno nuevo en cada intento. Si al implementar resulta que Inspect cubre k intentos aislados con grader de estado final sin fricción, se cambia y se anota.

**Files:**
- Create: `packages/evals/pyproject.toml`, `src/agent_toolkit/evals/{__init__,metrics,runner,pytest_plugin}.py`
- Test: `packages/evals/tests/test_metrics.py`, `test_runner.py`

**Interfaces:**
- Produces:
  - `pass_at_k(n: int, c: int, k: int) -> float` = `1 - C(n-c, k) / C(n, k)`; `pass_hat_k(n: int, c: int, k: int) -> float` = `C(c, k) / C(n, k)` (`math.comb`).
  - `Grade(passed: bool, reason: str)`
  - `EvalTask(id: str, input: Mapping[str, object], tags: frozenset[str], grade: Callable[[FinalStateT], Grade])`
  - `run_suite(tasks: Sequence[EvalTask], attempt: Callable[[EvalTask, int], FinalStateT], n: int, transcripts_dir: Path | None) -> SuiteReport`; `attempt` crea un entorno nuevo por intento (el llamador lo garantiza; el runner nunca reutiliza estado).
  - `TaskResult(task_id, n, c, pass_at: dict[int, float], pass_hat: dict[int, float], failures: tuple[Grade, ...])`; `SuiteReport(results: tuple[TaskResult, ...])` con `summary() -> str` y `to_json() -> str`.
  - Plugin pytest `--eval-n` (por defecto 5) para correr suites como tests.

- [ ] **Step 1: Tests de métricas**: `pass_at_k(5, 5, 3) == 1.0`; `pass_hat_k(5, 4, 5) == 0.0`; `pass_hat_k(5, 4, 1) == 0.8`; `pass_at_k(5, 0, 1) == 0.0`; `k > n` → `ValueError`.
- [ ] **Step 2: Tests del runner**: tarea que pasa en intentos pares → `c` correcto; una excepción dentro de `attempt` cuenta como fallo con su razón, no aborta la suite; transcripciones JSONL escritas por intento cuando se da `transcripts_dir`.
- [ ] **Step 3:** Run → FAIL. **Step 4:** Implementa. **Step 5:** Run → PASS.
- [ ] **Step 6:** Checks limpios. Commit `feat(evals): add pass@k and pass^k runner`.

### Task 9: Congelar v0.1.0

**Files:**
- Create/Modify: `packages/*/README.md` (propósito, contrato, ejemplo mínimo, referente que lo justifica, límites), `README.md` raíz (mapa de paquetes, regla de dependencias, cómo instalar por git tag), `CHANGELOG.md`

- [ ] **Step 1:** Ejemplo de cada README como doctest o test (`pytest --doctest-glob="*.md"`). Run → PASS.
- [ ] **Step 2:** `uv build --all-packages` genera 7 wheels; instalar cada wheel en un venv limpio y `python -c "import agent_toolkit.<pkg>"` sin extras → OK (prueba que el núcleo no requiere extras).
- [ ] **Step 3:** Prueba de consumo como la app: proyecto temporal fuera del repo con dependencia `agent-toolkit-policy-gate[langgraph] @ git+https://github.com/jovandyaz/agent-toolkit@v0.1.0rc1#subdirectory=packages/policy-gate`; `uv sync` resuelve también `mandates` y `budgets`. Si la resolución de dependencias entre paquetes por git falla, documentar en el README raíz la forma que sí funciona.
- [ ] **Step 4:** Commit `docs: add package readmes and changelog for v0.1.0`; merge; tag `v0.1.0`.

## Entregas a la app

| Tag | Cuándo (calendario del spec) | Contiene | Desbloquea en la app |
|---|---|---|---|
| `v0.1.0a1` | fin de semana 1 (22 nov) | `payments-core`, `webhooks`, `mandates` | conector pey, endpoint de webhooks, conciliación, Trusted Surface de mandatos |
| `v0.1.0a2` | fin de semana 2 (29 nov) | + `budgets`, `policy-gate` (+ adaptador), `audit-log` | grafo del agente con aprobaciones |
| `v0.1.0a3` | fin de semana 3 (6 dic) | + `evals` | suites del hackatón (viven en la app) |
| `v0.1.0` | 7–9 dic | interfaces congeladas | entrega |

Mientras tanto, el equipo de la app puede empezar con lo que no depende del toolkit: segunda API key y webhook, cliente HTTP de pey, primer cobro de $0.01, esquema de base de datos, UI y Trusted Surface en maqueta.
