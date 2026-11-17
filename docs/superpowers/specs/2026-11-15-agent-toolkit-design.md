# Diseño: agent-toolkit y su primer consumidor (Hackatón México)

- **Fecha:** 2026-11-15 (redactado el 2026-11-16). **Estado:** propuesta revisada el 2026-11-16 con las respuestas del usuario (presupuesto de mandatos, un comercio, canal de B, cuenta pey); pendiente de aprobación final.
- **Marcas de evidencia:** **[V]** verificado contra la fuente citada en la sesión de diseño; **[I]** inferencia o decisión de diseño; **[P]** pendiente de verificar antes de implementar; **[Decisión mía]** elección del diseñador que el usuario puede corregir.

## 1. Intención

El usuario quiere una base propia y reutilizable para construir agentes de IA, sin depender de un proveedor de modelos, de un framework ni de una nube, y sin reemplazar los frameworks que rigen el mercado (LangGraph, LangChain, CrewAI, SDKs de los proveedores). La base son piezas que se enchufan a esos frameworks.

Lo que motiva el momento: el **Hackatón México** (organizado por pey, AI Builders México y SpaceXAI; https://hackatonmexico.com/reglas). Su proyecto será el **primer consumidor real** de la base. El segundo será la pre-revisión de manifestaciones CDMX (`designs/prerevision-cdmx/arquitectura-prerevision-cdmx.md`, Python + LangGraph). La base es el entregable principal; el proyecto del hackatón es como se valida.

Las piezas se diseñan desde los referentes de la industria (sección 4). El código previo del usuario (pialy, Knowtis, llm-router) sirve solo como **casos de aceptación**, no como modelo a copiar.

## 2. Objetivos, no-objetivos y anti-objetivos

**Objetivos**
1. Un repo `agent-toolkit` con paquetes Python independientes de framework, cada uno justificado por un referente o estándar público.
2. Una app de hackatón que los consuma y se entregue antes del **12 dic 2026 23:59** (cierre de Entregas, sección 10 de las reglas) [V].
3. Que CDMX pueda consumir `budgets` y `audit-log` sin cambiarlos. CDMX no mueve dinero ni tiene herramientas financieras, y su diseño ya trae su propio harness de evaluación, así que `mandates`, `policy-gate`, `payments-core` y `webhooks` se justifican por este agente y por futuros agentes fintech, no por CDMX. `evals` se diseña para el hackatón [Decisión mía].

**No-objetivos**
- Un framework de agentes propio o un runtime que compita con LangGraph.
- Implementar el protocolo AP2 completo (SD-JWT, Credential Provider, etc.). Se adopta su modelo de mandato, no su protocolo.
- Guardrails de entrada propios: se adopta LlamaFirewall u otro existente cuando haga falta.
- Una plataforma de evals propia.
- Multi-agente, A2A, TypeScript.
- Publicar en PyPI antes de que CDMX consuma la base.

**Anti-objetivos** (fallaría aunque funcione)
- Un agente que pueda cobrar algo que el dueño del negocio no autorizó.
- Mensajes de cobro a personas que no aceptaron recibirlos (regla 7.3(b) [V]).
- Paquetes que solo sirven a esta app (acoplados a pey o a su dominio).
- Una integración CoDi "decorativa, simulada o solo prometida" (regla 6.2 [V]).

## 3. Restricciones

**Del hackatón** (https://hackatonmexico.com/reglas, actualizado 4 nov 2026) [V]
- Postulación hasta el 10 dic; entrega hasta el 12 dic; un proyecto postulado puede seguir construyéndose hasta el cierre (5.4, 5.5).
- Debe generar al menos un cobro CoDi con el API de pey y reaccionar a su resultado (6.2), y funcionar en producción (6.3).
- La integración con CoDi/pey debe ser nueva y construida desde el 14 nov (6.5, 9.1, 9.7). El código previo se permite si se declara (9.2).
- No se permite entregar un proyecto sustancialmente igual a otro (4.2, 10.9): debe diferenciarse claramente de pialy.
- 7.3: prohibido engañar al pagador, enviar mensajes de cobro masivos no solicitados y cobrar a favor de cuentas que no sean del titular de las credenciales.
- 7.4: prohibido revender el API u ofrecerlo como servicio a terceros; se permiten "puntos de venta, sistemas de gestión, comercios electrónicos, conciliadores".
- 16.3: el equipo es responsable de los datos personales de los pagadores (aviso de privacidad, consentimientos, LFPDPPP). 16.4: demos y repos sin datos reales de terceros.
- Entregables (10.1): descripción ≤500 palabras, demo pública, video ≤3 min con el flujo completo de un cobro CoDi, declaración de componentes previos, instrucciones de prueba ≤1000 palabras. Repo opcional.
- Evaluación (11.2): CoDi y API de pey 30 %, impacto 25 %, ejecución técnica 15 %, experiencia 15 %, originalidad 15 %.

**Del API de pey** (https://docs.pey.mx, OpenAPI 2026-10-20) [V]
- `POST /codi/pn` (push al celular; expiración ≥360 s; monto >0 y ≤1,000,000 MXN), `POST /codi/qr` (monto 0 = monto abierto), `POST /payment-link/create`, `POST /payment-link/update` (solo el monto, solo si está `pending`), `GET /transaction/...`, `POST /bank-account` (alta de CLABE, hasta 5 días hábiles).
- `bankAccountId` opcional por cobro (cuenta `ACTIVE` del titular).
- Webhooks con Standard Webhooks; uno por API key; eventos `codi.qr.payment_received`, `codi.pn.payment_received`, `payment_link.status_updated` (estados `pending`, `processing`, `partial`, `paid`), `bank.account.updated`.
- No existen: cancelar QR/push, reembolsos, pagos salientes, cobros recurrentes, sandbox. Se prueba en producción con $0.01. Gratis "por el momento".

**Técnicas** [Decisión mía salvo lo indicado]
- Python 3.12, `uv` (workspace en el toolkit; dependencias por ruta editable en desarrollo, por git tag en deploy; CI de la app con `no-sources`) [V: https://docs.astral.sh/uv/concepts/projects/dependencies/, https://docs.astral.sh/uv/reference/settings/].
- Framework de la app: **LangGraph** (decidido por el usuario).
- Modelo detrás de la interfaz de chat models de LangChain, configurable; se elige con evals (guía de OpenAI, p. 7 [V]).
- Pydantic v2 para modelos en fronteras de confianza (el SDK de AP2 también usa Pydantic [V]).

## 4. Referentes (fuentes primarias leídas)

| Referente | Principio que se adopta |
|---|---|
| Anthropic, *Building effective agents* (https://www.anthropic.com/research/building-effective-agents) | Distinguir workflows de agentes; la solución más simple; condiciones de parada; checkpoints humanos; empezar con APIs directas porque los frameworks "obscure the underlying prompts". |
| Anthropic, *Writing effective tools for agents* (https://www.anthropic.com/engineering/writing-tools-for-agents) | Pocas herramientas consolidadas, con namespace, respuestas concisas, errores accionables, "poka-yoke". |
| Anthropic, *Demystifying evals for AI agents* (https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) | Calificar el estado final; graders deterministas; pass^k; 20 a 50 tareas de fallas reales; leer transcripciones. |
| Anthropic, *Effective context engineering* (https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | Contexto finito, carga just-in-time, compactación. Guía para la app. |
| OpenAI, *A practical guide to building agents* (https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) | Un solo agente primero (p. 13); evals para la línea base; guardrails en capas; **tool safeguards** con riesgo low/medium/high por lectura/escritura, reversibilidad, permisos e impacto financiero; intervención humana por umbral de fallas o acciones de alto riesgo como "making payments" (p. 31). |
| AP2 v0.2 (https://ap2-protocol.org/ap2/specification/) | Mandatos *open* (restricciones: `amount_range`, `budget`, `allowed_payees`, `execution_date`) y *closed* (operación concreta); recibos; aprobación en una **Trusted Surface determinista**; el agente no firma mandatos abiertos; evidencia no repudiable. Montos en unidades menores ISO 4217. |
| OWASP Top 10 for Agentic Applications 2026 (PDF del OWASP GenAI Security Project, dic 2025) | ASI02: "Intent Gate" (punto de enforcement antes de ejecutar; salida del LLM no confiable), presupuestos de costo/tasa/tokens, logs inmutables. ASI01/ASI09: confirmación humana de acciones de alto impacto. ASI03: permisos acotados a la tarea. ASI08: cuotas y circuit breakers. |
| Meta, *Agents Rule of Two* (https://ai.meta.com/blog/practical-ai-agent-security/) | En una sesión, no más de dos de: [A] entradas no confiables, [B] datos o sistemas sensibles, [C] cambiar estado o comunicarse hacia afuera. Con las tres, supervisión humana u otra validación confiable. |
| Meta, LlamaFirewall (https://meta-llama.github.io/PurpleLlama/LlamaFirewall/) | Guardrail existente en Python para adoptar, no reconstruir. |
| Microsoft AI Red Team, *Taxonomy of failure modes in agentic AI systems* (PDF, 2025) | Identidad única por agente y auditoría de "qué componente hizo qué"; control determinista del flujo; UX de consentimiento informado; logging; controles XPIA. |
| Google Cloud, patrones agénticos (https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system) | Aprobación humana para "approving a large financial transaction"; mínimo privilegio; trazar cada acción. |
| MCP 2026-07-28, OTel GenAI semconv | Contratos de herramientas y de trazas. |

## 5. Enfoques considerados

**Estructura del código**
| Enfoque | Resultado |
|---|---|
| Monorepo con app y paquetes juntos | Descartado: ata los paquetes a la app y mezcla visibilidad (el conector pey es privado). |
| **Repo `agent-toolkit` solo con paquetes + app en repo aparte** | **Elegido por el usuario.** |
| Plantilla/scaffold (copier) | Descartado como base: las mejoras no llegan a proyectos generados. Puede añadirse después. |

**Orden de construcción**
| Enfoque | Resultado |
|---|---|
| Paquetes primero, app después | Descartado: diseño sin consumidor; consume el tiempo del hackatón. |
| **Cada pieza nace con su consumidor; lo genérico va directo al toolkit, lo dudoso nace en la app y se mueve cuando CDMX lo pida** | **Elegido.** |

**Framework de la app**: LangGraph (elegido por el usuario) frente a Pydantic AI, OpenAI Agents SDK, Microsoft Agent Framework, Google ADK, Strands y CrewAI. Razones: aprobación con `interrupt()` y esperas largas con checkpointer en Postgres sin servidores extra; ya lo usan melel y CDMX. Pydantic AI quedó como segunda opción (requiere motor durable externo).

**Idea del proyecto**: A (copiloto de cobros para el comerciante) con B (seguimiento de pendientes) como función opt-in, elegido por el usuario frente a B como foco y frente a un vendedor conversacional (C, parecido a pialy y a la idea 01 de los organizadores).

## 6. Arquitectura

```
agent-toolkit/                     (repo 1: solo paquetes, uv workspace)
  packages/
    mandates/        núcleo puro
    policy-gate/     núcleo puro + extra [langgraph]
    budgets/         núcleo puro + extra [langchain]
    audit-log/       núcleo puro + extra [postgres]
    evals/           núcleo puro, encima de una herramienta existente
    payments-core/   núcleo puro (puertos, sin I/O)
    webhooks/        núcleo puro + verificación Standard Webhooks

hackathon-app/                     (repo 2: privado mientras legal@pey.mx no confirme 7.4)
  conector pey, herramientas del agente, grafo LangGraph, UI web, Postgres
```

**Reglas de dependencia** [Decisión mía]
- El núcleo de cada paquete no importa LangGraph, LangChain, pey ni ningún SDK de proveedor. Se vigila con una regla de lint de imports que falla en CI (comprobada con una importación prohibida antes de dar el PR por hecho).
- Los adaptadores van como extras opcionales (`pip install policy-gate[langgraph]`).
- Ningún paquete depende de otro salvo `policy-gate` → `mandates` y `budgets`; `payments-core` no depende de ninguno.
- Nombres de distribución e import provisionales (`agent-toolkit-*`, `agent_toolkit.*`); verificar disponibilidad en PyPI antes de publicar [P].

## 7. Paquetes del primer corte

Para cada uno: responsabilidad, contrato mínimo, referente y límites. Los nombres de funciones son ilustrativos; el contrato es lo que importa.

### 7.1 `mandates`
- **Responsabilidad:** representar qué autorizó una persona que el agente puede hacer y comprobar si una operación concreta cabe ahí.
- **Adaptación de AP2 [I]:** en AP2 el usuario autoriza al agente a *pagar*. Aquí el dueño autoriza al agente a *emitir solicitudes de cobro* en su nombre; cada pagador aprueba aparte en su app bancaria (CoDi). El modelo de mandato es el mismo; el sentido del dinero es inverso. Se alinean nombres de campos con AP2 donde aplique.
- **Contrato:**
  - `OpenMandate`: `id`, `principal_id` (quien autoriza), `agent_id`, `purpose`, restricciones (`amount_range` en unidades menores + moneda, `budget` total y periodo, `receiving_accounts`, `allowed_payers`, `allowed_rails`, `execution_window`), `expires_at`, `revoked_at`.
    - `receiving_accounts` (obligatorio): solo cuentas del titular de las credenciales (regla 7.3(c)). Sustituye al `allowed_payees` de AP2, cuyo sentido se invierte aquí.
    - `allowed_payers` (opcional): a quién puede cobrar el agente sin pedir aprobación (p. ej., clientes ya registrados). Vacío = cualquier pagador exige aprobación.
  - **Consumo del presupuesto** (aprobado por el usuario):
    - Cuenta lo **solicitado**, no lo pagado: cada cobro emitido aparta su monto.
    - Un cobro expirado sin pago libera su monto solo cuando una consulta de estado al proveedor lo confirma (pey no permite cancelar).
    - Un QR de monto abierto no se puede acotar: siempre `require_approval`.
    - Aumentar el monto de un link pendiente consume el delta, y el nuevo total debe caber en `amount_range`.
    - Pagos parciales, tardíos o duplicados no consumen: son entradas, no solicitudes.
  - `ClosedMandate`: operación concreta (`payee`, `amount`, `rail`, `reference`), `open_mandate_id`, hash del contenido.
  - `Receipt`: resultado de verificar o ejecutar (`status`, `verifier`, `reference`, `at`).
  - `check(closed, open, consumed) -> Verdict` puro: lista de violaciones o aprobado. `consumed` lo aporta el llamador.
  - Firma opcional detrás de un puerto (`Signer`/`Verifier`); el MVP firma con una clave del servidor. Compatibilidad SD-JWT de AP2: fuera de alcance.
- **Límites:** no persiste, no ejecuta, no sabe de pey.

### 7.2 `policy-gate`
- **Responsabilidad:** el "Intent Gate" de OWASP ASI02. Recibe cada llamada a herramienta que propone el modelo y decide de forma determinista: `allow`, `require_approval(reason)` o `deny(reason)`.
- **Contrato:**
  - `ToolSpec`: `name`, `risk` (low/medium/high), `effects` (`reads_sensitive`, `changes_state`, `external_comm`), `reversible`, `financial`. Criterios de OpenAI (p. 26).
  - `SessionProfile`: banderas Rule of Two `untrusted_input`, `sensitive_access`, `state_change`, más `agent_id` y `principal_id`.
  - `decide(call, tool, session, mandate | None, budget_state) -> Decision`. Valida argumentos contra el esquema, el mandato (`mandates.check`) y el presupuesto (`budgets`). La salida del modelo se trata como no confiable.
  - Reglas base: herramienta `financial` o `risk=high` sin mandato abierto válido → `require_approval`; violación del mandato → `deny` o `require_approval` según configuración; sesión con las tres banderas Rule of Two → toda acción con `changes_state` exige mandato válido verificado de forma determinista o aprobación humana [I: se interpreta el mandato verificado como la "reliable means of validation" que menciona Meta].
- **Adaptador `[langgraph]`:** envuelve el nodo de herramientas; las llamadas con `require_approval` → un solo `interrupt(payload)` por turno, con `payload` en JSON plano: una lista de `ApprovalRequest.model_dump(mode="json")`, que incluye `closed` (el cargo tal como se ejecutará) y el `request_hash` que ata la aprobación al hilo, al mensaje, a la llamada y a ese cargo. Se retoma con `Command(resume={call_id: ApprovalDecision.model_dump(mode="json")})`, también JSON, porque el checkpoint guarda el valor de reanudación. La aprobación nunca se toma del texto del chat: viene de la Trusted Surface (sección 8).
- **Límites:** no detecta inyección (eso es de LlamaFirewall u otro guardrail); no persiste.

### 7.3 `budgets`
- **Responsabilidad:** topes por turno y por corrida: iteraciones, llamadas a herramientas, tokens, costo en USD y fallas consecutivas.
- **Contrato:** `BudgetLimits`, `BudgetState`, `charge(usage) -> BudgetState`, `check(state) -> ok | exceeded(kind)`. El precio por token lo inyecta el llamador (no hay catálogo en este corte).
- **Adaptador `[langchain]`:** callback que lee el uso de tokens reportado por el chat model.
- **Disparador de escalado:** `exceeded(consecutive_failures)` → la app transfiere a una persona (OpenAI p. 31).
- **Límites:** no reemplaza el `recursion_limit` nativo de LangGraph; lo complementa con costo y fallas.

### 7.4 `audit-log`
- **Responsabilidad:** registro append-only de decisiones del gate, mandatos, aprobaciones, recibos y efectos externos, con identidad del actor (agente o persona). Microsoft (identidad + auditoría), OWASP ASI02/ASI09 (logs inmutables), AP2 (evidencia).
- **Contrato:** `AuditEvent` (`actor_id`, `actor_kind`, `action`, `subject`, `payload_hash`, `decision`, `at`, `prev_hash`, `hash`); `append(event)`; `verify_chain(events)`. Puerto de almacenamiento.
- **Adaptador `[postgres]`:** tabla append-only; el rol de la app no tiene `UPDATE` ni `DELETE` sobre ella.
- **Límites:** no es telemetría (eso son trazas OTel/Langfuse); no guarda PII en claro, solo hashes o referencias.

### 7.5 `evals`
- **Responsabilidad:** formato de tareas, graders deterministas sobre el estado final y un runner que reporta pass@k y **pass^k** (Anthropic).
- **Herramienta base:** se adopta una existente (candidatas: pytest con un plugin mínimo propio, Inspect AI, DeepEval, datasets de Langfuse). Se elige en el plan con un criterio: ejecutar k intentos aislados y calificar contra el estado final [P].
- **Suites mínimas del hackatón:** 20 a 50 tareas: cobros dentro y fuera de mandato, montos ambiguos, cliente equivocado, aumento de link que se sale de `amount_range`, QR de monto abierto, inyección en campos que vienen de fuera (nombres/conceptos de clientes, datos de webhooks de pey), intentos de cobrar a una cuenta ajena, reintentos ante fallas de pey, cobro expirado que libera presupuesto. Casos positivos y negativos.
- **Límites:** no es plataforma ni dashboard.

### 7.6 `payments-core`
- **Responsabilidad:** dominio de cobros para rieles push o request-to-pay (CoDi; por extensión Pix, UPI), sin I/O.
- **Contrato:** `Money` (entero en unidades menores + moneda), `PaymentIntent` (referencia, monto, clave de idempotencia, máquina de estados), `Attempt` (riel, ids del proveedor, expiración), `apply_provider_payment` idempotente, `reconcile_attempt`. Puertos: `RailConnector` (solicitar, consultar), `Repository`, `UnitOfWork`.
- **Políticas** (casos de aceptación tomados de pialy, `README.md` y `packages/payments/test/` de `jovandyaz/pialy`): pago dentro de vigencia → confirmado; pago tardío → confirmado y marcado; pago duplicado o sin intent → reembolso marcado (pey no reembolsa por API); un POST al proveedor solo se reintenta si la respuesta prueba que no se procesó.
- **Límites:** no conoce pey (el conector vive en la app), no conoce reservas ni agenda.

### 7.7 `webhooks`
- **Responsabilidad:** verificar firmas Standard Webhooks sobre el cuerpo crudo y registrar cada `webhook-id` en un inbox dentro de la misma transacción que su efecto.
- **Contrato:** `verify(body, headers, secret)` (se adopta la librería `standardwebhooks`, la que usa el ejemplo Python de pey [V: https://docs.pey.mx/docs/webhooks/introduction]); puerto `Inbox` con `seen(id)`/`record(id)` transaccional.

### 7.8 Fuera del primer corte
- `telemetry`: en la app se usa el callback de Langfuse para LangChain/LangGraph; se extrae como paquete OTel GenAI con redacción cuando CDMX lo necesite.
- Guard de inyección: adoptar LlamaFirewall (requiere acceso a modelos Llama en Hugging Face; AlignmentCheck es experimental) cuando haya consumidor.
- Catálogo de modelos y precios.

## 8. App del hackatón (primer consumidor)

**Producto:** copiloto de cobros para un pequeño negocio. El dueño escribe en lenguaje natural ("cóbrale 350 a Juan por el corte", "súmale 100 al link de Ana", "¿quién no ha pagado?", "cierra el día") y el agente opera pey. Función opt-in: el agente da seguimiento a cobros pendientes solo con clientes que aceptaron recibir recordatorios.

**Herramientas** (consolidadas, con namespace, errores accionables; Anthropic) [Decisión mía, ajustable en el plan]
- `charges_create` (elige push, QR o link según el caso), `charges_update_amount` (solo links pendientes), `charges_list`/`charges_status`, `day_reconcile`, `followups_schedule` (B).
- Todas pasan por `policy-gate`. `charges_create` y `charges_update_amount` son `financial` y `changes_state`.

**Mandatos**
- El dueño crea un mandato abierto en la Trusted Surface: tope por cobro, presupuesto diario, rieles permitidos, vigencia. Dentro del mandato el agente cobra sin pedir nada; fuera, pide aprobación en la Trusted Surface.
- Todo cobro va a cuentas del titular de las credenciales (regla 7.3(c)), vía `receiving_accounts`. Semántica de consumo en 7.1.

**Un solo comercio** (aprobado por el usuario): la app del hackatón opera un negocio, pero todas las tablas llevan `principal_id` para que pasar a varios no obligue a rehacer el modelo. Multi-comercio (una key y un webhook por comercio, registro y aislamiento de datos) solo si legal@pey.mx lo autoriza antes del 30 nov.

**Cuenta pey**: los detalles operativos (API keys, webhooks, CLABEs y situación fiscal del titular) viven en el repo privado de la app, no en este repo público. Regla que importa aquí: cada app usa su propia API key con su propio webhook, y el demo cobra solo a cuentas propias del titular (7.3(c), 6.3).

**Trusted Surface** (AP2, Microsoft UX de consentimiento)
- Pantalla web determinista, separada del chat, que muestra el mandato o el cobro pendiente tal como se ejecutará y registra la aprobación en `audit-log`. El chat nunca aprueba.

**Perfiles Rule of Two** [I]
- Sesión del dueño: entrada autenticada del dueño (confiable), [B] datos del negocio, [C] cobros. Sin [A].
- Datos que vienen de fuera (nombres y conceptos de clientes, payloads de webhooks) se tratan como no confiables aunque la sesión sea del dueño; el gate valida argumentos de forma determinista.
- (Extra, solo si sobra tiempo) Sesión del deudor: [A] mensajes del deudor y [C] acotado a su propio cobro; sin [B]. Cualquier acción fuera del mandato escala al dueño.

**Seguimiento opt-in (B)** (aprobado por el usuario)
- v1: el recordatorio **es un nuevo push o link CoDi** que llega a la app bancaria del cliente. No hay conversación con el deudor: usa CoDi (30 % de la evaluación), no requiere WhatsApp y no deja entrar texto del deudor.
- Solo a clientes con consentimiento capturado y guardado antes del primer recordatorio. Cadencia y horarios configurables (fuentes no oficiales citan 7–22 h para cobranza comercial bajo PROFECO; verificar en DOF [P]). Cada recordatorio es un cobro y pasa por el mandato.
- Negociación con el deudor por chat (pago parcial, ajuste de monto): extra si sobra tiempo.
- B es lo primero que se recorta si el calendario se atrasa.

**Stack** [Decisión mía]: FastAPI + LangGraph (checkpointer Postgres) + Postgres + UI web simple + Langfuse. Canal del demo: chat web; Telegram o WhatsApp solo si sobra tiempo.

**Diferencia con pialy** (regla 4.2): pialy es agenda con anticipo para el cliente final; esta app es un agente para el dueño que opera cobros y seguimiento. No comparte producto ni pantallas.

## 9. Calendario (4 semanas)

| Semana | Toolkit | App | Hackatón |
|---|---|---|---|
| 16–22 nov | Repo, workspace, lint de imports; `payments-core` y `webhooks` con casos de aceptación; `mandates` | Repo, segunda API key + webhook propio, conector pey, primer cobro real de $0.01 contra una CLABE pre-autorizada | **Primer día: alta de la CLABE propia** (hasta 5 días hábiles); **postular el proyecto**; consulta a legal@pey.mx (7.3/7.4, multi-comercio y 8.2) |
| 23–29 nov | `policy-gate` (+ adaptador LangGraph), `budgets`, `audit-log` | Grafo del agente, herramientas, Trusted Surface, conciliación por webhook + sweeper | — |
| 30 nov–6 dic | `evals` y suites (20–50 tareas, pass^k) | Seguimiento opt-in (B), resumen diario | Revisar bounties (publicados a más tardar el 30 nov) |
| 7–12 dic | Congelar interfaces, README por paquete | Deploy en producción, endurecimiento, video ≤3 min, textos de entrega, declaración de componentes previos | Entregar antes del 12 dic 23:59 |

## 10. Riesgos y preguntas abiertas

**Riesgos**
- Alcance: 7 paquetes + app en 4 semanas. Capacidad: los paquetes los construye una persona; la app, un equipo de 3–4, con tareas asignadas en un plan propio. Orden de recorte si se atrasa: (1) B completo, (2) `evals` al mínimo (≈20 tareas), (3) `audit-log` sin adaptador Postgres. Nunca se recortan `policy-gate`, `mandates` ni la conciliación por webhook.
- Requisito de producción (6.3, 8.2): titular con actividad empresarial y CLABE propia. Si la CLABE no está `ACTIVE` antes del 7 dic, no hay demo en producción.
- Sin sandbox: cada prueba real cuesta $0.01 no reembolsable.
- Interpretación de 7.3/7.4 para un producto que otros comercios usarían con su propia API key: se consulta a legal@pey.mx; la respuesta escrita es vinculante (7.7).
- Rule of Two: la equivalencia "mandato verificado = validación confiable" es interpretación propia [I].

**Pendientes de verificar [P]**
- Herramienta base de `evals`.
- Disponibilidad de nombres en PyPI.
- Horarios oficiales de cobranza comercial (DOF/PROFECO) y aviso de privacidad para B.
- Posts de Anthropic sobre auto mode y sandboxing de Claude Code (relevantes para aprobaciones), no leídos.

## 11. Relación con trabajo previo
- **ADR 0002 (PR #724, knowtis-app):** queda superado por este diseño; se cierra sin merge o se reescribe cuando se apruebe este documento.
- **llm-router:** sigue en TS y con su plan de mantenimiento en t16; no forma parte de este corte.
- **pialy:** fuente de casos de aceptación para `payments-core` y `webhooks`; el diseño se declara en la entrega (9.2).

## 12. Qué queda al constructor
- Nombres exactos de funciones y módulos, estructura interna de cada paquete.
- Esquema de tablas de la app y del adaptador Postgres de `audit-log`.
- Prompts del agente, texto de las herramientas y diseño visual de la UI y la Trusted Surface.
- Elección concreta de la herramienta base de `evals` dentro del criterio de 7.5.
- Proveedor de modelo inicial, elegido con las evals.
