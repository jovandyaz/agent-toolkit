# agent-toolkit

Framework-agnostic Python building blocks for AI agents that act on money and other sensitive state.
They plug into LangGraph, LangChain or a provider SDK; they are not a framework.

| Package | What it does | Grounded in |
|---|---|---|
| `mandates` | What a person authorized the agent to do, and whether a concrete operation fits | AP2 open/closed mandates |
| `policy-gate` | Deterministic allow / require approval / deny for every tool call | OWASP Agentic ASI02 "Intent Gate", Meta Agents Rule of Two |
| `budgets` | Iteration, tool-call, token, cost and failure caps per run | OWASP ASI08, OpenAI agent guide |
| `audit-log` | Append-only, hash-chained record of decisions and effects | Microsoft agentic failure-mode taxonomy |
| `evals` | Final-state graders with pass@k and pass^k | Anthropic, *Demystifying evals for AI agents* |
| `payments-core` | Request-to-pay domain (intents, attempts, idempotent reconciliation), no I/O | — |
| `webhooks` | Standard Webhooks verification plus an idempotent inbox | Standard Webhooks |

Status: under construction. Design and plan in [`docs/superpowers/`](docs/superpowers/).

## Development

```bash
uv sync --all-packages
uv run pytest
uv run ruff check . && uv run mypy packages tests && uv run lint-imports
```

License: Apache-2.0.
