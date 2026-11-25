# agent-toolkit

Framework-agnostic Python building blocks for AI agents that act on money and other sensitive state.
They plug into LangGraph, LangChain or a provider SDK; they are not a framework.

## Packages

Each package installs on its own and imports as `agent_toolkit.<name>`. Extras add the framework
adapters; the cores never import a framework, a database driver or a provider SDK.

| Package | What it does | Extras | Grounded in |
|---|---|---|---|
| [`mandates`](packages/mandates) | What a person authorized the agent to do, and whether a concrete charge fits | none | AP2 open/closed mandates |
| [`policy-gate`](packages/policy-gate) | Deterministic allow / require approval / deny for every tool call | `langgraph` | OWASP Agentic ASI02 "Intent Gate", Meta Agents Rule of Two |
| [`budgets`](packages/budgets) | Iteration, tool-call, token, cost and failure caps per run | `langchain` | OWASP ASI08, OpenAI agent guide |
| [`audit-log`](packages/audit-log) | Append-only, hash-chained record of decisions and effects | `postgres` | Microsoft agentic failure-mode taxonomy, OWASP ASI02/ASI09 |
| [`evals`](packages/evals) | Final-state graders with pass@k and pass^k | `pytest` | Anthropic, *Demystifying evals for AI agents* |
| [`payments-core`](packages/payments-core) | Request-to-pay domain (intents, attempts, idempotent reconciliation), no I/O | none | prior collections project |
| [`webhooks`](packages/webhooks) | Standard Webhooks verification plus an idempotent inbox | none | Standard Webhooks |
| [`model-routing`](packages/model-routing) | Per-provider circuit breaker so fallback chains skip failing providers | `langchain` | Azure "Circuit Breaker pattern", Fowler |
| [`prompt-guard`](packages/prompt-guard) | Heuristic injection verdicts (es, en) with rule ids; wrap untrusted text as data | none | OWASP LLM01, OWASP prompt-injection cheat sheet |

Dependency rule: only `policy-gate` depends on other packages (`mandates` and `budgets`). Every
other package depends on nothing from this repo, and at most on `pydantic` (`model-routing`'s
core has no dependency at all).

## Install from a git tag

The packages are not on PyPI. Install one from a release tag:

```
agent-toolkit-<pkg>[extra] @ git+https://github.com/jovandyaz/agent-toolkit@<tag>#subdirectory=packages/<pkg>
```

For example, in a `pyproject.toml` managed by `uv`:

```toml
dependencies = [
    "agent-toolkit-policy-gate[langgraph] @ git+https://github.com/jovandyaz/agent-toolkit@v0.1.0#subdirectory=packages/policy-gate",
    "agent-toolkit-audit-log[postgres] @ git+https://github.com/jovandyaz/agent-toolkit@v0.1.0#subdirectory=packages/audit-log",
]
```

`uv` resolves `policy-gate`'s dependencies on `mandates` and `budgets` from the same repository and
tag, so listing the one package is enough. This was checked with `uv sync` against a local git
tag; plain `pip` was not verified, so with another installer list each package explicitly.

## Versioning

One lockstep version for all packages, as pydantic-ai does in its uv workspace: every package
carries the same PEP 440 version, and a release is one git tag `v{version}` (for example
`v0.1.0a1`, then `v0.1.0`). `scripts/check_versions.py` enforces it: CI fails when the packages
disagree or when a `v*` tag differs from the package version. See [CHANGELOG.md](CHANGELOG.md).

## Development

```bash
uv sync --all-packages
uv run pytest   # Postgres tests need DATABASE_URL; marker `postgres`
uv run ruff check . && uv run ruff format --check . && uv run scripts/typecheck.sh && uv run lint-imports
uv build --all-packages   # one wheel per package
```

The examples in every README are executed by `tests/test_readme_examples.py`; blocks marked
`python notest` need a database, a model or a graph and are not run.

Design and plan live in [`docs/superpowers/`](docs/superpowers/). License: Apache-2.0.
