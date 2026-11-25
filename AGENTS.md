# agent-toolkit: agent instructions

Framework-agnostic Python building blocks for safe AI agents. Pieces plug into LangGraph, LangChain,
CrewAI or provider SDKs; they never replace them. First consumer: a CoDi/pey collections agent for
Hackatón México (separate private repo). Second planned consumer: CDMX permit pre-review (uses
`budgets` and `audit-log`).

## Read first
- Design: `docs/superpowers/specs/2026-11-15-agent-toolkit-design.md` (approved 2026-11-16).
- Plans: `docs/superpowers/plans/2026-11-16-agent-toolkit-packages-plan.md` (first seven packages),
  `docs/superpowers/plans/2026-11-24-model-routing-prompt-guard-plan.md` (`model-routing`, `prompt-guard`). Tick steps as they land;
  the plan is the progress record.

## Rules
- uv workspace, one package per `packages/<name>`, import path `agent_toolkit.<name>`
  (namespace package: no `__init__.py` in `src/agent_toolkit/`).
- Cores never import `langgraph`, `langchain*`, `psycopg` or any provider SDK. Adapters live in
  `agent_toolkit.<name>.adapters.<x>` and ship as extras. Enforced by `lint-imports` and
  `tests/test_core_isolation.py`. Each new package adds its portion to `[tool.importlinter] root_packages`.
- Only `policy-gate` depends on other packages (`mandates`, `budgets`).
- Money is an integer in minor units plus an ISO 4217 code. Never `float`.
- Public types: Pydantic v2 frozen + strict; `mypy --strict`; no `Any` in public APIs.
- `mandates`, `policy-gate` and `payments-core` are money paths: TDD, one fresh reviewer per task,
  never weaken a test to make it pass.
- `audit-log` stores hashes or references, never PII in clear.
- This repo is public: no secrets, no real customer data, no pey operational details (those live in
  the app repo).

## Commands
- `uv sync --all-packages`
- `uv run pytest` (Postgres tests need `DATABASE_URL`; marker `postgres`)
- `uv run ruff check . && uv run ruff format --check . && uv run scripts/typecheck.sh && uv run lint-imports`
  (`scripts/typecheck.sh` runs mypy once per package plus the root `tests/`, since test basenames repeat
  across packages)

## Git
Conventional Commits, single line, English, no trailers. Branch per package, PR per package; never
push to `main`. Lockstep versioning: all packages share one PEP 440 version and one tag
`v{version}` (latest `v0.1.0rc1`, next `v0.1.0`). A release bumps every `pyproject.toml`, refreshes
`uv.lock` and moves `CHANGELOG.md` Unreleased; CI checks the tag with `scripts/check_versions.py`.
Revisit per-package versions once the packages are on PyPI.
