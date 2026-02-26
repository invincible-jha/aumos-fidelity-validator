# Contributing to aumos-fidelity-validator

## Development Setup

```bash
git clone <repo-url>
cd aumos-fidelity-validator
pip install -e ".[dev]"
cp .env.example .env
```

## Standards

- Type hints on all function signatures (mypy strict)
- Google-style docstrings on public classes and functions
- Max line length: 120 characters
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`

## Running Tests

```bash
make test          # full suite with coverage
make test-quick    # fast run, fail-fast
make lint          # ruff check
make typecheck     # mypy strict
```

## Pull Requests

- Branch from `main`: `feature/`, `fix/`, `docs/`
- All PRs require passing CI (lint + typecheck + tests ≥80% coverage)
- Squash-merge only

## Architecture

This repo follows hexagonal architecture:
- `api/` — FastAPI routes (thin, delegates to services)
- `core/` — Business logic services (no framework dependencies)
- `adapters/` — External integrations (SDMetrics, Anonymeter, storage)

Never put business logic in API routes. Never reimplement anything from `aumos-common`.
