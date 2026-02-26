# CLAUDE.md — AumOS Fidelity Validator

## Project Overview

AumOS Enterprise is a composable enterprise AI platform with 9 products + 2 services
across 62 repositories. This repo (`aumos-fidelity-validator`) is part of **Tier B (Open Core)**:
Data Factory validation layer consumed by ALL synthesis engines.

**Release Tier:** B: Open Core
**Product Mapping:** Product 1 — Data Factory (cross-cutting validator)
**Phase:** 1A (Months 3-8)

## Repo Purpose

Validates synthetic dataset quality across fidelity (statistical similarity), privacy (re-identification
risk), and memorization (model leakage) dimensions. Generates compliance certificates for tenant
audits and enforces data quality contracts before synthetic datasets are released downstream.
Consumed by every synthesis engine (tabular, text, image, audio, video, healthcare).

## Architecture Position

```
aumos-tabular-engine ──┐
aumos-text-engine ──────┤
aumos-image-engine ─────┼──► aumos-fidelity-validator ──► MinIO (certificates)
aumos-audio-engine ─────┤         │
aumos-video-engine ─────┘         └──► Kafka (fvl.validation.completed events)
aumos-healthcare-synth ──────────────► PostgreSQL (fvl_ tables)
```

**Upstream dependencies (this repo IMPORTS from):**
- `aumos-common` — auth, database, events, errors, config, health, pagination
- `aumos-proto` — Protobuf message definitions for Kafka events

**Downstream dependents (other repos IMPORT from this):**
- `aumos-tabular-engine` — calls /validate/full after generation
- `aumos-text-engine` — calls /validate/fidelity + /validate/privacy-risk
- `aumos-image-engine` — calls /validate/fidelity
- `aumos-audio-engine` — calls /validate/fidelity
- `aumos-video-engine` — calls /validate/fidelity
- `aumos-healthcare-synth` — calls /validate/full (strictest privacy thresholds)

## Tech Stack (DO NOT DEVIATE)

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.110+ | REST API framework |
| SQLAlchemy | 2.0+ (async) | Database ORM |
| asyncpg | 0.29+ | PostgreSQL async driver |
| Pydantic | 2.6+ | Data validation, settings, API schemas |
| confluent-kafka | 2.3+ | Kafka producer/consumer |
| structlog | 24.1+ | Structured JSON logging |
| OpenTelemetry | 1.23+ | Distributed tracing |
| pytest | 8.0+ | Testing framework |
| ruff | 0.3+ | Linting and formatting |
| mypy | 1.8+ | Type checking |
| sdmetrics | 0.14+ | 50+ fidelity evaluation metrics |
| anonymeter | 1.0+ | Re-identification risk (singling out, linkability, inference) |
| great-expectations | 0.18+ | Data quality contracts and assertions |
| scipy | 1.12+ | Statistical distance measures |
| scikit-learn | 1.4+ | Shadow models for membership inference |
| reportlab | 4.1+ | PDF certificate generation |

## Coding Standards

### ABSOLUTE RULES (violations will break integration with other repos)

1. **Import aumos-common, never reimplement.**
   ```python
   from aumos_common.auth import get_current_tenant, get_current_user
   from aumos_common.database import get_db_session, Base, AumOSModel, BaseRepository
   from aumos_common.events import EventPublisher, Topics
   from aumos_common.errors import NotFoundError, ErrorCode
   from aumos_common.config import AumOSSettings
   from aumos_common.health import create_health_router
   from aumos_common.pagination import PageRequest, PageResponse, paginate
   from aumos_common.app import create_app
   ```

2. **Type hints on EVERY function.** No exceptions.

3. **Pydantic models for ALL API inputs/outputs.** Never return raw dicts.

4. **RLS tenant isolation via aumos-common.** Never write raw SQL that bypasses RLS.

5. **Structured logging via structlog.** Never use print() or logging.getLogger().

6. **Publish domain events to Kafka after state changes.**

7. **Async by default.** All I/O operations must be async.

8. **Google-style docstrings** on all public classes and functions.

### Style Rules

- Max line length: **120 characters**
- Import order: stdlib → third-party → aumos-common → local
- Linter: `ruff` (select E, W, F, I, N, UP, ANN, B, A, COM, C4, PT, RUF)
- Type checker: `mypy` strict mode
- Formatter: `ruff format`

## Database Conventions

- **Table prefix:** `fvl_`  (e.g., `fvl_validation_jobs`, `fvl_quality_contracts`)
- ALL tenant-scoped tables extend `AumOSModel` (gets id, tenant_id, created_at, updated_at)
- RLS policy on every tenant table (created in migration)
- Migration naming: `{timestamp}_fvl_{description}.py`
- Foreign keys to other repos' tables: UUID type, no FK constraints (cross-service)

## Repo-Specific Context

### SDMetrics Integration
- Use `sdmetrics` v0.14+ for all fidelity evaluation
- Run `QualityReport` for aggregate score, individual metric classes for granular results
- 50+ metrics across: marginal (column-level), pairwise (column-pair), table-level, temporal, multi-table
- Metric results are stored as JSONB in `fvl_validation_jobs.fidelity_report`
- Aggregate score is stored in `fvl_validation_jobs.overall_score` (0.0–1.0 range, >0.82 = pass)

### Anonymeter Integration
- Use `anonymeter` v1.0+ for re-identification risk assessment
- Three attack types: `SinglingOutEvaluator`, `LinkabilityEvaluator`, `InferenceEvaluator`
- Risk scores stored in `fvl_validation_jobs.privacy_report` as JSONB
- Risk threshold: singling_out < 0.05, linkability < 0.1, inference < 0.15 to pass

### Memorization Attack Simulation
- Shadow model approach: train N shadow models on subsets of training data
- Membership inference: binary classifier on shadow model confidence scores
- Attack types modeled after Carlini et al. (2021), Nasr et al. (2019), Dai et al. (2023)
- Results stored in `fvl_validation_jobs.memorization_report` as JSONB
- Resistance threshold: membership inference AUC < 0.6 to pass

### Certificate Generation (ReportLab)
- Generate PDF certificates using `reportlab` (NOT WeasyPrint — license conflict)
- Certificate includes: fidelity scores, privacy scores, memorization scores, overall verdict
- Store PDF in MinIO: `{tenant_id}/certificates/{job_id}.pdf`
- URI stored in `fvl_validation_jobs.certificate_uri`
- Do NOT use python-docx (LGPL) — PDF only via reportlab

### Great Expectations Contracts
- `QualityContract` model stores GE suite as JSONB in `assertions` field
- `thresholds` JSONB stores per-metric minimum values
- ContractService runs GE checkpoint against dataset loaded from MinIO
- Contract violations return 422 with structured error listing failed assertions

### Performance Requirements
- Fidelity validation: < 5 minutes for datasets up to 100K rows
- Privacy risk assessment: < 10 minutes (anonymeter is CPU-bound)
- Memorization simulation: < 15 minutes (shadow models = CPU-bound)
- Full validation suite: < 30 minutes total
- Certificate generation: < 5 seconds

### Integration Notes
- All synthesis engines call this service BEFORE marking a generation job as `completed`
- If validation fails (passed=False), synthesis engine must mark job as `failed`
- Kafka event `fvl.validation.completed` published after every ValidationJob completion
- healthcare-synth uses stricter thresholds — pass these in the validation request payload

## What Claude Code Should NOT Do

1. **Do NOT reimplement anything in aumos-common.**
2. **Do NOT use print().** Use `get_logger(__name__)`.
3. **Do NOT return raw dicts from API endpoints.** Use Pydantic models.
4. **Do NOT write raw SQL.** Use SQLAlchemy ORM with BaseRepository.
5. **Do NOT hardcode configuration.** Use Pydantic Settings with env vars.
6. **Do NOT skip type hints.** Every function signature must be typed.
7. **Do NOT use python-docx or WeasyPrint** — license conflicts. PDF only via reportlab.
8. **Do NOT use anonymeter for anything outside privacy assessment** — it is a research tool.
9. **Do NOT store raw dataset content in the database** — store URIs and load from MinIO.
10. **Do NOT bypass RLS.**
