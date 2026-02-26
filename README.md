# aumos-fidelity-validator

Synthetic data quality validation service for AumOS Enterprise. Provides 50+ fidelity metrics,
re-identification risk assessment, memorization attack simulation, and compliance certificate generation.

## Overview

This service is consumed by all AumOS synthesis engines to validate that generated synthetic data
meets quality and privacy requirements before it is released to end users.

**Tier:** B (Open Core) | **Phase:** 1A | **Table prefix:** `fvl_`

## Capabilities

### Fidelity Evaluation (50+ Metrics)
- **Marginal metrics**: BoundaryAdherence, CategoryAdherence, KSComplement, TVComplement
- **Column-pair metrics**: CorrelationSimilarity, ContingencySimilarity, PairwiseCorrelationDifference
- **Table-level metrics**: NewRowSynthesis, LogisticDetection, SVCDetection, GMLogLikelihood
- **Temporal metrics**: TimeSeriesIntegrity, AutoCorrelationSimilarity
- **Multi-table metrics**: CardinalityShapeSimilarity, ReferentialIntegrity

### Privacy Risk Assessment
- **Singling out attacks** — can an adversary identify a unique record?
- **Linkability attacks** — can an adversary link records across datasets?
- **Inference attacks** — can an adversary infer sensitive attributes?

### Memorization Attack Simulation
- **Membership inference** (Carlini et al.) — shadow model attack
- **Attribute inference** (Nasr et al.) — gradient-based attribute recovery
- **Data extraction** (Dai et al.) — verbatim record reconstruction attempt

### Compliance Certificates
PDF certificates with:
- Fidelity score summary (pass/fail per metric category)
- Privacy risk scores (singling out / linkability / inference)
- Memorization resistance scores
- Overall pass/fail verdict
- Tenant and model version metadata

### Quality Contracts
Great Expectations-based contracts that define:
- Minimum thresholds per metric category
- Custom assertions on data distributions
- Automated enforcement before dataset release

## Architecture

```
aumos-tabular-engine ──┐
aumos-text-engine ──────┤
aumos-image-engine ─────┼──► aumos-fidelity-validator ──► MinIO (certificates)
aumos-audio-engine ─────┤         │
aumos-video-engine ─────┘         └──► Kafka (validation events)
aumos-healthcare-synth ───────────────► PostgreSQL (fvl_ tables)
```

## Quick Start

```bash
cp .env.example .env
pip install -e ".[dev]"
make test
make docker-run
```

## API

```
POST /api/v1/validate/fidelity          # Run fidelity metrics only
POST /api/v1/validate/privacy-risk      # Run re-identification risk only
POST /api/v1/validate/memorization      # Run memorization attack simulation
POST /api/v1/validate/full              # Run complete validation suite
GET  /api/v1/validate/{job_id}/report   # Retrieve validation report
GET  /api/v1/validate/{job_id}/certificate  # Retrieve PDF certificate
POST /api/v1/validate/contract          # Define and run quality contract
```

## Environment Variables

See `.env.example` for all required variables.
