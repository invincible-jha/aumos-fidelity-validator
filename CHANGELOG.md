# Changelog

All notable changes to aumos-fidelity-validator will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-26

### Added
- Initial scaffolding for aumos-fidelity-validator
- 50+ SDMetrics fidelity evaluation (marginal, pairwise, temporal, multi-table)
- Anonymeter re-identification risk assessment (singling out, linkability, inference)
- Carlini/Nasr/Dai memorization attack simulation (membership inference + extraction)
- Compliance certificate generation via ReportLab (PDF)
- Great Expectations data quality contracts
- Regression tracking across model versions
- Full hexagonal architecture (api/ + core/ + adapters/)
- ValidationJob, QualityContract, RegressionBaseline ORM models (fvl_ prefix)
- REST API: /validate/fidelity, /validate/privacy-risk, /validate/memorization, /validate/full
- Certificate retrieval endpoint
- Quality contract definition and enforcement
