"""Core business logic services for fidelity validation.

All services are framework-independent and depend on injected protocol
adapters for external integrations (SDMetrics, Anonymeter, storage, etc.).
"""

import uuid
from decimal import Decimal
from typing import Any

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.errors import NotFoundError
from aumos_common.events import EventPublisher, Topics
from aumos_common.observability import get_logger

from aumos_fidelity_validator.adapters.repositories import (
    QualityContractRepository,
    RegressionBaselineRepository,
    ValidationJobRepository,
)
from aumos_fidelity_validator.core.interfaces import (
    CertificateGeneratorProtocol,
    FidelityEvaluatorProtocol,
    MemorizationAttackProtocol,
    PrivacyRiskEvaluatorProtocol,
    QualityContractProtocol,
    StorageProtocol,
)
from aumos_fidelity_validator.core.models import (
    JobStatus,
    JobType,
    QualityContract,
    RegressionBaseline,
    ValidationJob,
)
from aumos_fidelity_validator.settings import Settings

logger = get_logger(__name__)


class FidelityService:
    """Orchestrates SDMetrics fidelity evaluation across 50+ metrics.

    Runs marginal (column-level), pairwise (column-pair), table-level,
    temporal, and multi-table SDMetrics metrics. Aggregates results into
    an overall fidelity score and persists the report to the ValidationJob.
    """

    def __init__(
        self,
        session: AsyncSession,
        evaluator: FidelityEvaluatorProtocol,
        storage: StorageProtocol,
        publisher: EventPublisher,
        settings: Settings,
    ) -> None:
        """Initialise the fidelity service with injected dependencies."""
        self._session = session
        self._evaluator = evaluator
        self._storage = storage
        self._publisher = publisher
        self._settings = settings
        self._job_repo = ValidationJobRepository(session)

    async def run_fidelity_validation(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> ValidationJob:
        """Run fidelity evaluation for an existing ValidationJob.

        Loads datasets from MinIO by URI, runs all fidelity metrics via
        SDMetrics, updates the job with results, and publishes a Kafka event.

        Args:
            job_id: UUID of the ValidationJob to process.
            tenant_id: Tenant context for RLS.
            metadata: Optional SDV-compatible column metadata. If not provided,
                SDMetrics will infer types automatically.

        Returns:
            The updated ValidationJob with fidelity_report and overall_score set.

        Raises:
            NotFoundError: If no job with the given ID exists for this tenant.
        """
        job = await self._job_repo.get_by_id(job_id)
        if job is None:
            raise NotFoundError(resource="ValidationJob", resource_id=str(job_id))

        await self._job_repo.update_status(job_id, JobStatus.RUNNING)
        logger.info(
            "Starting fidelity evaluation",
            job_id=str(job_id),
            tenant_id=tenant_id,
            source_uri=job.source_dataset_uri,
        )

        try:
            real_data = await self._storage.load_dataset(job.source_dataset_uri)
            synthetic_data = await self._storage.load_dataset(job.synthetic_dataset_uri)

            # Cap sample size for performance
            if len(real_data) > self._settings.max_sample_rows:
                real_data = real_data.sample(n=self._settings.max_sample_rows, random_state=42)
            if len(synthetic_data) > self._settings.max_sample_rows:
                synthetic_data = synthetic_data.sample(n=self._settings.max_sample_rows, random_state=42)

            report = await self._evaluator.evaluate(
                real_data=real_data,
                synthetic_data=synthetic_data,
                metadata=metadata or {},
            )

            overall_score = Decimal(str(report.get("overall_score", 0.0)))
            passed = float(overall_score) >= self._settings.fidelity_pass_threshold

            job = await self._job_repo.update_fidelity_report(
                job_id=job_id,
                fidelity_report=report,
                overall_score=overall_score,
                passed=passed,
                status=JobStatus.COMPLETED,
            )

            await self._publisher.publish(
                topic=Topics.FIDELITY_VALIDATION_COMPLETED,
                event={
                    "tenant_id": tenant_id,
                    "job_id": str(job_id),
                    "job_type": JobType.FIDELITY,
                    "overall_score": float(overall_score),
                    "passed": passed,
                },
            )

            logger.info(
                "Fidelity evaluation completed",
                job_id=str(job_id),
                overall_score=float(overall_score),
                passed=passed,
            )
            return job

        except Exception as exc:
            await self._job_repo.update_status(
                job_id,
                JobStatus.FAILED,
                error_message=str(exc),
            )
            logger.error("Fidelity evaluation failed", job_id=str(job_id), error=str(exc))
            raise

    async def get_metric_summary(
        self,
        fidelity_report: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract a human-readable metric summary from a fidelity report.

        Args:
            fidelity_report: The JSONB fidelity report from a ValidationJob.

        Returns:
            Dict with metric category scores and pass/fail per category.
        """
        categories = ["marginal", "pairwise", "table_level", "temporal"]
        threshold = self._settings.fidelity_pass_threshold
        summary: dict[str, Any] = {}

        for category in categories:
            category_data = fidelity_report.get(category, {})
            score = category_data.get("score", 0.0)
            summary[category] = {
                "score": score,
                "passed": score >= threshold,
                "metric_count": len(category_data.get("metrics", {})),
            }

        summary["overall_score"] = fidelity_report.get("overall_score", 0.0)
        summary["overall_passed"] = fidelity_report.get("overall_score", 0.0) >= threshold
        return summary


class PrivacyRiskService:
    """Orchestrates Anonymeter re-identification risk assessment.

    Runs three attack simulations: singling out, linkability, and inference.
    Each attack quantifies how easily an adversary could identify, link,
    or infer sensitive information from the synthetic dataset.
    """

    def __init__(
        self,
        session: AsyncSession,
        evaluator: PrivacyRiskEvaluatorProtocol,
        storage: StorageProtocol,
        publisher: EventPublisher,
        settings: Settings,
    ) -> None:
        """Initialise the privacy risk service with injected dependencies."""
        self._session = session
        self._evaluator = evaluator
        self._storage = storage
        self._publisher = publisher
        self._settings = settings
        self._job_repo = ValidationJobRepository(session)

    async def run_privacy_risk_assessment(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        aux_dataset_uri: str | None = None,
        secret_columns: list[str] | None = None,
        n_aux_cols: int = 2,
    ) -> ValidationJob:
        """Run privacy risk assessment for an existing ValidationJob.

        Args:
            job_id: UUID of the ValidationJob to process.
            tenant_id: Tenant context for RLS.
            aux_dataset_uri: Optional URI to auxiliary dataset for linkability attack.
            secret_columns: Columns to treat as sensitive for inference attack.
            n_aux_cols: Number of auxiliary columns used in linkability attack.

        Returns:
            The updated ValidationJob with privacy_report set.

        Raises:
            NotFoundError: If no job with the given ID exists for this tenant.
        """
        job = await self._job_repo.get_by_id(job_id)
        if job is None:
            raise NotFoundError(resource="ValidationJob", resource_id=str(job_id))

        await self._job_repo.update_status(job_id, JobStatus.RUNNING)
        logger.info(
            "Starting privacy risk assessment",
            job_id=str(job_id),
            tenant_id=tenant_id,
        )

        try:
            real_data = await self._storage.load_dataset(job.source_dataset_uri)
            synthetic_data = await self._storage.load_dataset(job.synthetic_dataset_uri)
            aux_data: pd.DataFrame | None = None

            if aux_dataset_uri:
                aux_data = await self._storage.load_dataset(aux_dataset_uri)

            report = await self._evaluator.evaluate(
                real_data=real_data,
                synthetic_data=synthetic_data,
                aux_data=aux_data,
            )

            # Assess pass/fail against configured thresholds
            singling_out_passed = (
                report.get("singling_out_risk", 1.0) <= self._settings.singling_out_risk_threshold
            )
            linkability_passed = (
                report.get("linkability_risk", 1.0) <= self._settings.linkability_risk_threshold
            )
            inference_passed = (
                report.get("inference_risk", 1.0) <= self._settings.inference_risk_threshold
            )
            passed = singling_out_passed and linkability_passed and inference_passed
            report["passed"] = passed

            job = await self._job_repo.update_privacy_report(
                job_id=job_id,
                privacy_report=report,
                passed=passed,
                status=JobStatus.COMPLETED,
            )

            await self._publisher.publish(
                topic=Topics.FIDELITY_VALIDATION_COMPLETED,
                event={
                    "tenant_id": tenant_id,
                    "job_id": str(job_id),
                    "job_type": JobType.PRIVACY_RISK,
                    "passed": passed,
                    "singling_out_risk": report.get("singling_out_risk"),
                    "linkability_risk": report.get("linkability_risk"),
                    "inference_risk": report.get("inference_risk"),
                },
            )

            logger.info(
                "Privacy risk assessment completed",
                job_id=str(job_id),
                passed=passed,
            )
            return job

        except Exception as exc:
            await self._job_repo.update_status(
                job_id,
                JobStatus.FAILED,
                error_message=str(exc),
            )
            logger.error("Privacy risk assessment failed", job_id=str(job_id), error=str(exc))
            raise


class MemorizationService:
    """Simulates memorization attacks to assess synthesis model leakage.

    Implements three attack paradigms:
    - Membership inference (Carlini et al. 2021): shadow model approach
    - Attribute inference (Nasr et al. 2019): gradient-based recovery
    - Data extraction (Dai et al. 2023): verbatim record reconstruction
    """

    def __init__(
        self,
        session: AsyncSession,
        attacker: MemorizationAttackProtocol,
        storage: StorageProtocol,
        publisher: EventPublisher,
        settings: Settings,
    ) -> None:
        """Initialise the memorization service with injected dependencies."""
        self._session = session
        self._attacker = attacker
        self._storage = storage
        self._publisher = publisher
        self._settings = settings
        self._job_repo = ValidationJobRepository(session)

    async def run_memorization_simulation(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        sensitive_columns: list[str] | None = None,
        n_extraction_candidates: int = 1000,
    ) -> ValidationJob:
        """Simulate memorization attacks for an existing ValidationJob.

        Args:
            job_id: UUID of the ValidationJob to process.
            tenant_id: Tenant context for RLS.
            sensitive_columns: Columns to target in attribute inference attack.
            n_extraction_candidates: Number of candidates for extraction attack.

        Returns:
            The updated ValidationJob with memorization_report set.

        Raises:
            NotFoundError: If no job with the given ID exists for this tenant.
        """
        job = await self._job_repo.get_by_id(job_id)
        if job is None:
            raise NotFoundError(resource="ValidationJob", resource_id=str(job_id))

        await self._job_repo.update_status(job_id, JobStatus.RUNNING)
        logger.info(
            "Starting memorization attack simulation",
            job_id=str(job_id),
            tenant_id=tenant_id,
            shadow_models=self._settings.memorization_shadow_models,
        )

        try:
            real_data = await self._storage.load_dataset(job.source_dataset_uri)
            synthetic_data = await self._storage.load_dataset(job.synthetic_dataset_uri)

            # Run membership inference (Carlini et al.)
            membership_report = await self._attacker.simulate_membership_inference(
                real_data=real_data,
                synthetic_data=synthetic_data,
                n_shadow_models=self._settings.memorization_shadow_models,
            )

            # Run attribute inference (Nasr et al.) if sensitive columns provided
            attribute_report: dict[str, Any] = {}
            if sensitive_columns:
                attribute_report = await self._attacker.simulate_attribute_inference(
                    real_data=real_data,
                    synthetic_data=synthetic_data,
                    sensitive_columns=sensitive_columns,
                )

            # Run extraction attack (Dai et al.)
            extraction_report = await self._attacker.simulate_extraction(
                real_data=real_data,
                synthetic_data=synthetic_data,
                n_candidates=n_extraction_candidates,
            )

            # Assess pass/fail: membership AUC < threshold = good (resistant)
            attack_auc = membership_report.get("attack_auc", 1.0)
            passed = attack_auc < self._settings.membership_inference_auc_threshold

            report = {
                "membership_inference": membership_report,
                "attribute_inference": attribute_report,
                "data_extraction": extraction_report,
                "attack_auc": attack_auc,
                "passed": passed,
            }

            job = await self._job_repo.update_memorization_report(
                job_id=job_id,
                memorization_report=report,
                passed=passed,
                status=JobStatus.COMPLETED,
            )

            await self._publisher.publish(
                topic=Topics.FIDELITY_VALIDATION_COMPLETED,
                event={
                    "tenant_id": tenant_id,
                    "job_id": str(job_id),
                    "job_type": JobType.MEMORIZATION,
                    "attack_auc": attack_auc,
                    "passed": passed,
                },
            )

            logger.info(
                "Memorization simulation completed",
                job_id=str(job_id),
                attack_auc=attack_auc,
                passed=passed,
            )
            return job

        except Exception as exc:
            await self._job_repo.update_status(
                job_id,
                JobStatus.FAILED,
                error_message=str(exc),
            )
            logger.error("Memorization simulation failed", job_id=str(job_id), error=str(exc))
            raise


class CertificateService:
    """Generates and stores PDF compliance certificates.

    Creates a PDF certificate summarising fidelity, privacy, and memorization
    scores. Uploads the certificate to MinIO and records the URI in the job.
    """

    def __init__(
        self,
        session: AsyncSession,
        generator: CertificateGeneratorProtocol,
        storage: StorageProtocol,
        settings: Settings,
    ) -> None:
        """Initialise the certificate service with injected dependencies."""
        self._session = session
        self._generator = generator
        self._storage = storage
        self._settings = settings
        self._job_repo = ValidationJobRepository(session)

    async def generate_certificate(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
    ) -> str:
        """Generate and store a PDF certificate for a completed validation job.

        Args:
            job_id: UUID of a completed ValidationJob.
            tenant_id: Tenant identifier for the certificate header.

        Returns:
            MinIO URI of the stored PDF certificate.

        Raises:
            NotFoundError: If no job with the given ID exists for this tenant.
            ValueError: If the job is not yet completed.
        """
        job = await self._job_repo.get_by_id(job_id)
        if job is None:
            raise NotFoundError(resource="ValidationJob", resource_id=str(job_id))

        if job.status != JobStatus.COMPLETED:
            raise ValueError(f"Job {job_id} is not completed (status={job.status})")

        logger.info("Generating compliance certificate", job_id=str(job_id), tenant_id=tenant_id)

        pdf_bytes = await self._generator.generate(
            job_id=job_id,
            tenant_id=tenant_id,
            fidelity_report=job.fidelity_report or {},
            privacy_report=job.privacy_report,
            memorization_report=job.memorization_report,
            overall_score=float(job.overall_score or 0),
            passed=job.passed or False,
        )

        certificate_uri = await self._storage.upload_certificate(
            tenant_id=tenant_id,
            job_id=job_id,
            pdf_bytes=pdf_bytes,
        )

        await self._job_repo.update_certificate_uri(
            job_id=job_id,
            certificate_uri=certificate_uri,
        )

        logger.info(
            "Certificate generated and stored",
            job_id=str(job_id),
            certificate_uri=certificate_uri,
        )
        return certificate_uri


class ContractService:
    """Defines and enforces data quality contracts using Great Expectations.

    Quality contracts specify minimum thresholds per metric category and
    custom assertions on dataset distributions. Contracts are evaluated
    before a synthetic dataset is released downstream.
    """

    def __init__(
        self,
        session: AsyncSession,
        contract_runner: QualityContractProtocol,
        storage: StorageProtocol,
        settings: Settings,
    ) -> None:
        """Initialise the contract service with injected dependencies."""
        self._session = session
        self._contract_runner = contract_runner
        self._storage = storage
        self._settings = settings
        self._contract_repo = QualityContractRepository(session)
        self._job_repo = ValidationJobRepository(session)

    async def define_contract(
        self,
        tenant_id: str,
        name: str,
        thresholds: dict[str, Any],
        assertions: dict[str, Any],
        description: str | None = None,
    ) -> QualityContract:
        """Create a new quality contract for a tenant.

        Args:
            tenant_id: Tenant ID for RLS.
            name: Human-readable contract name.
            thresholds: Per-metric minimum thresholds dict.
            assertions: Great Expectations expectation suite dict.
            description: Optional description of this contract's purpose.

        Returns:
            The newly created QualityContract.
        """
        contract = await self._contract_repo.create(
            tenant_id=tenant_id,
            name=name,
            description=description,
            thresholds=thresholds,
            assertions=assertions,
        )
        logger.info(
            "Quality contract defined",
            contract_id=str(contract.id),
            tenant_id=tenant_id,
            name=name,
        )
        return contract

    async def run_contract(
        self,
        contract_id: uuid.UUID,
        dataset_uri: str,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Run a quality contract against a dataset.

        Loads the dataset from MinIO, then runs Great Expectations assertions
        and validates metric thresholds. Returns detailed pass/fail results.

        Args:
            contract_id: UUID of the QualityContract to enforce.
            dataset_uri: MinIO URI of the dataset to validate.
            tenant_id: Tenant context for RLS.

        Returns:
            Results dict with keys: passed, failed_assertions, passed_assertions,
            success_percent.

        Raises:
            NotFoundError: If no contract with the given ID exists for this tenant.
        """
        contract = await self._contract_repo.get_by_id(contract_id)
        if contract is None:
            raise NotFoundError(resource="QualityContract", resource_id=str(contract_id))

        dataset = await self._storage.load_dataset(dataset_uri)

        results = await self._contract_runner.run_contract(
            data=dataset,
            contract_id=contract_id,
            assertions=contract.assertions,
            thresholds=contract.thresholds,
        )

        logger.info(
            "Quality contract run completed",
            contract_id=str(contract_id),
            tenant_id=tenant_id,
            passed=results.get("passed"),
            success_percent=results.get("success_percent"),
        )
        return results


class RegressionService:
    """Tracks fidelity score changes across synthesis model versions.

    Compares incoming validation scores against a stored baseline to detect
    regressions (score drops) that might indicate model degradation.
    """

    # Regression is flagged when score drops by more than this fraction
    REGRESSION_THRESHOLD: float = 0.05

    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
    ) -> None:
        """Initialise the regression service with injected dependencies."""
        self._session = session
        self._settings = settings
        self._baseline_repo = RegressionBaselineRepository(session)

    async def record_baseline(
        self,
        tenant_id: str,
        model_version: str,
        scores: dict[str, Any],
    ) -> RegressionBaseline:
        """Record or update a fidelity score baseline for a model version.

        Args:
            tenant_id: Tenant context for RLS.
            model_version: Version string from the synthesis engine.
            scores: Per-category score dict to store as baseline.

        Returns:
            The created or updated RegressionBaseline.
        """
        # Deactivate any existing current baseline for this version
        await self._baseline_repo.deactivate_current(
            tenant_id=tenant_id,
            model_version=model_version,
        )

        baseline = await self._baseline_repo.create(
            tenant_id=tenant_id,
            model_version=model_version,
            baseline_scores=scores,
            is_current=True,
        )

        logger.info(
            "Regression baseline recorded",
            baseline_id=str(baseline.id),
            tenant_id=tenant_id,
            model_version=model_version,
        )
        return baseline

    async def check_regression(
        self,
        tenant_id: str,
        model_version: str,
        current_scores: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare current scores against baseline to detect regressions.

        Args:
            tenant_id: Tenant context for RLS.
            model_version: Model version to compare against.
            current_scores: Latest validation scores.

        Returns:
            Dict with keys: regressed (bool), regression_details (per-category),
            baseline_version, current_version.
        """
        baseline = await self._baseline_repo.get_current(
            tenant_id=tenant_id,
            model_version=model_version,
        )

        if baseline is None:
            return {
                "regressed": False,
                "reason": "No baseline found — recording as new baseline",
                "baseline_scores": None,
                "current_scores": current_scores,
            }

        regressions: dict[str, Any] = {}
        for category, current_score in current_scores.items():
            baseline_score = baseline.baseline_scores.get(category)
            if baseline_score is None:
                continue
            drop = float(baseline_score) - float(current_score)
            if drop > self.REGRESSION_THRESHOLD:
                regressions[category] = {
                    "baseline": float(baseline_score),
                    "current": float(current_score),
                    "drop": drop,
                }

        regressed = len(regressions) > 0

        if regressed:
            logger.warning(
                "Fidelity regression detected",
                tenant_id=tenant_id,
                model_version=model_version,
                regression_categories=list(regressions.keys()),
            )

        return {
            "regressed": regressed,
            "regression_details": regressions,
            "baseline_scores": baseline.baseline_scores,
            "current_scores": current_scores,
        }


class FullValidationService:
    """Orchestrates the complete validation suite (fidelity + privacy + memorization).

    Runs all evaluation services in sequence and aggregates results into a
    single overall score and pass/fail verdict. Generates a certificate upon success.
    """

    def __init__(
        self,
        fidelity_service: FidelityService,
        privacy_service: PrivacyRiskService,
        memorization_service: MemorizationService,
        certificate_service: CertificateService,
        session: AsyncSession,
        settings: Settings,
    ) -> None:
        """Initialise the full validation service with all sub-services."""
        self._fidelity = fidelity_service
        self._privacy = privacy_service
        self._memorization = memorization_service
        self._certificate = certificate_service
        self._session = session
        self._settings = settings
        self._job_repo = ValidationJobRepository(session)

    async def run_full_validation(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
        aux_dataset_uri: str | None = None,
        sensitive_columns: list[str] | None = None,
    ) -> ValidationJob:
        """Run the complete validation suite and generate a certificate.

        Runs fidelity, privacy risk, and memorization evaluations sequentially.
        Computes a weighted overall score and generates a PDF certificate.

        Args:
            job_id: UUID of the ValidationJob to process.
            tenant_id: Tenant context for RLS.
            metadata: Optional SDV column metadata for fidelity evaluation.
            aux_dataset_uri: Optional auxiliary dataset URI for linkability attack.
            sensitive_columns: Columns to target in attribute inference attack.

        Returns:
            The fully updated ValidationJob.
        """
        logger.info(
            "Starting full validation suite",
            job_id=str(job_id),
            tenant_id=tenant_id,
        )

        # Run fidelity evaluation
        job = await self._fidelity.run_fidelity_validation(
            job_id=job_id,
            tenant_id=tenant_id,
            metadata=metadata,
        )

        # Run privacy risk assessment
        job = await self._privacy.run_privacy_risk_assessment(
            job_id=job_id,
            tenant_id=tenant_id,
            aux_dataset_uri=aux_dataset_uri,
            sensitive_columns=sensitive_columns,
        )

        # Run memorization simulation
        job = await self._memorization.run_memorization_simulation(
            job_id=job_id,
            tenant_id=tenant_id,
            sensitive_columns=sensitive_columns,
        )

        # Reload the job with all reports populated
        job = await self._job_repo.get_by_id(job_id)
        if job is None:
            raise NotFoundError(resource="ValidationJob", resource_id=str(job_id))

        # Compute weighted overall score
        fidelity_score = job.fidelity_report.get("overall_score", 0.0) if job.fidelity_report else 0.0
        privacy_passed = job.privacy_report.get("passed", False) if job.privacy_report else False
        memorization_passed = (
            job.memorization_report.get("passed", False) if job.memorization_report else False
        )

        # Weights: fidelity 60%, privacy (binary) 20%, memorization (binary) 20%
        privacy_contribution = 1.0 if privacy_passed else 0.0
        memorization_contribution = 1.0 if memorization_passed else 0.0
        overall = (fidelity_score * 0.6) + (privacy_contribution * 0.2) + (memorization_contribution * 0.2)
        passed = (
            fidelity_score >= self._settings.fidelity_pass_threshold
            and privacy_passed
            and memorization_passed
        )

        job = await self._job_repo.update_overall_score(
            job_id=job_id,
            overall_score=Decimal(str(overall)),
            passed=passed,
            status=JobStatus.COMPLETED,
        )

        # Generate certificate
        await self._certificate.generate_certificate(
            job_id=job_id,
            tenant_id=tenant_id,
        )

        logger.info(
            "Full validation suite completed",
            job_id=str(job_id),
            overall_score=overall,
            passed=passed,
        )
        return job
