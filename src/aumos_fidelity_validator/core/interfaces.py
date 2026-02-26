"""Protocol interfaces for the fidelity validator hexagonal architecture.

All adapters implement these protocols, allowing the core services to remain
framework-independent and testable with mock implementations.
"""

import uuid
from typing import Any, Protocol

import pandas as pd


class FidelityEvaluatorProtocol(Protocol):
    """Evaluates statistical fidelity between real and synthetic datasets.

    Runs 50+ SDMetrics metrics across marginal, pairwise, temporal, and
    multi-table dimensions to produce a comprehensive fidelity report.
    """

    async def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Run all fidelity metrics and return a structured report.

        Args:
            real_data: The original (source) dataset.
            synthetic_data: The generated synthetic dataset.
            metadata: SDV-compatible metadata dict describing column types.

        Returns:
            Report dict with keys: overall_score, marginal, pairwise,
            table_level, temporal, metric_details.
        """
        ...

    async def evaluate_marginal(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Run marginal (column-level) metrics only."""
        ...

    async def evaluate_pairwise(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Run column-pair correlation metrics only."""
        ...


class PrivacyRiskEvaluatorProtocol(Protocol):
    """Assesses re-identification risk using Anonymeter attack simulations.

    Runs three attack types — singling out, linkability, and inference —
    to quantify how easily an adversary could identify or link records.
    """

    async def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        aux_data: pd.DataFrame | None,
    ) -> dict[str, Any]:
        """Run all privacy risk evaluations and return a structured report.

        Args:
            real_data: The original dataset used for synthesis.
            synthetic_data: The generated synthetic dataset.
            aux_data: Optional auxiliary data available to an adversary.

        Returns:
            Report dict with keys: singling_out_risk, linkability_risk,
            inference_risk, overall_risk_level.
        """
        ...

    async def evaluate_singling_out(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Assess singling out risk only."""
        ...

    async def evaluate_linkability(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        aux_data: pd.DataFrame,
        n_aux_cols: int,
    ) -> dict[str, Any]:
        """Assess linkability risk only."""
        ...

    async def evaluate_inference(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        secret_columns: list[str],
    ) -> dict[str, Any]:
        """Assess attribute inference risk only."""
        ...


class MemorizationAttackProtocol(Protocol):
    """Simulates memorization attacks (membership inference + extraction).

    Models attacks described in Carlini et al. (2021), Nasr et al. (2019),
    and Dai et al. (2023) to assess whether the synthesis model leaks
    information about its training data.
    """

    async def simulate_membership_inference(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        n_shadow_models: int,
    ) -> dict[str, Any]:
        """Simulate membership inference attack using shadow models.

        Args:
            real_data: Training data used to fit the synthesis model.
            synthetic_data: Output of the synthesis model.
            n_shadow_models: Number of shadow models to train for the attack.

        Returns:
            Report dict with keys: attack_auc, attack_advantage,
            tpr_at_fpr_001, membership_inference_risk.
        """
        ...

    async def simulate_attribute_inference(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_columns: list[str],
    ) -> dict[str, Any]:
        """Simulate attribute inference attack (Nasr et al. approach)."""
        ...

    async def simulate_extraction(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        n_candidates: int,
    ) -> dict[str, Any]:
        """Simulate verbatim record extraction attack (Dai et al. approach)."""
        ...


class CertificateGeneratorProtocol(Protocol):
    """Generates PDF compliance certificates from validation reports."""

    async def generate(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        fidelity_report: dict[str, Any],
        privacy_report: dict[str, Any] | None,
        memorization_report: dict[str, Any] | None,
        overall_score: float,
        passed: bool,
    ) -> bytes:
        """Generate a PDF compliance certificate.

        Args:
            job_id: The validation job UUID.
            tenant_id: Tenant identifier for the certificate header.
            fidelity_report: SDMetrics evaluation results.
            privacy_report: Anonymeter evaluation results (optional).
            memorization_report: Memorization attack results (optional).
            overall_score: Aggregate score (0.0–1.0).
            passed: Whether all thresholds were met.

        Returns:
            PDF bytes ready for upload to MinIO.
        """
        ...


class QualityContractProtocol(Protocol):
    """Defines and enforces data quality contracts using Great Expectations."""

    async def run_contract(
        self,
        data: pd.DataFrame,
        contract_id: uuid.UUID,
        assertions: dict[str, Any],
        thresholds: dict[str, Any],
    ) -> dict[str, Any]:
        """Run Great Expectations assertions against a dataset.

        Args:
            data: The dataset to validate.
            contract_id: The quality contract UUID.
            assertions: Great Expectations expectation suite as a dict.
            thresholds: Per-metric minimum thresholds.

        Returns:
            Results dict with keys: passed, failed_assertions, passed_assertions,
            success_percent.
        """
        ...


class StorageProtocol(Protocol):
    """Abstracts MinIO dataset and certificate storage."""

    async def load_dataset(self, uri: str) -> pd.DataFrame:
        """Load a Parquet or CSV dataset from MinIO by URI."""
        ...

    async def upload_certificate(
        self,
        tenant_id: str,
        job_id: uuid.UUID,
        pdf_bytes: bytes,
    ) -> str:
        """Upload a PDF certificate to MinIO and return its URI."""
        ...
