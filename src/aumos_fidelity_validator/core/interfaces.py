"""Protocol interfaces for the fidelity validator hexagonal architecture.

All adapters implement these protocols, allowing the core services to remain
framework-independent and testable with mock implementations.
"""

import uuid
from typing import Any, Protocol

import numpy as np
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


class TabularMetricsProtocol(Protocol):
    """Evaluates detailed tabular data fidelity at the column level.

    Computes 1-Way Distribution, Wasserstein distance, KL divergence,
    column-level statistics comparison, and an aggregated fidelity score.
    """

    async def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run detailed tabular column-level fidelity evaluation.

        Args:
            real_data: The original source dataset.
            synthetic_data: The generated synthetic dataset.
            metadata: Optional column metadata with sdtype hints.

        Returns:
            Report dict with overall_score, column_metrics (per-column
            wasserstein_score, kl_divergence_score, statistics_score,
            one_wd_score, column_score), and aggregate counts.
        """
        ...


class TextMetricsProtocol(Protocol):
    """Evaluates quality of generated text against reference corpora.

    Computes BLEU (1-4 gram), ROUGE-1/2/L, semantic similarity,
    text coherence, and perplexity estimation.
    """

    async def evaluate(
        self,
        real_texts: list[str],
        synthetic_texts: list[str],
        model_name: str = "all-MiniLM-L6-v2",
    ) -> dict[str, Any]:
        """Run all text quality metrics.

        Args:
            real_texts: Reference (ground-truth) text samples.
            synthetic_texts: Generated text samples to evaluate.
            model_name: Sentence-transformer model for semantic similarity.

        Returns:
            Report dict with bleu, rouge, semantic_similarity, coherence,
            perplexity, and overall_score.
        """
        ...


class ImageMetricsProtocol(Protocol):
    """Evaluates image generation quality using distribution-level metrics.

    Computes FID, IS, LPIPS, and SSIM, then aggregates into a single
    image fidelity score.
    """

    async def evaluate(
        self,
        real_images: np.ndarray,
        synthetic_images: np.ndarray,
    ) -> dict[str, Any]:
        """Run full image fidelity evaluation.

        Args:
            real_images: Real image batch, shape (N, H, W, C), uint8, RGB.
            synthetic_images: Synthetic image batch, same format.

        Returns:
            Report dict with fid_score, inception_score, lpips_score,
            ssim_score, and overall_score.
        """
        ...


class AudioMetricsProtocol(Protocol):
    """Evaluates audio synthesis quality across perceptual and acoustic dimensions.

    Computes MOS estimation, speaker similarity, pitch contour matching,
    prosody alignment, and SNR comparison.
    """

    async def evaluate(
        self,
        real_audio_batch: list[np.ndarray],
        synthetic_audio_batch: list[np.ndarray],
        sample_rate: int = 16_000,
    ) -> dict[str, Any]:
        """Run full audio fidelity evaluation.

        Args:
            real_audio_batch: List of real waveforms (float32, normalised [-1, 1]).
            synthetic_audio_batch: List of synthetic waveforms.
            sample_rate: Sample rate in Hz for all waveforms.

        Returns:
            Report dict with mos_score, speaker_similarity, pitch_score,
            prosody_score, snr_score, and overall_score.
        """
        ...


class VideoMetricsProtocol(Protocol):
    """Evaluates video synthesis quality across temporal and spatial dimensions.

    Computes per-frame LPIPS, optical flow consistency, temporal coherence,
    and scene transition detection accuracy.
    """

    async def evaluate(
        self,
        real_video: np.ndarray,
        synthetic_video: np.ndarray,
    ) -> dict[str, Any]:
        """Run full video fidelity evaluation.

        Args:
            real_video: Real video frames, shape (T, H, W, C), uint8, RGB.
            synthetic_video: Synthetic video frames, same format.

        Returns:
            Report dict with lpips_score, optical_flow_score, temporal_score,
            scene_transition_score, and overall_score.
        """
        ...


class HealthcareMetricsProtocol(Protocol):
    """Validates healthcare data fidelity across clinical and technical dimensions.

    Evaluates FHIR bundle structure, clinical realism, ICD-10/CPT code
    alignment, lab value plausibility, and medication safety scoring.
    """

    async def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        fhir_bundles: list[dict[str, Any]] | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Run full healthcare fidelity evaluation.

        Args:
            real_data: Real healthcare records as a DataFrame.
            synthetic_data: Synthetic healthcare records as a DataFrame.
            fhir_bundles: Optional list of FHIR bundles from synthetic data.
            metadata: Optional column metadata mapping column names to clinical types.

        Returns:
            Report dict with fhir_validation, clinical_realism, code_alignment,
            lab_plausibility, medication_safety, and overall_score.
        """
        ...


class StatisticalTestRunnerProtocol(Protocol):
    """Runs a comprehensive statistical test suite comparing two datasets.

    Executes KS test, chi-squared test, Wasserstein distance, Anderson-Darling,
    and Jensen-Shannon divergence per column, with Bonferroni correction.
    """

    async def run_all_tests(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        alpha: float = 0.05,
        thresholds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run the full statistical test suite on two datasets.

        Args:
            real_data: The original source dataset.
            synthetic_data: The generated synthetic dataset.
            alpha: Significance level before Bonferroni correction.
            thresholds: Optional per-metric distance thresholds for pass/fail.

        Returns:
            Report dict with per-column results, Bonferroni-corrected p-values,
            overall pass/fail, and significantly different column lists.
        """
        ...

    async def run_ks_test(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run KS test per numeric column only."""
        ...

    async def run_chi_squared_test(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run chi-squared test per categorical column only."""
        ...


class FidelityReportGeneratorProtocol(Protocol):
    """Generates comprehensive fidelity reports in JSON and PDF formats."""

    async def generate_json_report(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        fidelity_report: dict[str, Any],
        privacy_report: dict[str, Any] | None,
        memorization_report: dict[str, Any] | None,
        overall_score: float,
        passed: bool,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON report from all evaluation results.

        Args:
            job_id: Validation job UUID.
            tenant_id: Tenant identifier.
            fidelity_report: Fidelity evaluation results.
            privacy_report: Privacy risk results (optional).
            memorization_report: Memorization attack results (optional).
            overall_score: Aggregate score in [0, 1].
            passed: Whether all thresholds were met.
            metadata: Optional column metadata.

        Returns:
            Structured JSON-serialisable report dict with executive summary,
            column breakdown, and visualisation data.
        """
        ...

    async def generate_pdf_report(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        fidelity_report: dict[str, Any],
        privacy_report: dict[str, Any] | None,
        memorization_report: dict[str, Any] | None,
        overall_score: float,
        passed: bool,
    ) -> bytes:
        """Generate a PDF compliance report using reportlab.

        Args:
            job_id: Validation job UUID.
            tenant_id: Tenant identifier.
            fidelity_report: Fidelity evaluation results.
            privacy_report: Privacy risk results (optional).
            memorization_report: Memorization attack results (optional).
            overall_score: Aggregate score in [0, 1].
            passed: Whether all thresholds were met.

        Returns:
            PDF bytes ready for storage or download.
        """
        ...
