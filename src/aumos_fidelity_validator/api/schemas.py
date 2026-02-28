"""Pydantic request and response schemas for the fidelity validator API."""

import uuid
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, field_validator

from aumos_fidelity_validator.core.models import JobStatus, JobType


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class FidelityValidationRequest(BaseModel):
    """Request body for POST /validate/fidelity."""

    source_dataset_uri: str = Field(
        ...,
        description="MinIO URI of the source (real) dataset.",
        examples=["minio://validation-datasets/tenant-123/source/dataset.parquet"],
    )
    synthetic_dataset_uri: str = Field(
        ...,
        description="MinIO URI of the synthetic dataset to validate.",
        examples=["minio://validation-datasets/tenant-123/synthetic/dataset.parquet"],
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="SDV-compatible column metadata. Auto-inferred if not provided.",
    )


class PrivacyRiskRequest(BaseModel):
    """Request body for POST /validate/privacy-risk."""

    source_dataset_uri: str = Field(..., description="MinIO URI of the source dataset.")
    synthetic_dataset_uri: str = Field(..., description="MinIO URI of the synthetic dataset.")
    aux_dataset_uri: str | None = Field(
        default=None,
        description="Optional MinIO URI of auxiliary dataset for linkability attack.",
    )
    secret_columns: list[str] | None = Field(
        default=None,
        description="Column names to treat as sensitive for inference attack.",
    )
    n_aux_cols: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of columns used by the adversary in linkability attack.",
    )


class MemorizationRequest(BaseModel):
    """Request body for POST /validate/memorization."""

    source_dataset_uri: str = Field(..., description="MinIO URI of the source dataset.")
    synthetic_dataset_uri: str = Field(..., description="MinIO URI of the synthetic dataset.")
    sensitive_columns: list[str] | None = Field(
        default=None,
        description="Columns to target in attribute inference attack.",
    )
    n_extraction_candidates: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of record candidates for the extraction attack.",
    )


class FullValidationRequest(BaseModel):
    """Request body for POST /validate/full — runs all evaluation suites."""

    source_dataset_uri: str = Field(..., description="MinIO URI of the source dataset.")
    synthetic_dataset_uri: str = Field(..., description="MinIO URI of the synthetic dataset.")
    metadata: dict[str, Any] | None = Field(default=None, description="SDV column metadata.")
    aux_dataset_uri: str | None = Field(
        default=None,
        description="Optional auxiliary dataset URI for linkability attack.",
    )
    sensitive_columns: list[str] | None = Field(
        default=None,
        description="Sensitive columns for inference and attribute attacks.",
    )


class QualityContractRequest(BaseModel):
    """Request body for POST /validate/contract — define a quality contract."""

    name: str = Field(..., min_length=1, max_length=255, description="Contract name.")
    description: str | None = Field(default=None, description="Optional description.")
    thresholds: dict[str, float] = Field(
        ...,
        description="Per-metric minimum threshold values.",
        examples=[{"fidelity": 0.85, "singling_out": 0.03, "linkability": 0.08}],
    )
    assertions: dict[str, Any] = Field(
        ...,
        description="Great Expectations expectation suite as a dict.",
    )

    @field_validator("thresholds")
    @classmethod
    def validate_thresholds(cls, value: dict[str, float]) -> dict[str, float]:
        """Ensure all threshold values are in valid 0.0–1.0 range."""
        for key, threshold in value.items():
            if not (0.0 <= threshold <= 1.0):
                raise ValueError(f"Threshold for '{key}' must be between 0.0 and 1.0")
        return value


class RunContractRequest(BaseModel):
    """Request body for POST /validate/contract/{contract_id}/run."""

    dataset_uri: str = Field(..., description="MinIO URI of the dataset to validate.")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class ValidationJobResponse(BaseModel):
    """Response model for a ValidationJob."""

    id: uuid.UUID
    tenant_id: str
    job_type: JobType
    status: JobStatus
    source_dataset_uri: str
    synthetic_dataset_uri: str
    overall_score: Decimal | None
    passed: bool | None
    certificate_uri: str | None
    error_message: str | None

    model_config = {"from_attributes": True}


class FidelityReportResponse(BaseModel):
    """Response model for a detailed fidelity report."""

    job_id: uuid.UUID
    status: JobStatus
    overall_score: float | None
    passed: bool | None
    fidelity_report: dict[str, Any] | None

    model_config = {"from_attributes": True}


class PrivacyReportResponse(BaseModel):
    """Response model for a detailed privacy risk report."""

    job_id: uuid.UUID
    status: JobStatus
    passed: bool | None
    privacy_report: dict[str, Any] | None

    model_config = {"from_attributes": True}


class MemorizationReportResponse(BaseModel):
    """Response model for a detailed memorization attack report."""

    job_id: uuid.UUID
    status: JobStatus
    passed: bool | None
    memorization_report: dict[str, Any] | None

    model_config = {"from_attributes": True}


class FullReportResponse(BaseModel):
    """Response model for the full validation suite report."""

    job_id: uuid.UUID
    status: JobStatus
    job_type: JobType
    overall_score: Decimal | None
    passed: bool | None
    fidelity_report: dict[str, Any] | None
    privacy_report: dict[str, Any] | None
    memorization_report: dict[str, Any] | None
    certificate_uri: str | None

    model_config = {"from_attributes": True}


class CertificateResponse(BaseModel):
    """Response model for certificate retrieval."""

    job_id: uuid.UUID
    certificate_uri: str
    passed: bool | None


class QualityContractResponse(BaseModel):
    """Response model for a QualityContract."""

    id: uuid.UUID
    tenant_id: str
    name: str
    description: str | None
    thresholds: dict[str, Any]
    assertions: dict[str, Any]
    is_active: bool

    model_config = {"from_attributes": True}


class ContractResultResponse(BaseModel):
    """Response model for quality contract run results."""

    contract_id: uuid.UUID
    passed: bool
    success_percent: float
    failed_assertions: list[dict[str, Any]]
    passed_assertions: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Multi-Modal Validation (GAP-108)
# ---------------------------------------------------------------------------


class DataModality(str):
    """Supported data modalities for multi-modal validation."""

    tabular = "tabular"
    image = "image"
    audio = "audio"
    video = "video"
    text = "text"


class MultiModalValidationRequest(BaseModel):
    """Request for multi-modal fidelity validation.

    Attributes:
        job_id: Generation job UUID that produced the synthetic data.
        source_engine: Name of the synthesis engine that generated the data.
        modality: Data modality — tabular, image, audio, video, or text.
        synthetic_data_uri: Storage URI for the synthetic dataset.
        real_data_sample_uri: Storage URI for real reference samples.
        custom_thresholds: Per-metric threshold overrides.
    """

    job_id: uuid.UUID
    source_engine: str
    modality: str = Field(pattern="^(tabular|image|audio|video|text)$")
    synthetic_data_uri: str
    real_data_sample_uri: str | None = None
    custom_thresholds: dict[str, float] | None = None


class MultiModalValidationResponse(BaseModel):
    """Response for multi-modal fidelity validation.

    Attributes:
        validation_job_id: Created validation job UUID.
        modality: Validated data modality.
        overall_score: Composite quality score in [0, 1].
        passed: Whether validation thresholds were met.
        report: Per-modality metric report dict.
    """

    validation_job_id: uuid.UUID
    modality: str
    overall_score: float
    passed: bool
    report: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validation Dashboard (GAP-110)
# ---------------------------------------------------------------------------


class DashboardMetricSummary(BaseModel):
    """Summary of a single metric for dashboard display.

    Attributes:
        metric_name: Human-readable metric name.
        score: Current score in [0, 1].
        threshold: Required minimum score.
        passed: Whether score meets threshold.
        trend: Optional comparison to previous validation.
    """

    metric_name: str
    score: float
    threshold: float
    passed: bool
    trend: float | None = None


class ValidationDashboardResponse(BaseModel):
    """Chart-ready dashboard data for a validation job.

    Attributes:
        job_id: Validation job UUID.
        overall_score: Aggregate score.
        passed: Whether all thresholds were met.
        metrics: Per-metric summaries.
        modality: Data modality validated.
        validated_at: Validation completion timestamp.
    """

    job_id: uuid.UUID
    overall_score: float | None
    passed: bool | None
    metrics: list[DashboardMetricSummary] = Field(default_factory=list)
    modality: str = "tabular"
    validated_at: str | None = None


# ---------------------------------------------------------------------------
# Regulatory Report (GAP-109)
# ---------------------------------------------------------------------------


class RegulatoryReportRequest(BaseModel):
    """Request to generate a regulatory compliance report for a validation job.

    Attributes:
        standard: Regulatory standard — "gdpr", "hipaa", or "soc2".
    """

    standard: str = Field(pattern="^(gdpr|hipaa|soc2)$")


class RegulatoryReportResponse(BaseModel):
    """Response containing a rendered regulatory compliance report.

    Attributes:
        report_id: UUID of the generated report.
        standard: Regulatory standard applied.
        report_uri: Storage URI of the report document.
        summary: One-line compliance summary.
    """

    report_id: uuid.UUID
    standard: str
    report_uri: str
    summary: str


# ---------------------------------------------------------------------------
# Custom Metric Plugins (GAP-112)
# ---------------------------------------------------------------------------


class PluginRegistration(BaseModel):
    """Request to register a custom validation metric plugin.

    Attributes:
        name: Unique plugin identifier.
        description: Human-readable plugin description.
        code_uri: MinIO URI of the Python plugin code.
        input_columns: Expected input column names (empty = all columns).
        output_metric_name: Metric key returned by the plugin callable.
    """

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="")
    code_uri: str
    input_columns: list[str] = Field(default_factory=list)
    output_metric_name: str


class PluginResponse(BaseModel):
    """Response after registering a custom metric plugin.

    Attributes:
        plugin_id: UUID assigned to the registered plugin.
        name: Plugin name.
        code_uri: Storage URI of the plugin code.
        active: Whether the plugin is enabled.
    """

    plugin_id: uuid.UUID
    name: str
    code_uri: str
    active: bool = True


# ---------------------------------------------------------------------------
# Calibration Benchmark (GAP-107)
# ---------------------------------------------------------------------------


class CalibrationRunResponse(BaseModel):
    """Response from a validator calibration benchmark run.

    Attributes:
        calibration_id: UUID of the calibration run.
        quality_level_scores: Map from quality level name to metric score dict.
        ranking_verified: Whether validators correctly ranked generators by quality.
        summary: Human-readable calibration summary.
    """

    calibration_id: uuid.UUID
    quality_level_scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    ranking_verified: bool
    summary: str
