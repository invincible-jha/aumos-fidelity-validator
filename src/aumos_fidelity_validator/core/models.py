"""SQLAlchemy ORM models for aumos-fidelity-validator.

All tables use the fvl_ prefix and extend AumOSModel for tenant isolation.
"""

import enum
from decimal import Decimal
from typing import Any

from sqlalchemy import Boolean, Enum, String, Text
from sqlalchemy.dialects.postgresql import JSONB, NUMERIC
from sqlalchemy.orm import Mapped, mapped_column

from aumos_common.database import AumOSModel


class JobType(str, enum.Enum):
    """Types of validation jobs."""

    FIDELITY = "fidelity"
    PRIVACY_RISK = "privacy_risk"
    MEMORIZATION = "memorization"
    FULL = "full"


class JobStatus(str, enum.Enum):
    """Status values for a validation job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationJob(AumOSModel):
    """Tracks a single validation run against a synthetic dataset.

    A validation job references source and synthetic datasets by URI (MinIO),
    runs the requested evaluations, and stores results as JSONB reports.
    A passing job generates a PDF compliance certificate.
    """

    __tablename__ = "fvl_validation_jobs"

    job_type: Mapped[JobType] = mapped_column(
        Enum(JobType, name="fvl_job_type"),
        nullable=False,
        index=True,
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="fvl_job_status"),
        nullable=False,
        default=JobStatus.PENDING,
        index=True,
    )

    # Dataset references (URIs into MinIO, never raw data)
    source_dataset_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    synthetic_dataset_uri: Mapped[str] = mapped_column(String(1024), nullable=False)

    # Evaluation results stored as JSONB
    fidelity_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    privacy_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    memorization_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Aggregate score (0.0–1.0, weighted combination of all enabled evaluations)
    overall_score: Mapped[Decimal | None] = mapped_column(
        NUMERIC(precision=5, scale=4),
        nullable=True,
    )

    # Certificate reference (MinIO URI for generated PDF)
    certificate_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Whether this job met all pass thresholds
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Error information if status=FAILED
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class QualityContract(AumOSModel):
    """Defines data quality thresholds and assertions for a tenant.

    Contracts are used to enforce minimum quality standards before
    synthetic datasets are released. Stored thresholds override the
    service defaults. Assertions use Great Expectations suite format.
    """

    __tablename__ = "fvl_quality_contracts"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Per-metric thresholds: {"fidelity": 0.85, "singling_out": 0.03, ...}
    thresholds: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # Great Expectations suite JSON
    assertions: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class RegressionBaseline(AumOSModel):
    """Tracks fidelity score baselines per model version for regression detection.

    When a new synthesis model version is deployed, its scores are compared
    against the baseline. A significant drop triggers an alert.
    """

    __tablename__ = "fvl_regression_baselines"

    # Model version identifier from the synthesis engine
    model_version: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Baseline scores per metric category: {"fidelity": 0.89, "privacy": 0.95, ...}
    baseline_scores: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # Number of validation runs that established this baseline
    sample_count: Mapped[int] = mapped_column(nullable=False, default=1)

    # Whether this is the current active baseline for this model version
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
