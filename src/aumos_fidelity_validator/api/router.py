"""FastAPI router for fidelity validator endpoints.

All routes are thin — they delegate all business logic to core services.
Authentication and tenant context are handled by aumos-common middleware.
"""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext, get_current_tenant
from aumos_common.database import get_db_session
from aumos_common.errors import NotFoundError
from aumos_common.observability import get_logger

from aumos_fidelity_validator.adapters.anonymeter_evaluator import AnonymeterEvaluator
from aumos_fidelity_validator.adapters.certificate_generator import PdfCertificateGenerator
from aumos_fidelity_validator.adapters.memorization_attacker import ShadowModelAttacker
from aumos_fidelity_validator.adapters.quality_contracts import GreatExpectationsRunner
from aumos_fidelity_validator.adapters.repositories import (
    QualityContractRepository,
    ValidationJobRepository,
)
from aumos_fidelity_validator.adapters.sdmetrics_evaluator import SDMetricsEvaluator
from aumos_fidelity_validator.adapters.storage import MinIOStorage
from aumos_fidelity_validator.api.schemas import (
    CertificateResponse,
    ContractResultResponse,
    FullReportResponse,
    FullValidationRequest,
    MemorizationRequest,
    MemorizationReportResponse,
    PrivacyReportResponse,
    PrivacyRiskRequest,
    QualityContractRequest,
    QualityContractResponse,
    RunContractRequest,
    ValidationJobResponse,
)
from aumos_fidelity_validator.core.models import JobType, ValidationJob
from aumos_fidelity_validator.core.services import (
    CertificateService,
    ContractService,
    FidelityService,
    FullValidationService,
    MemorizationService,
    PrivacyRiskService,
)
from aumos_fidelity_validator.settings import Settings
from aumos_fidelity_validator.api.schemas import FidelityValidationRequest

logger = get_logger(__name__)

router = APIRouter(prefix="/validate", tags=["validation"])

# ---------------------------------------------------------------------------
# Dependency factories
# ---------------------------------------------------------------------------


def get_settings() -> Settings:
    """Return global Settings instance."""
    return Settings()


def get_fidelity_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> FidelityService:
    """Build FidelityService with injected adapters."""
    from aumos_common.events import EventPublisher

    return FidelityService(
        session=session,
        evaluator=SDMetricsEvaluator(),
        storage=MinIOStorage(settings),
        publisher=EventPublisher(),
        settings=settings,
    )


def get_privacy_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> PrivacyRiskService:
    """Build PrivacyRiskService with injected adapters."""
    from aumos_common.events import EventPublisher

    return PrivacyRiskService(
        session=session,
        evaluator=AnonymeterEvaluator(settings),
        storage=MinIOStorage(settings),
        publisher=EventPublisher(),
        settings=settings,
    )


def get_memorization_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> MemorizationService:
    """Build MemorizationService with injected adapters."""
    from aumos_common.events import EventPublisher

    return MemorizationService(
        session=session,
        attacker=ShadowModelAttacker(settings),
        storage=MinIOStorage(settings),
        publisher=EventPublisher(),
        settings=settings,
    )


def get_certificate_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> CertificateService:
    """Build CertificateService with injected adapters."""
    return CertificateService(
        session=session,
        generator=PdfCertificateGenerator(),
        storage=MinIOStorage(settings),
        settings=settings,
    )


def get_full_validation_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> FullValidationService:
    """Build FullValidationService with all sub-services."""
    from aumos_common.events import EventPublisher

    publisher = EventPublisher()
    storage = MinIOStorage(settings)

    fidelity = FidelityService(
        session=session,
        evaluator=SDMetricsEvaluator(),
        storage=storage,
        publisher=publisher,
        settings=settings,
    )
    privacy = PrivacyRiskService(
        session=session,
        evaluator=AnonymeterEvaluator(settings),
        storage=storage,
        publisher=publisher,
        settings=settings,
    )
    memorization = MemorizationService(
        session=session,
        attacker=ShadowModelAttacker(settings),
        storage=storage,
        publisher=publisher,
        settings=settings,
    )
    certificate = CertificateService(
        session=session,
        generator=PdfCertificateGenerator(),
        storage=storage,
        settings=settings,
    )
    return FullValidationService(
        fidelity_service=fidelity,
        privacy_service=privacy,
        memorization_service=memorization,
        certificate_service=certificate,
        session=session,
        settings=settings,
    )


def get_contract_service(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ContractService:
    """Build ContractService with injected adapters."""
    return ContractService(
        session=session,
        contract_runner=GreatExpectationsRunner(),
        storage=MinIOStorage(settings),
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Helper to create a ValidationJob record before dispatching to services
# ---------------------------------------------------------------------------


async def _create_job(
    session: AsyncSession,
    tenant_id: str,
    job_type: JobType,
    source_uri: str,
    synthetic_uri: str,
) -> ValidationJob:
    """Create a new ValidationJob record and return it."""
    repo = ValidationJobRepository(session)
    return await repo.create(
        tenant_id=tenant_id,
        job_type=job_type,
        source_dataset_uri=source_uri,
        synthetic_dataset_uri=synthetic_uri,
    )


# ---------------------------------------------------------------------------
# Fidelity routes
# ---------------------------------------------------------------------------


@router.post(
    "/fidelity",
    response_model=ValidationJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start fidelity-only validation",
)
async def validate_fidelity(
    request: FidelityValidationRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    service: Annotated[FidelityService, Depends(get_fidelity_service)],
) -> ValidationJob:
    """Run SDMetrics fidelity evaluation on a synthetic dataset.

    Loads source and synthetic datasets from MinIO by URI, runs 50+ metrics,
    and returns the job. Results are stored asynchronously — poll /validate/{job_id}/report.
    """
    job = await _create_job(
        session=session,
        tenant_id=tenant.tenant_id,
        job_type=JobType.FIDELITY,
        source_uri=request.source_dataset_uri,
        synthetic_uri=request.synthetic_dataset_uri,
    )

    # Run synchronously in this implementation (async task queue in production)
    job = await service.run_fidelity_validation(
        job_id=job.id,
        tenant_id=tenant.tenant_id,
        metadata=request.metadata,
    )
    return job


# ---------------------------------------------------------------------------
# Privacy risk routes
# ---------------------------------------------------------------------------


@router.post(
    "/privacy-risk",
    response_model=ValidationJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start privacy risk assessment",
)
async def validate_privacy_risk(
    request: PrivacyRiskRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    service: Annotated[PrivacyRiskService, Depends(get_privacy_service)],
) -> ValidationJob:
    """Run Anonymeter re-identification risk assessment.

    Evaluates singling out, linkability, and inference attack vulnerabilities.
    """
    job = await _create_job(
        session=session,
        tenant_id=tenant.tenant_id,
        job_type=JobType.PRIVACY_RISK,
        source_uri=request.source_dataset_uri,
        synthetic_uri=request.synthetic_dataset_uri,
    )

    job = await service.run_privacy_risk_assessment(
        job_id=job.id,
        tenant_id=tenant.tenant_id,
        aux_dataset_uri=request.aux_dataset_uri,
        secret_columns=request.secret_columns,
        n_aux_cols=request.n_aux_cols,
    )
    return job


# ---------------------------------------------------------------------------
# Memorization routes
# ---------------------------------------------------------------------------


@router.post(
    "/memorization",
    response_model=ValidationJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start memorization attack simulation",
)
async def validate_memorization(
    request: MemorizationRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    service: Annotated[MemorizationService, Depends(get_memorization_service)],
) -> ValidationJob:
    """Simulate membership inference and data extraction attacks.

    Models Carlini/Nasr/Dai attack paradigms to measure memorization resistance.
    """
    job = await _create_job(
        session=session,
        tenant_id=tenant.tenant_id,
        job_type=JobType.MEMORIZATION,
        source_uri=request.source_dataset_uri,
        synthetic_uri=request.synthetic_dataset_uri,
    )

    job = await service.run_memorization_simulation(
        job_id=job.id,
        tenant_id=tenant.tenant_id,
        sensitive_columns=request.sensitive_columns,
        n_extraction_candidates=request.n_extraction_candidates,
    )
    return job


# ---------------------------------------------------------------------------
# Full validation route
# ---------------------------------------------------------------------------


@router.post(
    "/full",
    response_model=FullReportResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run complete validation suite",
)
async def validate_full(
    request: FullValidationRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    service: Annotated[FullValidationService, Depends(get_full_validation_service)],
) -> ValidationJob:
    """Run fidelity + privacy risk + memorization validation and generate certificate.

    This is the primary endpoint used by synthesis engines before releasing datasets.
    Generates a PDF compliance certificate if the dataset passes all thresholds.
    """
    job = await _create_job(
        session=session,
        tenant_id=tenant.tenant_id,
        job_type=JobType.FULL,
        source_uri=request.source_dataset_uri,
        synthetic_uri=request.synthetic_dataset_uri,
    )

    job = await service.run_full_validation(
        job_id=job.id,
        tenant_id=tenant.tenant_id,
        metadata=request.metadata,
        aux_dataset_uri=request.aux_dataset_uri,
        sensitive_columns=request.sensitive_columns,
    )
    return job


# ---------------------------------------------------------------------------
# Report retrieval routes
# ---------------------------------------------------------------------------


@router.get(
    "/{job_id}/report",
    response_model=FullReportResponse,
    summary="Retrieve validation report",
)
async def get_report(
    job_id: uuid.UUID,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ValidationJob:
    """Retrieve the full validation report for a job."""
    repo = ValidationJobRepository(session)
    job = await repo.get_by_id(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"ValidationJob {job_id} not found")
    return job


@router.get(
    "/{job_id}/certificate",
    response_model=CertificateResponse,
    summary="Retrieve compliance certificate URI",
)
async def get_certificate(
    job_id: uuid.UUID,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> dict[str, Any]:
    """Retrieve the MinIO URI of the generated PDF compliance certificate."""
    repo = ValidationJobRepository(session)
    job = await repo.get_by_id(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"ValidationJob {job_id} not found")
    if not job.certificate_uri:
        raise HTTPException(
            status_code=404,
            detail="Certificate not yet generated for this job",
        )
    return {
        "job_id": job.id,
        "certificate_uri": job.certificate_uri,
        "passed": job.passed,
    }


# ---------------------------------------------------------------------------
# Quality contract routes
# ---------------------------------------------------------------------------


@router.post(
    "/contract",
    response_model=QualityContractResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Define a quality contract",
)
async def create_contract(
    request: QualityContractRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    service: Annotated[ContractService, Depends(get_contract_service)],
) -> Any:
    """Define a new data quality contract with thresholds and GE assertions."""
    contract = await service.define_contract(
        tenant_id=tenant.tenant_id,
        name=request.name,
        thresholds=request.thresholds,
        assertions=request.assertions,
        description=request.description,
    )
    return contract


@router.post(
    "/contract/{contract_id}/run",
    response_model=ContractResultResponse,
    summary="Run a quality contract against a dataset",
)
async def run_contract(
    contract_id: uuid.UUID,
    request: RunContractRequest,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    service: Annotated[ContractService, Depends(get_contract_service)],
) -> Any:
    """Run a defined quality contract against a dataset URI."""
    try:
        results = await service.run_contract(
            contract_id=contract_id,
            dataset_uri=request.dataset_uri,
            tenant_id=tenant.tenant_id,
        )
        return {**results, "contract_id": contract_id}
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/contract/{contract_id}",
    response_model=QualityContractResponse,
    summary="Retrieve a quality contract",
)
async def get_contract(
    contract_id: uuid.UUID,
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> Any:
    """Retrieve a quality contract by ID."""
    repo = QualityContractRepository(session)
    contract = await repo.get_by_id(contract_id)
    if contract is None:
        raise HTTPException(status_code=404, detail=f"QualityContract {contract_id} not found")
    return contract
