"""Service-specific settings extending AumOS base config."""

from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class Settings(AumOSSettings):
    """Fidelity Validator service settings.

    All env vars use prefix AUMOS_FVL_.
    """

    service_name: str = "aumos-fidelity-validator"

    # MinIO buckets
    certificate_bucket: str = "fidelity-certificates"
    dataset_bucket: str = "validation-datasets"

    # Validation limits
    max_sample_rows: int = 100_000
    privacy_attack_iterations: int = 1_000
    memorization_shadow_models: int = 10

    # Report retention
    report_expiry_days: int = 90

    # Pass thresholds (defaults, can be overridden per-request)
    fidelity_pass_threshold: float = 0.82
    singling_out_risk_threshold: float = 0.05
    linkability_risk_threshold: float = 0.10
    inference_risk_threshold: float = 0.15
    membership_inference_auc_threshold: float = 0.60

    model_config = SettingsConfigDict(env_prefix="AUMOS_FVL_")
