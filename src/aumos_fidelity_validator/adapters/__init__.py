"""External integration adapters for aumos-fidelity-validator.

All adapters implement protocols defined in core.interfaces and are
injected into core services via dependency injection at startup.
"""

from aumos_fidelity_validator.adapters.anonymeter_evaluator import AnonymeterEvaluator
from aumos_fidelity_validator.adapters.audio_metrics import AudioMetricsEvaluator
from aumos_fidelity_validator.adapters.healthcare_metrics import HealthcareMetricsEvaluator
from aumos_fidelity_validator.adapters.image_metrics import ImageMetricsEvaluator
from aumos_fidelity_validator.adapters.report_generator import FidelityReportGenerator
from aumos_fidelity_validator.adapters.sdmetrics_evaluator import SDMetricsEvaluator
from aumos_fidelity_validator.adapters.statistical_tests import StatisticalTestRunner
from aumos_fidelity_validator.adapters.tabular_metrics import TabularMetricsEvaluator
from aumos_fidelity_validator.adapters.text_metrics import TextMetricsEvaluator
from aumos_fidelity_validator.adapters.video_metrics import VideoMetricsEvaluator

__all__ = [
    "AnonymeterEvaluator",
    "AudioMetricsEvaluator",
    "FidelityReportGenerator",
    "HealthcareMetricsEvaluator",
    "ImageMetricsEvaluator",
    "SDMetricsEvaluator",
    "StatisticalTestRunner",
    "TabularMetricsEvaluator",
    "TextMetricsEvaluator",
    "VideoMetricsEvaluator",
]
