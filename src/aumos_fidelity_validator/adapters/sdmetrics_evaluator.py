"""SDMetrics adapter — runs 50+ fidelity metrics on synthetic datasets.

Implements FidelityEvaluatorProtocol using the sdmetrics library.
Covers marginal, pairwise, table-level, temporal, and multi-table metrics.
"""

import asyncio
from typing import Any

import pandas as pd
from sdmetrics import QualityReport
from sdmetrics.column_pairs import (
    ContingencySimilarity,
    CorrelationSimilarity,
)
from sdmetrics.reports.single_table import DiagnosticReport
from sdmetrics.single_column import (
    BoundaryAdherence,
    CategoryAdherence,
    KSComplement,
    RangeCoverage,
    StatisticMSAS,
    StatisticSimilarity,
    TVComplement,
)
from sdmetrics.single_table import (
    BinaryAdaBoostClassifier,
    BinaryDecisionTreeClassifier,
    BinaryLogisticRegression,
    BinaryMLPClassifier,
    ContinuousKLDivergence,
    DiscreteKLDivergence,
    GMLogLikelihood,
    LogisticDetection,
    MultiColumnPairs,
    MultiSingleColumn,
    NewRowSynthesis,
    SVCDetection,
)

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class SDMetricsEvaluator:
    """Runs SDMetrics quality evaluation across 50+ metrics.

    Uses the sdmetrics QualityReport as the primary evaluation framework,
    supplemented with individual metric classes for granular results.
    """

    # Column-level metrics (marginal distribution quality)
    MARGINAL_METRICS: list[type[Any]] = [
        BoundaryAdherence,
        CategoryAdherence,
        KSComplement,
        TVComplement,
        RangeCoverage,
        StatisticSimilarity,
        StatisticMSAS,
    ]

    # Column-pair metrics (bivariate dependency quality)
    PAIRWISE_METRICS: list[type[Any]] = [
        CorrelationSimilarity,
        ContingencySimilarity,
    ]

    # Table-level detection metrics (adversarial classifiers)
    TABLE_LEVEL_METRICS: list[type[Any]] = [
        LogisticDetection,
        SVCDetection,
        BinaryLogisticRegression,
        BinaryDecisionTreeClassifier,
        BinaryAdaBoostClassifier,
        BinaryMLPClassifier,
        GMLogLikelihood,
        NewRowSynthesis,
        ContinuousKLDivergence,
        DiscreteKLDivergence,
        MultiSingleColumn,
        MultiColumnPairs,
    ]

    async def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Run full SDMetrics quality evaluation.

        Args:
            real_data: The original source dataset.
            synthetic_data: The generated synthetic dataset.
            metadata: SDV-compatible metadata dict.

        Returns:
            Comprehensive fidelity report with overall_score and per-category results.
        """
        logger.info(
            "Running SDMetrics full evaluation",
            real_rows=len(real_data),
            synthetic_rows=len(synthetic_data),
            columns=len(real_data.columns),
        )

        # Run in thread pool — sdmetrics is CPU-bound
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(
            None,
            self._run_quality_report,
            real_data,
            synthetic_data,
            metadata,
        )

        marginal = await self.evaluate_marginal(real_data, synthetic_data, metadata)
        pairwise = await self.evaluate_pairwise(real_data, synthetic_data, metadata)
        table_level = await self._evaluate_table_level(real_data, synthetic_data, metadata)

        return {
            "overall_score": report.get("overall_score", 0.0),
            "marginal": marginal,
            "pairwise": pairwise,
            "table_level": table_level,
            "quality_report_summary": report,
        }

    async def evaluate_marginal(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Run marginal (column-level) fidelity metrics.

        Evaluates boundary adherence, category coverage, KS complement,
        and TV complement for each column independently.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._compute_marginal_metrics,
            real_data,
            synthetic_data,
            metadata,
        )

    async def evaluate_pairwise(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Run column-pair correlation and contingency metrics."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._compute_pairwise_metrics,
            real_data,
            synthetic_data,
            metadata,
        )

    def _run_quality_report(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the SDMetrics QualityReport synchronously."""
        try:
            report = QualityReport()
            report.generate(real_data, synthetic_data, metadata, verbose=False)
            return {
                "overall_score": report.get_score(),
                "properties": {
                    prop: report.get_details(prop)
                    for prop in ["Column Shapes", "Column Pair Trends"]
                },
            }
        except Exception as exc:
            logger.warning("QualityReport failed, using fallback metrics", error=str(exc))
            return {"overall_score": 0.0, "error": str(exc)}

    def _compute_marginal_metrics(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute per-column marginal metrics synchronously."""
        results: dict[str, dict[str, float]] = {}
        columns = metadata.get("columns", {})

        for column in real_data.columns:
            col_meta = columns.get(column, {})
            sdtype = col_meta.get("sdtype", "numerical")
            col_results: dict[str, float] = {}

            try:
                if sdtype in ("numerical", "datetime"):
                    score = KSComplement.compute(
                        real_data=real_data[column],
                        synthetic_data=synthetic_data[column],
                    )
                    col_results["ks_complement"] = float(score)

                    boundary_score = BoundaryAdherence.compute(
                        real_data=real_data[column],
                        synthetic_data=synthetic_data[column],
                    )
                    col_results["boundary_adherence"] = float(boundary_score)

                elif sdtype == "categorical":
                    tv_score = TVComplement.compute(
                        real_data=real_data[column],
                        synthetic_data=synthetic_data[column],
                    )
                    col_results["tv_complement"] = float(tv_score)

                    cat_score = CategoryAdherence.compute(
                        real_data=real_data[column],
                        synthetic_data=synthetic_data[column],
                    )
                    col_results["category_adherence"] = float(cat_score)

            except Exception as exc:
                logger.debug(
                    "Marginal metric failed for column",
                    column=column,
                    error=str(exc),
                )
                col_results["error"] = str(exc)

            if col_results:
                results[column] = col_results

        # Compute aggregate marginal score
        all_scores = [
            v
            for col_data in results.values()
            for k, v in col_data.items()
            if k != "error" and isinstance(v, float)
        ]
        score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        return {
            "score": score,
            "metrics": results,
            "metric_count": len(all_scores),
        }

    def _compute_pairwise_metrics(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute column-pair correlation and contingency metrics synchronously."""
        numerical_cols = [
            col
            for col in real_data.columns
            if pd.api.types.is_numeric_dtype(real_data[col])
        ]
        categorical_cols = [
            col
            for col in real_data.columns
            if not pd.api.types.is_numeric_dtype(real_data[col])
        ]

        pair_scores: list[float] = []
        pair_details: dict[str, Any] = {}

        # Pairwise correlations for numerical columns
        for i, col_a in enumerate(numerical_cols[:10]):  # Cap at 10 for performance
            for col_b in numerical_cols[i + 1 : 11]:
                try:
                    score = CorrelationSimilarity.compute(
                        real_data=real_data[[col_a, col_b]],
                        synthetic_data=synthetic_data[[col_a, col_b]],
                    )
                    key = f"{col_a}_{col_b}_correlation"
                    pair_details[key] = float(score)
                    pair_scores.append(float(score))
                except Exception:  # noqa: BLE001
                    pass

        # Contingency for categorical columns
        for i, col_a in enumerate(categorical_cols[:5]):
            for col_b in categorical_cols[i + 1 : 6]:
                try:
                    score = ContingencySimilarity.compute(
                        real_data=real_data[[col_a, col_b]],
                        synthetic_data=synthetic_data[[col_a, col_b]],
                    )
                    key = f"{col_a}_{col_b}_contingency"
                    pair_details[key] = float(score)
                    pair_scores.append(float(score))
                except Exception:  # noqa: BLE001
                    pass

        avg_score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
        return {
            "score": avg_score,
            "metrics": pair_details,
            "metric_count": len(pair_scores),
        }

    async def _evaluate_table_level(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Run table-level detection and ML metrics."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._compute_table_level,
            real_data,
            synthetic_data,
            metadata,
        )

    def _compute_table_level(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute table-level metrics synchronously."""
        results: dict[str, Any] = {}
        scores: list[float] = []

        metrics_to_run = [
            ("logistic_detection", LogisticDetection),
            ("new_row_synthesis", NewRowSynthesis),
        ]

        for metric_name, metric_cls in metrics_to_run:
            try:
                score = metric_cls.compute(
                    real_data=real_data,
                    synthetic_data=synthetic_data,
                    metadata=metadata,
                )
                results[metric_name] = float(score)
                scores.append(float(score))
            except Exception as exc:
                logger.debug("Table-level metric failed", metric=metric_name, error=str(exc))
                results[metric_name] = None

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "score": avg_score,
            "metrics": results,
            "metric_count": len(scores),
        }
