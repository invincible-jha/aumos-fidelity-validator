"""Tabular metrics adapter — detailed column-level fidelity evaluation.

Implements TabularMetricsProtocol using scipy statistical distances and
numpy descriptive statistics. Covers 1-Way Distribution, Wasserstein
distance, KL divergence, quantile comparison, and overall aggregation.
"""

import asyncio
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Histogram bins for distribution comparison
_DEFAULT_BINS = 50


class TabularMetricsEvaluator:
    """Evaluates detailed tabular data fidelity at the column level.

    Runs 1-Way Distribution (1WD) comparison, Wasserstein distance,
    KL divergence, and descriptive statistics per column, then
    aggregates into a single tabular fidelity score.
    """

    # Weight each distance category contributes to the overall score
    _WASSERSTEIN_WEIGHT: float = 0.35
    _KL_WEIGHT: float = 0.25
    _STATS_WEIGHT: float = 0.25
    _ONE_WD_WEIGHT: float = 0.15

    async def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run full tabular fidelity evaluation.

        Args:
            real_data: The original source dataset.
            synthetic_data: The generated synthetic dataset.
            metadata: Optional column metadata with sdtype hints.

        Returns:
            Report dict with overall_score, per-column breakdowns, and
            aggregated category scores.
        """
        logger.info(
            "Running tabular metrics evaluation",
            real_rows=len(real_data),
            synthetic_rows=len(synthetic_data),
            columns=len(real_data.columns),
        )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._compute_all_metrics,
            real_data,
            synthetic_data,
            metadata or {},
        )

    def _compute_all_metrics(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Synchronous computation of all tabular metrics."""
        column_results: dict[str, Any] = {}
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []

        for column in real_data.columns:
            if column not in synthetic_data.columns:
                logger.warning("Column missing from synthetic data", column=column)
                continue

            real_col = real_data[column].dropna()
            synthetic_col = synthetic_data[column].dropna()

            if len(real_col) == 0 or len(synthetic_col) == 0:
                continue

            if pd.api.types.is_numeric_dtype(real_col):
                numeric_cols.append(column)
                col_metrics = self._evaluate_numeric_column(real_col, synthetic_col, column)
            else:
                categorical_cols.append(column)
                col_metrics = self._evaluate_categorical_column(real_col, synthetic_col, column)

            column_results[column] = col_metrics

        overall_score = self._aggregate_score(column_results, numeric_cols, categorical_cols)

        return {
            "overall_score": overall_score,
            "column_metrics": column_results,
            "numeric_columns_evaluated": len(numeric_cols),
            "categorical_columns_evaluated": len(categorical_cols),
            "total_columns": len(real_data.columns),
        }

    def _evaluate_numeric_column(
        self,
        real_col: pd.Series,
        synthetic_col: pd.Series,
        column_name: str,
    ) -> dict[str, Any]:
        """Compute numeric fidelity metrics for a single column.

        Computes Wasserstein distance, KL divergence, 1WD, and descriptive
        statistics comparison. All distances are normalised to [0, 1] where
        1 means identical distributions.

        Args:
            real_col: Real data values (no NaN).
            synthetic_col: Synthetic data values (no NaN).
            column_name: Column identifier for logging.

        Returns:
            Dict of metric name to normalised score.
        """
        results: dict[str, Any] = {"column_type": "numeric"}

        try:
            real_vals = real_col.astype(float).values
            synthetic_vals = synthetic_col.astype(float).values

            # Wasserstein distance (normalised by real range)
            real_range = float(np.ptp(real_vals)) if np.ptp(real_vals) > 0 else 1.0
            raw_wasserstein = float(wasserstein_distance(real_vals, synthetic_vals))
            normalised_wasserstein = max(0.0, 1.0 - (raw_wasserstein / real_range))
            results["wasserstein_distance_raw"] = raw_wasserstein
            results["wasserstein_score"] = normalised_wasserstein

            # KL divergence via histogram approximation
            kl_score = self._kl_divergence_score(real_vals, synthetic_vals)
            results["kl_divergence_score"] = kl_score

            # 1-Way Distribution: KS statistic complement
            ks_stat, ks_pvalue = stats.ks_2samp(real_vals, synthetic_vals)
            results["one_wd_score"] = float(1.0 - ks_stat)
            results["ks_statistic"] = float(ks_stat)
            results["ks_pvalue"] = float(ks_pvalue)

            # Descriptive statistics comparison
            stat_score = self._statistics_similarity(real_vals, synthetic_vals)
            results["statistics_score"] = stat_score
            results["real_mean"] = float(np.mean(real_vals))
            results["synthetic_mean"] = float(np.mean(synthetic_vals))
            results["real_std"] = float(np.std(real_vals))
            results["synthetic_std"] = float(np.std(synthetic_vals))
            results["real_median"] = float(np.median(real_vals))
            results["synthetic_median"] = float(np.median(synthetic_vals))

            # Quantile comparison
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
            real_quantiles = np.quantile(real_vals, quantiles)
            synth_quantiles = np.quantile(synthetic_vals, quantiles)
            quantile_errors = np.abs(real_quantiles - synth_quantiles) / (np.abs(real_quantiles) + 1e-8)
            results["quantile_similarity"] = float(1.0 - np.mean(np.clip(quantile_errors, 0, 1)))

            # Column-level overall score
            results["column_score"] = (
                results["wasserstein_score"] * self._WASSERSTEIN_WEIGHT
                + results["kl_divergence_score"] * self._KL_WEIGHT
                + results["statistics_score"] * self._STATS_WEIGHT
                + results["one_wd_score"] * self._ONE_WD_WEIGHT
            )

        except Exception as exc:
            logger.warning("Numeric column metric failed", column=column_name, error=str(exc))
            results["error"] = str(exc)
            results["column_score"] = 0.0

        return results

    def _evaluate_categorical_column(
        self,
        real_col: pd.Series,
        synthetic_col: pd.Series,
        column_name: str,
    ) -> dict[str, Any]:
        """Compute categorical fidelity metrics for a single column.

        Uses Total Variation distance (TVD) and Jensen-Shannon divergence
        over the category probability distributions.

        Args:
            real_col: Real data values (no NaN).
            synthetic_col: Synthetic data values (no NaN).
            column_name: Column identifier for logging.

        Returns:
            Dict of metric name to normalised score.
        """
        results: dict[str, Any] = {"column_type": "categorical"}

        try:
            real_freqs = real_col.value_counts(normalize=True)
            synth_freqs = synthetic_col.value_counts(normalize=True)

            # Align on union of categories
            all_categories = real_freqs.index.union(synth_freqs.index)
            real_dist = real_freqs.reindex(all_categories, fill_value=0.0).values.astype(float)
            synth_dist = synth_freqs.reindex(all_categories, fill_value=0.0).values.astype(float)

            # Smooth to avoid log(0)
            epsilon = 1e-10
            real_dist_smooth = real_dist + epsilon
            synth_dist_smooth = synth_dist + epsilon
            real_dist_smooth /= real_dist_smooth.sum()
            synth_dist_smooth /= synth_dist_smooth.sum()

            # Total Variation Distance complement
            tvd = float(0.5 * np.sum(np.abs(real_dist - synth_dist)))
            results["tv_complement"] = float(1.0 - tvd)
            results["tv_distance"] = tvd

            # Jensen-Shannon divergence score
            m = 0.5 * (real_dist_smooth + synth_dist_smooth)
            js_div = float(0.5 * np.sum(rel_entr(real_dist_smooth, m)) + 0.5 * np.sum(rel_entr(synth_dist_smooth, m)))
            js_score = float(1.0 - np.clip(js_div, 0.0, 1.0))
            results["js_divergence"] = js_div
            results["js_score"] = js_score

            # Category coverage (fraction of real categories present in synthetic)
            real_categories = set(real_col.unique())
            synth_categories = set(synthetic_col.unique())
            coverage = len(real_categories & synth_categories) / max(len(real_categories), 1)
            results["category_coverage"] = float(coverage)
            results["real_category_count"] = len(real_categories)
            results["synthetic_category_count"] = len(synth_categories)

            results["column_score"] = float(
                0.4 * results["tv_complement"] + 0.4 * results["js_score"] + 0.2 * coverage
            )

        except Exception as exc:
            logger.warning("Categorical column metric failed", column=column_name, error=str(exc))
            results["error"] = str(exc)
            results["column_score"] = 0.0

        return results

    def _kl_divergence_score(self, real_vals: np.ndarray, synthetic_vals: np.ndarray) -> float:
        """Estimate KL divergence via equal-width histograms.

        Args:
            real_vals: Numeric real data array.
            synthetic_vals: Numeric synthetic data array.

        Returns:
            Normalised KL divergence score in [0, 1] where 1 = identical.
        """
        try:
            combined_min = min(real_vals.min(), synthetic_vals.min())
            combined_max = max(real_vals.max(), synthetic_vals.max())
            if combined_min == combined_max:
                return 1.0
            bins = np.linspace(combined_min, combined_max, _DEFAULT_BINS + 1)
            epsilon = 1e-10

            real_hist, _ = np.histogram(real_vals, bins=bins, density=False)
            synth_hist, _ = np.histogram(synthetic_vals, bins=bins, density=False)

            real_prob = (real_hist.astype(float) + epsilon) / (real_hist.sum() + epsilon * len(real_hist))
            synth_prob = (synth_hist.astype(float) + epsilon) / (synth_hist.sum() + epsilon * len(synth_hist))

            kl_div = float(np.sum(rel_entr(real_prob, synth_prob)))
            # KL divergence is unbounded; cap at 10 and normalise
            score = float(1.0 - np.clip(kl_div / 10.0, 0.0, 1.0))
            return score
        except Exception:  # noqa: BLE001
            return 0.0

    def _statistics_similarity(self, real_vals: np.ndarray, synthetic_vals: np.ndarray) -> float:
        """Score based on how closely mean, std, skew, and kurtosis match.

        Args:
            real_vals: Numeric real data array.
            synthetic_vals: Numeric synthetic data array.

        Returns:
            Similarity score in [0, 1].
        """
        try:
            scale = float(np.std(real_vals)) + 1e-8
            mean_diff = abs(float(np.mean(real_vals)) - float(np.mean(synthetic_vals))) / scale
            std_diff = abs(float(np.std(real_vals)) - float(np.std(synthetic_vals))) / scale
            skew_diff = abs(float(stats.skew(real_vals)) - float(stats.skew(synthetic_vals)))
            kurt_diff = abs(float(stats.kurtosis(real_vals)) - float(stats.kurtosis(synthetic_vals)))

            # Each difference penalises the score; clamp to [0, 1]
            mean_score = float(1.0 - np.clip(mean_diff, 0, 1))
            std_score = float(1.0 - np.clip(std_diff, 0, 1))
            skew_score = float(1.0 - np.clip(skew_diff / 5.0, 0, 1))
            kurt_score = float(1.0 - np.clip(kurt_diff / 10.0, 0, 1))

            return float(0.35 * mean_score + 0.35 * std_score + 0.15 * skew_score + 0.15 * kurt_score)
        except Exception:  # noqa: BLE001
            return 0.0

    def _aggregate_score(
        self,
        column_results: dict[str, Any],
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> float:
        """Compute overall tabular fidelity score from column-level scores.

        Args:
            column_results: Per-column metric dicts.
            numeric_cols: List of numeric column names.
            categorical_cols: List of categorical column names.

        Returns:
            Weighted overall score in [0, 1].
        """
        scores = [
            column_results[col].get("column_score", 0.0)
            for col in (numeric_cols + categorical_cols)
            if col in column_results
        ]
        if not scores:
            return 0.0
        return float(np.mean(scores))
