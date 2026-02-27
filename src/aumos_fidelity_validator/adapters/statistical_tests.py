"""Statistical tests adapter — distribution comparison test suite.

Implements StatisticalTestRunnerProtocol running KS test, chi-squared test,
Wasserstein distance, Anderson-Darling test, Jensen-Shannon divergence, and
multi-test Bonferroni correction across all columns of a dataset pair.
"""

import asyncio
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import rel_entr
from scipy.stats import (
    anderson_ksamp,
    chi2_contingency,
    ks_2samp,
    wasserstein_distance,
)

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Default significance level before correction
_DEFAULT_ALPHA: float = 0.05


class StatisticalTestRunner:
    """Runs a comprehensive statistical test suite comparing two datasets.

    Executes KS test, chi-squared test, Wasserstein distance, Anderson-Darling,
    and Jensen-Shannon divergence per column. Applies Bonferroni correction for
    multiple comparisons and summarises pass/fail per threshold.
    """

    async def run_all_tests(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        alpha: float = _DEFAULT_ALPHA,
        thresholds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run the full statistical test suite on two datasets.

        Args:
            real_data: The original source dataset.
            synthetic_data: The generated synthetic dataset.
            alpha: Significance level before Bonferroni correction.
            thresholds: Optional per-metric distance thresholds for pass/fail.
                Keys: wasserstein_max, js_divergence_max.

        Returns:
            Report dict with per-column results, bonferroni-corrected p-values,
            overall pass/fail, and summary statistics.
        """
        logger.info(
            "Running statistical test suite",
            real_rows=len(real_data),
            synthetic_rows=len(synthetic_data),
            columns=len(real_data.columns),
            alpha=alpha,
        )
        effective_thresholds = {
            "wasserstein_max": 0.1,
            "js_divergence_max": 0.1,
            **(thresholds or {}),
        }

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._run_tests_sync,
            real_data,
            synthetic_data,
            alpha,
            effective_thresholds,
        )

    async def run_ks_test(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run KS test per numeric column only.

        Args:
            real_data: Real dataset.
            synthetic_data: Synthetic dataset.

        Returns:
            Per-column KS statistics and p-values.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._ks_per_column, real_data, synthetic_data)

    async def run_chi_squared_test(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run chi-squared test per categorical column only.

        Args:
            real_data: Real dataset.
            synthetic_data: Synthetic dataset.

        Returns:
            Per-column chi-squared statistics and p-values.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._chi2_per_column, real_data, synthetic_data)

    def _run_tests_sync(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        alpha: float,
        thresholds: dict[str, float],
    ) -> dict[str, Any]:
        """Synchronous full test suite computation."""
        column_results: dict[str, Any] = {}
        all_p_values: list[float] = []
        column_order: list[str] = []

        for column in real_data.columns:
            if column not in synthetic_data.columns:
                continue

            real_col = real_data[column].dropna()
            synth_col = synthetic_data[column].dropna()

            if len(real_col) < 5 or len(synth_col) < 5:
                continue

            col_result: dict[str, Any] = {}

            if pd.api.types.is_numeric_dtype(real_col):
                col_result.update(self._test_numeric_column(real_col, synth_col))
                if "ks_pvalue" in col_result:
                    all_p_values.append(col_result["ks_pvalue"])
                    column_order.append(column)
            else:
                col_result.update(self._test_categorical_column(real_col, synth_col))
                if "chi2_pvalue" in col_result:
                    all_p_values.append(col_result["chi2_pvalue"])
                    column_order.append(column)

            column_results[column] = col_result

        # Apply Bonferroni correction
        n_tests = len(all_p_values)
        bonferroni_alpha = alpha / max(n_tests, 1)
        bonferroni_results: dict[str, bool] = {}

        for col_name, p_value in zip(column_order, all_p_values):
            # Reject null hypothesis (distributions differ) if p < corrected alpha
            bonferroni_results[col_name] = p_value < bonferroni_alpha

        # Columns where we reject the null (distributions are significantly different)
        significantly_different_columns = [c for c, rejected in bonferroni_results.items() if rejected]
        fraction_passing = float(
            1.0 - len(significantly_different_columns) / max(len(bonferroni_results), 1)
        )

        # Threshold-based pass/fail on distance metrics
        wasserstein_threshold = thresholds["wasserstein_max"]
        js_threshold = thresholds["js_divergence_max"]

        failing_wasserstein = [
            col
            for col, result in column_results.items()
            if result.get("wasserstein_distance", 0.0) > wasserstein_threshold
        ]
        failing_js = [
            col
            for col, result in column_results.items()
            if result.get("js_divergence", 0.0) > js_threshold
        ]

        overall_passed = (
            fraction_passing >= 0.85
            and len(failing_wasserstein) == 0
            and len(failing_js) == 0
        )

        return {
            "overall_passed": overall_passed,
            "fraction_distributions_matching": fraction_passing,
            "total_columns_tested": len(column_results),
            "total_tests_run": n_tests,
            "bonferroni_alpha": bonferroni_alpha,
            "original_alpha": alpha,
            "significantly_different_columns": significantly_different_columns,
            "failing_wasserstein_columns": failing_wasserstein,
            "failing_js_columns": failing_js,
            "column_results": column_results,
            "thresholds": thresholds,
        }

    def _test_numeric_column(
        self,
        real_col: pd.Series,
        synth_col: pd.Series,
    ) -> dict[str, Any]:
        """Run all numeric tests on a single column.

        Args:
            real_col: Real column values (no NaN).
            synth_col: Synthetic column values (no NaN).

        Returns:
            Dict with ks_statistic, ks_pvalue, wasserstein_distance,
            js_divergence, anderson_darling_statistic.
        """
        result: dict[str, Any] = {"column_type": "numeric"}

        real_vals = real_col.astype(float).values
        synth_vals = synth_col.astype(float).values

        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_pvalue = ks_2samp(real_vals, synth_vals)
            result["ks_statistic"] = float(ks_stat)
            result["ks_pvalue"] = float(ks_pvalue)
        except Exception as exc:
            logger.debug("KS test failed", error=str(exc))

        # Wasserstein (Earth Mover's) distance, normalised by real std
        try:
            raw_w = float(wasserstein_distance(real_vals, synth_vals))
            real_std = float(np.std(real_vals)) + 1e-8
            result["wasserstein_distance"] = raw_w / real_std
            result["wasserstein_raw"] = raw_w
        except Exception as exc:
            logger.debug("Wasserstein test failed", error=str(exc))

        # Anderson-Darling k-sample test
        try:
            ad_result = anderson_ksamp([real_vals, synth_vals])
            result["anderson_darling_statistic"] = float(ad_result.statistic)
            result["anderson_darling_pvalue"] = float(getattr(ad_result, "pvalue", 0.0))
        except Exception as exc:
            logger.debug("Anderson-Darling test failed", error=str(exc))

        # Jensen-Shannon divergence via histogram
        try:
            result["js_divergence"] = self._js_divergence_histogram(real_vals, synth_vals)
        except Exception as exc:
            logger.debug("JS divergence failed", error=str(exc))

        # Summary statistics
        result["real_mean"] = float(np.mean(real_vals))
        result["synthetic_mean"] = float(np.mean(synth_vals))
        result["real_std"] = float(np.std(real_vals))
        result["synthetic_std"] = float(np.std(synth_vals))
        result["mean_difference"] = float(abs(np.mean(real_vals) - np.mean(synth_vals)))

        return result

    def _test_categorical_column(
        self,
        real_col: pd.Series,
        synth_col: pd.Series,
    ) -> dict[str, Any]:
        """Run all categorical tests on a single column.

        Args:
            real_col: Real column values (no NaN).
            synth_col: Synthetic column values (no NaN).

        Returns:
            Dict with chi2_statistic, chi2_pvalue, js_divergence, tvd.
        """
        result: dict[str, Any] = {"column_type": "categorical"}

        # Chi-squared test via frequency table
        try:
            all_categories = list(set(real_col.unique()) | set(synth_col.unique()))
            real_counts = real_col.value_counts().reindex(all_categories, fill_value=0).values
            synth_counts = synth_col.value_counts().reindex(all_categories, fill_value=0).values

            # chi2_contingency requires 2D array
            contingency = np.array([real_counts, synth_counts])
            chi2_stat, chi2_pvalue, dof, _ = chi2_contingency(contingency)
            result["chi2_statistic"] = float(chi2_stat)
            result["chi2_pvalue"] = float(chi2_pvalue)
            result["degrees_of_freedom"] = int(dof)
        except Exception as exc:
            logger.debug("Chi-squared test failed", error=str(exc))
            result["chi2_pvalue"] = 1.0

        # Total Variation Distance
        try:
            all_categories_union = list(set(real_col.unique()) | set(synth_col.unique()))
            real_freq = real_col.value_counts(normalize=True).reindex(all_categories_union, fill_value=0).values
            synth_freq = synth_col.value_counts(normalize=True).reindex(all_categories_union, fill_value=0).values
            tvd = float(0.5 * np.sum(np.abs(real_freq - synth_freq)))
            result["total_variation_distance"] = tvd
        except Exception as exc:
            logger.debug("TVD computation failed", error=str(exc))

        # Jensen-Shannon divergence
        try:
            epsilon = 1e-10
            real_prob = (real_freq + epsilon) / (real_freq + epsilon).sum()
            synth_prob = (synth_freq + epsilon) / (synth_freq + epsilon).sum()
            m = 0.5 * (real_prob + synth_prob)
            js_div = float(
                0.5 * np.sum(rel_entr(real_prob, m)) + 0.5 * np.sum(rel_entr(synth_prob, m))
            )
            result["js_divergence"] = float(np.clip(js_div, 0.0, 1.0))
        except Exception as exc:
            logger.debug("JS divergence (categorical) failed", error=str(exc))

        result["real_unique_count"] = int(real_col.nunique())
        result["synthetic_unique_count"] = int(synth_col.nunique())

        return result

    def _ks_per_column(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run KS test per numeric column (synchronous).

        Args:
            real_data: Real dataset.
            synthetic_data: Synthetic dataset.

        Returns:
            Per-column KS results.
        """
        results: dict[str, Any] = {}
        for col in real_data.columns:
            if col not in synthetic_data.columns:
                continue
            real_col = real_data[col].dropna()
            synth_col = synthetic_data[col].dropna()
            if not pd.api.types.is_numeric_dtype(real_col):
                continue
            try:
                stat, pvalue = ks_2samp(real_col.astype(float).values, synth_col.astype(float).values)
                results[col] = {"ks_statistic": float(stat), "ks_pvalue": float(pvalue)}
            except Exception as exc:
                results[col] = {"error": str(exc)}
        return results

    def _chi2_per_column(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run chi-squared test per categorical column (synchronous).

        Args:
            real_data: Real dataset.
            synthetic_data: Synthetic dataset.

        Returns:
            Per-column chi-squared results.
        """
        results: dict[str, Any] = {}
        for col in real_data.columns:
            if col not in synthetic_data.columns:
                continue
            real_col = real_data[col].dropna()
            synth_col = synthetic_data[col].dropna()
            if pd.api.types.is_numeric_dtype(real_col):
                continue
            try:
                all_categories = list(set(real_col.unique()) | set(synth_col.unique()))
                real_counts = real_col.value_counts().reindex(all_categories, fill_value=0).values
                synth_counts = synth_col.value_counts().reindex(all_categories, fill_value=0).values
                chi2_stat, chi2_pvalue, dof, _ = chi2_contingency(np.array([real_counts, synth_counts]))
                results[col] = {
                    "chi2_statistic": float(chi2_stat),
                    "chi2_pvalue": float(chi2_pvalue),
                    "degrees_of_freedom": int(dof),
                }
            except Exception as exc:
                results[col] = {"error": str(exc)}
        return results

    def _js_divergence_histogram(
        self,
        real_vals: np.ndarray,
        synth_vals: np.ndarray,
        n_bins: int = 50,
    ) -> float:
        """Compute JS divergence between two numeric arrays via histograms.

        Args:
            real_vals: Real numeric values.
            synth_vals: Synthetic numeric values.
            n_bins: Number of histogram bins.

        Returns:
            JS divergence in [0, 1].
        """
        combined_min = min(real_vals.min(), synth_vals.min())
        combined_max = max(real_vals.max(), synth_vals.max())
        if combined_min == combined_max:
            return 0.0
        bins = np.linspace(combined_min, combined_max, n_bins + 1)
        epsilon = 1e-10

        real_hist, _ = np.histogram(real_vals, bins=bins)
        synth_hist, _ = np.histogram(synth_vals, bins=bins)

        real_prob = (real_hist.astype(float) + epsilon) / (real_hist.sum() + epsilon * n_bins)
        synth_prob = (synth_hist.astype(float) + epsilon) / (synth_hist.sum() + epsilon * n_bins)

        m = 0.5 * (real_prob + synth_prob)
        js_div = float(0.5 * np.sum(rel_entr(real_prob, m)) + 0.5 * np.sum(rel_entr(synth_prob, m)))
        return float(np.clip(js_div, 0.0, 1.0))
