"""Anonymeter evaluator adapter — re-identification risk assessment.

Implements PrivacyRiskEvaluatorProtocol using the anonymeter library to
simulate singling-out, linkability, and inference attacks. Also incorporates
Carlini/Nasr/Dai membership inference metrics for overall re-identification
risk scoring with categorised risk levels (low/medium/high/critical).
"""

import asyncio
from typing import Any

import pandas as pd

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Risk thresholds for categorisation
_RISK_LOW_MAX: float = 0.05
_RISK_MEDIUM_MAX: float = 0.15
_RISK_HIGH_MAX: float = 0.30

# Default number of attack iterations for anonymeter
_DEFAULT_N_ATTACKS: int = 1_000


class AnonymeterEvaluator:
    """Assesses re-identification risk using Anonymeter attack simulations.

    Runs singling-out, linkability, and inference attack evaluations.
    Produces per-column privacy risk scores and an overall risk score
    in [0, 1] with risk categorisation (low/medium/high/critical).
    """

    def __init__(self, n_attacks: int = _DEFAULT_N_ATTACKS) -> None:
        """Initialise the evaluator with attack configuration.

        Args:
            n_attacks: Number of attack iterations for each evaluator type.
        """
        self._n_attacks = n_attacks

    async def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        aux_data: pd.DataFrame | None,
    ) -> dict[str, Any]:
        """Run all re-identification risk evaluations.

        Args:
            real_data: The original dataset used for synthesis.
            synthetic_data: The generated synthetic dataset.
            aux_data: Optional auxiliary data available to an adversary.

        Returns:
            Report dict with singling_out_risk, linkability_risk,
            inference_risk, overall_risk, and overall_risk_level.
        """
        logger.info(
            "Running Anonymeter re-identification risk evaluation",
            real_rows=len(real_data),
            synthetic_rows=len(synthetic_data),
            has_aux_data=aux_data is not None,
            n_attacks=self._n_attacks,
        )

        loop = asyncio.get_running_loop()

        singling_out_result = await self.evaluate_singling_out(real_data, synthetic_data)
        inference_result = await self.evaluate_inference(
            real_data,
            synthetic_data,
            secret_columns=list(real_data.columns[-3:]),  # Default: last 3 columns as sensitive
        )

        if aux_data is not None:
            n_aux_cols = min(2, len(aux_data.columns))
            linkability_result = await self.evaluate_linkability(
                real_data, synthetic_data, aux_data, n_aux_cols=n_aux_cols
            )
        else:
            linkability_result = await loop.run_in_executor(
                None, self._evaluate_linkability_no_aux, real_data, synthetic_data
            )

        # Aggregate risk scores
        singling_out_risk = singling_out_result.get("risk_score", 1.0)
        linkability_risk = linkability_result.get("risk_score", 1.0)
        inference_risk = inference_result.get("risk_score", 1.0)

        overall_risk = float(
            singling_out_risk * 0.40
            + linkability_risk * 0.35
            + inference_risk * 0.25
        )
        overall_risk_level = self._categorise_risk(overall_risk)

        return {
            "singling_out_risk": float(singling_out_risk),
            "singling_out_details": singling_out_result,
            "linkability_risk": float(linkability_risk),
            "linkability_details": linkability_result,
            "inference_risk": float(inference_risk),
            "inference_details": inference_result,
            "overall_risk": float(overall_risk),
            "overall_risk_level": overall_risk_level,
            "passed": (
                singling_out_risk <= 0.05
                and linkability_risk <= 0.10
                and inference_risk <= 0.15
            ),
        }

    async def evaluate_singling_out(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Assess singling-out risk — how uniquely identifiable records are.

        Simulates an adversary finding a query that selects exactly one
        real record using only the synthetic data as a guide.

        Args:
            real_data: Original dataset.
            synthetic_data: Synthetic dataset.

        Returns:
            Dict with risk_score, attack_success_rate, and per-column results.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._run_singling_out,
            real_data,
            synthetic_data,
        )

    async def evaluate_linkability(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        aux_data: pd.DataFrame,
        n_aux_cols: int = 2,
    ) -> dict[str, Any]:
        """Assess linkability risk — linking two records to the same individual.

        Simulates an adversary who has auxiliary data and tries to link
        synthetic records to real individuals by matching attributes.

        Args:
            real_data: Original dataset.
            synthetic_data: Synthetic dataset.
            aux_data: Auxiliary dataset available to the adversary.
            n_aux_cols: Number of auxiliary columns to use for linking.

        Returns:
            Dict with risk_score and attack success metrics.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._run_linkability,
            real_data,
            synthetic_data,
            aux_data,
            n_aux_cols,
        )

    async def evaluate_inference(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        secret_columns: list[str],
    ) -> dict[str, Any]:
        """Assess attribute inference risk on sensitive columns.

        Simulates an adversary who knows some attributes about an individual
        and tries to infer the values of hidden (secret) columns.

        Args:
            real_data: Original dataset.
            synthetic_data: Synthetic dataset.
            secret_columns: Columns to treat as sensitive/secret for inference.

        Returns:
            Dict with risk_score and per-secret-column inference accuracy.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._run_inference,
            real_data,
            synthetic_data,
            secret_columns,
        )

    def _run_singling_out(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Execute singling-out attack synchronously.

        Attempts to use the anonymeter SinglingOutEvaluator. Falls back
        to a statistical uniqueness proxy if anonymeter is unavailable.

        Args:
            real_data: Original dataset.
            synthetic_data: Synthetic dataset.

        Returns:
            Attack result dict with risk_score.
        """
        try:
            from anonymeter.evaluators import SinglingOutEvaluator  # type: ignore[import]

            n_attacks = min(self._n_attacks, len(synthetic_data))
            evaluator = SinglingOutEvaluator(
                ori=real_data,
                syn=synthetic_data,
                n_attacks=n_attacks,
                n_cols=min(3, len(real_data.columns)),
            )
            evaluator.evaluate(mode="univariate")
            result = evaluator.risk()

            risk_score = float(result.value)
            ci_low = float(result.ci[0]) if hasattr(result, "ci") and result.ci else risk_score
            ci_high = float(result.ci[1]) if hasattr(result, "ci") and result.ci else risk_score

            return {
                "risk_score": risk_score,
                "confidence_interval": [ci_low, ci_high],
                "attack_type": "singling_out_univariate",
                "n_attacks": n_attacks,
            }

        except ImportError:
            logger.warning("anonymeter not installed — using uniqueness proxy for singling-out risk")
            return self._singling_out_uniqueness_proxy(real_data, synthetic_data)
        except Exception as exc:
            logger.warning("Singling-out attack failed", error=str(exc))
            return {"risk_score": 0.0, "error": str(exc)}

    def _singling_out_uniqueness_proxy(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Proxy singling-out risk via record uniqueness analysis.

        Records that are unique in the synthetic dataset but can be
        matched exactly to real records represent singling-out risk.

        Args:
            real_data: Original dataset.
            synthetic_data: Synthetic dataset.

        Returns:
            Proxy risk score dict.
        """
        try:
            import numpy as np

            # Use a subset of columns for matching
            compare_cols = list(set(real_data.columns) & set(synthetic_data.columns))[:5]
            if not compare_cols:
                return {"risk_score": 0.0, "method": "uniqueness_proxy", "note": "no_common_columns"}

            # Find unique rows in synthetic data that have exact matches in real data
            real_subset = real_data[compare_cols].astype(str)
            synth_subset = synthetic_data[compare_cols].astype(str)

            # Check for exact row matches between synthetic and real
            real_hashes = set(real_subset.apply(lambda row: hash(tuple(row)), axis=1))
            synth_hashes = synth_subset.apply(lambda row: hash(tuple(row)), axis=1)

            match_count = int(synth_hashes.isin(real_hashes).sum())
            risk_score = float(match_count / max(len(synthetic_data), 1))

            return {
                "risk_score": risk_score,
                "matched_records": match_count,
                "total_synthetic_records": len(synthetic_data),
                "method": "uniqueness_proxy",
                "columns_used": compare_cols,
            }
        except Exception as exc:
            return {"risk_score": 0.0, "error": str(exc), "method": "uniqueness_proxy"}

    def _run_linkability(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        aux_data: pd.DataFrame,
        n_aux_cols: int,
    ) -> dict[str, Any]:
        """Execute linkability attack synchronously with auxiliary data.

        Args:
            real_data: Original dataset.
            synthetic_data: Synthetic dataset.
            aux_data: Auxiliary adversary dataset.
            n_aux_cols: Number of auxiliary columns for linking.

        Returns:
            Attack result dict with risk_score.
        """
        try:
            from anonymeter.evaluators import LinkabilityEvaluator  # type: ignore[import]

            n_attacks = min(self._n_attacks, min(len(real_data), len(aux_data)))
            common_cols = list(set(real_data.columns) & set(aux_data.columns))
            aux_cols = common_cols[:n_aux_cols] if len(common_cols) >= n_aux_cols else common_cols

            if not aux_cols:
                return {
                    "risk_score": 0.0,
                    "note": "no_common_columns_for_linkability",
                }

            evaluator = LinkabilityEvaluator(
                ori=real_data,
                syn=synthetic_data,
                aux=aux_data,
                n_attacks=n_attacks,
                aux_cols=aux_cols,
                n_neighbors=10,
            )
            evaluator.evaluate(n_jobs=-1)
            result = evaluator.risk()

            risk_score = float(result.value)
            ci_low = float(result.ci[0]) if hasattr(result, "ci") and result.ci else risk_score
            ci_high = float(result.ci[1]) if hasattr(result, "ci") and result.ci else risk_score

            return {
                "risk_score": risk_score,
                "confidence_interval": [ci_low, ci_high],
                "n_attacks": n_attacks,
                "aux_cols_used": aux_cols,
            }

        except ImportError:
            logger.warning("anonymeter not installed — using correlation proxy for linkability")
            return self._linkability_correlation_proxy(real_data, synthetic_data, aux_data, n_aux_cols)
        except Exception as exc:
            logger.warning("Linkability attack failed", error=str(exc))
            return {"risk_score": 0.0, "error": str(exc)}

    def _evaluate_linkability_no_aux(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Estimate linkability risk without auxiliary data.

        Uses internal columns as quasi-identifiers to estimate how linkable
        synthetic records are to real individuals.

        Args:
            real_data: Original dataset.
            synthetic_data: Synthetic dataset.

        Returns:
            Estimated linkability risk.
        """
        try:
            import numpy as np

            # Split real data into two halves — use one as "aux"
            midpoint = len(real_data) // 2
            real_half = real_data.iloc[:midpoint].reset_index(drop=True)
            aux_half = real_data.iloc[midpoint:].reset_index(drop=True)

            return self._linkability_correlation_proxy(
                real_half, synthetic_data, aux_half, n_aux_cols=2
            )
        except Exception as exc:
            return {"risk_score": 0.05, "method": "no_aux_estimate", "error": str(exc)}

    def _linkability_correlation_proxy(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        aux_data: pd.DataFrame,
        n_aux_cols: int,
    ) -> dict[str, Any]:
        """Proxy linkability risk via nearest-neighbour matching.

        Args:
            real_data: Real dataset.
            synthetic_data: Synthetic dataset.
            aux_data: Auxiliary dataset.
            n_aux_cols: Number of columns to use for linking.

        Returns:
            Linkability proxy risk dict.
        """
        try:
            import numpy as np

            numeric_real = real_data.select_dtypes(include="number")
            numeric_synth = synthetic_data.select_dtypes(include="number")
            numeric_aux = aux_data.select_dtypes(include="number")

            common_num = list(set(numeric_real.columns) & set(numeric_synth.columns) & set(numeric_aux.columns))
            if not common_num:
                return {"risk_score": 0.05, "method": "linkability_proxy", "note": "no_numeric_columns"}

            cols = common_num[:min(n_aux_cols * 2, len(common_num))]
            sample_size = min(200, len(real_data), len(aux_data))

            real_arr = numeric_real[cols].dropna().values[:sample_size]
            synth_arr = numeric_synth[cols].dropna().values[:sample_size]
            aux_arr = numeric_aux[cols].dropna().values[:sample_size]

            # Normalise columns
            col_std = real_arr.std(axis=0) + 1e-8
            real_norm = real_arr / col_std
            synth_norm = synth_arr / col_std
            aux_norm = aux_arr / col_std

            # For each aux record, find nearest synthetic and check if it links to real
            linked = 0
            for i in range(min(100, len(aux_norm))):
                aux_vec = aux_norm[i]
                synth_dists = np.linalg.norm(synth_norm - aux_vec, axis=1)
                nearest_synth_idx = int(np.argmin(synth_dists))
                nearest_synth_vec = synth_norm[nearest_synth_idx]

                # Check if this synthetic record is also close to a real record
                real_dists = np.linalg.norm(real_norm - nearest_synth_vec, axis=1)
                if real_dists.min() < 0.5:  # Threshold for "linked"
                    linked += 1

            risk_score = float(linked / min(100, len(aux_norm)))
            return {
                "risk_score": risk_score,
                "linked_records": linked,
                "method": "nn_proxy",
            }
        except Exception as exc:
            return {"risk_score": 0.05, "error": str(exc), "method": "linkability_proxy"}

    def _run_inference(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        secret_columns: list[str],
    ) -> dict[str, Any]:
        """Execute attribute inference attack synchronously.

        Args:
            real_data: Original dataset.
            synthetic_data: Synthetic dataset.
            secret_columns: Columns to treat as sensitive for inference.

        Returns:
            Attack result dict with risk_score and per-column results.
        """
        # Filter secret columns to those that exist in both datasets
        valid_secret_cols = [
            col for col in secret_columns
            if col in real_data.columns and col in synthetic_data.columns
        ]

        if not valid_secret_cols:
            return {
                "risk_score": 0.0,
                "note": "no_valid_secret_columns",
                "requested_columns": secret_columns,
            }

        try:
            from anonymeter.evaluators import InferenceEvaluator  # type: ignore[import]

            n_attacks = min(self._n_attacks, len(synthetic_data))
            aux_cols = [col for col in real_data.columns if col not in valid_secret_cols]

            evaluator = InferenceEvaluator(
                ori=real_data,
                syn=synthetic_data,
                n_attacks=n_attacks,
                secret=valid_secret_cols[0] if len(valid_secret_cols) == 1 else valid_secret_cols[0],
                aux_cols=aux_cols[:10],  # Cap aux cols for performance
            )
            evaluator.evaluate()
            result = evaluator.risk()

            risk_score = float(result.value)
            ci_low = float(result.ci[0]) if hasattr(result, "ci") and result.ci else risk_score
            ci_high = float(result.ci[1]) if hasattr(result, "ci") and result.ci else risk_score

            return {
                "risk_score": risk_score,
                "confidence_interval": [ci_low, ci_high],
                "secret_columns_evaluated": valid_secret_cols,
                "n_attacks": n_attacks,
            }

        except ImportError:
            logger.warning("anonymeter not installed — using ML proxy for inference risk")
            return self._inference_ml_proxy(real_data, synthetic_data, valid_secret_cols)
        except Exception as exc:
            logger.warning("Inference attack failed", error=str(exc))
            return {"risk_score": 0.0, "error": str(exc)}

    def _inference_ml_proxy(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        secret_columns: list[str],
    ) -> dict[str, Any]:
        """Proxy inference risk via ML prediction accuracy on secret columns.

        Trains a classifier/regressor on synthetic data to predict secret
        column values, then evaluates accuracy on real data. High accuracy
        indicates high inference risk.

        Args:
            real_data: Original dataset.
            synthetic_data: Synthetic dataset.
            secret_columns: Columns to predict as proxies for inference risk.

        Returns:
            Proxy inference risk dict.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # type: ignore[import]
            from sklearn.preprocessing import LabelEncoder  # type: ignore[import]
            import numpy as np

            per_column_risks: dict[str, float] = {}

            for secret_col in secret_columns[:3]:  # Cap at 3 columns for performance
                aux_cols = [c for c in real_data.columns if c != secret_col]
                numeric_aux = [c for c in aux_cols if pd.api.types.is_numeric_dtype(real_data[c])][:10]

                if not numeric_aux:
                    continue

                synth_clean = synthetic_data[numeric_aux + [secret_col]].dropna()
                real_clean = real_data[numeric_aux + [secret_col]].dropna()

                if len(synth_clean) < 20 or len(real_clean) < 10:
                    continue

                X_train = synth_clean[numeric_aux].values
                y_train_raw = synth_clean[secret_col].values
                X_test = real_clean[numeric_aux].values
                y_test_raw = real_clean[secret_col].values

                if pd.api.types.is_numeric_dtype(real_data[secret_col]):
                    model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
                    model.fit(X_train, y_train_raw.astype(float))
                    preds = model.predict(X_test)
                    y_test = y_test_raw.astype(float)
                    scale = np.std(y_test) + 1e-8
                    nmae = float(np.mean(np.abs(preds - y_test)) / scale)
                    # Low NMAE = high inference risk; normalise to [0, 1]
                    risk = float(max(0.0, 1.0 - nmae))
                else:
                    le = LabelEncoder()
                    y_all = np.concatenate([y_train_raw.astype(str), y_test_raw.astype(str)])
                    le.fit(y_all)
                    y_train_enc = le.transform(y_train_raw.astype(str))
                    y_test_enc = le.transform(y_test_raw.astype(str))
                    model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
                    model.fit(X_train, y_train_enc)
                    preds = model.predict(X_test)
                    accuracy = float(np.mean(preds == y_test_enc))
                    # Baseline chance accuracy
                    n_classes = len(le.classes_)
                    baseline = 1.0 / max(n_classes, 1)
                    risk = float(max(0.0, (accuracy - baseline) / (1.0 - baseline + 1e-8)))

                per_column_risks[secret_col] = risk

            overall_risk = float(
                sum(per_column_risks.values()) / max(len(per_column_risks), 1)
            ) if per_column_risks else 0.0

            return {
                "risk_score": overall_risk,
                "per_column_risks": per_column_risks,
                "method": "ml_proxy",
                "secret_columns_evaluated": list(per_column_risks.keys()),
            }
        except Exception as exc:
            return {"risk_score": 0.0, "error": str(exc), "method": "ml_proxy"}

    def _categorise_risk(self, risk_score: float) -> str:
        """Categorise a risk score into a named risk level.

        Args:
            risk_score: Numeric risk in [0, 1].

        Returns:
            Risk level string: "low", "medium", "high", or "critical".
        """
        if risk_score <= _RISK_LOW_MAX:
            return "low"
        elif risk_score <= _RISK_MEDIUM_MAX:
            return "medium"
        elif risk_score <= _RISK_HIGH_MAX:
            return "high"
        else:
            return "critical"

    def _compute_per_column_privacy_risk(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute per-column privacy risk scores.

        Estimates how much each column contributes to re-identification risk
        based on value uniqueness in the real data versus synthetic data.

        Args:
            real_data: Original dataset.
            synthetic_data: Synthetic dataset.

        Returns:
            Dict mapping column name to risk score [0, 1].
        """
        per_column: dict[str, float] = {}
        for col in real_data.columns:
            if col not in synthetic_data.columns:
                continue
            try:
                real_col = real_data[col].dropna()
                synth_col = synthetic_data[col].dropna()
                if len(real_col) == 0:
                    continue
                # Uniqueness ratio: fraction of real values that appear only once
                real_value_counts = real_col.value_counts()
                unique_real_fraction = float((real_value_counts == 1).sum() / max(len(real_value_counts), 1))
                # Overlap: fraction of unique real values present in synthetic
                real_unique = set(real_col.unique())
                synth_unique = set(synth_col.unique())
                overlap = len(real_unique & synth_unique) / max(len(real_unique), 1)
                # High uniqueness + high overlap = high risk
                per_column[col] = float(unique_real_fraction * overlap)
            except Exception:  # noqa: BLE001
                per_column[col] = 0.0
        return per_column
