"""Healthcare metrics adapter — evaluates healthcare-specific data fidelity.

Implements HealthcareMetricsProtocol validating FHIR resource structure,
clinical realism (diagnosis-treatment consistency), ICD-10/CPT code alignment,
lab value plausibility, and medication interaction safety scoring.
"""

import asyncio
import re
from typing import Any

import pandas as pd

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Normal reference ranges for common lab values
# Format: (low_critical, low_normal, high_normal, high_critical)
_LAB_REFERENCE_RANGES: dict[str, tuple[float, float, float, float]] = {
    "glucose_mg_dl": (40.0, 70.0, 100.0, 500.0),
    "hba1c_percent": (2.0, 4.0, 5.7, 15.0),
    "creatinine_mg_dl": (0.2, 0.6, 1.2, 15.0),
    "sodium_meq_l": (100.0, 136.0, 145.0, 170.0),
    "potassium_meq_l": (2.0, 3.5, 5.0, 8.0),
    "hemoglobin_g_dl": (5.0, 12.0, 17.5, 25.0),
    "wbc_k_ul": (0.5, 4.5, 11.0, 100.0),
    "platelet_k_ul": (20.0, 150.0, 400.0, 2000.0),
    "alt_u_l": (0.0, 0.0, 56.0, 1000.0),
    "ast_u_l": (0.0, 0.0, 40.0, 1000.0),
    "total_cholesterol_mg_dl": (50.0, 0.0, 200.0, 600.0),
    "ldl_mg_dl": (10.0, 0.0, 130.0, 500.0),
    "bmi": (10.0, 18.5, 30.0, 70.0),
    "systolic_bp_mmhg": (60.0, 90.0, 120.0, 250.0),
    "diastolic_bp_mmhg": (30.0, 60.0, 80.0, 150.0),
    "heart_rate_bpm": (20.0, 60.0, 100.0, 250.0),
    "temperature_f": (90.0, 97.0, 99.5, 107.0),
    "oxygen_saturation_percent": (70.0, 95.0, 100.0, 100.0),
}

# FHIR resource required fields
_FHIR_REQUIRED_FIELDS: dict[str, list[str]] = {
    "Patient": ["id", "resourceType"],
    "Condition": ["id", "resourceType", "subject", "clinicalStatus"],
    "Observation": ["id", "resourceType", "status", "code", "subject"],
    "MedicationRequest": ["id", "resourceType", "status", "intent", "subject"],
    "Encounter": ["id", "resourceType", "status", "subject"],
    "DiagnosticReport": ["id", "resourceType", "status", "code", "subject"],
}

# ICD-10 format regex: letter + 2 digits + optional decimal + alphanumeric
_ICD10_PATTERN = re.compile(r"^[A-Z][0-9]{2}(\.[A-Z0-9]{1,4})?$")
# CPT code format: 5 digits (possibly with suffix)
_CPT_PATTERN = re.compile(r"^[0-9]{5}[FUMT]?$")


class HealthcareMetricsEvaluator:
    """Validates healthcare data fidelity across clinical and technical dimensions.

    Evaluates FHIR bundle structure, clinical realism of diagnosis-treatment
    pairs, ICD-10/CPT code validity, lab value plausibility, and medication
    interaction safety for synthetic healthcare datasets.
    """

    async def evaluate(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        fhir_bundles: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
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
        logger.info(
            "Running healthcare metrics evaluation",
            real_rows=len(real_data),
            synthetic_rows=len(synthetic_data),
            fhir_bundle_count=len(fhir_bundles) if fhir_bundles else 0,
        )
        loop = asyncio.get_running_loop()

        fhir_result: dict[str, Any] = {}
        if fhir_bundles:
            fhir_result = await loop.run_in_executor(
                None, self._validate_fhir_bundles, fhir_bundles
            )

        clinical_result, code_result, lab_result, med_result = await asyncio.gather(
            loop.run_in_executor(None, self._score_clinical_realism, real_data, synthetic_data, metadata or {}),
            loop.run_in_executor(None, self._validate_code_alignment, real_data, synthetic_data, metadata or {}),
            loop.run_in_executor(None, self._check_lab_plausibility, synthetic_data, metadata or {}),
            loop.run_in_executor(None, self._score_medication_safety, synthetic_data, metadata or {}),
        )

        fhir_score = fhir_result.get("validation_score", 1.0) if fhir_bundles else 1.0
        clinical_score = clinical_result.get("clinical_realism_score", 0.0)
        code_score = code_result.get("code_alignment_score", 0.0)
        lab_score = lab_result.get("lab_plausibility_score", 0.0)
        med_score = med_result.get("medication_safety_score", 0.0)

        # FHIR only contributes if bundles were provided
        if fhir_bundles:
            overall_score = (
                fhir_score * 0.20
                + clinical_score * 0.30
                + code_score * 0.20
                + lab_score * 0.20
                + med_score * 0.10
            )
        else:
            overall_score = (
                clinical_score * 0.35
                + code_score * 0.25
                + lab_score * 0.25
                + med_score * 0.15
            )

        return {
            "overall_score": float(overall_score),
            "fhir_validation": fhir_result,
            "fhir_score": float(fhir_score),
            "clinical_realism": clinical_result,
            "clinical_score": float(clinical_score),
            "code_alignment": code_result,
            "code_score": float(code_score),
            "lab_plausibility": lab_result,
            "lab_score": float(lab_score),
            "medication_safety": med_result,
            "medication_score": float(med_score),
        }

    def _validate_fhir_bundles(
        self,
        fhir_bundles: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Validate FHIR resource bundles for required fields and structure.

        Checks that each resource in each bundle has the required FHIR fields
        for its resourceType, and that the bundle itself is well-formed.

        Args:
            fhir_bundles: List of FHIR bundle dicts.

        Returns:
            Dict with validation_score, total_resources, invalid_resources.
        """
        total_resources = 0
        invalid_resources = 0
        error_details: list[dict[str, Any]] = []

        for bundle_idx, bundle in enumerate(fhir_bundles):
            # Bundle must have resourceType = Bundle and entry list
            if bundle.get("resourceType") != "Bundle":
                invalid_resources += 1
                error_details.append({"bundle": bundle_idx, "error": "missing_bundle_resource_type"})
                continue

            entries = bundle.get("entry", [])
            if not isinstance(entries, list):
                invalid_resources += 1
                error_details.append({"bundle": bundle_idx, "error": "invalid_entry_format"})
                continue

            for entry in entries:
                resource = entry.get("resource", {})
                if not isinstance(resource, dict):
                    total_resources += 1
                    invalid_resources += 1
                    error_details.append({"bundle": bundle_idx, "error": "non_dict_resource"})
                    continue

                resource_type = resource.get("resourceType", "")
                required_fields = _FHIR_REQUIRED_FIELDS.get(resource_type, ["id", "resourceType"])
                total_resources += 1

                missing_fields = [f for f in required_fields if f not in resource]
                if missing_fields:
                    invalid_resources += 1
                    error_details.append({
                        "bundle": bundle_idx,
                        "resource_type": resource_type,
                        "missing_fields": missing_fields,
                    })

        validation_score = float(1.0 - (invalid_resources / max(total_resources, 1)))

        return {
            "validation_score": validation_score,
            "total_resources": total_resources,
            "invalid_resources": invalid_resources,
            "error_sample": error_details[:20],
        }

    def _score_clinical_realism(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Score clinical realism of diagnosis-treatment pairs.

        Checks for plausible age-diagnosis combinations, consistent
        treatment-diagnosis mappings, and physiologically reasonable
        vital sign combinations in synthetic records.

        Args:
            real_data: Real patient records.
            synthetic_data: Synthetic patient records.
            metadata: Column metadata with clinical type hints.

        Returns:
            Dict with clinical_realism_score and per-check results.
        """
        checks: dict[str, Any] = {}
        scores: list[float] = []

        # Age distribution plausibility
        age_cols = self._find_columns(synthetic_data, metadata, "age")
        if age_cols:
            age_col = age_cols[0]
            if synthetic_data[age_col].dtype in (float, int) or pd.api.types.is_numeric_dtype(synthetic_data[age_col]):
                ages = pd.to_numeric(synthetic_data[age_col], errors="coerce").dropna()
                valid_age = float(((ages >= 0) & (ages <= 120)).mean())
                checks["age_plausibility"] = float(valid_age)
                scores.append(valid_age)

        # Vital sign co-plausibility (BP systolic > diastolic)
        sbp_cols = self._find_columns(synthetic_data, metadata, "systolic_bp")
        dbp_cols = self._find_columns(synthetic_data, metadata, "diastolic_bp")
        if sbp_cols and dbp_cols:
            sbp = pd.to_numeric(synthetic_data[sbp_cols[0]], errors="coerce")
            dbp = pd.to_numeric(synthetic_data[dbp_cols[0]], errors="coerce")
            valid_mask = sbp.notna() & dbp.notna()
            if valid_mask.sum() > 0:
                bp_valid = float((sbp[valid_mask] > dbp[valid_mask]).mean())
                checks["bp_systolic_gt_diastolic"] = float(bp_valid)
                scores.append(bp_valid)

        # Distribution similarity with real data for numerical columns
        numeric_real = real_data.select_dtypes(include="number")
        numeric_synth = synthetic_data.select_dtypes(include="number")
        common_cols = list(set(numeric_real.columns) & set(numeric_synth.columns))

        if common_cols:
            from scipy.stats import ks_2samp

            ks_scores: list[float] = []
            for col in common_cols[:10]:
                real_vals = numeric_real[col].dropna()
                synth_vals = numeric_synth[col].dropna()
                if len(real_vals) > 10 and len(synth_vals) > 10:
                    stat, _ = ks_2samp(real_vals.values, synth_vals.values)
                    ks_scores.append(float(1.0 - stat))
            if ks_scores:
                distribution_score = float(sum(ks_scores) / len(ks_scores))
                checks["distribution_similarity"] = distribution_score
                scores.append(distribution_score)

        clinical_realism_score = float(sum(scores) / len(scores)) if scores else 0.5

        return {
            "clinical_realism_score": clinical_realism_score,
            "checks": checks,
        }

    def _validate_code_alignment(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate ICD-10 and CPT code format and alignment.

        Checks that synthetic ICD-10 diagnosis codes and CPT procedure codes
        match the expected format and that the distribution of code prefixes
        is similar to the real data.

        Args:
            real_data: Real healthcare records.
            synthetic_data: Synthetic healthcare records.
            metadata: Column metadata.

        Returns:
            Dict with code_alignment_score, icd10_validity, cpt_validity.
        """
        scores: list[float] = []
        result: dict[str, Any] = {}

        # Find ICD-10 columns
        icd10_cols = self._find_columns(synthetic_data, metadata, "icd10") or self._find_columns(
            synthetic_data, metadata, "diagnosis_code"
        )
        if icd10_cols:
            col = icd10_cols[0]
            codes = synthetic_data[col].dropna().astype(str)
            if len(codes) > 0:
                valid_mask = codes.str.strip().apply(lambda c: bool(_ICD10_PATTERN.match(c.upper())))
                validity_rate = float(valid_mask.mean())
                result["icd10_validity_rate"] = validity_rate
                scores.append(validity_rate)

                # Code prefix distribution alignment
                if col in real_data.columns:
                    real_codes = real_data[col].dropna().astype(str)
                    real_prefixes = real_codes.str[:1].value_counts(normalize=True)
                    synth_prefixes = codes.str[:1].value_counts(normalize=True)
                    all_prefixes = set(real_prefixes.index) | set(synth_prefixes.index)
                    tvd = float(
                        0.5
                        * sum(
                            abs(real_prefixes.get(p, 0) - synth_prefixes.get(p, 0))
                            for p in all_prefixes
                        )
                    )
                    result["icd10_prefix_tvd"] = tvd
                    scores.append(float(1.0 - tvd))

        # Find CPT columns
        cpt_cols = self._find_columns(synthetic_data, metadata, "cpt") or self._find_columns(
            synthetic_data, metadata, "procedure_code"
        )
        if cpt_cols:
            col = cpt_cols[0]
            codes = synthetic_data[col].dropna().astype(str)
            if len(codes) > 0:
                valid_mask = codes.str.strip().apply(lambda c: bool(_CPT_PATTERN.match(c)))
                validity_rate = float(valid_mask.mean())
                result["cpt_validity_rate"] = validity_rate
                scores.append(validity_rate)

        result["code_alignment_score"] = float(sum(scores) / len(scores)) if scores else 0.5
        return result

    def _check_lab_plausibility(
        self,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Check lab value plausibility against clinical reference ranges.

        Validates that synthetic lab values fall within physiologically
        possible ranges (not just normal ranges). Values outside the
        critical range are marked as implausible.

        Args:
            synthetic_data: Synthetic patient records.
            metadata: Column metadata with lab type hints.

        Returns:
            Dict with lab_plausibility_score and per-lab results.
        """
        per_lab: dict[str, Any] = {}
        plausibility_scores: list[float] = []

        for lab_name, (critical_low, _normal_low, _normal_high, critical_high) in _LAB_REFERENCE_RANGES.items():
            cols = self._find_columns(synthetic_data, metadata, lab_name)
            if not cols:
                # Try matching column names directly
                matching = [c for c in synthetic_data.columns if lab_name.replace("_", " ") in c.lower().replace("_", " ")]
                if matching:
                    cols = [matching[0]]

            if not cols:
                continue

            col = cols[0]
            values = pd.to_numeric(synthetic_data[col], errors="coerce").dropna()
            if len(values) == 0:
                continue

            within_critical = float(((values >= critical_low) & (values <= critical_high)).mean())
            per_lab[lab_name] = {
                "column": col,
                "plausible_fraction": within_critical,
                "sample_count": len(values),
                "critical_low": critical_low,
                "critical_high": critical_high,
            }
            plausibility_scores.append(within_critical)

        overall_plausibility = float(sum(plausibility_scores) / len(plausibility_scores)) if plausibility_scores else 0.8

        return {
            "lab_plausibility_score": overall_plausibility,
            "labs_evaluated": len(plausibility_scores),
            "per_lab_results": per_lab,
        }

    def _score_medication_safety(
        self,
        synthetic_data: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Score medication interaction safety in synthetic records.

        Detects obvious contraindicated medication combinations and
        checks that medication codes follow expected formats. A score
        of 1.0 means no unsafe combinations detected.

        Args:
            synthetic_data: Synthetic patient records.
            metadata: Column metadata.

        Returns:
            Dict with medication_safety_score and flagged_combinations.
        """
        # Known dangerous drug combinations (simplified subset for scoring)
        # In production this would use a full drug interaction database
        _CONTRAINDICATED_PAIRS: list[tuple[str, str]] = [
            ("warfarin", "aspirin"),
            ("warfarin", "ibuprofen"),
            ("maoi", "ssri"),
            ("maoi", "tramadol"),
            ("metformin", "contrast_dye"),
            ("sildenafil", "nitrate"),
        ]

        med_cols = self._find_columns(synthetic_data, metadata, "medication") or self._find_columns(
            synthetic_data, metadata, "drug"
        )

        if not med_cols:
            return {"medication_safety_score": 1.0, "note": "no_medication_columns_found"}

        flagged: list[dict[str, Any]] = []
        total_records_checked = 0

        # Group medications by patient if possible
        patient_id_cols = self._find_columns(synthetic_data, metadata, "patient_id")
        if patient_id_cols and len(med_cols) > 0:
            patient_col = patient_id_cols[0]
            med_col = med_cols[0]
            try:
                patient_meds = (
                    synthetic_data.groupby(patient_col)[med_col]
                    .apply(lambda x: [str(v).lower() for v in x.dropna()])
                )
                total_records_checked = len(patient_meds)

                for patient_id, med_list in patient_meds.items():
                    for drug_a, drug_b in _CONTRAINDICATED_PAIRS:
                        if any(drug_a in m for m in med_list) and any(drug_b in m for m in med_list):
                            flagged.append({
                                "patient_id": str(patient_id),
                                "drug_a": drug_a,
                                "drug_b": drug_b,
                            })
            except Exception as exc:
                logger.debug("Medication grouping failed", error=str(exc))

        flagged_rate = float(len(flagged) / max(total_records_checked, 1))
        safety_score = float(max(0.0, 1.0 - flagged_rate * 10))  # Penalise heavily

        return {
            "medication_safety_score": safety_score,
            "flagged_combination_count": len(flagged),
            "total_records_checked": total_records_checked,
            "flagged_sample": flagged[:10],
        }

    def _find_columns(
        self,
        data: pd.DataFrame,
        metadata: dict[str, Any],
        clinical_type: str,
    ) -> list[str]:
        """Find columns in a DataFrame that match a clinical type.

        Checks metadata first, then falls back to fuzzy column name matching.

        Args:
            data: DataFrame to search.
            metadata: Column metadata dict.
            clinical_type: Clinical type string (e.g., "age", "icd10").

        Returns:
            List of matching column names.
        """
        # Check metadata for explicit mapping
        column_meta = metadata.get("columns", {})
        explicit = [
            col
            for col, meta in column_meta.items()
            if meta.get("clinical_type") == clinical_type and col in data.columns
        ]
        if explicit:
            return explicit

        # Fuzzy name matching
        search_term = clinical_type.replace("_", " ").lower()
        fuzzy = [
            col
            for col in data.columns
            if search_term in col.lower().replace("_", " ")
        ]
        return fuzzy
