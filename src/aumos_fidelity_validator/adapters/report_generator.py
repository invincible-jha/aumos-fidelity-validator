"""Report generator adapter — produces JSON and PDF fidelity reports.

Implements FidelityReportGeneratorProtocol generating structured JSON reports
with per-column breakdowns, executive summaries, comparison tables, and
PDF reports via reportlab. Stores reports to MinIO/S3 via StorageProtocol.
"""

import asyncio
import io
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# ReportLab colours and fonts
_PASS_COLOR_RGB = (0.18, 0.64, 0.34)   # Green
_FAIL_COLOR_RGB = (0.83, 0.18, 0.18)   # Red
_HEADER_COLOR_RGB = (0.07, 0.20, 0.45)  # Dark navy
_SECTION_COLOR_RGB = (0.93, 0.95, 0.98)  # Light blue-grey


class FidelityReportGenerator:
    """Generates comprehensive fidelity reports in JSON and PDF formats.

    Produces structured JSON reports with all metric results, per-column
    breakdowns, metric visualisation data, and cross-run comparison tables.
    Also generates PDF reports via reportlab with executive summary and
    pass/fail verdicts.
    """

    # Maximum number of column entries to include in the PDF table
    _PDF_MAX_COLUMN_ROWS: int = 40

    async def generate_json_report(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        fidelity_report: dict[str, Any],
        privacy_report: dict[str, Any] | None,
        memorization_report: dict[str, Any] | None,
        overall_score: float,
        passed: bool,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON report from all evaluation results.

        Args:
            job_id: Validation job UUID.
            tenant_id: Tenant identifier.
            fidelity_report: SDMetrics or tabular fidelity evaluation results.
            privacy_report: Anonymeter privacy risk results (optional).
            memorization_report: Memorization attack results (optional).
            overall_score: Aggregate score in [0, 1].
            passed: Whether all thresholds were met.
            metadata: Optional column metadata for richer report detail.

        Returns:
            Structured JSON-serialisable report dict.
        """
        logger.info(
            "Generating JSON fidelity report",
            job_id=str(job_id),
            tenant_id=tenant_id,
            overall_score=overall_score,
            passed=passed,
        )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._build_json_report,
            job_id,
            tenant_id,
            fidelity_report,
            privacy_report,
            memorization_report,
            overall_score,
            passed,
            metadata or {},
        )

    async def generate_pdf_report(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        fidelity_report: dict[str, Any],
        privacy_report: dict[str, Any] | None,
        memorization_report: dict[str, Any] | None,
        overall_score: float,
        passed: bool,
    ) -> bytes:
        """Generate a PDF compliance report using reportlab.

        Args:
            job_id: Validation job UUID.
            tenant_id: Tenant identifier.
            fidelity_report: Fidelity evaluation results.
            privacy_report: Privacy risk results (optional).
            memorization_report: Memorization attack results (optional).
            overall_score: Aggregate score in [0, 1].
            passed: Whether all thresholds were met.

        Returns:
            PDF bytes ready for storage or download.
        """
        logger.info(
            "Generating PDF fidelity report",
            job_id=str(job_id),
            tenant_id=tenant_id,
            overall_score=overall_score,
        )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._build_pdf_report,
            job_id,
            tenant_id,
            fidelity_report,
            privacy_report,
            memorization_report,
            overall_score,
            passed,
        )

    def _build_json_report(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        fidelity_report: dict[str, Any],
        privacy_report: dict[str, Any] | None,
        memorization_report: dict[str, Any] | None,
        overall_score: float,
        passed: bool,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Synchronous JSON report construction."""
        executive_summary = self._build_executive_summary(
            fidelity_report, privacy_report, memorization_report, overall_score, passed
        )

        column_breakdown = self._build_column_breakdown(fidelity_report)
        visualisation_data = self._build_visualisation_data(fidelity_report)

        report = {
            "report_version": "1.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "job_id": str(job_id),
            "tenant_id": tenant_id,
            "overall_score": float(overall_score),
            "passed": bool(passed),
            "executive_summary": executive_summary,
            "fidelity_report": fidelity_report,
            "privacy_report": privacy_report,
            "memorization_report": memorization_report,
            "column_breakdown": column_breakdown,
            "visualisation_data": visualisation_data,
        }
        return report

    def _build_executive_summary(
        self,
        fidelity_report: dict[str, Any],
        privacy_report: dict[str, Any] | None,
        memorization_report: dict[str, Any] | None,
        overall_score: float,
        passed: bool,
    ) -> dict[str, Any]:
        """Build a concise executive summary section.

        Args:
            fidelity_report: Full fidelity evaluation results.
            privacy_report: Privacy risk results.
            memorization_report: Memorization results.
            overall_score: Aggregate score.
            passed: Overall pass/fail verdict.

        Returns:
            Executive summary dict.
        """
        fidelity_score = fidelity_report.get("overall_score", 0.0)
        fidelity_passed = fidelity_score >= 0.82

        privacy_risk_level = "not_evaluated"
        if privacy_report:
            overall_risk = privacy_report.get("overall_risk_level", "unknown")
            privacy_risk_level = str(overall_risk)

        memorization_risk = "not_evaluated"
        if memorization_report:
            attack_auc = memorization_report.get("attack_auc", 0.0)
            if attack_auc < 0.55:
                memorization_risk = "low"
            elif attack_auc < 0.65:
                memorization_risk = "medium"
            else:
                memorization_risk = "high"

        verdict = "PASS" if passed else "FAIL"
        verdict_explanation = (
            "All fidelity, privacy, and memorization thresholds met."
            if passed
            else self._build_failure_explanation(
                fidelity_passed, privacy_report, memorization_report
            )
        )

        return {
            "verdict": verdict,
            "overall_score": float(overall_score),
            "verdict_explanation": verdict_explanation,
            "fidelity_score": float(fidelity_score),
            "fidelity_passed": bool(fidelity_passed),
            "privacy_risk_level": privacy_risk_level,
            "memorization_risk": memorization_risk,
            "category_scores": {
                "marginal": fidelity_report.get("marginal", {}).get("score", None),
                "pairwise": fidelity_report.get("pairwise", {}).get("score", None),
                "table_level": fidelity_report.get("table_level", {}).get("score", None),
            },
        }

    def _build_failure_explanation(
        self,
        fidelity_passed: bool,
        privacy_report: dict[str, Any] | None,
        memorization_report: dict[str, Any] | None,
    ) -> str:
        """Build a human-readable explanation of why validation failed.

        Args:
            fidelity_passed: Whether fidelity threshold was met.
            privacy_report: Privacy evaluation results.
            memorization_report: Memorization evaluation results.

        Returns:
            Failure explanation string.
        """
        reasons: list[str] = []
        if not fidelity_passed:
            reasons.append("Fidelity score below threshold (0.82)")
        if privacy_report and not privacy_report.get("passed", True):
            failing = []
            if privacy_report.get("singling_out_risk", 0) > 0.05:
                failing.append("singling-out risk too high")
            if privacy_report.get("linkability_risk", 0) > 0.10:
                failing.append("linkability risk too high")
            if privacy_report.get("inference_risk", 0) > 0.15:
                failing.append("inference risk too high")
            reasons.append(f"Privacy thresholds failed: {', '.join(failing)}")
        if memorization_report and not memorization_report.get("passed", True):
            auc = memorization_report.get("attack_auc", 1.0)
            reasons.append(f"Memorization AUC ({auc:.3f}) exceeds threshold (0.60)")
        return "; ".join(reasons) if reasons else "Validation failed for unknown reasons."

    def _build_column_breakdown(
        self,
        fidelity_report: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Extract per-column metric breakdown for the report.

        Args:
            fidelity_report: Fidelity evaluation results with marginal data.

        Returns:
            List of per-column metric rows sorted by score ascending.
        """
        column_metrics: dict[str, Any] = (
            fidelity_report.get("marginal", {}).get("metrics", {})
            or fidelity_report.get("column_metrics", {})
        )
        rows: list[dict[str, Any]] = []

        for column_name, metrics in column_metrics.items():
            if not isinstance(metrics, dict):
                continue
            row: dict[str, Any] = {"column": column_name}
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[metric_name] = float(value)
                elif isinstance(value, str):
                    row[metric_name] = value
            rows.append(row)

        # Sort by column_score ascending (worst first for attention)
        rows.sort(key=lambda r: float(r.get("column_score", 1.0)))
        return rows

    def _build_visualisation_data(
        self,
        fidelity_report: dict[str, Any],
    ) -> dict[str, Any]:
        """Build histogram and distribution data for visualisation.

        Extracts score distributions and category-level summaries
        suitable for rendering bar charts and histograms.

        Args:
            fidelity_report: Full fidelity report.

        Returns:
            Dict with category_scores, column_score_histogram.
        """
        column_metrics = (
            fidelity_report.get("marginal", {}).get("metrics", {})
            or fidelity_report.get("column_metrics", {})
        )

        column_scores = [
            float(metrics.get("column_score", 0.0))
            for metrics in column_metrics.values()
            if isinstance(metrics, dict) and "column_score" in metrics
        ]

        # Build histogram buckets for column scores
        import numpy as np
        if column_scores:
            hist, bin_edges = np.histogram(column_scores, bins=10, range=(0.0, 1.0))
            histogram_data = [
                {
                    "bucket_start": float(bin_edges[i]),
                    "bucket_end": float(bin_edges[i + 1]),
                    "count": int(hist[i]),
                }
                for i in range(len(hist))
            ]
        else:
            histogram_data = []

        category_scores = {
            "marginal": fidelity_report.get("marginal", {}).get("score", None),
            "pairwise": fidelity_report.get("pairwise", {}).get("score", None),
            "table_level": fidelity_report.get("table_level", {}).get("score", None),
        }

        return {
            "category_scores": {k: float(v) for k, v in category_scores.items() if v is not None},
            "column_score_histogram": histogram_data,
            "total_columns": len(column_scores),
            "mean_column_score": float(sum(column_scores) / len(column_scores)) if column_scores else 0.0,
            "min_column_score": float(min(column_scores)) if column_scores else 0.0,
            "max_column_score": float(max(column_scores)) if column_scores else 0.0,
        }

    def _build_pdf_report(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        fidelity_report: dict[str, Any],
        privacy_report: dict[str, Any] | None,
        memorization_report: dict[str, Any] | None,
        overall_score: float,
        passed: bool,
    ) -> bytes:
        """Synchronous PDF report generation using reportlab.

        Args:
            job_id: Validation job UUID.
            tenant_id: Tenant identifier.
            fidelity_report: Fidelity evaluation results.
            privacy_report: Privacy risk results.
            memorization_report: Memorization attack results.
            overall_score: Aggregate score.
            passed: Overall pass/fail verdict.

        Returns:
            PDF file as bytes.
        """
        try:
            from reportlab.lib import colors  # type: ignore[import]
            from reportlab.lib.pagesizes import A4  # type: ignore[import]
            from reportlab.lib.styles import getSampleStyleSheet  # type: ignore[import]
            from reportlab.lib.units import cm  # type: ignore[import]
            from reportlab.platypus import (  # type: ignore[import]
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                topMargin=2 * cm,
                bottomMargin=2 * cm,
                leftMargin=2 * cm,
                rightMargin=2 * cm,
            )
            styles = getSampleStyleSheet()
            story = []

            # Title
            title_style = styles["Title"]
            story.append(Paragraph("AumOS Fidelity Validation Report", title_style))
            story.append(Spacer(1, 0.5 * cm))

            # Header info table
            generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
            header_data = [
                ["Job ID:", str(job_id)],
                ["Tenant:", tenant_id],
                ["Generated:", generated_at],
                ["Overall Score:", f"{overall_score:.4f}"],
                ["Verdict:", "PASS" if passed else "FAIL"],
            ]
            header_table = Table(header_data, colWidths=[4 * cm, 12 * cm])
            verdict_color = colors.Color(*_PASS_COLOR_RGB) if passed else colors.Color(*_FAIL_COLOR_RGB)
            header_table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("TEXTCOLOR", (1, 4), (1, 4), verdict_color),
                ("FONTNAME", (1, 4), (1, 4), "Helvetica-Bold"),
                ("FONTSIZE", (1, 4), (1, 4), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(header_table)
            story.append(Spacer(1, 0.8 * cm))

            # Fidelity scores section
            story.append(Paragraph("Fidelity Scores", styles["Heading2"]))
            fidelity_data = [["Category", "Score", "Status"]]
            categories = [
                ("Overall", fidelity_report.get("overall_score", 0.0), 0.82),
                ("Marginal", fidelity_report.get("marginal", {}).get("score", 0.0), 0.82),
                ("Pairwise", fidelity_report.get("pairwise", {}).get("score", 0.0), 0.82),
                ("Table Level", fidelity_report.get("table_level", {}).get("score", 0.0), 0.82),
            ]
            for cat_name, score, threshold in categories:
                if score is not None:
                    status = "Pass" if float(score) >= threshold else "Fail"
                    fidelity_data.append([cat_name, f"{float(score):.4f}", status])

            fidelity_table = Table(fidelity_data, colWidths=[7 * cm, 5 * cm, 4 * cm])
            fidelity_table.setStyle(self._get_table_style(colors))
            story.append(fidelity_table)
            story.append(Spacer(1, 0.6 * cm))

            # Privacy section
            if privacy_report:
                story.append(Paragraph("Privacy Risk Assessment", styles["Heading2"]))
                privacy_data = [["Attack Type", "Risk Score", "Threshold", "Status"]]
                risk_rows = [
                    ("Singling Out", privacy_report.get("singling_out_risk", "N/A"), 0.05),
                    ("Linkability", privacy_report.get("linkability_risk", "N/A"), 0.10),
                    ("Inference", privacy_report.get("inference_risk", "N/A"), 0.15),
                ]
                for attack_name, risk_value, threshold in risk_rows:
                    if isinstance(risk_value, (int, float)):
                        status = "Pass" if float(risk_value) <= threshold else "Fail"
                        privacy_data.append([
                            attack_name, f"{float(risk_value):.4f}", f"≤ {threshold}", status
                        ])
                privacy_table = Table(privacy_data, colWidths=[5 * cm, 4 * cm, 4 * cm, 3 * cm])
                privacy_table.setStyle(self._get_table_style(colors))
                story.append(privacy_table)
                story.append(Spacer(1, 0.6 * cm))

            # Memorization section
            if memorization_report:
                story.append(Paragraph("Memorization Attack Results", styles["Heading2"]))
                attack_auc = memorization_report.get("attack_auc", "N/A")
                mem_passed = memorization_report.get("passed", False)
                mem_data = [
                    ["Metric", "Value", "Status"],
                    [
                        "Membership Inference AUC",
                        f"{float(attack_auc):.4f}" if isinstance(attack_auc, (int, float)) else "N/A",
                        "Pass" if mem_passed else "Fail",
                    ],
                ]
                mem_table = Table(mem_data, colWidths=[8 * cm, 5 * cm, 3 * cm])
                mem_table.setStyle(self._get_table_style(colors))
                story.append(mem_table)
                story.append(Spacer(1, 0.6 * cm))

            # Column breakdown table (worst performing columns)
            column_metrics = (
                fidelity_report.get("marginal", {}).get("metrics", {})
                or fidelity_report.get("column_metrics", {})
            )
            if column_metrics:
                story.append(Paragraph("Column-Level Metrics (Bottom 20)", styles["Heading2"]))
                col_rows = sorted(
                    [
                        (col_name, float(metrics.get("column_score", 0.0)))
                        for col_name, metrics in column_metrics.items()
                        if isinstance(metrics, dict)
                    ],
                    key=lambda x: x[1],
                )[:20]

                col_data = [["Column", "Score", "Status"]]
                for col_name, score in col_rows:
                    col_data.append([col_name[:40], f"{score:.4f}", "Pass" if score >= 0.82 else "Fail"])

                col_table = Table(col_data, colWidths=[9 * cm, 4 * cm, 3 * cm])
                col_table.setStyle(self._get_table_style(colors))
                story.append(col_table)

            doc.build(story)
            return buffer.getvalue()

        except ImportError:
            logger.warning("reportlab not available — generating minimal text-based PDF substitute")
            return self._generate_text_fallback_pdf(
                job_id, tenant_id, overall_score, passed, fidelity_report
            )
        except Exception as exc:
            logger.error("PDF generation failed", job_id=str(job_id), error=str(exc))
            raise

    def _get_table_style(self, colors: Any) -> Any:
        """Return a standard TableStyle for report tables.

        Args:
            colors: reportlab colors module.

        Returns:
            TableStyle instance.
        """
        from reportlab.platypus import TableStyle  # type: ignore[import]

        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(*_HEADER_COLOR_RGB)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 1), (-1, -1), colors.Color(*_SECTION_COLOR_RGB)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(*_SECTION_COLOR_RGB)]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ])

    def _generate_text_fallback_pdf(
        self,
        job_id: uuid.UUID,
        tenant_id: str,
        overall_score: float,
        passed: bool,
        fidelity_report: dict[str, Any],
    ) -> bytes:
        """Generate a plain text fallback when reportlab is unavailable.

        Args:
            job_id: Validation job UUID.
            tenant_id: Tenant identifier.
            overall_score: Overall fidelity score.
            passed: Verdict.
            fidelity_report: Fidelity evaluation results.

        Returns:
            UTF-8 encoded text as bytes (not real PDF but valid content).
        """
        lines = [
            "AumOS Fidelity Validation Report",
            "=" * 50,
            f"Job ID: {job_id}",
            f"Tenant: {tenant_id}",
            f"Generated: {datetime.now(UTC).isoformat()}",
            f"Overall Score: {overall_score:.4f}",
            f"Verdict: {'PASS' if passed else 'FAIL'}",
            "",
            "Fidelity Summary:",
            f"  Overall: {fidelity_report.get('overall_score', 0.0):.4f}",
            f"  Marginal: {fidelity_report.get('marginal', {}).get('score', 'N/A')}",
            f"  Pairwise: {fidelity_report.get('pairwise', {}).get('score', 'N/A')}",
            f"  Table Level: {fidelity_report.get('table_level', {}).get('score', 'N/A')}",
        ]
        return "\n".join(lines).encode("utf-8")
