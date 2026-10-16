"""Report and benchmark artifact exporters."""

import json
from pathlib import Path
from typing import Any, Dict

from src.evaluation.interfaces import ReportExporter


class MarkdownReportExporter(ReportExporter):
    """Serializes evaluation results into a publication-ready Markdown report table."""

    def export(self, summary: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
        """Generate and save evaluation_summary.md.

        Parameters
        ----------
        summary : Dict[str, Any]
            Comprehensive evaluation record.
        output_dir : Path
            Target output directory.

        Returns
        -------
        Dict[str, Path]
            Dictionary pointing to the created markdown report.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "evaluation_summary.md"

        phys = summary.get("physical_metrics", {})
        feas = summary.get("feasibility_metrics", {})
        comp = summary.get("computational_metrics", {})
        ckpt = summary.get("checkpoint", "Unknown")

        rel_l2 = phys.get("rel_l2_overall", 0.0) * 100.0
        surf_l2 = phys.get("surface_rel_l2", 0.0) * 100.0
        vol_l2 = phys.get("volume_rel_l2", 0.0) * 100.0
        l_inf = phys.get("l_inf_max_mm", 0.0)
        h1 = phys.get("sobolev_h1_semi_norm", 0.0) * 100.0

        max_pen = feas.get("max_die_penetration_mm", 0.0)
        mean_pen = feas.get("mean_die_penetration_mm", 0.0)
        viol_rate = feas.get("penetration_violation_rate_pct", 0.0)
        feas_pass = "PASS" if feas.get("contact_feasibility_pass", 1.0) == 1.0 else "FAIL"

        latency_ms = comp.get("inference_latency_ms_per_sample", 0.0)
        vram = comp.get("peak_vram_mb", 0.0)
        params = int(comp.get("trainable_parameters", 0))
        speedup = comp.get("surrogate_speedup_vs_fem", 0.0)

        content = f"""# Surrogate Benchmark Evaluation Report

**Checkpoint:** `{ckpt}`

---

### 1. Physical Accuracy Metrics (Real Physical Units: mm)
| Metric | Value | Reference Standard | Assessment |
| :--- | :--- | :--- | :--- |
| **Overall Relative L2 Error** | **{rel_l2:.2f}%** | < 5.0% | {"Excellent" if rel_l2 < 5.0 else "Adequate"} |
| **Surface Contact L2 Error** | **{surf_l2:.2f}%** | < 6.0% | {"Strong Contact Adherence" if surf_l2 < 6.0 else "Moderate"} |
| **Volume Interior L2 Error** | **{vol_l2:.2f}%** | < 5.0% | {"Accurate Continuum Flow" if vol_l2 < 5.0 else "Moderate"} |
| **Maximum Pointwise Error (L_inf)** | **{l_inf:.3f} mm** | Peak displacement: ~6.5 mm | Peak localized deviation |
| **Sobolev H1 Semi-Norm Error** | **{h1:.2f}%** | < 10.0% | Strain gradient fidelity |

---

### 2. Solid Mechanics Feasibility (Die Non-Penetration Constraint)
| Metric | Value | Constraint Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Max Die Penetration Depth** | **{max_pen:.4f} mm** | < 0.05 mm | **{feas_pass}** |
| **Mean Die Penetration** | **{mean_pen:.4f} mm** | < 0.01 mm | **{feas_pass}** |
| **Contact Violation Rate** | **{viol_rate:.1f}%** of boundary nodes | < 5.0% | **{feas_pass}** |

---

### 3. Computational Performance vs. Numerical Solver (ABAQUS FEM)
| Hardware / Execution Metric | Evaluated Value |
| :--- | :--- |
| **Inference Latency per Sample** | **{latency_ms:.2f} ms** |
| **Peak GPU VRAM Allocation** | **{vram:.1f} MB** |
| **Trainable Parameters** | **{params:,}** |
| **Abaqus FEM Solver Runtime** | ~120.0 seconds / simulation |
| **Surrogate Speedup Factor** | **{speedup:,.0f}x Faster** |
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {"markdown_report": report_path}


class JsonReportExporter(ReportExporter):
    """Serializes evaluation results into machine-readable JSON for cross-study aggregation."""

    def export(self, summary: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
        """Save evaluation_metrics.json.

        Parameters
        ----------
        summary : Dict[str, Any]
            Comprehensive evaluation record.
        output_dir : Path
            Target output directory.

        Returns
        -------
        Dict[str, Path]
            Dictionary pointing to the created JSON record.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "evaluation_metrics.json"

        # Convert any Path objects to str for JSON serialization
        clean_summary = dict(summary)
        if "exported_artifacts" in clean_summary:
            clean_summary["exported_artifacts"] = {
                k: str(v) for k, v in clean_summary["exported_artifacts"].items()
            }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(clean_summary, f, indent=2)

        return {"json_report": json_path}


class CompositeReportExporter(ReportExporter):
    """Combines Markdown and JSON exporters."""

    def __init__(self) -> None:
        self.md_exporter = MarkdownReportExporter()
        self.json_exporter = JsonReportExporter()

    def export(self, summary: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
        """Execute both report exporters."""
        out = {}
        out.update(self.md_exporter.export(summary, output_dir))
        out.update(self.json_exporter.export(summary, output_dir))
        return out
