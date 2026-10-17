"""Training telemetry parsing, convergence KPI computation, and publication visualization."""

import csv
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.interfaces import TrainingHistoryAnalyzer


class TrainingTelemetryAnalyzer(TrainingHistoryAnalyzer):
    """Parses training telemetry, computes solid mechanics convergence KPIs, and exports artifacts.

    Generates:
    - Tabular CSV dataset of error per epoch (training_history.csv).
    - Preserved copy of raw JSONL telemetry (training_metrics.jsonl).
    - Multi-panel publication-grade KPI dashboard visualization (training_curves.png).
    """

    def analyze_and_export(
        self,
        metrics_file: Path,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Parse training metrics, compute convergence KPIs, export tables and plots.

        Parameters
        ----------
        metrics_file : Path
            Path to training_metrics.jsonl log file.
        output_dir : Path
            Destination directory under results/ to write training data and KPI plots.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing 'training_kpis', 'artifacts', and 'history_summary'.
        """
        metrics_path = Path(metrics_file)
        if not metrics_path.exists():
            raise FileNotFoundError(f"Training metrics log file not found at: {metrics_path}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        records = self.parse_telemetry(metrics_path)
        if not records:
            raise ValueError(f"No valid telemetry records found in: {metrics_path}")

        kpis = self.compute_kpis(records)

        # 1. Export CSV table of errors per epoch
        csv_path = out_dir / "training_history.csv"
        self.export_csv(records, csv_path)

        # 2. Copy raw JSONL training log into results directory
        dest_jsonl = out_dir / "training_metrics.jsonl"
        if metrics_path.resolve() != dest_jsonl.resolve():
            shutil.copy2(metrics_path, dest_jsonl)

        # 3. Generate high-resolution 4-panel KPI chart
        plot_path = out_dir / "training_curves.png"
        self.plot_training_kpis(records, kpis, plot_path)

        return {
            "training_kpis": kpis,
            "artifacts": {
                "training_curves_plot": plot_path,
                "training_history_csv": csv_path,
                "training_metrics_log": dest_jsonl,
            },
            "history_summary": records,
        }

    def parse_telemetry(self, metrics_file: Path) -> List[Dict[str, Any]]:
        """Parse line-delimited JSON training records and compute timing deltas.

        Parameters
        ----------
        metrics_file : Path
            Path to JSONL file.

        Returns
        -------
        List[Dict[str, Any]]
            List of validated epoch records sorted by epoch number.
        """
        records: List[Dict[str, Any]] = []
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean = line.strip()
                if not clean:
                    continue
                try:
                    data = json.loads(clean)
                    if "epoch" in data and ("train" in data or "val" in data):
                        records.append(data)
                except json.JSONDecodeError:
                    continue

        records.sort(key=lambda r: int(r.get("epoch", 0)))

        # Add per-epoch duration calculations if timestamps exist
        for idx in range(len(records)):
            curr_ts = records[idx].get("timestamp", 0.0)
            if idx == 0:
                records[idx]["epoch_duration_s"] = 0.0
            else:
                prev_ts = records[idx - 1].get("timestamp", 0.0)
                records[idx]["epoch_duration_s"] = max(0.0, curr_ts - prev_ts)

        return records

    def compute_kpis(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quantitative optimization and convergence KPIs across epochs.

        Parameters
        ----------
        records : List[Dict[str, Any]]
            Parsed epoch telemetry records.

        Returns
        -------
        Dict[str, Any]
            Calculated key performance indicators dictionary.
        """
        total_epochs = len(records)
        first = records[0]
        last = records[-1]

        def get_loss(record: Dict[str, Any], split: str, key: str = "loss_total") -> float:
            sub = record.get(split, {})
            if isinstance(sub, dict):
                return float(sub.get(key, sub.get("loss", 0.0)))
            return float(sub)

        # Identify best epoch based on validation total loss
        best_record = min(records, key=lambda r: get_loss(r, "val", "loss_total"))
        best_epoch = int(best_record.get("epoch", 1))

        init_train = get_loss(first, "train", "loss_total")
        init_val = get_loss(first, "val", "loss_total")
        final_train = get_loss(last, "train", "loss_total")
        final_val = get_loss(last, "val", "loss_total")

        best_val = get_loss(best_record, "val", "loss_total")
        best_train = get_loss(best_record, "train", "loss_total")

        # Surface and volume components if present
        best_val_surf = get_loss(best_record, "val", "loss_surf")
        best_val_vol = get_loss(best_record, "val", "loss_vol")
        best_train_surf = get_loss(best_record, "train", "loss_surf")
        best_train_vol = get_loss(best_record, "train", "loss_vol")

        final_train_surf = get_loss(last, "train", "loss_surf")
        final_train_vol = get_loss(last, "train", "loss_vol")
        final_val_surf = get_loss(last, "val", "loss_surf")
        final_val_vol = get_loss(last, "val", "loss_vol")

        train_red_pct = max(0.0, (1.0 - (final_train / max(1e-12, init_train))) * 100.0)
        val_red_pct = max(0.0, (1.0 - (best_val / max(1e-12, init_val))) * 100.0)

        best_gap = best_val - best_train
        final_gap = final_val - final_train

        init_lr = float(first.get("lr", 0.0))
        final_lr = float(last.get("lr", 0.0))
        min_lr = min(float(r.get("lr", 0.0)) for r in records)

        # Assessment heuristic
        if final_val > 1.25 * best_val and best_epoch < total_epochs * 0.7:
            status = f"Mild Overfitting Detected after Epoch {best_epoch}"
        elif val_red_pct >= 50.0:
            status = f"Strong Convergence ({val_red_pct:.1f}% Error Drop, Peak at Epoch {best_epoch})"
        elif val_red_pct >= 20.0:
            status = f"Moderate Convergence (Peak at Epoch {best_epoch})"
        else:
            status = f"Plateaued / Baseline Improvement (Epoch {best_epoch})"

        return {
            "total_epochs": total_epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "best_train_loss": best_train,
            "initial_train_loss": init_train,
            "initial_val_loss": init_val,
            "final_train_loss": final_train,
            "final_val_loss": final_val,
            "best_val_loss_surf": best_val_surf,
            "best_val_loss_vol": best_val_vol,
            "best_train_loss_surf": best_train_surf,
            "best_train_loss_vol": best_train_vol,
            "final_val_loss_surf": final_val_surf,
            "final_val_loss_vol": final_val_vol,
            "train_loss_reduction_pct": train_red_pct,
            "val_loss_reduction_pct": val_red_pct,
            "generalization_gap_best": best_gap,
            "generalization_gap_final": final_gap,
            "initial_lr": init_lr,
            "final_lr": final_lr,
            "min_lr": min_lr,
            "convergence_status": status,
        }

    def export_csv(self, records: List[Dict[str, Any]], output_path: Path) -> Path:
        """Write tabular epoch progression data into a CSV file.

        Parameters
        ----------
        records : List[Dict[str, Any]]
            Parsed epoch records.
        output_path : Path
            Destination CSV file path.

        Returns
        -------
        Path
            Path to created CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "epoch",
            "train_loss_total",
            "train_loss_surf",
            "train_loss_vol",
            "val_loss_total",
            "val_loss_surf",
            "val_loss_vol",
            "lr",
            "timestamp",
            "epoch_duration_s",
        ]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                tr = r.get("train", {})
                va = r.get("val", {})
                row = {
                    "epoch": r.get("epoch", 0),
                    "train_loss_total": tr.get("loss_total", tr.get("loss", "")),
                    "train_loss_surf": tr.get("loss_surf", ""),
                    "train_loss_vol": tr.get("loss_vol", ""),
                    "val_loss_total": va.get("loss_total", va.get("loss", "")),
                    "val_loss_surf": va.get("loss_surf", ""),
                    "val_loss_vol": va.get("loss_vol", ""),
                    "lr": r.get("lr", ""),
                    "timestamp": r.get("timestamp", ""),
                    "epoch_duration_s": f"{r.get('epoch_duration_s', 0.0):.2f}",
                }
                writer.writerow(row)

        return output_path

    def plot_training_kpis(
        self,
        records: List[Dict[str, Any]],
        kpis: Dict[str, Any],
        output_path: Path,
    ) -> Path:
        """Render publication-grade 4-panel KPI diagnostic chart.

        Parameters
        ----------
        records : List[Dict[str, Any]]
            Telemetry history records.
        kpis : Dict[str, Any]
            Computed summary KPIs.
        output_path : Path
            Destination PNG image path.

        Returns
        -------
        Path
            Path to saved figure.
        """
        epochs = np.array([int(r["epoch"]) for r in records])
        train_tot = np.array([float(r.get("train", {}).get("loss_total", 0.0)) for r in records])
        val_tot = np.array([float(r.get("val", {}).get("loss_total", 0.0)) for r in records])

        has_components = all(
            "loss_surf" in r.get("train", {}) and "loss_vol" in r.get("train", {})
            for r in records
        )
        if has_components:
            tr_surf = np.array([float(r["train"]["loss_surf"]) for r in records])
            va_surf = np.array([float(r["val"]["loss_surf"]) for r in records])
            tr_vol = np.array([float(r["train"]["loss_vol"]) for r in records])
            va_vol = np.array([float(r["val"]["loss_vol"]) for r in records])

        lrs = np.array([float(r.get("lr", 0.0)) for r in records])
        gaps = val_tot - train_tot
        best_epoch = kpis["best_epoch"]
        best_val = kpis["best_val_loss"]

        # Ensure positive values for log scaling
        safe_train_tot = np.maximum(train_tot, 1e-12)
        safe_val_tot = np.maximum(val_tot, 1e-12)

        fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
        if len(epochs) == 1:
            axes[0, 0].set_xlim(0.5, 1.5)

        # Panel 1: Total Loss Progression (Semilog-y)
        ax1 = axes[0, 0]
        ax1.semilogy(epochs, safe_train_tot, color="#1f77b4", linewidth=2.0, label="Train Loss (Rel L2)")
        ax1.semilogy(epochs, safe_val_tot, color="#d62728", linewidth=2.0, label="Val Loss (Rel L2)")
        ax1.axvline(
            best_epoch,
            color="#2ca02c",
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            label=f"Best Model (Epoch {best_epoch})",
        )
        ax1.scatter([best_epoch], [best_val], color="#2ca02c", s=80, zorder=5)
        ax1.annotate(
            f"  Best Val: {best_val:.4e}",
            xy=(best_epoch, best_val),
            xytext=(best_epoch + 1, best_val * 1.05),
            fontweight="bold",
            color="#1b631b",
            fontsize=10,
        )
        ax1.set_title("Total Relative L2 Loss Progression (Log Scale)", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Relative L2 Loss (Log Scale)", fontsize=11)
        ax1.grid(True, which="both", linestyle="--", alpha=0.4)
        ax1.legend(loc="upper right", framealpha=0.9)

        # Panel 2: Component Breakdown (Surface Contact vs Volume Continuum)
        ax2 = axes[0, 1]
        if has_components:
            ax2.semilogy(epochs, np.maximum(tr_surf, 1e-12), color="#008080", linewidth=1.8, label="Train Surface Loss")
            ax2.semilogy(epochs, np.maximum(va_surf, 1e-12), color="#008080", linestyle="--", linewidth=1.8, label="Val Surface Loss")
            ax2.semilogy(epochs, np.maximum(tr_vol, 1e-12), color="#ff7f0e", linewidth=1.8, label="Train Volume Loss")
            ax2.semilogy(epochs, np.maximum(va_vol, 1e-12), color="#ff7f0e", linestyle="--", linewidth=1.8, label="Val Volume Loss")
            ax2.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.6)
            ax2.set_title("Physical Loss Decomposition (Surface Contact vs Volume)", fontsize=13, fontweight="bold")
            ax2.set_ylabel("Component Rel L2 Loss", fontsize=11)
            ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)
        else:
            ax2.plot(epochs, train_tot, color="#1f77b4", linewidth=2.0, label="Train Loss (Linear)")
            ax2.plot(epochs, val_tot, color="#d62728", linewidth=2.0, label="Val Loss (Linear)")
            ax2.set_title("Total Loss Progression (Linear Scale)", fontsize=13, fontweight="bold")
            ax2.set_ylabel("Relative L2 Loss", fontsize=11)
            ax2.legend(loc="upper right")
        ax2.grid(True, which="both", linestyle="--", alpha=0.4)

        # Panel 3: Generalization Gap (Val - Train)
        ax3 = axes[1, 0]
        ax3.plot(epochs, gaps, color="#8c564b", linewidth=1.8, label=r"Generalization Gap ($\mathcal{L}_{val} - \mathcal{L}_{train}$)")
        ax3.axhline(0.0, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
        ax3.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.7, label=f"Best Model (Epoch {best_epoch})")
        ax3.fill_between(
            epochs,
            0,
            gaps,
            where=(gaps >= 0),
            color="#2ca02c",
            alpha=0.15,
            label="Standard Gap Range",
        )
        ax3.fill_between(
            epochs,
            0,
            gaps,
            where=(gaps < 0),
            color="#1f77b4",
            alpha=0.15,
            label="Val < Train",
        )
        ax3.set_title(r"Generalization Dynamics: Gap ($\mathcal{L}_{val} - \mathcal{L}_{train}$)", fontsize=13, fontweight="bold")
        ax3.set_xlabel("Epoch Number", fontsize=11)
        ax3.set_ylabel("Loss Difference", fontsize=11)
        ax3.grid(True, linestyle="--", alpha=0.4)
        ax3.legend(loc="upper right", framealpha=0.9, fontsize=9)

        # Panel 4: Learning Rate Schedule & Training Dynamics
        ax4 = axes[1, 1]
        ax4.plot(epochs, lrs, color="#9467bd", linewidth=2.0, label="Learning Rate Schedule")
        if np.max(lrs) / max(1e-12, np.min(lrs)) > 10.0:
            ax4.set_yscale("log")
        ax4.set_title("Optimization Dynamics: Learning Rate Annealing", fontsize=13, fontweight="bold")
        ax4.set_xlabel("Epoch Number", fontsize=11)
        ax4.set_ylabel("Learning Rate", fontsize=11)
        ax4.grid(True, which="both", linestyle="--", alpha=0.4)
        ax4.legend(loc="upper right", framealpha=0.9)

        # Top summary title
        red_pct = kpis.get("val_loss_reduction_pct", 0.0)
        best_gap = kpis.get("generalization_gap_best", 0.0)
        subtitle = (
            f"Optimization Summary: {len(epochs)} Epochs | "
            f"Best Val Rel-L2: {best_val:.4e} (Epoch {best_epoch}) | "
            f"Val Error Reduction: {red_pct:.1f}% | "
            f"Best Gap: {best_gap:+.4e}"
        )

        plt.suptitle(
            "Surrogate Neural Operator Training Telemetry & Optimization KPIs\n" + subtitle,
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        fig.tight_layout(rect=[0, 0.02, 1, 0.93])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        return output_path
