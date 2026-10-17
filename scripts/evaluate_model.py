#!/usr/bin/env python3
"""Scientific evaluation and benchmarking script for surrogate models.

Evaluates trained model checkpoints (e.g., SMART) on unobserved test partitions
using the post-training EvaluationEngine subsystem in src/evaluation. Results
are persisted in structured directories under results/ adhering to the hierarchy:
    results/<name of the dataset>/<ModelName>/<training parameters>/<experiment>/

Artifacts generated:
- evaluation_summary.md: Publication-ready markdown report with benchmark tables.
- evaluation_metrics.json: Machine-readable metric records for downstream aggregation.
- evaluation_contours.png: High-resolution visual comparisons of displacements and error fields.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    BatchedInferenceRunner,
    CompositeReportExporter,
    DiePenetrationValidator,
    EvaluationEngine,
    HardwareProfiler,
    PlasticityMetricsCalculator,
    StatisticalDenormalizer,
)
from src.training.batch_loaders import SMARTBatchLoader
from src.training.wrappers.smart_wrapper import SMARTModelWrapper


class PlasticityPartitionDataset(Dataset):
    """PyTorch Dataset loading simulation samples from a preprocessed .npz archive.

    Parameters
    ----------
    npz_path : Path
        Path to partition archive (e.g. test.npz, val.npz).
    """

    def __init__(self, npz_path: Path) -> None:
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Partition archive not found at: {self.npz_path}")

        data = np.load(self.npz_path)
        self.die_boundary: np.ndarray = data["die_boundary"]
        self.coords_initial: np.ndarray = data["coords_initial"]
        self.displacements: np.ndarray = data["displacements"]
        self.coords: np.ndarray = (
            data["coords"] if "coords" in data else self.coords_initial + self.displacements
        )
        self.num_samples: int = self.die_boundary.shape[0]

    def __len__(self) -> int:
        """Return the number of simulation samples in this partition."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Retrieve a single simulation sample dictionary.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing 'die_boundary', 'coords_initial', 'coords',
            and 'displacements'.
        """
        return {
            "die_boundary": self.die_boundary[idx],
            "coords_initial": self.coords_initial[idx],
            "coords": self.coords[idx],
            "displacements": self.displacements[idx],
        }


def infer_training_params(
    model_dir: Path,
    explicit_params: Optional[str] = None,
) -> str:
    """Infer or format a standardized training parameter descriptor string.

    Inspects training telemetry logs in the model directory if no explicit
    descriptor is passed via command-line arguments.

    Parameters
    ----------
    model_dir : Path
        Directory where model checkpoints and metrics reside.
    explicit_params : Optional[str], default=None
        User-provided training parameters string. If provided, returned as-is.

    Returns
    -------
    str
        Training parameter identifier (e.g., 'epochs_50_batch_16_lr_1e-4').
    """
    if explicit_params:
        return explicit_params.strip()

    metrics_file = model_dir / "training_metrics.jsonl"
    total_epochs = 50
    lr_str = "1e-4"
    batch_size = 16

    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                first_line = json.loads(f.readline().strip())
                if "total_epochs" in first_line:
                    total_epochs = int(first_line["total_epochs"])
                if "lr" in first_line:
                    lr_val = float(first_line["lr"])
                    if abs(lr_val - 1e-4) < 1e-6:
                        lr_str = "1e-4"
                    else:
                        lr_str = f"{lr_val:.1e}"
        except Exception:
            pass

    return f"epochs_{total_epochs}_batch_{batch_size}_lr_{lr_str}"


def plot_displacement_contours(
    coords_initial: np.ndarray,
    die_contour: np.ndarray,
    true_u: np.ndarray,
    pred_u: np.ndarray,
    output_path: Path,
    sample_idx: int = 0,
) -> Path:
    """Export contour visualization comparing ground truth, surrogate, and error.

    Parameters
    ----------
    coords_initial : np.ndarray
        Initial undeformed coordinates (N, 101, 31, 2) in mm.
    die_contour : np.ndarray
        Rigid die boundary coordinates (N, 101, 2) in mm.
    true_u : np.ndarray
        Ground-truth physical displacements (N, 101, 31, 2) in mm.
    pred_u : np.ndarray
        Surrogate predicted physical displacements (N, 101, 31, 2) in mm.
    output_path : Path
        Target image file path (.png).
    sample_idx : int, default=0
        Index of the test sample to visualize.

    Returns
    -------
    Path
        Path to the saved contour plot.
    """
    x0 = coords_initial[sample_idx, :, :, 0]
    y0 = coords_initial[sample_idx, :, :, 1]
    die_x = die_contour[sample_idx, :, 0]
    die_y = die_contour[sample_idx, :, 1]

    u_true = true_u[sample_idx]
    u_pred = pred_u[sample_idx]

    mag_true = np.linalg.norm(u_true, axis=-1)
    mag_pred = np.linalg.norm(u_pred, axis=-1)
    mag_diff = np.linalg.norm(u_pred - u_true, axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # 1. ABAQUS Ground Truth
    ax1 = axes[0]
    cf1 = ax1.contourf(x0, y0, mag_true, levels=50, cmap="viridis")
    ax1.plot(die_x, die_y, "r--", linewidth=1.8, label="Rigid Die Profile")
    ax1.set_title("ABAQUS Ground Truth ||u|| [mm]", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X Coordinate [mm]")
    ax1.set_ylabel("Y Coordinate [mm]")
    ax1.legend(loc="upper right")
    fig.colorbar(cf1, ax=ax1, label="Displacement [mm]")

    # 2. Surrogate Prediction
    ax2 = axes[1]
    cf2 = ax2.contourf(x0, y0, mag_pred, levels=50, cmap="viridis")
    ax2.plot(die_x, die_y, "r--", linewidth=1.8, label="Rigid Die Profile")
    ax2.set_title("SMART Predicted ||u|| [mm]", fontsize=12, fontweight="bold")
    ax2.set_xlabel("X Coordinate [mm]")
    ax2.legend(loc="upper right")
    fig.colorbar(cf2, ax=ax2, label="Displacement [mm]")

    # 3. Absolute Error
    ax3 = axes[2]
    cf3 = ax3.contourf(x0, y0, mag_diff, levels=50, cmap="inferno")
    ax3.plot(die_x, die_y, "r--", linewidth=1.8, label="Rigid Die Profile")
    ax3.set_title("Absolute Error ||u_pred - u_true|| [mm]", fontsize=12, fontweight="bold")
    ax3.set_xlabel("X Coordinate [mm]")
    ax3.legend(loc="upper right")
    fig.colorbar(cf3, ax=ax3, label="Error [mm]")

    plt.suptitle(
        f"Surrogate Physical Verification (Sample {sample_idx:02d})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for surrogate model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate surrogate models on solid mechanics benchmarks using EvaluationEngine."
    )
    # Model and experiment locations
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "smart_experiment_1",
        help="Directory containing the trained model checkpoints and logs.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="best_model.pt",
        help="Filename of the checkpoint to evaluate within model_dir.",
    )
    # Dataset paths
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ds_plasticity_onlyFinalStep",
        help="Directory containing test.npz and statistical_data.json.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Base results directory where evaluation benchmarks are stored.",
    )
    # Output hierarchy configuration: <dataset>/<model>/<training_params>/<experiment>
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset folder name under results/ (default: inferred from data_dir).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="SMART",
        help="Model architecture name under results/ (default: SMART).",
    )
    parser.add_argument(
        "--training_params",
        type=str,
        default=None,
        help="Training parameters folder name under results/ (default: inferred from training logs).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment folder name under results/ (default: inferred from model_dir).",
    )
    # Evaluation execution parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for test evaluation DataLoader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Hardware execution device ('cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--fem_reference_time_s",
        type=float,
        default=120.0,
        help="Reference FEM runtime per simulation in seconds for speedup estimation.",
    )
    parser.add_argument(
        "--tolerance_mm",
        type=float,
        default=0.05,
        help="Allowable rigid die penetration tolerance threshold in millimeters.",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=5,
        help="Number of GPU warmup iterations before latency profiling.",
    )
    parser.add_argument(
        "--timing_runs",
        type=int,
        default=20,
        help="Number of timed iterations for latency profiling.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable contour visualization plot generation.",
    )
    # SMART Architecture hyperparameters
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent feature dimension.")
    parser.add_argument("--latent_geo_points", type=int, default=64, help="Latent geometry points.")
    parser.add_argument("--subsampled_points", type=int, default=101, help="Subsampled geometry points.")
    parser.add_argument("--num_blocks", type=int, default=6, help="Encoder-decoder transformer blocks.")
    parser.add_argument("--num_heads", type=int, default=8, help="Attention heads.")

    return parser.parse_args()


def main() -> None:
    """Execute end-to-end scientific evaluation pipeline."""
    args = parse_args()

    # 1. Resolve Output Directory Hierarchy: <dataset>/<model>/<training_params>/<experiment>
    dataset_name = args.dataset_name or args.data_dir.name
    model_name = args.model_name
    training_params = infer_training_params(args.model_dir, args.training_params)
    experiment_name = args.experiment or args.model_dir.name

    output_dir = (
        args.results_dir
        / dataset_name
        / model_name
        / training_params
        / experiment_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Hardware Device Setup
    device = torch.device(args.device)

    print("=" * 80)
    print("SURROGATE MODEL SCIENTIFIC BENCHMARK EVALUATION")
    print(f"Target Device:            {device}")
    print(f"Model Directory:          {args.model_dir}")
    print(f"Checkpoint Target:        {args.checkpoint_name}")
    print(f"Dataset Name:             {dataset_name}")
    print(f"Model Name:               {model_name}")
    print(f"Training Parameters:      {training_params}")
    print(f"Experiment Name:          {experiment_name}")
    print(f"Destination Results Path: {output_dir}")
    print("=" * 80)

    # 3. Verify Files and Paths
    checkpoint_path = args.model_dir / args.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint archive not found at: {checkpoint_path}")

    stats_path = args.data_dir / "statistical_data.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Statistical moments missing at: {stats_path}")

    test_npz_path = args.data_dir / "test.npz"
    if not test_npz_path.exists():
        raise FileNotFoundError(f"Test partition archive missing at: {test_npz_path}")

    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    # 4. Create Test Partition DataLoader
    test_dataset = PlasticityPartitionDataset(test_npz_path)
    batch_loader = SMARTBatchLoader(stats=stats)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=batch_loader.loadDataset,
        num_workers=0,
    )
    print(f"Loaded Test Partition:    {len(test_dataset)} simulations ({len(test_loader)} batches)")

    # 5. Instantiate SMART ModelWrapper
    smart_config = {
        "spatial_dim": 2,
        "surface_channels": 2,
        "volume_channels": 2,
        "parameter_channels": 0,
        "latent_dim": args.latent_dim,
        "latent_geometry_points": args.latent_geo_points,
        "subsampled_geometry_points": args.subsampled_points,
        "num_encoder_decoder_blocks": args.num_blocks,
        "num_heads": args.num_heads,
        "pos_scale_factor": 1000.0,
        "dropout": 0.0,
    }
    model_wrapper = SMARTModelWrapper(config=smart_config)

    # 6. Instantiate Modular EvaluationEngine Subsystem
    engine = EvaluationEngine(
        runner=BatchedInferenceRunner(),
        denormalizer=StatisticalDenormalizer(),
        metrics_calc=PlasticityMetricsCalculator(),
        validator=DiePenetrationValidator(tolerance_mm=args.tolerance_mm),
        profiler=HardwareProfiler(
            warmup_runs=args.warmup_runs,
            timing_runs=args.timing_runs,
        ),
        exporter=CompositeReportExporter(),
    )

    # 7. Execute Benchmark Pipeline
    print("\nRunning evaluation through EvaluationEngine...")
    summary = engine.evaluate(
        model=model_wrapper,
        checkpoint_path=checkpoint_path,
        test_loader=test_loader,
        stats_path=stats_path,
        output_dir=output_dir,
        device=device,
        test_npz_path=test_npz_path,
        fem_reference_time_s=args.fem_reference_time_s,
    )

    # 8. Generate Visual Contour Comparisons
    if not args.no_plots:
        print("Generating displacement contour comparison plots...")
        test_npz_data = np.load(test_npz_path)
        raw_predictions = engine.runner.predict(model_wrapper, test_loader, device)
        targets_dict, initial_mesh, die_contours = engine._extract_ground_truth(
            test_loader=test_loader,
            test_npz_path=test_npz_path,
        )
        phys_preds, phys_targets = engine.denormalizer.denormalize(
            pred_norm=raw_predictions,
            target_norm=targets_dict,
            stats=stats,
        )
        plot_path = output_dir / "evaluation_contours.png"
        plot_displacement_contours(
            coords_initial=initial_mesh,
            die_contour=die_contours,
            true_u=phys_targets["displacements"],
            pred_u=phys_preds["displacements"],
            output_path=plot_path,
            sample_idx=0,
        )
        summary["exported_artifacts"]["contour_plot"] = str(plot_path)
        # Update JSON record with plot artifact path
        with open(output_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # 9. Render Terminal Summary
    phys = summary["physical_metrics"]
    feas = summary["feasibility_metrics"]
    comp = summary["computational_metrics"]
    pass_str = "PASS" if feas["contact_feasibility_pass"] == 1.0 else "FAIL"

    print("\n" + "=" * 80)
    print("SCIENTIFIC BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Overall Rel L2 Error:     {phys['rel_l2_overall'] * 100.0:.2f}%")
    print(f"Surface Contact L2 Error: {phys['surface_rel_l2'] * 100.0:.2f}%")
    print(f"Volume Interior L2 Error: {phys['volume_rel_l2'] * 100.0:.2f}%")
    print(f"Max Pointwise Error:      {phys['l_inf_max_mm']:.4f} mm")
    print(f"Sobolev H1 Semi-Norm:     {phys['sobolev_h1_semi_norm'] * 100.0:.2f}%")
    print(f"Max Die Penetration:      {feas['max_die_penetration_mm']:.4f} mm [{pass_str}]")
    print(f"Violation Rate:           {feas['penetration_violation_rate_pct']:.1f}%")
    print(f"Inference Latency:        {comp['inference_latency_ms_per_sample']:.2f} ms / sample")
    print(f"Surrogate Speedup vs FEM: {comp['surrogate_speedup_vs_fem']:,.0f}x Faster")
    print(f"Artifacts Saved To:       {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
