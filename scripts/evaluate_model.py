#!/usr/bin/env python3
"""Scientific evaluation and benchmarking script for surrogate models.

Evaluates trained model checkpoints (e.g., SMART) on unobserved test partitions
using the post-training EvaluationEngine subsystem in src/evaluation. Results
are persisted in structured directories under results/ adhering to the hierarchy:
    results/TrainingAnalysis/<name of the dataset>/<ModelName>/<training parameters>/<experiment>/

Artifacts generated:
- evaluation_summary.md: Publication-ready markdown report with benchmark tables and embedded KPI charts.
- evaluation_metrics.json: Machine-readable metric records with physical, feasibility, computational, and training KPIs.
- evaluation_contours.png: High-resolution visual comparisons of displacements and error fields.
- training_curves.png: 4-panel publication-grade KPI dashboard (total loss, surface vs. volume, generalization gap, LR schedule).
- training_history.csv: Tabular numerical dataset of error per epoch.
- training_metrics.jsonl: Preserved copy of raw training telemetry.
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
        self.time_params: Optional[np.ndarray] = (
            data["time_params"] if "time_params" in data else None
        )
        self.simulation_idx: Optional[np.ndarray] = (
            data["simulation_idx"] if "simulation_idx" in data else None
        )
        self.time_step_idx: Optional[np.ndarray] = (
            data["time_step_idx"] if "time_step_idx" in data else None
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
            'displacements', and optional 'time_params'.
        """
        item = {
            "die_boundary": self.die_boundary[idx],
            "coords_initial": self.coords_initial[idx],
            "coords": self.coords[idx],
            "displacements": self.displacements[idx],
        }
        if self.time_params is not None:
            item["time_params"] = self.time_params[idx]
        return item


def infer_training_params(
    model_dir: Path,
    explicit_params: Optional[str] = None,
    batch_size: Optional[int] = None,
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
    batch_size : Optional[int], default=None
        Evaluation batch size fallback if not explicitly defined.

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
    b_size = batch_size or 16

    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                first_line = json.loads(f.readline().strip())
                if "total_epochs" in first_line:
                    total_epochs = int(first_line["total_epochs"])
                if "lr" in first_line:
                    lr_val = float(first_line["lr"])
                    raw_lr = f"{lr_val:.1e}".replace("e-0", "e-").replace("e+0", "e+")
                    lr_str = raw_lr.replace(".0e", "e") if ".0e" in raw_lr else raw_lr
        except Exception:
            pass

    return f"epochs_{total_epochs}_batch_{b_size}_lr_{lr_str}"


def plot_displacement_contours(
    coords_initial: np.ndarray,
    die_contour: np.ndarray,
    true_u: np.ndarray,
    pred_u: np.ndarray,
    output_path: Path,
    sample_idx: int = 0,
    title: Optional[str] = None,
) -> Path:
    """Export contour visualization comparing ground truth, surrogate, and error.

    Plots the spatial deformation on the deformed mesh configuration at the final
    load increment, with the rigid die profile positioned at its active final stroke
    and the undeformed reference outline marked for geometric comparison.

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
    title : Optional[str], default=None
        Custom suptitle for the 3-panel figure.

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

    norm_true = np.linalg.norm(u_true)
    norm_diff = np.linalg.norm(u_pred - u_true)
    rel_l2 = (norm_diff / (norm_true + 1e-8)) * 100.0 if norm_true > 1e-4 else 0.0

    # Deformed spatial mesh coordinates at final time step
    x_true = x0 + u_true[:, :, 0]
    y_true = y0 + u_true[:, :, 1]
    x_pred = x0 + u_pred[:, :, 0]
    y_pred = y0 + u_pred[:, :, 1]

    # Rigid die position at active stroke:
    # In the raw dataset, die_contour is defined at its initial elevation (t=0).
    # At active deformation increments, the rigid die strokes downwards into the workpiece.
    # The vertical stroke matches the deepest indentation of the contacting top surface (y-index 0).
    stroke = np.min(die_y) - np.min(y_true[:, 0])
    die_y_stroked = die_y - stroke

    vmax = max(float(np.max(mag_true)), float(np.max(mag_pred)), 0.01)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)

    # 1. ABAQUS Ground Truth
    ax1 = axes[0]
    cf1 = ax1.contourf(x_true, y_true, mag_true, levels=50, cmap="viridis", vmin=0.0, vmax=vmax)
    ax1.plot(die_x, die_y_stroked, "r--", linewidth=1.8, label=f"Rigid Die (Stroke={stroke:.2f} mm)")
    ax1.plot(x0[:, 0], y0[:, 0], color="gray", linestyle=":", linewidth=1.2, alpha=0.8, label="Initial Outline (t=0)")
    ax1.set_title(f"ABAQUS Ground Truth ||u|| [mm]\nPeak = {np.max(mag_true):.2f} mm", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X Coordinate [mm]")
    ax1.set_ylabel("Y Coordinate [mm]")
    ax1.legend(loc="lower right", fontsize=8)
    fig.colorbar(cf1, ax=ax1, label="Displacement [mm]")

    # 2. Surrogate Prediction
    ax2 = axes[1]
    cf2 = ax2.contourf(x_pred, y_pred, mag_pred, levels=50, cmap="viridis", vmin=0.0, vmax=vmax)
    ax2.plot(die_x, die_y_stroked, "r--", linewidth=1.8, label=f"Rigid Die (Stroke={stroke:.2f} mm)")
    ax2.plot(x0[:, 0], y0[:, 0], color="gray", linestyle=":", linewidth=1.2, alpha=0.8, label="Initial Outline (t=0)")
    ax2.set_title(f"SMART Predicted ||u|| [mm]\nPeak = {np.max(mag_pred):.2f} mm", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X Coordinate [mm]")
    ax2.legend(loc="lower right", fontsize=8)
    fig.colorbar(cf2, ax=ax2, label="Displacement [mm]")

    # 3. Absolute Error
    ax3 = axes[2]
    cf3 = ax3.contourf(x_true, y_true, mag_diff, levels=50, cmap="inferno")
    ax3.plot(die_x, die_y_stroked, "r--", linewidth=1.8, label=f"Rigid Die (Stroke={stroke:.2f} mm)")
    ax3.plot(x0[:, 0], y0[:, 0], color="gray", linestyle=":", linewidth=1.2, alpha=0.8, label="Initial Outline (t=0)")
    ax3.set_title(f"Absolute Error ||u_pred - u_true|| [mm]\nRel L2 = {rel_l2:.1f}% | L_inf = {np.max(mag_diff):.2f} mm", fontsize=11, fontweight="bold")
    ax3.set_xlabel("X Coordinate [mm]")
    ax3.legend(loc="lower right", fontsize=8)
    fig.colorbar(cf3, ax=ax3, label="Error [mm]")

    main_title = title if title is not None else f"Surrogate Physical Verification (Sample {sample_idx:02d})"
    plt.suptitle(
        main_title,
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_temporal_trajectory_grid(
    coords_initial: np.ndarray,
    die_contour: np.ndarray,
    true_u: np.ndarray,
    pred_u: np.ndarray,
    time_step_idx: np.ndarray,
    time_params: np.ndarray,
    sim_idx: np.ndarray,
    target_sim_id: int,
    output_path: Path,
    title: Optional[str] = None,
) -> Path:
    """Render 5x3 multi-timestep trajectory rollout matrix comparison for a single simulation.

    Columns: Ground Truth ||u||, SMART Predicted ||u||, Absolute Error ||u_pred - u_true||
    Rows: t=0, t=5, t=10, t=15, t=19
    """
    match_indices = np.where(sim_idx == target_sim_id)[0]
    if len(match_indices) == 0:
        return output_path

    steps_to_plot = [0, 5, 10, 15, 19]
    fig, axes = plt.subplots(len(steps_to_plot), 3, figsize=(18, 18), sharex=True, sharey=True)

    # Compute global vmax across steps for this simulation
    vmax_list = []
    for s in steps_to_plot:
        s_match = np.where((sim_idx == target_sim_id) & (time_step_idx == s))[0]
        if len(s_match) > 0:
            idx = s_match[0]
            vmax_list.append(np.max(np.linalg.norm(true_u[idx], axis=-1)))
            vmax_list.append(np.max(np.linalg.norm(pred_u[idx], axis=-1)))
    vmax = max(max(vmax_list) if vmax_list else 1.0, 0.01)

    for row_i, step in enumerate(steps_to_plot):
        s_match = np.where((sim_idx == target_sim_id) & (time_step_idx == step))[0]
        if len(s_match) == 0:
            continue
        idx = int(s_match[0])
        x0 = coords_initial[idx, :, :, 0]
        y0 = coords_initial[idx, :, :, 1]
        die_x = die_contour[idx, :, 0]
        die_y = die_contour[idx, :, 1]

        u_tr = true_u[idx]
        u_pr = pred_u[idx]

        m_tr = np.linalg.norm(u_tr, axis=-1)
        m_pr = np.linalg.norm(u_pr, axis=-1)
        m_diff = np.linalg.norm(u_pr - u_tr, axis=-1)

        x_tr = x0 + u_tr[:, :, 0]
        y_tr = y0 + u_tr[:, :, 1]
        x_pr = x0 + u_pr[:, :, 0]
        y_pr = y0 + u_pr[:, :, 1]

        stroke = float(np.min(die_y) - np.min(y_tr[:, 0]))
        die_y_st = die_y - stroke

        # Col 0: ABAQUS GT
        ax0 = axes[row_i, 0]
        cf0 = ax0.contourf(x_tr, y_tr, m_tr, levels=50, cmap="viridis", vmin=0.0, vmax=vmax)
        ax0.plot(die_x, die_y_st, "r--", linewidth=1.5)
        ax0.plot(x0[:, 0], y0[:, 0], color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
        t_val = time_params[idx] if idx < len(time_params) else 0.0
        ax0.set_ylabel(f"Step {step:02d} (t*={t_val:.3f})\nY [mm]", fontsize=10, fontweight="bold")
        if row_i == 0:
            ax0.set_title("ABAQUS Ground Truth ||u|| [mm]", fontsize=11, fontweight="bold")

        # Col 1: SMART Predicted
        ax1 = axes[row_i, 1]
        cf1 = ax1.contourf(x_pr, y_pr, m_pr, levels=50, cmap="viridis", vmin=0.0, vmax=vmax)
        ax1.plot(die_x, die_y_st, "r--", linewidth=1.5)
        ax1.plot(x0[:, 0], y0[:, 0], color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
        if row_i == 0:
            ax1.set_title("SMART Predicted ||u|| [mm]", fontsize=11, fontweight="bold")

        # Col 2: Error
        ax2 = axes[row_i, 2]
        cf2 = ax2.contourf(x_tr, y_tr, m_diff, levels=50, cmap="inferno")
        ax2.plot(die_x, die_y_st, "r--", linewidth=1.5)
        ax2.plot(x0[:, 0], y0[:, 0], color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
        norm_tr = np.linalg.norm(u_tr)
        norm_df = np.linalg.norm(u_pr - u_tr)
        rel_l2 = (norm_df / (norm_tr + 1e-8)) * 100.0 if norm_tr > 1e-4 else 0.0
        ax2.set_ylabel(f"Rel L2: {rel_l2:.1f}%\nMax: {np.max(m_diff):.2f} mm", fontsize=9)
        ax2.yaxis.set_label_position("right")
        if row_i == 0:
            ax2.set_title("Absolute Error ||u_pred - u_true|| [mm]", fontsize=11, fontweight="bold")

    axes[-1, 0].set_xlabel("X Coordinate [mm]")
    axes[-1, 1].set_xlabel("X Coordinate [mm]")
    axes[-1, 2].set_xlabel("X Coordinate [mm]")

    main_title = title or f"Spatiotemporal Trajectory Rollout: Simulation {target_sim_id}"
    plt.suptitle(main_title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
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
    parser.add_argument(
        "--training_metrics",
        type=Path,
        default=None,
        help="Path to training_metrics.jsonl log file (default: inferred from model_dir).",
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
        default=PROJECT_ROOT / "results" / "TrainingAnalysis",
        help="Base results directory where evaluation benchmarks are stored (default: results/TrainingAnalysis).",
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
    parser.add_argument(
        "--parameter_channels",
        type=int,
        default=None,
        help="Conditioning parameter channels (auto-detected from test partition if None).",
    )

    return parser.parse_args()


def main() -> None:
    """Execute end-to-end scientific evaluation pipeline."""
    args = parse_args()

    # 1. Resolve Output Directory Hierarchy: <dataset>/<model>/<training_params>/<experiment>
    dataset_name = args.dataset_name or args.data_dir.name
    model_name = args.model_name
    training_params = infer_training_params(
        args.model_dir, args.training_params, batch_size=args.batch_size
    )
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
    print(f"Loaded Test Partition:    {len(test_dataset)} instances ({len(test_loader)} batches)")

    # Auto-detect parameter conditioning
    parameter_channels = args.parameter_channels
    if parameter_channels is None:
        parameter_channels = 1 if test_dataset.time_params is not None else 0
    print(f"Conditioning Channels:    parameter_channels={parameter_channels}")

    # 5. Instantiate SMART ModelWrapper
    smart_config = {
        "spatial_dim": 2,
        "surface_channels": 2,
        "volume_channels": 2,
        "parameter_channels": parameter_channels,
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
    training_metrics_path = args.training_metrics
    if training_metrics_path is None:
        candidate_metrics = args.model_dir / "training_metrics.jsonl"
        if candidate_metrics.exists():
            training_metrics_path = candidate_metrics

    summary = engine.evaluate(
        model=model_wrapper,
        checkpoint_path=checkpoint_path,
        test_loader=test_loader,
        stats_path=stats_path,
        output_dir=output_dir,
        device=device,
        test_npz_path=test_npz_path,
        fem_reference_time_s=args.fem_reference_time_s,
        training_metrics_path=training_metrics_path,
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
        # Detect if dataset is temporal
        is_temporal = "time_step_idx" in test_npz_data
        if is_temporal:
            time_step_idx = test_npz_data["time_step_idx"].flatten()
            time_params = test_npz_data["time_params"].flatten()
            sim_idx = test_npz_data["simulation_idx"].flatten()
            # Select final time step for the first simulation
            final_step_val = int(np.max(time_step_idx))
            final_matches = np.where((time_step_idx == final_step_val) & (sim_idx == sim_idx[0]))[0]
            primary_idx = int(final_matches[0]) if len(final_matches) > 0 else 0
            sim_id = sim_idx[primary_idx]
            step_id = time_step_idx[primary_idx]
            t_val = time_params[primary_idx]
            primary_title = f"Surrogate Physical Verification (Simulation {sim_id}, Final Step {step_id:02d}, t*={t_val:.3f})"
        else:
            primary_idx = 0
            primary_title = f"Surrogate Physical Verification (Final Step, Sample {primary_idx:02d})"

        plot_path = output_dir / "evaluation_contours.png"
        plot_displacement_contours(
            coords_initial=initial_mesh,
            die_contour=die_contours,
            true_u=phys_targets["displacements"],
            pred_u=phys_preds["displacements"],
            output_path=plot_path,
            sample_idx=primary_idx,
            title=primary_title,
        )
        summary["exported_artifacts"]["contour_plot"] = str(plot_path)

        # For temporal datasets, generate additional cross-time and cross-sample verification contours
        if is_temporal:
            # 1. Multi-step time series for Sample 00
            for step in [0, 5, 10, 15, 19]:
                match_idx = np.where((time_step_idx == step) & (sim_idx == sim_idx[0]))[0]
                if len(match_idx) > 0:
                    s_idx = int(match_idx[0])
                    s_path = output_dir / f"evaluation_contours_sample00_step{step:02d}_t{time_params[s_idx]:.3f}.png"
                    s_title = f"Surrogate Verification: Simulation {sim_idx[s_idx]} (Sample 00), Step {step:02d}/19 (t*={time_params[s_idx]:.3f})"
                    plot_displacement_contours(
                        coords_initial=initial_mesh,
                        die_contour=die_contours,
                        true_u=phys_targets["displacements"],
                        pred_u=phys_preds["displacements"],
                        output_path=s_path,
                        sample_idx=s_idx,
                        title=s_title,
                    )
            # 2. Diverse simulation samples at final step
            unique_sims = np.unique(sim_idx)
            for k, sid in enumerate(unique_sims[:4]):
                match_idx = np.where((time_step_idx == final_step_val) & (sim_idx == sid))[0]
                if len(match_idx) > 0:
                    s_idx = int(match_idx[0])
                    s_path = output_dir / f"evaluation_contours_sample{k:02d}_sim{sid}_step19_t1.000.png"
                    s_title = f"Surrogate Verification: Simulation {sid} (Sample {k:02d}), Step {final_step_val:02d}/19 (t*={time_params[s_idx]:.3f})"
                    plot_displacement_contours(
                        coords_initial=initial_mesh,
                        die_contour=die_contours,
                        true_u=phys_targets["displacements"],
                        pred_u=phys_preds["displacements"],
                        output_path=s_path,
                        sample_idx=s_idx,
                        title=s_title,
                    )
            # 3. Diverse simulation samples at intermediate step (step 10, t*=0.526)
            for k, sid in enumerate(unique_sims[:4]):
                match_idx = np.where((time_step_idx == 10) & (sim_idx == sid))[0]
                if len(match_idx) > 0:
                    s_idx = int(match_idx[0])
                    s_path = output_dir / f"evaluation_contours_sample{k:02d}_sim{sid}_step10_t0.526.png"
                    s_title = f"Surrogate Verification: Simulation {sid} (Sample {k:02d}), Step 10/19 (t*={time_params[s_idx]:.3f})"
                    plot_displacement_contours(
                        coords_initial=initial_mesh,
                        die_contour=die_contours,
                        true_u=phys_targets["displacements"],
                        pred_u=phys_preds["displacements"],
                        output_path=s_path,
                        sample_idx=s_idx,
                        title=s_title,
                    )

            # 4. Spatiotemporal Trajectory Evolution Grid Panels
            for k, sid in enumerate(unique_sims[:4]):
                t_panel_path = output_dir / f"evaluation_trajectory_sim{sid}_sample{k:02d}.png"
                plot_temporal_trajectory_grid(
                    coords_initial=initial_mesh,
                    die_contour=die_contours,
                    true_u=phys_targets["displacements"],
                    pred_u=phys_preds["displacements"],
                    time_step_idx=time_step_idx,
                    time_params=time_params,
                    sim_idx=sim_idx,
                    target_sim_id=int(sid),
                    output_path=t_panel_path,
                    title=f"Spatiotemporal Trajectory Rollout: Simulation {sid} (Test Sample {k:02d})",
                )
                if k == 0:
                    import shutil
                    shutil.copy2(t_panel_path, output_dir / "evaluation_trajectory_sample00.png")

        # Refresh exported Markdown and JSON reports with contour plot artifact
        engine.exporter.export(summary, output_dir)

    # 9. Render Terminal Summary
    phys = summary["physical_metrics"]
    feas = summary["feasibility_metrics"]
    comp = summary["computational_metrics"]
    pass_str = "PASS" if feas["contact_feasibility_pass"] == 1.0 else "FAIL"

    train_kpis = summary.get("training_kpis")
    if train_kpis:
        print("\n" + "=" * 80)
        print("TRAINING CONVERGENCE & OPTIMIZATION KPIS")
        print("=" * 80)
        print(f"Total Completed Epochs:   {train_kpis['total_epochs']}")
        print(f"Optimal Epoch (Best Val): Epoch {train_kpis['best_epoch']}")
        print(f"Initial Validation Loss:  {train_kpis['initial_val_loss']:.4e}")
        print(f"Best Validation Loss:     {train_kpis['best_val_loss']:.4e} ({train_kpis['val_loss_reduction_pct']:.1f}% reduction)")
        print(f"Final Train / Val Loss:   {train_kpis['final_train_loss']:.4e} / {train_kpis['final_val_loss']:.4e}")
        print(f"Generalization Gap:       {train_kpis['generalization_gap_best']:+.4e}")
        print(f"Learning Rate Transition: {train_kpis['initial_lr']:.2e} -> {train_kpis['final_lr']:.2e}")
        print(f"Convergence Assessment:   {train_kpis['convergence_status']}")

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
