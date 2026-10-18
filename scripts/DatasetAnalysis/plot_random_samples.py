"""Diagnostic plotting script to visualize random samples from processed datasets.

General and unopinionated: plots what the dataset actually contains, with a prominent
telemetry banner/tag embedded directly into every PNG showing the exact input and
output tensor dimensions.

Handles both:
1. Time-dependent datasets (5D tensors): plots the complete temporal evolution
   across all stored time steps for each random sample, plus an overview figure.
2. Static datasets (4D tensors, e.g. onlyFinalStep): plots the stored static solution
   fields (deformed mesh, channels, die profile) for each sample and an overview.

Usage:
------
python scripts/DatasetAnalysis/plot_random_samples.py \\
    --data_dir data/processed/ds_plasticity_nochange \\
    --output_dir results/DatasetAnalysis/ds_plasticity_nochange \\
    --num_samples 3 --seed 42

Or using positional argument:
python scripts/DatasetAnalysis/plot_random_samples.py data/processed/ds_plasticity_onlyFinalStep
"""

import argparse
import math
from pathlib import Path
import sys
from timeit import default_timer
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Suppress curvilinear mesh warning in pcolormesh
warnings.filterwarnings("ignore", message=".*input coordinates to pcolormesh.*")
warnings.filterwarnings("ignore", message=".*This figure includes Axes that are not compatible with tight_layout.*")

# Ensure workspace root in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="General diagnostic visualizer plotting raw dataset samples with dimension tags."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="Path to processed dataset directory (e.g. data/processed/ds_plasticity_nochange).",
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir_flag",
        type=str,
        default=None,
        help="Explicit flag for data directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to results/DatasetAnalysis/<dataset_name>).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of random samples to visualize (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sample selection (default: 42).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to sample from (default: train).",
    )
    return parser.parse_args()


# ==============================================================================
# DIMENSION INSPECTION & TELEMETRY TAGGING
# ==============================================================================

def categorize_dataset_keys(
    data: Dict[str, np.ndarray],
) -> Tuple[Dict[str, Tuple[int, ...]], Dict[str, Tuple[int, ...]], Dict[str, Tuple[int, ...]]]:
    """Categorize dataset keys into inputs, outputs, and metadata."""
    inputs_info: Dict[str, Tuple[int, ...]] = {}
    outputs_info: Dict[str, Tuple[int, ...]] = {}
    other_info: Dict[str, Tuple[int, ...]] = {}

    for k, v in data.items():
        shape = tuple(v.shape)
        if any(term in k.lower() for term in ["die", "boundary", "profile", "initial", "input", "in"]):
            inputs_info[k] = shape
        elif any(term in k.lower() for term in ["coords", "displacement", "target", "output", "out"]):
            outputs_info[k] = shape
        else:
            other_info[k] = shape

    return inputs_info, outputs_info, other_info


def render_dimension_tag(
    fig: plt.Figure,
    dataset_name: str,
    split: str,
    total_samples: int,
    inputs_info: Dict[str, Tuple[int, ...]],
    outputs_info: Dict[str, Tuple[int, ...]],
    other_info: Dict[str, Tuple[int, ...]],
    is_temporal: bool,
    sample_id: Optional[int] = None,
    y_pos: float = 0.985,
    fontsize: float = 8.0,
) -> None:
    """Render a prominent diagnostic telemetry badge at the top of the PNG."""
    mode_str = "TEMPORAL (5D)" if is_temporal else "STATIC (4D)"
    sample_str = f"  |  INSPECTING SAMPLE: #{sample_id}" if sample_id is not None else f"  |  RANDOM SUBSET ({total_samples} total)"
    header_line = f"[ DATASET: {dataset_name} | SPLIT: {split} (N={total_samples}) | FORMULATION: {mode_str}{sample_str} ]"

    in_items = [f"{k}: {shape}" for k, shape in inputs_info.items()]
    in_line = "INPUT DIMS:  " + (" | ".join(in_items) if in_items else "None specified")

    out_items = [f"{k}: {shape}" for k, shape in outputs_info.items()]
    out_line = "OUTPUT DIMS: " + (" | ".join(out_items) if out_items else "None specified")

    lines = [header_line, in_line, out_line]
    if other_info:
        other_items = [f"{k}: {shape}" for k, shape in other_info.items()]
        lines.append("AUX DIMS:    " + " | ".join(other_items))

    tag_text = "\n".join(lines)

    fig.text(
        0.5,
        y_pos,
        tag_text,
        ha="center",
        va="top",
        fontsize=fontsize,
        fontfamily="monospace",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#0f172a",
            edgecolor="#38bdf8",
            linewidth=1.2,
            alpha=0.95,
        ),
        color="#f8fafc",
    )


# ==============================================================================
# TIME-DEPENDENT PLOTTING (5D)
# ==============================================================================

def plot_sample_temporal_all_steps(
    coords: np.ndarray,
    displacements: np.ndarray,
    die_boundary: Optional[np.ndarray],
    sample_id: int,
    out_path: Path,
    dataset_name: str,
    split: str,
    total_samples: int,
    inputs_info: Dict[str, Tuple[int, ...]],
    outputs_info: Dict[str, Tuple[int, ...]],
    other_info: Dict[str, Tuple[int, ...]],
    step_sub: int = 4,
) -> None:
    """Plot every single stored timestep for one time-dependent sample."""
    nt = coords.shape[2]  # (nx, ny, nt, 2)
    cols = 5
    rows = math.ceil(nt / cols)

    disp_mag_all = np.linalg.norm(displacements, axis=-1)  # (nx, ny, nt)
    vmax = float(np.max(disp_mag_all))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 3.4, rows * 2.8 + 1.2),
        sharex=True,
        sharey=True,
        dpi=180,
    )
    axes_flat = axes.ravel() if isinstance(axes, np.ndarray) else [axes]

    pm_ref = None
    for t in range(nt):
        ax = axes_flat[t]
        x_g = coords[:, :, t, 0]
        y_g = coords[:, :, t, 1]
        u_mag = disp_mag_all[:, :, t]

        pm = ax.pcolormesh(
            x_g,
            y_g,
            u_mag,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            shading="auto",
        )
        if pm_ref is None:
            pm_ref = pm

        # Wireframe grid lines
        for i in range(0, x_g.shape[0], step_sub):
            ax.plot(x_g[i, :], y_g[i, :], color="black", linewidth=0.25, alpha=0.35)
        for j in range(0, x_g.shape[1], step_sub):
            ax.plot(x_g[:, j], y_g[:, j], color="black", linewidth=0.25, alpha=0.35)

        # Die boundary overlay
        if die_boundary is not None:
            ax.plot(
                die_boundary[:, 0],
                die_boundary[:, 1],
                color="#e11d48",
                linestyle="--",
                linewidth=0.9,
                alpha=0.85,
            )

        t_norm = t / max(nt - 1, 1)
        ax.set_title(
            f"Step {t} (t*={t_norm:.2f})\nmax|u|={np.max(u_mag):.2f} mm | slice={x_g.shape}",
            fontsize=8.0,
            fontweight="bold",
        )
        ax.set_aspect("equal", "box")
        if t % cols == 0:
            ax.set_ylabel("y [mm]", fontsize=8.5)
        if t >= (rows - 1) * cols:
            ax.set_xlabel("x [mm]", fontsize=8.5)

    # Hide extra subplots if any
    for t in range(nt, len(axes_flat)):
        axes_flat[t].set_visible(False)

    if pm_ref is not None:
        cbar = fig.colorbar(
            pm_ref,
            ax=axes_flat[:nt],
            orientation="horizontal",
            fraction=0.03,
            pad=0.07,
        )
        cbar.set_label("Displacement Magnitude |u| [mm]", fontsize=9.5, fontweight="bold")

    # Render Dimension & Formulation Telemetry Tag
    render_dimension_tag(
        fig=fig,
        dataset_name=dataset_name,
        split=split,
        total_samples=total_samples,
        inputs_info=inputs_info,
        outputs_info=outputs_info,
        other_info=other_info,
        is_temporal=True,
        sample_id=sample_id,
        y_pos=0.992,
        fontsize=8.5,
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved sample temporal PNG: {out_path.name}")


def plot_temporal_overview_summary(
    coords: np.ndarray,
    displacements: np.ndarray,
    die_boundary: Optional[np.ndarray],
    sample_indices: List[int],
    out_path: Path,
    dataset_name: str,
    split: str,
    total_samples: int,
    inputs_info: Dict[str, Tuple[int, ...]],
    outputs_info: Dict[str, Tuple[int, ...]],
    other_info: Dict[str, Tuple[int, ...]],
    num_time_cols: int = 5,
    step_sub: int = 4,
) -> None:
    """Big overview comparing all selected random samples over time."""
    nt = coords.shape[3]  # (N, nx, ny, nt, 2)
    selected_steps = np.linspace(0, nt - 1, num_time_cols, dtype=int).tolist()
    num_samples = len(sample_indices)

    total_cols = 1 + len(selected_steps)
    fig, axes = plt.subplots(
        num_samples,
        total_cols,
        figsize=(total_cols * 3.4, num_samples * 2.8 + 1.2),
        dpi=180,
    )
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for r_idx, s_idx in enumerate(sample_indices):
        s_coords = coords[s_idx]
        s_disp = displacements[s_idx]
        s_mag = np.linalg.norm(s_disp, axis=-1)
        s_vmax = float(np.max(s_mag))

        # Column 0: Die Profile
        ax_die = axes[r_idx, 0]
        if die_boundary is not None:
            die_x = die_boundary[s_idx, :, 0]
            die_y = die_boundary[s_idx, :, 1]
            ax_die.plot(die_x, die_y, color="#e11d48", linewidth=1.5)
            ax_die.set_title(f"Sample #{s_idx}\nDie Profile $S_d(x)$\nshape={die_boundary[s_idx].shape}", fontsize=8.5, fontweight="bold")
            ax_die.set_xlabel("x [mm]", fontsize=8)
            ax_die.set_ylabel("Elevation y [mm]", fontsize=8)
            ax_die.grid(True, linestyle="--", alpha=0.4)
        else:
            ax_die.text(0.5, 0.5, "No die data", ha="center", va="center")

        # Columns 1..N: Stored Timesteps
        for c_idx, t_step in enumerate(selected_steps, start=1):
            ax = axes[r_idx, c_idx]
            x_g = s_coords[:, :, t_step, 0]
            y_g = s_coords[:, :, t_step, 1]
            u_mag = s_mag[:, :, t_step]

            ax.pcolormesh(
                x_g,
                y_g,
                u_mag,
                cmap="viridis",
                vmin=0.0,
                vmax=s_vmax,
                shading="auto",
            )
            for i in range(0, x_g.shape[0], step_sub):
                ax.plot(x_g[i, :], y_g[i, :], color="black", linewidth=0.2, alpha=0.3)
            for j in range(0, x_g.shape[1], step_sub):
                ax.plot(x_g[:, j], y_g[:, j], color="black", linewidth=0.2, alpha=0.3)

            if die_boundary is not None:
                ax.plot(die_boundary[s_idx, :, 0], die_boundary[s_idx, :, 1], color="#e11d48", linestyle="--", linewidth=0.8)

            t_norm = t_step / max(nt - 1, 1)
            ax.set_title(f"Step t={t_step} (t*={t_norm:.2f})\nmax|u|={np.max(u_mag):.2f} mm", fontsize=8.5)
            ax.set_aspect("equal", "box")
            if c_idx == 1:
                ax.set_ylabel("y [mm]", fontsize=8)
            if r_idx == num_samples - 1:
                ax.set_xlabel("x [mm]", fontsize=8)

    # Render Dimension & Formulation Telemetry Tag
    render_dimension_tag(
        fig=fig,
        dataset_name=dataset_name,
        split=split,
        total_samples=total_samples,
        inputs_info=inputs_info,
        outputs_info=outputs_info,
        other_info=other_info,
        is_temporal=True,
        sample_id=None,
        y_pos=0.992,
        fontsize=8.5,
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.89])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved big temporal overview PNG: {out_path.name}")


# ==============================================================================
# STATIC PLOTTING (4D)
# ==============================================================================

def plot_sample_static_solution(
    coords: np.ndarray,
    coords_0: np.ndarray,
    displacements: np.ndarray,
    die_boundary: Optional[np.ndarray],
    sample_id: int,
    out_path: Path,
    dataset_name: str,
    split: str,
    total_samples: int,
    inputs_info: Dict[str, Tuple[int, ...]],
    outputs_info: Dict[str, Tuple[int, ...]],
    other_info: Dict[str, Tuple[int, ...]],
    step_sub: int = 4,
) -> None:
    """Plot stored static solution fields for a single sample."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=200)

    x_g = coords[:, :, 0]
    y_g = coords[:, :, 1]
    x_0 = coords_0[:, :, 0]
    y_0 = coords_0[:, :, 1]
    ux = displacements[:, :, 0]
    uy = displacements[:, :, 1]
    u_mag = np.sqrt(ux**2 + uy**2)

    # 1. Undeformed vs Deformed Mesh with Die
    ax0 = axes[0, 0]
    for i in range(0, x_0.shape[0], step_sub):
        ax0.plot(x_0[i, :], y_0[i, :], color="gray", linewidth=0.4, linestyle=":", alpha=0.6)
    for j in range(0, x_0.shape[1], step_sub):
        ax0.plot(x_0[:, j], y_0[:, j], color="gray", linewidth=0.4, linestyle=":", alpha=0.6)

    pm0 = ax0.pcolormesh(x_g, y_g, u_mag, cmap="viridis", shading="auto")
    for i in range(0, x_g.shape[0], step_sub):
        ax0.plot(x_g[i, :], y_g[i, :], color="black", linewidth=0.25, alpha=0.35)
    for j in range(0, x_g.shape[1], step_sub):
        ax0.plot(x_g[:, j], y_g[:, j], color="black", linewidth=0.25, alpha=0.35)

    if die_boundary is not None:
        ax0.plot(die_boundary[:, 0], die_boundary[:, 1], color="#e11d48", linewidth=1.5, label="Die Profile $S_d(x)$")
        ax0.legend(loc="upper right", fontsize=8)

    ax0.set_title(f"Deformed Mesh (coords: {coords.shape}) vs Ref (dotted)", fontsize=9.5, fontweight="bold")
    ax0.set_aspect("equal", "box")
    ax0.set_ylabel("y [mm]", fontsize=9)
    fig.colorbar(pm0, ax=ax0, label="|u| [mm]", fraction=0.046, pad=0.04)

    # 2. Total displacement magnitude |u|
    ax1 = axes[0, 1]
    cf1 = ax1.contourf(x_g, y_g, u_mag, levels=30, cmap="plasma")
    ax1.set_title(f"Displacement Magnitude |u| (max={np.max(u_mag):.2f} mm | shape: {u_mag.shape})", fontsize=9.5, fontweight="bold")
    ax1.set_aspect("equal", "box")
    fig.colorbar(cf1, ax=ax1, label="|u| [mm]", fraction=0.046, pad=0.04)

    # 3. Horizontal displacement ux
    ax2 = axes[1, 0]
    cf2 = ax2.contourf(x_g, y_g, ux, levels=30, cmap="coolwarm")
    ax2.set_title(f"Horizontal ux [mm] (span [{np.min(ux):.2f}, {np.max(ux):.2f}] | shape: {ux.shape})", fontsize=9.5, fontweight="bold")
    ax2.set_aspect("equal", "box")
    ax2.set_xlabel("x [mm]", fontsize=9)
    ax2.set_ylabel("y [mm]", fontsize=9)
    fig.colorbar(cf2, ax=ax2, label="ux [mm]", fraction=0.046, pad=0.04)

    # 4. Vertical displacement uy
    ax3 = axes[1, 1]
    cf3 = ax3.contourf(x_g, y_g, uy, levels=30, cmap="coolwarm_r")
    ax3.set_title(f"Vertical uy [mm] (indentation={abs(np.min(uy)):.2f} mm | shape: {uy.shape})", fontsize=9.5, fontweight="bold")
    ax3.set_aspect("equal", "box")
    ax3.set_xlabel("x [mm]", fontsize=9)
    fig.colorbar(cf3, ax=ax3, label="uy [mm]", fraction=0.046, pad=0.04)

    # Render Dimension & Formulation Telemetry Tag
    render_dimension_tag(
        fig=fig,
        dataset_name=dataset_name,
        split=split,
        total_samples=total_samples,
        inputs_info=inputs_info,
        outputs_info=outputs_info,
        other_info=other_info,
        is_temporal=False,
        sample_id=sample_id,
        y_pos=0.99,
        fontsize=8.5,
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved sample static PNG: {out_path.name}")


def plot_static_overview_summary(
    coords: np.ndarray,
    coords_0: np.ndarray,
    displacements: np.ndarray,
    die_boundary: Optional[np.ndarray],
    sample_indices: List[int],
    out_path: Path,
    dataset_name: str,
    split: str,
    total_samples: int,
    inputs_info: Dict[str, Tuple[int, ...]],
    outputs_info: Dict[str, Tuple[int, ...]],
    other_info: Dict[str, Tuple[int, ...]],
    step_sub: int = 4,
) -> None:
    """Big overview comparing all selected static random samples."""
    num_samples = len(sample_indices)
    cols = 4  # Die, Deformed Mesh, uy, ux
    fig, axes = plt.subplots(
        num_samples,
        cols,
        figsize=(cols * 3.6, num_samples * 2.8 + 1.2),
        dpi=180,
    )
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for r_idx, s_idx in enumerate(sample_indices):
        x_g = coords[s_idx, :, :, 0]
        y_g = coords[s_idx, :, :, 1]
        ux = displacements[s_idx, :, :, 0]
        uy = displacements[s_idx, :, :, 1]
        u_mag = np.sqrt(ux**2 + uy**2)

        # Col 0: Die Profile
        ax_die = axes[r_idx, 0]
        if die_boundary is not None:
            die_x = die_boundary[s_idx, :, 0]
            die_y = die_boundary[s_idx, :, 1]
            ax_die.plot(die_x, die_y, color="#e11d48", linewidth=1.5)
            ax_die.set_title(f"Sample #{s_idx}\nDie Profile $S_d(x)$\nshape={die_boundary[s_idx].shape}", fontsize=8.5, fontweight="bold")
            ax_die.set_xlabel("x [mm]", fontsize=8)
            ax_die.set_ylabel("Elevation y [mm]", fontsize=8)
            ax_die.grid(True, linestyle="--", alpha=0.4)
        else:
            ax_die.text(0.5, 0.5, "No die data", ha="center", va="center")

        # Col 1: Deformed Mesh & |u|
        ax_mesh = axes[r_idx, 1]
        pm_mesh = ax_mesh.pcolormesh(x_g, y_g, u_mag, cmap="viridis", shading="auto")
        for i in range(0, x_g.shape[0], step_sub):
            ax_mesh.plot(x_g[i, :], y_g[i, :], color="black", linewidth=0.2, alpha=0.3)
        for j in range(0, x_g.shape[1], step_sub):
            ax_mesh.plot(x_g[:, j], y_g[:, j], color="black", linewidth=0.2, alpha=0.3)
        if die_boundary is not None:
            ax_mesh.plot(die_boundary[s_idx, :, 0], die_boundary[s_idx, :, 1], color="#e11d48", linestyle="--", linewidth=0.8)
        ax_mesh.set_title(f"Deformed Mesh (Static Final Step)\nmax|u|={np.max(u_mag):.2f} mm | shape={x_g.shape}", fontsize=8.5)
        ax_mesh.set_aspect("equal", "box")
        fig.colorbar(pm_mesh, ax=ax_mesh, fraction=0.046, pad=0.04)

        # Col 2: Vertical displacement uy
        ax_uy = axes[r_idx, 2]
        cf_uy = ax_uy.contourf(x_g, y_g, uy, levels=25, cmap="coolwarm_r")
        ax_uy.set_title(f"Vertical $u_y$ [mm]\nindentation={abs(np.min(uy)):.2f} mm | shape={uy.shape}", fontsize=8.5)
        ax_uy.set_aspect("equal", "box")
        fig.colorbar(cf_uy, ax=ax_uy, fraction=0.046, pad=0.04)

        # Col 3: Horizontal displacement ux
        ax_ux = axes[r_idx, 3]
        cf_ux = ax_ux.contourf(x_g, y_g, ux, levels=25, cmap="coolwarm")
        ax_ux.set_title(f"Horizontal $u_x$ [mm]\nspan=[{np.min(ux):.2f}, {np.max(ux):.2f}] | shape={ux.shape}", fontsize=8.5)
        ax_ux.set_aspect("equal", "box")
        fig.colorbar(cf_ux, ax=ax_ux, fraction=0.046, pad=0.04)

    # Render Dimension & Formulation Telemetry Tag
    render_dimension_tag(
        fig=fig,
        dataset_name=dataset_name,
        split=split,
        total_samples=total_samples,
        inputs_info=inputs_info,
        outputs_info=outputs_info,
        other_info=other_info,
        is_temporal=False,
        sample_id=None,
        y_pos=0.992,
        fontsize=8.5,
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.89])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] Saved big static overview PNG: {out_path.name}")


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def main() -> None:
    args = parse_arguments()
    data_dir_str = args.data_dir_flag or args.data_dir
    if not data_dir_str:
        print("Error: data_dir is required. Example: python plot_random_samples.py data/processed/ds_plasticity_nochange", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(data_dir_str).resolve()
    if not data_dir.exists():
        print(f"Error: dataset directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Derive output directory if not explicitly provided
    dataset_name = data_dir.name
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = (WORKSPACE_ROOT / "results" / "DatasetAnalysis" / dataset_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_file = data_dir / f"{args.split}.npz"
    if not split_file.exists():
        print(f"Error: Split file '{split_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    t0 = default_timer()
    print("=" * 70)
    print("      DATASET RANDOM SAMPLES DIAGNOSTIC VISUALIZER       ")
    print("=" * 70)
    print(f"Dataset Input Path:  {data_dir}")
    print(f"Output Directory:    {out_dir}")
    print(f"Split:               {args.split}")
    print(f"Number of Samples:   {args.num_samples}")
    print(f"Random Seed:         {args.seed}")
    print("-" * 70)

    print(f"Loading split: {split_file.name} ...")
    data = dict(np.load(split_file))

    # General inspection of what the dataset actually contains
    inputs_info, outputs_info, other_info = categorize_dataset_keys(data)

    print("Detected Dataset Content:")
    print("  Inputs: ", inputs_info)
    print("  Outputs:", outputs_info)
    if other_info:
        print("  Other:  ", other_info)

    coords = data["coords"]
    coords_0 = data.get("coords_initial")
    displacements = data["displacements"]
    die_boundary = data.get("die_boundary")

    total_samples = coords.shape[0]
    is_temporal = (coords.ndim == 5)
    is_unrolled_temporal = (
        coords.ndim == 4 and "simulation_idx" in data and "time_step_idx" in data
    )

    if is_unrolled_temporal:
        sim_ids = np.unique(data["simulation_idx"])
        num_sims = len(sim_ids)
        num_t = len(np.unique(data["time_step_idx"]))
        coords = coords.reshape(num_sims, num_t, coords.shape[1], coords.shape[2], coords.shape[3]).transpose(0, 2, 3, 1, 4)
        displacements = displacements.reshape(num_sims, num_t, displacements.shape[1], displacements.shape[2], displacements.shape[3]).transpose(0, 2, 3, 1, 4)
        if coords_0 is not None:
            coords_0 = coords_0[::num_t]
        if die_boundary is not None:
            die_boundary = die_boundary[::num_t]
        total_samples = num_sims
        is_temporal = True

    formulation_desc = (
        "Temporal (Unrolled Trajectory 4D -> Reshaped 5D)"
        if is_unrolled_temporal
        else ("Temporal (Time-dependent 5D)" if is_temporal else "Static (Final Step only 4D)")
    )
    print(f"Formulation:         {formulation_desc}")
    print(f"Total Simulations:   {total_samples}")

    # Randomly select samples
    rng = np.random.default_rng(args.seed)
    k = min(args.num_samples, total_samples)
    chosen_indices = sorted(rng.choice(total_samples, size=k, replace=False).tolist())
    print(f"Selected Sample IDs: {chosen_indices}")
    print("-" * 70)

    common_kwargs = {
        "dataset_name": dataset_name,
        "split": args.split,
        "total_samples": total_samples,
        "inputs_info": inputs_info,
        "outputs_info": outputs_info,
        "other_info": other_info,
    }

    if is_temporal:
        nt = coords.shape[3]
        print(f"Plotting full temporal progression ({nt} steps) for each sample with dimension tags...")
        # 1. Full temporal process for each individual sample
        for sid in chosen_indices:
            s_coords = coords[sid]  # (nx, ny, nt, 2)
            s_disp = displacements[sid]  # (nx, ny, nt, 2)
            s_die = die_boundary[sid] if die_boundary is not None else None
            out_sample_png = out_dir / f"sample_{sid:03d}_temporal_all_timesteps.png"
            plot_sample_temporal_all_steps(
                coords=s_coords,
                displacements=s_disp,
                die_boundary=s_die,
                sample_id=sid,
                out_path=out_sample_png,
                **common_kwargs,
            )

        # 2. Big comparative temporal overview across all selected samples
        overview_png = out_dir / "random_samples_temporal_overview.png"
        plot_temporal_overview_summary(
            coords=coords,
            displacements=displacements,
            die_boundary=die_boundary,
            sample_indices=chosen_indices,
            out_path=overview_png,
            **common_kwargs,
        )

    else:
        print("Plotting static solution fields for each sample with dimension tags...")
        # 1. Solution fields for each individual sample
        for sid in chosen_indices:
            s_coords = coords[sid]  # (nx, ny, 2)
            s_coords_0 = coords_0[sid] if coords_0 is not None else s_coords
            s_disp = displacements[sid]  # (nx, ny, 2)
            s_die = die_boundary[sid] if die_boundary is not None else None
            out_sample_png = out_dir / f"sample_{sid:03d}_static_solution.png"
            plot_sample_static_solution(
                coords=s_coords,
                coords_0=s_coords_0,
                displacements=s_disp,
                die_boundary=s_die,
                sample_id=sid,
                out_path=out_sample_png,
                **common_kwargs,
            )

        # 2. Big comparative static overview across all selected samples
        overview_png = out_dir / "random_samples_static_overview.png"
        plot_static_overview_summary(
            coords=coords,
            coords_0=coords_0,
            displacements=displacements,
            die_boundary=die_boundary,
            sample_indices=chosen_indices,
            out_path=overview_png,
            **common_kwargs,
        )

    elapsed = default_timer() - t0
    print("=" * 70)
    print(f"VISUALIZATION COMPLETED IN {elapsed:.2f}s")
    print(f"Artifacts successfully written to: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
