#!/usr/bin/env python3
"""Diagnostic script to plot time-dependent output of a random sample with the press line overlay.

Generates a publication-grade diagnostic PNG visualizing the complete 20-step spatiotemporal
deformation field of a metal workpiece indented by a rigid press tool.

Key Features:
- Exact rigid kinematic translation of the press surface profile at each time step.
- Press line plotted prominently in red color in its physically correct location.
- Explicit legend indicating that the red line emulates the press surface location.
- High-resolution (300 DPI) output saved to results/DatasetAnalysis/ds_plasticity_nochange/.

Usage:
------
python scripts/DatasetAnalysis/plot_temporal_output_with_press.py
python scripts/DatasetAnalysis/plot_temporal_output_with_press.py --sample_idx 80
python scripts/DatasetAnalysis/plot_temporal_output_with_press.py --seed 42 --filename random_sample_temporal_output.png
"""

import argparse
import math
from pathlib import Path
import sys
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Suppress pcolormesh and layout warnings
warnings.filterwarnings("ignore", message=".*input coordinates to pcolormesh.*")
warnings.filterwarnings("ignore", message=".*This figure includes Axes that are not compatible with tight_layout.*")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot time-dependent output of a dataset sample with the press surface location."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/ds_plasticity_nochange",
        help="Path to processed dataset directory (default: data/processed/ds_plasticity_nochange).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/DatasetAnalysis/ds_plasticity_nochange",
        help="Target output directory (default: results/DatasetAnalysis/ds_plasticity_nochange).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset partition to sample from (default: train).",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="Specific sample index to visualize. If None, chosen randomly.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42).",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="random_sample_temporal_output.png",
        help="Output filename (default: random_sample_temporal_output.png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution in dots per inch (default: 300).",
    )
    return parser.parse_args()


def plot_sample_temporal_output(
    coords: np.ndarray,
    displacements: np.ndarray,
    die_boundary: np.ndarray,
    sample_id: int,
    total_samples: int,
    split: str,
    output_path: Path,
    dpi: int = 300,
    step_sub: int = 4,
) -> Path:
    """Render and save a 20-step spatiotemporal grid with press line in red and legend.

    Parameters
    ----------
    coords : np.ndarray
        Deformed coordinates over time, shape (101, 31, 20, 2).
    displacements : np.ndarray
        Displacement fields over time, shape (101, 31, 20, 2).
    die_boundary : np.ndarray
        Rigid die profile [x, S_d(x)], shape (101, 2).
    sample_id : int
        Sample identifier index.
    total_samples : int
        Total samples in partition.
    split : str
        Partition name ('train', 'val', or 'test').
    output_path : Path
        Destination PNG path.
    dpi : int, default=300
        Image resolution.
    step_sub : int, default=4
        Subsampling interval for finite element mesh wireframe.

    Returns
    -------
    Path
        Destination path of the generated figure.
    """
    nt = coords.shape[2]  # 20 timesteps
    cols = 5
    rows = math.ceil(nt / cols)

    disp_mag_all = np.linalg.norm(displacements, axis=-1)  # (101, 31, 20)
    vmax = float(np.max(disp_mag_all))
    vmax = max(vmax, 1e-4)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 3.6, rows * 3.0 + 1.4),
        sharex=True,
        sharey=True,
        dpi=dpi,
    )
    axes_flat = axes.ravel()

    die_x = die_boundary[:, 0]
    die_y = die_boundary[:, 1]
    die_y_min = float(np.min(die_y))

    pm_ref = None
    press_line_handle = None

    for t in range(nt):
        ax = axes_flat[t]
        x_g = coords[:, :, t, 0]
        y_g = coords[:, :, t, 1]
        u_mag = disp_mag_all[:, :, t]

        # 1. Continuous displacement magnitude heatmap
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

        # 2. Deformed finite element wireframe lines
        for i in range(0, x_g.shape[0], step_sub):
            ax.plot(x_g[i, :], y_g[i, :], color="black", linewidth=0.25, alpha=0.35)
        for j in range(0, x_g.shape[1], step_sub):
            ax.plot(x_g[:, j], y_g[:, j], color="black", linewidth=0.25, alpha=0.35)

        # 3. Rigid press surface line placed in the exact right physical location
        # The workpiece top boundary is indexed at y-index 0.
        top_y_min = float(np.min(y_g[:, 0]))
        stroke = die_y_min - top_y_min
        die_y_stroked = die_y - stroke

        (press_line,) = ax.plot(
            die_x,
            die_y_stroked,
            color="red",
            linestyle="-",
            linewidth=1.5,
            alpha=0.95,
            label="Press surface location",
            zorder=5,
        )
        if press_line_handle is None:
            press_line_handle = press_line

        # 4. Legend indicating the red line emulates the press surface location
        ax.legend(
            handles=[press_line],
            labels=["Press surface location"],
            loc="upper right",
            fontsize=6.5,
            framealpha=0.88,
            edgecolor="#e2e8f0",
        )

        # 5. Panel metadata titles
        t_norm = t / max(nt - 1, 1)
        ax.set_title(
            f"Step {t:02d} ($t^*={t_norm:.2f}$)\nStroke={stroke:.2f} mm | max|u|={np.max(u_mag):.2f} mm",
            fontsize=8.5,
            fontweight="bold",
        )
        ax.set_aspect("equal", "box")
        if t % cols == 0:
            ax.set_ylabel("$y$ [mm]", fontsize=9.0)
        if t >= (rows - 1) * cols:
            ax.set_xlabel("$x$ [mm]", fontsize=9.0)

    # Global shared colorbar
    if pm_ref is not None:
        cbar = fig.colorbar(
            pm_ref,
            ax=axes_flat.tolist(),
            orientation="horizontal",
            fraction=0.03,
            pad=0.06,
        )
        cbar.set_label(r"Displacement Field Magnitude $|\mathbf{u}(t)|$ [mm]", fontsize=10.5, fontweight="bold")

    # Global Diagnostic Suptitle with explicit note
    fig.suptitle(
        f"2D Elastoplastic Forging Benchmark: Time-Dependent Output (Sample #{sample_id} from {split} set, N={total_samples})\n"
        r"$\mathbf{Legend:}$ The red solid line emulates the rigid press surface location at each active indentation stroke",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_arguments()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    split_file = data_dir / f"{args.split}.npz"

    if not split_file.exists():
        print(f"Error: Split file '{split_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print("=" * 75)
    print("      DATASET TEMPORAL OUTPUT PLOTTER WITH PRESS SURFACE OVERLAY       ")
    print("=" * 75)
    print(f"Dataset directory: {data_dir}")
    print(f"Partition:         {args.split}")
    print(f"Target directory:  {output_dir}")

    # Load partition using mmap for fast random-access
    data = np.load(split_file, mmap_mode="r")
    total_samples = data["coords"].shape[0]

    # Select sample
    if args.sample_idx is not None:
        if args.sample_idx < 0 or args.sample_idx >= total_samples:
            print(f"Error: sample_idx {args.sample_idx} out of range [0, {total_samples - 1}].", file=sys.stderr)
            sys.exit(1)
        sample_id = args.sample_idx
    else:
        rng = np.random.default_rng(args.seed)
        sample_id = int(rng.integers(0, total_samples))

    print(f"Selected sample:   #{sample_id} (out of {total_samples} samples)")
    print(f"Output filename:   {args.filename}")
    print(f"Plot DPI:          {args.dpi}")
    print("-" * 75)

    coords = data["coords"][sample_id]            # (101, 31, 20, 2)
    displacements = data["displacements"][sample_id]  # (101, 31, 20, 2)
    die_boundary = data["die_boundary"][sample_id]    # (101, 2)

    out_path = output_dir / args.filename
    print(f"Rendering spatiotemporal trajectory with press surface overlay...")
    plot_sample_temporal_output(
        coords=coords,
        displacements=displacements,
        die_boundary=die_boundary,
        sample_id=sample_id,
        total_samples=total_samples,
        split=args.split,
        output_path=out_path,
        dpi=args.dpi,
    )

    print(f"[✓] Saved successfully: {out_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
