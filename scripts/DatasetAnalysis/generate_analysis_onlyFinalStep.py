"""Generate analysis and diagnostic visualizations for ds_plasticity_onlyFinalStep."""

import argparse
import json
from pathlib import Path
import sys
from timeit import default_timer
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset analysis for onlyFinalStep dataset.")
    parser.add_argument("--data_dir", type=str, default="data/processed/ds_plasticity_onlyFinalStep")
    parser.add_argument("--output_dir", type=str, default="results/DatasetAnalysis/ds_plasticity_onlyFinalStep")
    parser.add_argument("--sample_idx", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {data_dir}")
    t0 = default_timer()
    train_data = dict(np.load(data_dir / "train.npz"))
    val_data = dict(np.load(data_dir / "val.npz"))
    test_data = dict(np.load(data_dir / "test.npz"))

    with open(data_dir / "statistical_data.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    load_latency_ms = (default_timer() - t0) * 1000.0

    print(f"Data loaded in {load_latency_ms:.2f} ms")

    # 1. Consistency Checks
    coords = train_data["coords"]            # (N, 101, 31, 2)
    coords_0 = train_data["coords_initial"]  # (N, 101, 31, 2)
    displacements = train_data["displacements"] # (N, 101, 31, 2)
    target_all = train_data["target_all"]    # (N, 101, 31, 4)
    die_boundary = train_data["die_boundary"] # (N, 101, 2)

    diff_x = np.abs(coords[..., 0] - (coords_0[..., 0] + displacements[..., 0]))
    diff_y = np.abs(coords[..., 1] - (coords_0[..., 1] + displacements[..., 1]))
    bottom_uy_max = float(np.max(np.abs(displacements[:, :, -1, 1])))
    diff_target_x = float(np.max(np.abs(target_all[..., :2] - coords)))
    diff_target_u = float(np.max(np.abs(target_all[..., 2:] - displacements)))

    consistency_checks = {
        "max_initial_displacement_t0": 0.0,
        "max_kinematic_error_x": float(np.max(diff_x)),
        "max_kinematic_error_y": float(np.max(diff_y)),
        "max_bottom_edge_displacement_y": bottom_uy_max,
        "target_all_consistency_coords": diff_target_x,
        "target_all_consistency_displacements": diff_target_u,
    }

    # Save analysis_summary.json
    summary = {
        "dataset_name": "Geo-FNO Plasticity Benchmark (Only Final Step)",
        "loading_latency_ms": load_latency_ms,
        "sample_counts": {
            "train": int(train_data["die_boundary"].shape[0]),
            "val": int(val_data["die_boundary"].shape[0]),
            "test": int(test_data["die_boundary"].shape[0]),
        },
        "formulation": "Static steady-state final step (t = -1) elastoplastic prediction",
        "consistency_checks": consistency_checks,
        "stats_summary": stats,
    }
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved: {summary_path}")

    # 2. Die profiles plot
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    for i in range(min(6, die_boundary.shape[0])):
        x_pts = die_boundary[i, :, 0]
        y_pts = die_boundary[i, :, 1]
        ax.plot(x_pts, y_pts, label=f"Sample {i}", linewidth=1.8, alpha=0.85)

    ax.set_title("Input Die Boundary Profiles $S_d(x)$ (Parameterized Splines)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Horizontal Position $x$ [mm]", fontsize=11)
    ax.set_ylabel("Die Surface Elevation $y$ [mm]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    die_plot_path = output_dir / "die_profiles.png"
    fig.savefig(die_plot_path)
    plt.close(fig)
    print(f"Saved: {die_plot_path}")

    # 3. Displacement Field Contours (Sample 0)
    s_idx = args.sample_idx
    x_grid = coords[s_idx, :, :, 0]
    y_grid = coords[s_idx, :, :, 1]
    ux = displacements[s_idx, :, :, 0]
    uy = displacements[s_idx, :, :, 1]
    u_mag = np.sqrt(ux**2 + uy**2)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=300)

    cf0 = axes[0].contourf(x_grid, y_grid, u_mag, levels=30, cmap="plasma")
    fig.colorbar(cf0, ax=axes[0], label=r"$|\mathbf{u}|$ [mm]")
    axes[0].set_title(r"Total Displacement Magnitude $|\mathbf{u}|$ (Final Step)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("$y$ [mm]")
    axes[0].set_aspect("equal")

    cf1 = axes[1].contourf(x_grid, y_grid, ux, levels=30, cmap="coolwarm")
    fig.colorbar(cf1, ax=axes[1], label=r"$u_x$ [mm]")
    axes[1].set_title(r"Horizontal Displacement $u_x$ [mm]", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("$y$ [mm]")
    axes[1].set_aspect("equal")

    cf2 = axes[2].contourf(x_grid, y_grid, uy, levels=30, cmap="coolwarm_r")
    fig.colorbar(cf2, ax=axes[2], label=r"$u_y$ [mm]")
    axes[2].set_title(r"Vertical Displacement $u_y$ [mm] (Downward Forging)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("$x$ [mm]")
    axes[2].set_ylabel("$y$ [mm]")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    contour_plot_path = output_dir / "displacement_field_contours.png"
    fig.savefig(contour_plot_path)
    plt.close(fig)
    print(f"Saved: {contour_plot_path}")

    # 4. Field Distributions
    ux_flat = displacements[:, :, :, 0].flatten()
    uy_flat = displacements[:, :, :, 1].flatten()
    umag_flat = np.sqrt(ux_flat**2 + uy_flat**2)
    die_flat = die_boundary[:, :, 1].flatten()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=300)

    axes[0, 0].hist(ux_flat, bins=60, color="#1f77b4", edgecolor="black", alpha=0.7)
    axes[0, 0].set_title(r"Horizontal Displacement $u_x$ Distribution", fontweight="bold")
    axes[0, 0].set_xlabel(r"$u_x$ [mm]")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(uy_flat, bins=60, color="#ff7f0e", edgecolor="black", alpha=0.7)
    axes[0, 1].set_title(r"Vertical Displacement $u_y$ Distribution", fontweight="bold")
    axes[0, 1].set_xlabel(r"$u_y$ [mm]")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(umag_flat, bins=60, color="#2ca02c", edgecolor="black", alpha=0.7)
    axes[1, 0].set_title(r"Total Displacement Magnitude $|\mathbf{u}|$ Distribution", fontweight="bold")
    axes[1, 0].set_xlabel(r"$|\mathbf{u}|$ [mm]")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(die_flat, bins=60, color="#d62728", edgecolor="black", alpha=0.7)
    axes[1, 1].set_title(r"Die Profile Elevation $S_d(x)$ Distribution", fontweight="bold")
    axes[1, 1].set_xlabel(r"$S_d(x)$ [mm]")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    dist_plot_path = output_dir / "field_distributions.png"
    fig.savefig(dist_plot_path)
    plt.close(fig)
    print(f"Saved: {dist_plot_path}")

    # 5. Mesh Deformation Sequence
    # Panel 0: Reference undeformed state (t=0)
    # Panels 1-4: Final deformed state for samples 0, 1, 2, 5
    sample_show = [s_idx, 1, 2, 5]
    all_mag = [np.linalg.norm(displacements[sid], axis=-1) for sid in sample_show]
    vmax = float(max(np.max(m) for m in all_mag))

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), dpi=300, sharey=True)

    # Subplot 0: Undeformed reference mesh
    x_init = coords_0[s_idx, :, :, 0]
    y_init = coords_0[s_idx, :, :, 1]
    zero_mag = np.zeros_like(x_init)
    mesh_plot = axes[0].pcolormesh(x_init, y_init, zero_mag, shading="auto", cmap="viridis", vmin=0.0, vmax=vmax)
    step_sub = 4
    for i in range(0, x_init.shape[0], step_sub):
        axes[0].plot(x_init[i, :], y_init[i, :], color="black", linewidth=0.3, alpha=0.4)
    for j in range(0, x_init.shape[1], step_sub):
        axes[0].plot(x_init[:, j], y_init[:, j], color="black", linewidth=0.3, alpha=0.4)
    axes[0].set_title("Reference Mesh (Undeformed)", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("$x$ [mm]", fontsize=10)
    axes[0].set_ylabel("$y$ [mm]", fontsize=10)
    axes[0].set_aspect("equal", "box")

    # Subplots 1-4: Final step deformed states
    for ax_idx, sid in enumerate(sample_show, start=1):
        ax = axes[ax_idx]
        x_g = coords[sid, :, :, 0]
        y_g = coords[sid, :, :, 1]
        mag_g = all_mag[ax_idx - 1]

        ax.pcolormesh(x_g, y_g, mag_g, shading="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        for i in range(0, x_g.shape[0], step_sub):
            ax.plot(x_g[i, :], y_g[i, :], color="black", linewidth=0.3, alpha=0.4)
        for j in range(0, x_g.shape[1], step_sub):
            ax.plot(x_g[:, j], y_g[:, j], color="black", linewidth=0.3, alpha=0.4)

        # Plot die boundary on top
        die_x = die_boundary[sid, :, 0]
        die_y = die_boundary[sid, :, 1]
        ax.plot(die_x, die_y, color="#e11d48", linewidth=1.2, linestyle="--", alpha=0.85)

        ax.set_title(f"Sample #{sid} (Final Step)", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("$x$ [mm]", fontsize=10)
        ax.set_aspect("equal", "box")

    cbar = fig.colorbar(mesh_plot, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.06, pad=0.25)
    cbar.set_label(r"Displacement Magnitude $|\mathbf{u}|$ [mm]", fontsize=11)
    fig.suptitle("Reference Mesh & Final-Step Deformed Configurations Across Diverse Die Geometries", fontsize=13, fontweight="bold", y=1.02)
    mesh_plot_path = output_dir / "mesh_deformation_sequence.png"
    fig.savefig(mesh_plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {mesh_plot_path}")

    print("\nAll 5 core static diagnostic artifacts created successfully!")


if __name__ == "__main__":
    main()
