# Dataset Analysis & Visualization Scripts Documentation

This document describes the utilities located in `scripts/DatasetAnalysis/` used for diagnostic visualization, statistical profiling, physical consistency validation, and interactive exploration of surrogate PDE datasets.

---

## Overview of Scripts

| Script | Purpose | Supported Formulations | Default Output Directory |
| :--- | :--- | :--- | :--- |
| [`plot_random_samples.py`](../../scripts/DatasetAnalysis/plot_random_samples.py) | Quick visual diagnostics & telemetry verification for random samples. | Native 5D, Unrolled 4D Temporal, Static 4D | `results/DatasetAnalysis/<dataset_name>/` |
| [`generate_analysis_temporal.py`](../../scripts/DatasetAnalysis/generate_analysis_temporal.py) | Comprehensive continuum mechanics validation, KPI trajectories, and interactive HTML. | Unrolled 4D Temporal (`ds_plasticity_temporal`) | `results/DatasetAnalysis/ds_plasticity_temporal/` |
| [`generate_analysis_onlyFinalStep.py`](../../scripts/DatasetAnalysis/generate_analysis_onlyFinalStep.py) | Verification & visualization suite for static final-step elastoplastic datasets. | Static 4D (`ds_plasticity_onlyFinalStep`) | `results/DatasetAnalysis/ds_plasticity_onlyFinalStep/` |
| [`analyzeDS_plasticity.py`](../../scripts/DatasetAnalysis/analyzeDS_plasticity.py) | Statistical characterization and sample plotting for raw 5D trajectories. | Native 5D (`ds_plasticity_nochange`) | `results/DatasetAnalysis/ds_plasticity_nochange/` |

---

## Dimension & Formulation Detection Logic

When inspecting `.npz` dataset split files (`train.npz`, `val.npz`, `test.npz`), the plotting and analysis tools automatically inspect tensor ranks and dictionary metadata to determine whether the problem is static or dynamic.

### How the Three Cases are Distinguished:

1. **Native 5D (`ds_plasticity_nochange`):**
    - `coords.ndim == 5` → Detected immediately as temporal.
    - Tensor shape is `(N_sims, Nx, Ny, Nt, C)` where time is the 4th explicit axis.
2. **Unrolled Temporal 4D (`ds_plasticity_temporal`):**
    - `coords.ndim == 4`, BUT the file contains `"simulation_idx"` and `"time_step_idx"` (and `"time_params"`).
    - The script sees these tracking keys, calculates $N_\text{sims} = 987$ and $N_t = 20$, and reconstructs the 5D trajectory in memory so it can plot the time progression.
3. **Truly Static 4D (`ds_plasticity_onlyFinalStep`):**
    - `coords.ndim == 4`, and there is no `simulation_idx` or `time_step_idx`.
    - The script recognizes it as purely static and draws the single-snapshot row `[Die | Mesh | uy | ux]`.

---

## Detailed Script Descriptions

### 1. `plot_random_samples.py`
Designed to be an unopinionated diagnostic tool. Every generated PNG includes an embedded diagnostic telemetry banner displaying:
- Dataset name and split partition (`train`, `val`, `test`).
- Number of samples in the split.
- Formulation mode (`TEMPORAL (5D)` vs `STATIC (4D)`).
- Input and output tensor shapes.

**Usage Examples:**
```bash
# Plot 3 random temporal simulations across all 20 time increments:
python scripts/DatasetAnalysis/plot_random_samples.py data/processed/ds_plasticity_temporal --num_samples 3 --seed 42

# Plot 3 random samples from static final step dataset:
python scripts/DatasetAnalysis/plot_random_samples.py data/processed/ds_plasticity_onlyFinalStep --num_samples 3 --seed 42
```

### 2. `generate_analysis_temporal.py`
Generates publication-quality physical continuum diagnostics (300 DPI) and an interactive Plotly dashboard for the continuous unrolled dataset:
- Verifies base platen clamping: $u_y(x, y_\text{bottom}, t) \equiv 0$.
- Checks relative continuum kinematic precision: $|(\mathbf{x}(t) - \mathbf{X}_0) - (\mathbf{u}(t) - \mathbf{u}_0)| < 10^{-4}$ mm.
- Computes population-wide stroke penetration and displacement evolution trajectories.
- Exports structured report: `DATASET_STUDY_REPORT.md` and `analysis_summary.json`.

**Usage Example:**
```bash
python scripts/DatasetAnalysis/generate_analysis_temporal.py \
    --data_dir data/processed/ds_plasticity_temporal \
    --output_dir results/DatasetAnalysis/ds_plasticity_temporal
```
