# Dataset Analysis & Physical Diagnostic Studies

This directory houses comprehensive dataset characterization studies, physical consistency benchmarks, continuum mechanics diagnostics, and interactive visualization tools for the simulation datasets used to train neural operator surrogates.

---

## Dataset Studies Index

| Dataset Subfolder | Formulation | Total Samples | Unique Sims | Time Steps | Status / Study Report |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`ds_plasticity_temporal/`](ds_plasticity_temporal/) | **Continuous Trajectory (Unrolled 4D)** | 19,740 | 987 | 20 increments ($t^* \in [0, 1]$) | [DATASET_STUDY_REPORT.md](ds_plasticity_temporal/DATASET_STUDY_REPORT.md) |
| `ds_plasticity_onlyFinalStep/` | **Static Steady-State (4D)** | 987 | 987 | Final Step only ($t = -1$) | See `scripts/DatasetAnalysis/generate_analysis_onlyFinalStep.py` |
| `ds_plasticity_nochange/` | **Full Trajectory (5D)** | 987 | 987 | 20 increments | See `scripts/DatasetAnalysis/analyzeDS_plasticity.py` |

---

## `ds_plasticity_temporal` Result Study Overview

The primary temporal benchmark for training continuous spatiotemporal surrogate models (e.g. SMART).

- **Full Report:** [DATASET_STUDY_REPORT.md](ds_plasticity_temporal/DATASET_STUDY_REPORT.md)
- **Machine Telemetry:** [analysis_summary.json](ds_plasticity_temporal/analysis_summary.json)
- **Interactive Explorer:** [interactive_explorer.html](ds_plasticity_temporal/interactive_explorer.html)

### Key Physical & Continuum Mechanics Metrics:
- **Zero Simulation Leakage:** Train (900 sims, 18,000 steps), Val (7 sims, 140 steps), Test (80 sims, 1,600 steps) are strictly disjoint.
- **Kinematic Precision Identity:** Relative continuum deformation identity $(\mathbf{x}(t) - \mathbf{x}_0) \equiv (\mathbf{u}(t) - \mathbf{u}_0)$ is satisfied within **$3.49 \times 10^{-5}$ mm** across all snapshots.
- **Base Platen Clamping:** Vertical displacement along the bottom tool support ($y = -0.1$ mm) is identically **$0.0000$ mm** ($u_y \equiv 0$).
- **Die Indentation Stroke:** Peak vertical downward stroke indents up to **$6.511$ mm** into the billet, driving plastic flow with peak magnitude $|\mathbf{u}| = 6.603$ mm.
- **Pre-Step Contact Initialization:** 582 of 900 training simulations in the ABAQUS source data contain an initial die touch offset $\mathbf{u}(t=0) \neq \mathbf{0}$ (up to $0.797$ mm), faithfully preserved and documented for continuous time parameter conditioning.

### Diagnostic Visualizations:
- `die_profiles_variation.png`: 12 representative punch-die spline geometries $S_d(x)$ across varying curvatures.
- `spatiotemporal_mesh_sequence.png`: Multi-increment 2D curvilinear mesh deformation progression over $t^* \in \{0.00, 0.21, 0.42, 0.63, 0.84, 1.00\}$.
- `displacement_field_contours.png`: High-resolution 2D contour fields of $u_x$, $u_y$, and $|\mathbf{u}|$ at final punch penetration.
- `spatiotemporal_evolution_trajectories.png`: Population median and 10th–90th percentile trajectories for stroke depth and deformation magnitude.
- `field_distributions_and_correlations.png`: 4-panel histogram distributions of kinematic fields and die elevations.
- `split_comparative_distributions.png`: Partition distributions verifying domain alignment between Train, Val, and Test splits.
- `continuum_mechanics_diagnostics.png`: Validation curves for base platen clamping and continuum kinematic precision.
- `random_samples_temporal_overview.png`: Comparative rollout of randomly sampled simulation trajectories.
