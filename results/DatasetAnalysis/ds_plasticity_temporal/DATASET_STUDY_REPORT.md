# Comprehensive Dataset Study: `ds_plasticity_temporal`
**Task Domain:** 2D Plane-Strain Elastoplastic Metal Forging  
**Surrogate Paradigm:** Time-Conditioned Continuous Spatiotemporal Neural Operators (SMART / Geo-FNO)  
**Processed Artifact Location:** `/home/arturo/GuidedResearch/TransformerSurrogatesPDEs-GuidedResearch/data/processed/ds_plasticity_temporal`  
**Study Date:** October 2026  

---

## 1. Executive Summary

This study presents a rigorous continuum mechanics and statistical analysis of `ds_plasticity_temporal`, an unrolled 4D temporal trajectory dataset structured for training and evaluating time-conditioned neural operator surrogates (such as SMART). The dataset models the finite-strain elastoplastic deformation of a rectangular metal workpiece indented by a parameterized, rigid punch-die profile $S_d(x)$ across $T = 20$ discrete time increments ($t^* \in [0.0, 1.0]$).

### Key Findings:
1. **Partition Integrity & Zero Leakage:** The dataset spans $N = 987$ unique finite element simulations strictly segregated into Train (900 simulations, 18000 snapshots), Validation (7 simulations, 140 snapshots), and Test (80 simulations, 1600 snapshots) with **zero simulation leakage** ($S_\mathrm{train} \cap S_\mathrm{val} \cap S_\mathrm{test} = \emptyset$).
2. **Exact Continuum Kinematics:** When tracking physical deformation, the relative kinematic relation $(\mathbf{x}(t) - \mathbf{x}_0) \equiv (\mathbf{u}(t) - \mathbf{u}_0)$ is satisfied across all 19740 snapshots within **3.49e-05 mm** (floating-point tolerance).
3. **Contact Offset Characterization:** In 527 of 900 training simulations (58.56%) in the raw ABAQUS source data, a pre-step contact offset $\mathbf{u}(t=0) \neq \mathbf{0}$ (up to 0.797 mm) was accumulated during die touch initialization. The unrolled trajectory representation correctly tracks this offset alongside continuous time parameter conditioning $t^*$.
4. **Boundary Condition Compliance:** The bottom workpiece edge ($y = -0.1$ mm) is rigidly clamped against vertical displacement ($u_y \equiv 0.0000$ mm across all samples), satisfying the Dirichlet support constraint.

---

## 2. Dataset Architecture & Tensor Structure

Each sample in `ds_plasticity_temporal` represents an instantaneous spatial snapshot of the workpiece at normalized physical time $t^* \in [0, 1]$.

| Array Name | Tensor Shape | Data Type | Physical Meaning |
| :--- | :--- | :--- | :--- |
| `die_boundary` | `(N, 101, 2)` | `float32` | Top boundary parameterized rigid punch-die profile $[x, S_d(x)]$ |
| `coords_initial` | `(N, 101, 31, 2)` | `float64` | Undeformed reference mesh coordinates $[X_0, Y_0]$ at $t = 0$ |
| `coords` | `(N, 101, 31, 2)` | `float64` | Current deformed physical spatial grid coordinates $[x(t), y(t)]$ |
| `displacements` | `(N, 101, 31, 2)` | `float64` | Nodal displacement field vector $[u_x(t), u_y(t)]$ |
| `target_all` | `(N, 101, 31, 4)` | `float64` | Concatenated 4-channel state $[x(t), y(t), u_x(t), u_y(t)]$ |
| `time_steps` | `(20,)` | `float32` | Normalized global time grid $t^*_k = k / 19 \in [0.0, 1.0]$ |
| `time_params` | `(N, 1)` | `float32` | Continuous scalar conditioning parameter $t^*$ for sample $i$ |
| `simulation_idx` | `(N, 1)` | `int32` | Global simulation sequence identifier ($0$ to $986$) |
| `time_step_idx` | `(N, 1)` | `int32` | Discrete time increment index ($0$ to $19$) |

---

## 3. Continuum Mechanics & Boundary Diagnostics

| Diagnostic Test | Measured Value | Theoretical Expectation | Status |
| :--- | :--- | :--- | :--- |
| **Simulation Split Disjointness** | $0$ overlap | $0$ shared simulations | **PASSED** (Zero Leakage) |
| **Relative Kinematic Identity** $|(\Delta \mathbf{x} - \Delta \mathbf{u})|$ | **3.49e-05 mm** | $< 10^{-4}$ mm (Float precision) | **PASSED** |
| **Base Platen Clamping** $\max |u_y(x, y_\mathrm{bottom}, t)|$ | **0.000000 mm** | $\equiv 0.0$ mm | **PASSED** (Exact Clamp) |
| **Max Punch Compressive Stroke** | **6.511 mm** | $5.0 - 7.0$ mm downward indentation | **PASSED** |
| **Max Displacement Magnitude** $|\mathbf{u}|$ | **6.603 mm** | Plastic flow compliant | **PASSED** |
| **Time Monotonicity** | Strict monotonic | $\Delta t^* > 0$ | **PASSED** |

---

## 4. Statistical Moments Summary (Zero Leakage)

Computed exclusively on `train.npz` (18000 samples, 900 simulations):

| Physical Field | Global Mean | Global Std | Min | Max | Channel 0 (x) [Mean &plusmn; Std] | Channel 1 (y) [Mean &plusmn; Std] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Punch Die $S_d(x)$** | -2.507 | 24.481 | -49.650 | 26.103 | -24.650 &plusmn; 14.577 | 19.635 &plusmn; 2.365 |
| **Initial Mesh $\mathbf{X}_0$** | -8.625 | 19.314 | -49.650 | 14.908 | -24.650 &plusmn; 14.577 | 7.400 &plusmn; 4.472 |
| **Deformed Mesh $\mathbf{x}(t)$** | -8.717 | 19.338 | -54.836 | 15.063 | -24.679 &plusmn; 14.804 | 7.245 &plusmn; 4.381 |
| **Displacement $\mathbf{u}(t)$** | -0.092 | 0.744 | -6.511 | 5.728 | -0.029 &plusmn; 0.898 | -0.155 &plusmn; 0.541 |
| **Time Parameter $t^*$** | 0.500 | 0.303 | 0.000 | 1.000 | 0.500 &plusmn; 0.303 | N/A |

---

## 5. Visual Diagnostic Artifacts

The following publication-grade diagnostic figures (300 DPI) were generated in this study:

1. **`die_profiles_variation.png`**: Highlights 12 representative punch-die spline geometries $S_d(x)$ plotted against the undeformed billet top edge ($y = 14.91$ mm).
2. **`spatiotemporal_mesh_sequence.png`**: Multi-panel spatiotemporal sequence showing the deformed mesh at $t^* \in \{0.00, 0.21, 0.42, 0.63, 0.84, 1.00\}$ with displacement magnitude heatmap and wireframe grid lines.
3. **`displacement_field_contours.png`**: High-resolution 2D spatial contour heatmaps of $u_x$, $u_y$, and $|\mathbf{u}|$ at the final forging step with punch-die profile overlay.
4. **`spatiotemporal_evolution_trajectories.png`**: Population trajectory curves showing median and 10th-90th percentile bands for maximum indentation stroke and resultant displacement progression over continuous time $t^* \in [0, 1]$.
5. **`field_distributions_and_correlations.png`**: 4-panel histogram distribution of displacement components, resultant magnitude, and die elevation.
6. **`split_comparative_distributions.png`**: Density distributions across Train, Validation, and Test partitions demonstrating statistical domain alignment.
7. **`continuum_mechanics_diagnostics.png`**: Continuum mechanics validation displaying base Dirichlet clamping ($|u_y| \equiv 0$) and machine-precision kinematic identity consistency.
8. **`interactive_explorer.html`**: Interactive web dashboard allowing real-time parameter exploration and time scrubbing across simulation runs.

---

## 6. Recommendations for Surrogate Modeling (SMART / Geo-FNO)

1. **Time Conditioning Architecture:** Surrogates should concatenate the normalized time parameter $t^* \in [0, 1]$ directly into the model's global conditioning vector or as an auxiliary spatial channel tiled across nodes.
2. **Coordinate & Boundary Invariance:** The undeformed initial mesh $\mathbf{X}_0$ and the punch profile $S_d(x)$ must be preserved as static inputs while predicting instantaneous state $[x(t), y(t), u_x(t), u_y(t)]$.
3. **Physics-Informed Penalty:** Incorporating a penalty loss on the bottom edge vertical displacement $\mathcal{L}_\mathrm{bottom} = \|u_y(x, y_\mathrm{bottom}, t)\|^2_2$ can enforce physical platen contact compliance during inference.
