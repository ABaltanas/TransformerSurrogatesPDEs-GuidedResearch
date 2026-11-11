# Relative L2 Error Metrics in the Plasticity Benchmark

In this benchmark, two different relative $L_2$ error formulations are used: one used in **this repository**, and one used in the **literature** (such as PLT, Geo-FNO, and Transolver).

---

## 1. Pure Displacement Relative L2 Error (Used in This Repository)

This metric evaluates the relative error strictly on the physical displacement field of the nodes at time $t$.

### Formula
$$\text{Rel-}L_2 = \frac{\|\hat{\mathbf{u}} - \mathbf{u}\|_2}{\|\mathbf{u}\|_2}$$

### Meaning of Each Element
* $\mathbf{u}(t) = [u_x(t), u_y(t)]$: Ground-truth displacement vector of a node at time $t$ (how much the node has moved from its undeformed reference position).
* $\hat{\mathbf{u}}(t) = [\hat{u}_x(t), \hat{u}_y(t)]$: Predicted displacement vector of the node at time $t$.
* $\|\hat{\mathbf{u}} - \mathbf{u}\|_2$: $L_2$ (Frobenius) norm of the difference between predicted and true displacements across all nodes and time steps.
* $\|\mathbf{u}\|_2$: $L_2$ (Frobenius) norm of the ground-truth displacements across all nodes and time steps.

---

## 2. Joint 4-Channel Relative L2 Error (Used in the Literature)

This metric evaluates the relative error jointly across four output channels: the deformed spatial positions and the physical displacements of the nodes at time $t$.

### Formula
$$\text{Rel-}L_2 = \frac{\|\hat{\mathbf{Y}} - \mathbf{Y}\|_2}{\|\mathbf{Y}\|_2}$$

### Meaning of Each Element
* $\mathbf{Y}(t) = [x(t), y(t), u_x(t), u_y(t)]$: Ground-truth 4-channel target vector of a node at time $t$, where:
  * $x(t)$: Ground-truth spatial $x$-position of the deformed node at time $t$ ($x(t) = x_0 + u_x(t)$).
  * $y(t)$: Ground-truth spatial $y$-position of the deformed node at time $t$ ($y(t) = y_0 + u_y(t)$).
  * $u_x(t)$: Ground-truth displacement of the node along the $x$-axis at time $t$.
  * $u_y(t)$: Ground-truth displacement of the node along the $y$-axis at time $t$.
* $\hat{\mathbf{Y}}(t) = [\hat{x}(t), \hat{y}(t), \hat{u}_x(t), \hat{u}_y(t)]$: Predicted 4-channel vector for the node at time $t$ (predicted deformed positions and displacements).
* $\|\hat{\mathbf{Y}} - \mathbf{Y}\|_2$: $L_2$ (Frobenius) norm of the difference between predictions and targets across all 4 channels, nodes, and time steps.
* $\|\mathbf{Y}\|_2$: $L_2$ (Frobenius) norm of the complete 4-channel target tensor across all channels, nodes, and time steps.

---

## 3. Mathematical Relationship & Scale Discrepancy

Because the initial undeformed mesh $\mathbf{x}_0$ is fixed, the difference in the deformed spatial coordinates equals the difference in displacements:
$$(\mathbf{x}_0 + \hat{\mathbf{u}}) - (\mathbf{x}_0 + \mathbf{u}) = \hat{\mathbf{u}} - \mathbf{u}$$

Consequently:
$$\|\hat{\mathbf{Y}} - \mathbf{Y}\|_2 = \sqrt{\|\hat{\mathbf{u}} - \mathbf{u}\|_2^2 + \|\hat{\mathbf{u}} - \mathbf{u}\|_2^2} = \sqrt{2} \, \|\hat{\mathbf{u}} - \mathbf{u}\|_2$$

The target norm is:
$$\|\mathbf{Y}\|_2 = \sqrt{\|\mathbf{x}_0 + \mathbf{u}\|_2^2 + \|\mathbf{u}\|_2^2}$$

In solid mechanics benchmarks, spatial coordinate magnitudes $\|\mathbf{x}_0 + \mathbf{u}\|_2$ (tens to hundreds of millimeters) are orders of magnitude larger than relative deformations $\|\mathbf{u}\|_2$ (fractions of a millimeter to few millimeters). As a result:
$$\|\mathbf{Y}\|_2 \gg \|\mathbf{u}\|_2 \implies \text{Rel-}L_{2,\text{Joint}} \ll \text{Rel-}L_{2,\text{Pure}}$$

Typically, a pure displacement relative error of **3% - 10%** scales down to **0.06% - 0.8%** in the 4-channel literature metric (a factor of **10x - 58x** lower).

---

## 4. Empirical Benchmark Comparison on Test Partition

Empirical results obtained across trained SMART surrogate checkpoints evaluated on the unobserved test partition:

| Experiment / Configuration | Pure Displacement Rel $L_2$ | Joint 4-Channel Rel $L_2$ (Literature) | Reduction Factor |
| :--- | :---: | :---: | :---: |
| **LION temporal (150 ep, bs 16, swap test - best)** | **3.16%** | **0.06%** | **50.5x** |
| **LION temporal (150 ep, bs 16)** | **6.50%** | **0.13%** | **50.2x** |
| **LION temporal (50 ep, bs 16)** | **7.48%** | **0.13%** | **57.2x** |
| **LION temporal (50 ep, bs 32)** | **8.02%** | **0.14%** | **58.5x** |
| **LION temporal (50 ep, bs 8)** | **11.44%** | **0.21%** | **54.3x** |
| **SMART temporal (300 ep, bs 64)** | **11.45%** | **0.20%** | **56.6x** |
| **LION temporal (25 ep, bs 1)** | **11.93%** | **0.25%** | **47.6x** |
| **LION temporal (50 ep, bs 1, lr 5e-5)** | **24.31%** | **0.57%** | **42.4x** |
| **LION temporal (50 ep, bs 1, lr 1e-4)** | **30.89%** | **0.71%** | **43.5x** |
| **SMART temporal (10 ep, bs 32)** | **83.32%** | **2.27%** | **36.7x** |
| **SMART static final step (200 ep, bs 16)** | **9.11%** | **0.82%** | **11.1x** |
| **SMART static final step (50 ep, bs 16)** | **27.92%** | **2.59%** | **10.8x** |
