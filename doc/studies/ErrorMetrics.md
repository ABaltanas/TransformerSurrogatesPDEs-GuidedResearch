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
