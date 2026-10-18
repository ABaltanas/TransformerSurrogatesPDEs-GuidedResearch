# SMART Model Architecture, Input Requirements, and Dataset Adaptation Guide

Adapting the **SMART** (*Scalable Mesh Autoregressive Transformer*) architecture to a new dataset requires understanding both the tensor input signatures required during the forward pass and the model hyperparameters that define its physical dimensionality and expressive capacity.

The reference implementation of the core model is located in [`SMART`](file:///home/arturobaltanas/University/GuidedResearch/external/smart/smart/models/smart/smart.py#L436), while its integration within this repository is governed by [`SMARTModelWrapper`](file:///home/arturobaltanas/University/GuidedResearch/src/training/wrappers/smart_wrapper.py#L23) and [`SMARTBatchLoader`](file:///home/arturobaltanas/University/GuidedResearch/src/training/batch_loaders.py#L15).

---

## 1. How SMART Works and Required Tensor Inputs

SMART is a *mesh-free*, query-based encoder-decoder Transformer surrogate:
1. **Encoder:** Accepts a point cloud representing the **input geometry or boundary condition** (`geo`), downsamples it, and projects it into a reduced set of latent points via cross-attention.
2. **Decoder:** Receives arbitrary spatial coordinates where solutions are requested (*query points*) and attends to the latent geometry representations to predict field variables.

### Input Tensors in `forward(geo, surf_query_pos, vol_query_pos, params)`

| Input Tensor | Shape | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **`geo`** | `(B, N_geo, spatial_dim)` | `torch.float32` | Point coordinates defining the **input geometry or physical boundary condition** (e.g., rigid die contour, airfoil surface mesh, or fixed domain boundary). |
| **`surf_query_pos`** | `(B, N_surf, spatial_dim)` | `torch.float32` | Spatial coordinates of query points on the **surface or interface** where solution fields are evaluated. |
| **`vol_query_pos`** | `(B, N_vol, spatial_dim)` | `torch.float32` | Spatial coordinates of query points in the **volume or domain interior** where continuum solution fields are evaluated. |
| **`params`** *(optional)* | `(B, parameter_channels)` | `torch.float32` | Global scalar simulation parameters (e.g., normalized time $t \in [0, 1]$, material parameters such as $E$ or $\nu$, Reynolds number, angle of attack). Pass `None` if no scalar conditioning is used. |

> [!IMPORTANT]
> **Spatial Coordinate Normalization:**
> SMART employs modulated sine-cosine and Rotary Positional Embeddings (RoPE) calibrated for bounded coordinate domains. All spatial point clouds (`geo`, `surf_query_pos`, `vol_query_pos`) must be **normalized to the $[0, 1]$ interval** using the global bounding box of the physical domain:
> $$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x} - \mathbf{x}_{\min}}{\mathbf{x}_{\max} - \mathbf{x}_{\min}}$$

### Output Tensors
* **`pred_surf`**: `(B, N_surf, surface_channels)` — Predicted physical field quantities at the surface query coordinates.
* **`pred_vol`**: `(B, N_vol, volume_channels)` — Predicted physical field quantities at the interior volume query coordinates.

---

## 2. Modifiable Parameters and Hyperparameters

When instantiating [`SMART`](file:///home/arturobaltanas/University/GuidedResearch/external/smart/smart/models/smart/smart.py#L454-L466), the following hyperparameters can be adjusted:

### A. Physical Problem Dimensions
* **`spatial_dim`** *(int, default=3 or 2)*: Spatial dimensionality of the domain. Set to `2` for planar/2D problems or `3` for 3D domains. This directly configures the positional embedding channels.
* **`surface_channels`** *(int)*: Number of physical target variables predicted on the surface (e.g., surface pressure coefficient $C_p = 1$, boundary displacement vector $(u_x, u_y) = 2$, or traction components).
* **`volume_channels`** *(int)*: Number of physical target variables predicted in the interior volume (e.g., displacement vector $(u_x, u_y) = 2$, velocity and pressure field $(u, v, w, p) = 4$, or equivalent plastic strain).
* **`parameter_channels`** *(int, default=0 or 2)*: Dimensionality of global conditioning parameters (`cond_dim`).
  - `0`: If the solution depends exclusively on geometry `geo`.
  - `1`: If conditioning on normalized time $t \in [0, 1]$ for dynamic trajectories.
  - `$k$`: If conditioning on multiple material, operating, or boundary parameters.

### B. Latent Geometry Sampling
* **`latent_geometry_points`** *(int)*: Number of latent geometry query points ($N_{\text{lat}}$) to which the encoder compresses the input geometry.
  - For small or 2D benchmarks: `32` to `128`.
  - For large or 3D benchmarks: `1024` to `4096`.
  - *Constraint:* Must satisfy $N_{\text{lat}} \le N_{\text{geo}}$.
* **`subsampled_geometry_points`** *(int)*: Number of points randomly subsampled from `geo` in each encoder cross-attention block. If $N_{\text{geo}}$ is small (e.g., 101 points), set equal to $N_{\text{geo}}$. For dense 3D point clouds, set to a representative subset (e.g., `16384`).
* **`pos_scale_factor`** *(float, default=1000.0)*: Multiplicative scaling factor applied to normalized $[0, 1]$ coordinates prior to computing trigonometric positional embeddings.

### C. Transformer Expressive Capacity
* **`latent_dim`** *(int, e.g., 128, 256, 512)*: Hidden feature dimension across all attention and MLP layers.
* **`num_heads`** *(int, e.g., 4, 8, 16)*: Number of parallel attention heads. Must evenly divide `latent_dim`.
* **`num_encoder_decoder_blocks`** *(int, e.g., 4, 6, 8)*: Number of sequential encoder and decoder Transformer blocks.
* **`dropout`** *(float, e.g., 0.0 to 0.1)*: Dropout probability applied within attention and feed-forward layers.

---

## 3. Practical Example: Instantiating SMART for a New Dataset

Below is a self-contained demonstration configuring SMART for a 3D steady-state fluid flow problem with 5,000 geometry points, predicting 1 surface scalar field (pressure), 3 volumetric vector fields (velocity components), and conditioning on 1 global physical parameter (Reynolds number):

```python
import torch
from external.smart.smart.models.smart.smart import SMART

# 1. Hyperparameter configuration for the target problem
config = {
    "spatial_dim": 3,
    "surface_channels": 1,
    "volume_channels": 3,
    "parameter_channels": 1,
    "latent_dim": 256,
    "latent_geometry_points": 512,
    "subsampled_geometry_points": 2048,
    "num_encoder_decoder_blocks": 6,
    "num_heads": 8,
    "pos_scale_factor": 1000.0,
    "dropout": 0.05,
}

# 2. Instantiate model and initialize weights
model = SMART(**config)
model.initialize_weights()

# 3. Create synthetic batch with spatial coordinates normalized to [0, 1]
batch_size = 4
geo = torch.rand(batch_size, 5000, 3, dtype=torch.float32)
surf_query_pos = torch.rand(batch_size, 1000, 3, dtype=torch.float32)
vol_query_pos = torch.rand(batch_size, 8000, 3, dtype=torch.float32)
params = torch.rand(batch_size, 1, dtype=torch.float32)

# 4. Forward execution
pred_surf, pred_vol = model(
    geo=geo,
    surf_query_pos=surf_query_pos,
    vol_query_pos=vol_query_pos,
    params=params,
)

print(f"Surface predictions shape: {pred_surf.shape}")  # (4, 1000, 1)
print(f"Volume predictions shape:  {pred_vol.shape}")   # (4, 8000, 3)
```

---

## 4. Components to Implement for a New Dataset

Following the modular architecture in [`src/`](file:///home/arturobaltanas/University/GuidedResearch/src):

1. **Custom `BatchLoader`** (implementing [`BatchLoader`](file:///home/arturobaltanas/University/GuidedResearch/src/training/interfaces.py#L15)):
   - Loads raw samples from partition files.
   - Partitions spatial coordinates into `geo`, `surf_query_pos`, and `vol_query_pos`.
   - Normalizes spatial coordinates to $[0, 1]$ via the domain bounding box.
   - Standardizes target values using dataset statistics (zero mean, unit variance).
2. **Custom or Adapted `ModelWrapper`** (implementing [`ModelWrapper`](file:///home/arturobaltanas/University/GuidedResearch/src/training/interfaces.py#L35)):
   - Encapsulates the SMART instance with dataset-specific channel counts and spatial dimensions.
   - Unpacks incoming batch dictionaries and handles tensor reshaping or domain reconstruction (e.g., [`_reconstruct_displacement_field`](file:///home/arturobaltanas/University/GuidedResearch/src/training/wrappers/smart_wrapper.py#L120)).
3. **Loss Engine (`LossEngine`)**:
   - Choose or extend [`RelativeL2LossEngine`](file:///home/arturobaltanas/University/GuidedResearch/src/training/losses.py#L10) (relative Frobenius error) or [`MSELossEngine`](file:///home/arturobaltanas/University/GuidedResearch/src/training/losses.py#L85) according to benchmarking standards.
