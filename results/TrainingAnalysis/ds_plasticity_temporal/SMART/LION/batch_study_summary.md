# Lion Optimizer: Batch Size & Sensitivity Study Summary

**Date:** 2026-11-06 17:51:30 UTC
**Architecture:** SMART (11.38M parameters, latent_dim=256, num_blocks=8, 101 geometry points)
**Optimization:** Lion (warmup_cosine scheduler, 50 epochs, AMP FP16)

| Run Identifier | Batch Size | Learning Rate | Status | Best Val Rel-L2 | Best Epoch | Test Rel-L2 (%) | Contact Violations (%) | Latency (ms) | Wall Time (h) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `smart_temporal_experiment_3` | **1** | `1.0e-04` | `DIVERGED_AT_EP5 (NaN)` | `0.3217` | 2 / 50 | **30.89%** | 0.0% | 10.53 | 24.83 h |
| `batch_size_32` | **32** | `1.0e-04` | `IN_PROGRESS` | `N/A` | N/A | **N/A** | N/A | N/A | N/A |
