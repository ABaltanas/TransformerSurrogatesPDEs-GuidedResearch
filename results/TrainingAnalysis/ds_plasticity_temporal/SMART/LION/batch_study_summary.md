# Lion Optimizer: Batch Size & Sensitivity Study Summary

**Date:** 2026-11-08 09:57:35 UTC
**Architecture:** SMART (11.38M parameters, latent_dim=256, num_blocks=8, 101 geometry points)
**Optimization:** Lion (warmup_cosine scheduler, 50 epochs, AMP FP16)

| Run Identifier | Batch Size | Learning Rate | Status | Best Val Rel-L2 | Best Epoch | Test Rel-L2 (%) | Contact Violations (%) | Latency (ms) | Wall Time (h) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `smart_temporal_experiment_3` | **1** | `1.0e-04` | `DIVERGED_AT_EP5 (NaN)` | `0.3217` | 2 / 50 | **30.89%** | 0.0% | 10.53 | 24.83 h |
| `batch_size_1_lr_5e-5` | **1** | `5.0e-05` | `DIVERGED (NaN, Best Ep saved)` | `0.2661` | 4 / 50 | **24.31%** | 0.0% | 10.54 | 28.79 h |
| `batch_size_8` | **8** | `1.0e-04` | `DIVERGED (NaN, Best Ep saved)` | `0.0892` | 24 / 50 | **11.44%** | 0.0% | 10.49 | 5.54 h |
| `batch_size_16` | **16** | `1.0e-04` | `COMPLETED` | `0.0320` | 50 / 50 | **7.48%** | 0.0% | 10.53 | 4.59 h |
| `batch_size_32` | **32** | `1.0e-04` | `COMPLETED` | `0.0341` | 50 / 50 | **8.02%** | 0.0% | 10.54 | 2.86 h |
