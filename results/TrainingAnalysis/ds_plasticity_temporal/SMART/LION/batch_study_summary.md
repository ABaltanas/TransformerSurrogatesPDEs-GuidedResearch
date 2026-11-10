# Lion Optimizer: Batch Size & Sensitivity Study Summary

**Date:** 2026-11-10 05:16:37 UTC
**Architecture:** SMART (11.38M parameters, latent_dim=256, num_blocks=8, 101 geometry points)
**Optimization:** Lion (warmup_cosine scheduler, 50 epochs, AMP FP16)

| Run Identifier | Batch Size | Learning Rate | Status | Best Val Rel-L2 | Best Epoch | Test Rel-L2 (%) | Contact Violations (%) | Latency (ms) | Wall Time (h) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `smart_temporal_experiment_3` | **1** | `1.0e-04` | `DIVERGED_AT_EP5 (NaN)` | `0.3217` | 2 / 50 | **30.89%** | 0.0% | 10.53 | 24.83 h |
| `batch_size_1_fp32_ep50` | **1** | `1.0e-04` | `NOT_STARTED` | `N/A` | N/A | **N/A** | N/A | N/A | N/A |
| `batch_size_1_lr_5e-5` | **1** | `5.0e-05` | `DIVERGED (NaN, Best Ep saved)` | `0.2661` | 4 / 50 | **24.31%** | 0.0% | 10.54 | 28.79 h |
| `batch_size_8` | **8** | `1.0e-04` | `DIVERGED (NaN, Best Ep saved)` | `0.0892` | 24 / 50 | **11.44%** | 0.0% | 10.49 | 5.54 h |
| `batch_size_16_fp32_swap_test` | **16** | `1.0e-04` | `COMPLETED` | `0.0139` | 149 / 150 | **3.16%** | 0.0% | 10.49 | 13.76 h |
| `batch_size_16` | **16** | `1.0e-04` | `COMPLETED` | `0.0320` | 50 / 50 | **7.48%** | 0.0% | 10.53 | 4.59 h |
| `batch_size_16_epochs_150` | **16** | `1.0e-04` | `DIVERGED (NaN, Best Ep saved)` | `0.0484` | 72 / 150 | **6.50%** | 0.0% | 10.49 | 16.87 h |
| `batch_size_32` | **32** | `1.0e-04` | `COMPLETED` | `0.0341` | 50 / 50 | **8.02%** | 0.0% | 10.54 | 2.86 h |
