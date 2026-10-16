"""Computational performance, latency, memory, and speedup profiling."""

import time
from typing import Any, Dict
import torch
import torch.nn as nn

from src.evaluation.interfaces import BenchmarkProfiler
from src.training.interfaces import ModelWrapper


class HardwareProfiler(BenchmarkProfiler):
    """Profiles inference latency, GPU memory footprint, and numerical acceleration.

    Parameters
    ----------
    warmup_runs : int, default=5
        Number of warm-up iterations prior to timing.
    timing_runs : int, default=20
        Number of iterations averaged for timing estimation.
    """

    def __init__(self, warmup_runs: int = 5, timing_runs: int = 20) -> None:
        self.warmup_runs = warmup_runs
        self.timing_runs = timing_runs

    def profile_inference(
        self,
        model: Any,
        sample_batch: Dict[str, Any],
        device: torch.device,
        fem_reference_time_s: float = 120.0,
    ) -> Dict[str, float]:
        """Measure inference runtime, memory usage, and speedup over FEM.

        Parameters
        ----------
        model : Any
            The neural operator module (nn.Module) or ModelWrapper instance.
        sample_batch : Dict[str, Any]
            Sample batch for forward execution.
        device : torch.device
            Target execution device.
        fem_reference_time_s : float, default=120.0
            Reference numerical solver time per sample (default: 120s in ABAQUS).

        Returns
        -------
        Dict[str, float]
            Dictionary of profiled metrics.
        """
        if isinstance(model, ModelWrapper):
            torch_module = model.get_torch_module()
        elif isinstance(model, nn.Module):
            torch_module = model
        else:
            raise TypeError(f"Expected nn.Module or ModelWrapper, got {type(model)}")

        torch_module.to(device)
        torch_module.eval()

        batch = self._transfer_to_device(sample_batch, device)

        # Batch size
        batch_size = 1
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    batch_size = v.shape[0]
                    break

        # 1. Warm-up runs
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                if isinstance(model, ModelWrapper):
                    _ = model.forward(batch)
                elif isinstance(batch, dict) and "inputs" in batch:
                    _ = torch_module(batch["inputs"])
                else:
                    _ = torch_module(batch)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        # 2. Timing benchmark
        total_time_s = 0.0
        with torch.no_grad():
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                for _ in range(self.timing_runs):
                    if isinstance(model, ModelWrapper):
                        _ = model.forward(batch)
                    elif isinstance(batch, dict) and "inputs" in batch:
                        _ = torch_module(batch["inputs"])
                    else:
                        _ = torch_module(batch)
                end_event.record()
                torch.cuda.synchronize(device)
                total_time_s = (start_event.elapsed_time(end_event) / 1000.0)
            else:
                t0 = time.perf_counter()
                for _ in range(self.timing_runs):
                    if isinstance(model, ModelWrapper):
                        _ = model.forward(batch)
                    elif isinstance(batch, dict) and "inputs" in batch:
                        _ = torch_module(batch["inputs"])
                    else:
                        _ = torch_module(batch)
                t1 = time.perf_counter()
                total_time_s = t1 - t0

        avg_latency_per_sample_s = (total_time_s / (self.timing_runs * batch_size))
        avg_latency_ms = avg_latency_per_sample_s * 1000.0

        # 3. Peak VRAM
        peak_vram_mb = 0.0
        if device.type == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        # 4. Parameter counts
        total_params = sum(p.numel() for p in torch_module.parameters())
        trainable_params = sum(p.numel() for p in torch_module.parameters() if p.requires_grad)

        # 5. Speedup factor vs ABAQUS FEM
        speedup = fem_reference_time_s / max(1e-6, avg_latency_per_sample_s)

        return {
            "inference_latency_ms_per_sample": float(avg_latency_ms),
            "peak_vram_mb": float(peak_vram_mb),
            "total_parameters": float(total_params),
            "trainable_parameters": float(trainable_params),
            "surrogate_speedup_vs_fem": float(speedup),
        }

    def _transfer_to_device(self, batch: Any, device: torch.device) -> Any:
        """Recursively transfer batch tensors to target device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: self._transfer_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._transfer_to_device(v, device) for v in batch]
        return batch
