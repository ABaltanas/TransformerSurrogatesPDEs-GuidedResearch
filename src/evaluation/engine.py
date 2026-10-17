"""Orchestrator for surrogate physical evaluation, feasibility checks, and benchmarking."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.interfaces import (
    BenchmarkProfiler,
    Denormalizer,
    FeasibilityValidator,
    InferenceRunner,
    PhysicalMetricsCalculator,
    ReportExporter,
    TrainingHistoryAnalyzer,
)
from src.evaluation.training_history import TrainingTelemetryAnalyzer
from src.training.interfaces import ModelWrapper


class EvaluationEngine:
    """Coordinates post-training physical evaluation, validation, and benchmarking.

    Parameters
    ----------
    runner : InferenceRunner
        Batch prediction executor.
    denormalizer : Denormalizer
        Statistical rescaling strategy.
    metrics_calc : PhysicalMetricsCalculator
        Scientific PDE error metrics calculator.
    validator : FeasibilityValidator
        Physical boundary condition constraint validator.
    profiler : BenchmarkProfiler
        Inference latency, memory, and throughput profiler.
    exporter : ReportExporter
        Report and artifact generation strategy.
    training_analyzer : Optional[TrainingHistoryAnalyzer], default=None
        Training telemetry parser, convergence KPI calculator, and plotter.
    """

    def __init__(
        self,
        runner: InferenceRunner,
        denormalizer: Denormalizer,
        metrics_calc: PhysicalMetricsCalculator,
        validator: FeasibilityValidator,
        profiler: BenchmarkProfiler,
        exporter: ReportExporter,
        training_analyzer: Optional[TrainingHistoryAnalyzer] = None,
    ) -> None:
        self.runner = runner
        self.denormalizer = denormalizer
        self.metrics_calc = metrics_calc
        self.validator = validator
        self.profiler = profiler
        self.exporter = exporter
        self.training_analyzer = training_analyzer or TrainingTelemetryAnalyzer()

    def evaluate(
        self,
        model: Any,
        checkpoint_path: Optional[Path],
        test_loader: DataLoader,
        stats_path: Path,
        output_dir: Optional[Path] = None,
        device: torch.device = torch.device("cpu"),
        test_npz_path: Optional[Path] = None,
        fem_reference_time_s: float = 120.0,
        training_metrics_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute complete scientific benchmark evaluation.

        Parameters
        ----------
        model : Any
            The neural operator module (nn.Module) or ModelWrapper instance.
        checkpoint_path : Optional[Path]
            Path to the saved checkpoint file to load. If None, uses existing weights.
        test_loader : DataLoader
            DataLoader wrapping the test partition.
        stats_path : Path
            Path to statistical_data.json.
        output_dir : Optional[Path], default=None
            Directory where evaluation reports and plots will be written (default: results/TrainingAnalysis).
        device : torch.device, default=cpu
            Evaluation hardware device.
        test_npz_path : Optional[Path], default=None
            Direct path to test.npz for ground truth extraction if available.
        fem_reference_time_s : float, default=120.0
            Abaqus FEM simulation time per sample for speedup calculation.
        training_metrics_path : Optional[Path], default=None
            Direct path to training_metrics.jsonl log file. If None, checks checkpoint directory.

        Returns
        -------
        Dict[str, Any]
            Comprehensive evaluation results dictionary.
        """
        resolved_output_dir = (
            Path(output_dir) if output_dir is not None else Path("results") / "TrainingAnalysis"
        )
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = resolved_output_dir


        # 1. Load trained checkpoint weights if path provided
        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            torch_module = model.get_torch_module() if isinstance(model, ModelWrapper) else model
            torch_module.load_state_dict(checkpoint["model_state_dict"])

        # 2. Execute deterministic inference on the test partition
        raw_predictions = self.runner.predict(model, test_loader, device)

        # 3. Ingest normalization statistics
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        # 4. Extract ground truth arrays
        targets_dict, initial_mesh, die_contours = self._extract_ground_truth(
            test_loader=test_loader,
            test_npz_path=test_npz_path,
        )

        # 5. Denormalize predictions and ground-truth targets into physical millimeters
        phys_preds, phys_targets = self.denormalizer.denormalize(
            pred_norm=raw_predictions,
            target_norm=targets_dict,
            stats=stats,
        )

        # 6. Compute scientific PDE error metrics (Rel L2, H1, L_inf)
        error_metrics = self.metrics_calc.compute_metrics(
            pred_phys=phys_preds,
            target_phys=phys_targets,
            initial_mesh=initial_mesh,
        )

        # 7. Verify physical mechanics constraints (die non-penetration)
        feasibility_metrics = self.validator.check_constraints(
            pred_coords=phys_preds["coords"],
            boundary_constraints=die_contours,
        )

        # 8. Profile computational performance (latency, VRAM, speedup)
        sample_batch = next(iter(test_loader))
        hardware_metrics = self.profiler.profile_inference(
            model=model,
            sample_batch=sample_batch,
            device=device,
            fem_reference_time_s=fem_reference_time_s,
        )

        # 9. Ingest and analyze training telemetry and convergence KPIs (if available)
        target_metrics_file: Optional[Path] = None
        if training_metrics_path is not None and Path(training_metrics_path).exists():
            target_metrics_file = Path(training_metrics_path)
        elif checkpoint_path is not None:
            candidate = Path(checkpoint_path).parent / "training_metrics.jsonl"
            if candidate.exists():
                target_metrics_file = candidate

        training_data: Optional[Dict[str, Any]] = None
        if target_metrics_file is not None and self.training_analyzer is not None:
            try:
                training_data = self.training_analyzer.analyze_and_export(
                    metrics_file=target_metrics_file,
                    output_dir=output_dir,
                )
            except Exception as e:
                print(f"[Warning] Failed to analyze training telemetry: {e}")

        # 10. Aggregate complete benchmark record
        evaluation_summary: Dict[str, Any] = {
            "checkpoint": str(checkpoint_path) if checkpoint_path else "in_memory",
            "physical_metrics": error_metrics,
            "feasibility_metrics": feasibility_metrics,
            "computational_metrics": hardware_metrics,
        }

        if training_data is not None:
            evaluation_summary["training_kpis"] = training_data["training_kpis"]
            evaluation_summary["training_artifacts"] = {
                k: str(v) for k, v in training_data["artifacts"].items()
            }

        # 11. Export publication markdown report and JSON record
        saved_artifacts = self.exporter.export(evaluation_summary, output_dir)
        evaluation_summary["exported_artifacts"] = {k: str(v) for k, v in saved_artifacts.items()}
        if training_data is not None:
            for k, v in training_data["artifacts"].items():
                evaluation_summary["exported_artifacts"][k] = str(v)

        return evaluation_summary


    def _extract_ground_truth(
        self,
        test_loader: DataLoader,
        test_npz_path: Optional[Path],
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Extract unnormalized reference meshes and targets from file or DataLoader dataset."""
        if test_npz_path is not None and Path(test_npz_path).exists():
            data = np.load(test_npz_path)
            targets_dict = {
                "displacements": data["displacements"],
                "coords_initial": data["coords_initial"],
                "coords": data["coords"] if "coords" in data else data["coords_initial"] + data["displacements"],
            }
            initial_mesh = data["coords_initial"]
            die_contours = data["die_boundary"]
            return targets_dict, initial_mesh, die_contours

        # Fallback: inspect dataset methods or iterate
        dataset = test_loader.dataset
        if hasattr(dataset, "get_all_targets") and hasattr(dataset, "get_initial_mesh"):
            return dataset.get_all_targets(), dataset.get_initial_mesh(), dataset.get_die_boundaries()

        # Extract by iterating over DataLoader batches
        all_x0 = []
        all_u = []
        all_die = []
        for batch in test_loader:
            if "coords_initial" in batch:
                x0_tensor = batch["coords_initial"]
                all_x0.append(x0_tensor.cpu().numpy() if isinstance(x0_tensor, torch.Tensor) else x0_tensor)
            if "die_boundary_raw" in batch:
                die_tensor = batch["die_boundary_raw"]
                all_die.append(die_tensor.cpu().numpy() if isinstance(die_tensor, torch.Tensor) else die_tensor)
            elif "die_boundary" in batch:
                die_tensor = batch["die_boundary"]
                all_die.append(die_tensor.cpu().numpy() if isinstance(die_tensor, torch.Tensor) else die_tensor)
            if "targets" in batch and "displacements" in batch["targets"]:
                u_tensor = batch["targets"]["displacements"]
                all_u.append(u_tensor.cpu().numpy() if isinstance(u_tensor, torch.Tensor) else u_tensor)

        x0_arr = np.concatenate(all_x0, axis=0) if all_x0 else np.zeros((1, 101, 31, 2))
        u_arr = np.concatenate(all_u, axis=0) if all_u else np.zeros((1, 101, 31, 2))
        die_arr = np.concatenate(all_die, axis=0) if all_die else np.zeros((1, 101, 2))

        targets = {
            "displacements": u_arr,
            "coords_initial": x0_arr,
            "coords": x0_arr + u_arr,
        }
        return targets, x0_arr, die_arr
