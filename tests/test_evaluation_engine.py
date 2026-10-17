"""Unit and integration tests for the EvaluationEngine subsystem."""

import json
from pathlib import Path
import shutil
import tempfile
from typing import Dict
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.evaluation.denormalizer import StatisticalDenormalizer
from src.evaluation.engine import EvaluationEngine
from src.evaluation.exporters import CompositeReportExporter
from src.evaluation.metrics import PlasticityMetricsCalculator
from src.evaluation.profiler import HardwareProfiler
from src.evaluation.runner import BatchedInferenceRunner
from src.evaluation.validators import DiePenetrationValidator
from src.training.batch_loaders import SMARTBatchLoader
from src.training.wrappers.smart_wrapper import SMARTModelWrapper


class MockEvalPlasticityDataset(Dataset):
    """Generates synthetic test partition data for evaluation testing."""

    def __init__(self, num_samples: int = 4) -> None:
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # Die contour: (101, 2)
        die_x = np.linspace(-49.65, 0.35, 101, dtype=np.float32)
        die_y = 20.0 + 2.0 * np.sin(die_x / 5.0).astype(np.float32)
        die = np.stack([die_x, die_y], axis=-1)

        # Initial coordinates: (101, 31, 2)
        x_grid = np.linspace(-49.65, 0.35, 101, dtype=np.float32)
        y_grid = np.linspace(0.0, 14.9, 31, dtype=np.float32)
        xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
        coords_initial = np.stack([xx, yy], axis=-1)

        # Displacements in mm: (101, 31, 2)
        ux = 0.2 * np.sin(xx).astype(np.float32)
        uy = -0.5 * np.cos(yy).astype(np.float32)
        displacements = np.stack([ux, uy], axis=-1)

        return {
            "die_boundary": die,
            "coords_initial": coords_initial,
            "coords": coords_initial + displacements,
            "displacements": displacements,
        }


def test_batched_inference_runner() -> None:
    """Test BatchedInferenceRunner on mock dataset with ModelWrapper."""
    dataset = MockEvalPlasticityDataset(num_samples=4)
    loader = SMARTBatchLoader()
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=loader.loadDataset)

    config = {
        "spatial_dim": 2,
        "surface_channels": 2,
        "volume_channels": 2,
        "latent_dim": 32,
        "latent_geometry_points": 16,
        "subsampled_geometry_points": 32,
        "num_encoder_decoder_blocks": 1,
        "num_heads": 2,
    }
    wrapper = SMARTModelWrapper(config=config)

    runner = BatchedInferenceRunner()
    preds = runner.predict(wrapper, dataloader, device=torch.device("cpu"))

    assert "displacements" in preds
    assert preds["displacements"].shape == (4, 101, 31, 2)
    assert "surf_pred" in preds
    assert preds["surf_pred"].shape == (4, 101, 2)
    assert "vol_pred" in preds
    assert preds["vol_pred"].shape == (4, 3030, 2)


def test_statistical_denormalizer() -> None:
    """Test exact mathematical inverse scaling to physical millimeters."""
    denormalizer = StatisticalDenormalizer()

    stats = {
        "displacements": {
            "per_channel": {
                "mean": [0.5, -2.0],
                "std": [1.5, 3.0],
            }
        }
    }

    # Normalized input: zeros should become mean, ones should become mean + std
    pred_norm = {"displacements": np.zeros((2, 10, 10, 2), dtype=np.float32)}
    target_norm = {
        "displacements": np.ones((2, 10, 10, 2), dtype=np.float32),
        "coords_initial": np.zeros((2, 10, 10, 2), dtype=np.float32),
    }

    phys_preds, phys_targets = denormalizer.denormalize(pred_norm, target_norm, stats)

    # Check that zeros scaled to mean
    assert np.allclose(phys_preds["displacements"][:, :, :, 0], 0.5)
    assert np.allclose(phys_preds["displacements"][:, :, :, 1], -2.0)

    # Check coords = x0 + u
    assert np.allclose(phys_preds["coords"], phys_preds["displacements"])


def test_plasticity_metrics_calculator() -> None:
    """Test Relative L2, L_infinity, surface vs volume, and Sobolev H1 semi-norm."""
    calc = PlasticityMetricsCalculator()
    shape = (2, 101, 31, 2)

    # Identical predictions -> 0 error
    true_u = np.ones(shape, dtype=np.float64)
    pred_u = np.ones(shape, dtype=np.float64)

    metrics = calc.compute_metrics(
        pred_phys={"displacements": pred_u},
        target_phys={"displacements": true_u},
        initial_mesh=np.zeros(shape, dtype=np.float64),
    )

    assert metrics["rel_l2_overall"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["l_inf_max_mm"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["surface_rel_l2"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["volume_rel_l2"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["sobolev_h1_semi_norm"] == pytest.approx(0.0, abs=1e-6)

    # Shifted prediction
    pred_u_shifted = true_u + 0.1
    metrics_shifted = calc.compute_metrics(
        pred_phys={"displacements": pred_u_shifted},
        target_phys={"displacements": true_u},
        initial_mesh=np.zeros(shape, dtype=np.float64),
    )
    assert metrics_shifted["l_inf_max_mm"] == pytest.approx(0.1, abs=1e-6)
    assert metrics_shifted["rel_l2_overall"] > 0.0


def test_die_penetration_validator() -> None:
    """Test DiePenetrationValidator detects contact violations."""
    validator = DiePenetrationValidator(tolerance_mm=0.05)
    n, nx, ny = 2, 101, 31

    coords = np.zeros((n, nx, ny, 2), dtype=np.float64)
    die = np.zeros((n, nx, 2), dtype=np.float64)

    # Scenario 1: Workpiece top surface at y = 15.0, Die at y = 20.0 -> No penetration
    coords[:, :, -1, 1] = 15.0
    die[:, :, 1] = 20.0
    res_pass = validator.check_constraints(coords, die)
    assert res_pass["max_die_penetration_mm"] == 0.0
    assert res_pass["contact_feasibility_pass"] == 1.0
    assert res_pass["penetration_violation_rate_pct"] == 0.0

    # Scenario 2: Workpiece surface at y = 22.0, Die at y = 20.0 -> 2.0mm penetration
    coords[:, :, -1, 1] = 22.0
    res_fail = validator.check_constraints(coords, die)
    assert res_fail["max_die_penetration_mm"] == pytest.approx(2.0, abs=1e-5)
    assert res_fail["contact_feasibility_pass"] == 0.0
    assert res_fail["penetration_violation_rate_pct"] == 100.0


def test_hardware_profiler() -> None:
    """Test HardwareProfiler measuring latency, parameter counts, and speedup."""
    profiler = HardwareProfiler(warmup_runs=2, timing_runs=3)

    model = nn.Linear(10, 5)
    sample_batch = {"inputs": torch.randn(2, 10)}

    res = profiler.profile_inference(
        model=model,
        sample_batch=sample_batch,
        device=torch.device("cpu"),
        fem_reference_time_s=120.0,
    )

    assert res["inference_latency_ms_per_sample"] > 0.0
    assert res["total_parameters"] == (10 * 5 + 5)
    assert res["surrogate_speedup_vs_fem"] > 100.0


def test_evaluation_engine_end_to_end() -> None:
    """Test full EvaluationEngine lifecycle and report artifact export."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        dataset = MockEvalPlasticityDataset(num_samples=4)
        loader = SMARTBatchLoader()
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=loader.loadDataset)

        config = {
            "spatial_dim": 2,
            "surface_channels": 2,
            "volume_channels": 2,
            "latent_dim": 32,
            "latent_geometry_points": 16,
            "subsampled_geometry_points": 32,
            "num_encoder_decoder_blocks": 1,
            "num_heads": 2,
        }
        wrapper = SMARTModelWrapper(config=config)

        # Mock stats file
        stats_file = temp_dir / "statistical_data.json"
        stats_data = {
            "displacements": {
                "per_channel": {
                    "mean": [0.0, 0.0],
                    "std": [1.0, 1.0],
                }
            }
        }
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_data, f)

        # Initialize engine
        engine = EvaluationEngine(
            runner=BatchedInferenceRunner(),
            denormalizer=StatisticalDenormalizer(),
            metrics_calc=PlasticityMetricsCalculator(),
            validator=DiePenetrationValidator(),
            profiler=HardwareProfiler(warmup_runs=1, timing_runs=2),
            exporter=CompositeReportExporter(),
        )

        results = engine.evaluate(
            model=wrapper,
            checkpoint_path=None,
            test_loader=dataloader,
            stats_path=stats_file,
            output_dir=temp_dir / "report",
            device=torch.device("cpu"),
            fem_reference_time_s=120.0,
        )

        assert "physical_metrics" in results
        assert "feasibility_metrics" in results
        assert "computational_metrics" in results
        assert "exported_artifacts" in results

        # Verify exported artifacts
        assert (temp_dir / "report" / "evaluation_summary.md").exists()
        assert (temp_dir / "report" / "evaluation_metrics.json").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_training_telemetry_analyzer() -> None:
    """Test TrainingTelemetryAnalyzer parsing, KPI calculation, CSV export, and plot generation."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        from src.evaluation.training_history import TrainingTelemetryAnalyzer

        metrics_file = temp_dir / "training_metrics.jsonl"
        records = [
            {
                "epoch": 1,
                "total_epochs": 3,
                "timestamp": 100.0,
                "train": {"loss_surf": 0.5, "loss_vol": 0.5, "loss_total": 1.0},
                "val": {"loss_surf": 0.4, "loss_vol": 0.4, "loss_total": 0.8},
                "lr": 1e-4,
            },
            {
                "epoch": 2,
                "total_epochs": 3,
                "timestamp": 110.0,
                "train": {"loss_surf": 0.2, "loss_vol": 0.3, "loss_total": 0.5},
                "val": {"loss_surf": 0.15, "loss_vol": 0.2, "loss_total": 0.35},
                "lr": 5e-5,
            },
            {
                "epoch": 3,
                "total_epochs": 3,
                "timestamp": 120.0,
                "train": {"loss_surf": 0.1, "loss_vol": 0.2, "loss_total": 0.3},
                "val": {"loss_surf": 0.1, "loss_vol": 0.15, "loss_total": 0.25},
                "lr": 1e-5,
            },
        ]
        with open(metrics_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        analyzer = TrainingTelemetryAnalyzer()
        res = analyzer.analyze_and_export(metrics_file=metrics_file, output_dir=temp_dir / "out")

        kpis = res["training_kpis"]
        assert kpis["total_epochs"] == 3
        assert kpis["best_epoch"] == 3
        assert kpis["best_val_loss"] == pytest.approx(0.25, abs=1e-5)
        assert kpis["initial_val_loss"] == pytest.approx(0.8, abs=1e-5)
        assert kpis["val_loss_reduction_pct"] == pytest.approx((1.0 - 0.25 / 0.8) * 100.0, abs=1e-3)
        assert kpis["best_val_loss_surf"] == pytest.approx(0.1, abs=1e-5)
        assert kpis["best_val_loss_vol"] == pytest.approx(0.15, abs=1e-5)

        # Artifacts verification
        artifacts = res["artifacts"]
        assert (temp_dir / "out" / "training_history.csv").exists()
        assert (temp_dir / "out" / "training_metrics.jsonl").exists()
        assert (temp_dir / "out" / "training_curves.png").exists()
        assert (temp_dir / "out" / "training_curves.png").stat().st_size > 1000

        # Check CSV content
        with open(artifacts["training_history_csv"], "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 4  # header + 3 rows
            assert "train_loss_total" in lines[0]
            assert "val_loss_surf" in lines[0]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_evaluation_engine_with_training_telemetry() -> None:
    """Test full EvaluationEngine end-to-end integration with training telemetry."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        dataset = MockEvalPlasticityDataset(num_samples=2)
        loader = SMARTBatchLoader()
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=loader.loadDataset)

        config = {
            "spatial_dim": 2,
            "surface_channels": 2,
            "volume_channels": 2,
            "latent_dim": 16,
            "latent_geometry_points": 8,
            "subsampled_geometry_points": 16,
            "num_encoder_decoder_blocks": 1,
            "num_heads": 2,
        }
        wrapper = SMARTModelWrapper(config=config)

        stats_file = temp_dir / "statistical_data.json"
        stats_data = {
            "displacements": {
                "per_channel": {
                    "mean": [0.0, 0.0],
                    "std": [1.0, 1.0],
                }
            }
        }
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_data, f)

        # Create training metrics
        metrics_file = temp_dir / "training_metrics.jsonl"
        with open(metrics_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": 1, "total_epochs": 1, "train": {"loss_total": 0.5, "loss_surf": 0.2, "loss_vol": 0.3}, "val": {"loss_total": 0.4, "loss_surf": 0.15, "loss_vol": 0.25}, "lr": 1e-4}) + "\n")

        engine = EvaluationEngine(
            runner=BatchedInferenceRunner(),
            denormalizer=StatisticalDenormalizer(),
            metrics_calc=PlasticityMetricsCalculator(),
            validator=DiePenetrationValidator(),
            profiler=HardwareProfiler(warmup_runs=1, timing_runs=1),
            exporter=CompositeReportExporter(),
        )

        output_dir = temp_dir / "eval_out"
        summary = engine.evaluate(
            model=wrapper,
            checkpoint_path=None,
            test_loader=dataloader,
            stats_path=stats_file,
            output_dir=output_dir,
            device=torch.device("cpu"),
            training_metrics_path=metrics_file,
        )

        assert "training_kpis" in summary
        assert summary["training_kpis"]["total_epochs"] == 1
        assert "training_curves_plot" in summary["exported_artifacts"]
        assert "training_history_csv" in summary["exported_artifacts"]

        # Check markdown report includes Section 4
        md_path = output_dir / "evaluation_summary.md"
        assert md_path.exists()
        md_text = md_path.read_text(encoding="utf-8")
        assert "### 4. Training Convergence & Optimization KPIs" in md_text
        assert "training_curves.png" in md_text
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

