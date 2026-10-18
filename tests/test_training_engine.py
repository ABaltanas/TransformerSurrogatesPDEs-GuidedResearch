"""Unit and integration tests for the TrainingEngine subsystem."""

import shutil
import tempfile
from pathlib import Path
from typing import Dict
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.training.batch_loaders import SMARTBatchLoader, StandardBatchLoader
from src.training.checkpoints import TorchCheckpointHandler
from src.training.engine import TrainingEngine
from src.training.loggers import CliProgressMonitor, JsonMetricsLogger
from src.training.losses import (
    CompositePhysicsLossEngine,
    MSELossEngine,
    RelativeL2LossEngine,
)
from src.training.optimizers import StandardOptimizer
from src.training.wrappers.smart_wrapper import SMARTModelWrapper


class DummyPlasticityDataset(Dataset):
    """Generates synthetic 2D plasticity simulation samples for unit testing."""

    def __init__(self, num_samples: int = 4) -> None:
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        # Die contour: (101, 2)
        die_x = np.linspace(-49.65, 0.35, 101, dtype=np.float32)
        die_y = 20.0 + 5.0 * np.sin(die_x / 10.0).astype(np.float32)
        die = np.stack([die_x, die_y], axis=-1)

        # Initial coords: (101, 31, 2)
        x_grid = np.linspace(-49.65, 0.35, 101, dtype=np.float32)
        y_grid = np.linspace(0.0, 14.9, 31, dtype=np.float32)
        xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
        coords_initial = np.stack([xx, yy], axis=-1)

        # Displacements: (101, 31, 2)
        ux = 0.5 * np.sin(xx).astype(np.float32)
        uy = -1.0 * np.cos(yy).astype(np.float32)
        displacements = np.stack([ux, uy], axis=-1)

        return {
            "die_boundary": die,
            "coords_initial": coords_initial,
            "displacements": displacements,
        }


def test_smart_batch_loader() -> None:
    """Test SMARTBatchLoader collating and normalizing samples."""
    dataset = DummyPlasticityDataset(num_samples=2)
    loader = SMARTBatchLoader()
    batch = loader.loadDataset([dataset[0], dataset[1]])

    assert "geo" in batch
    assert "surf_query_pos" in batch
    assert "vol_query_pos" in batch
    assert "targets" in batch

    # Check tensor shapes
    assert batch["geo"].shape == (2, 101, 2)
    assert batch["surf_query_pos"].shape == (2, 101, 2)
    assert batch["vol_query_pos"].shape == (2, 3030, 2)
    assert batch["targets"]["surf_data"].shape == (2, 101, 2)
    assert batch["targets"]["vol_data"].shape == (2, 3030, 2)
    assert batch["targets"]["displacements"].shape == (2, 101, 31, 2)

    # Check normalization to [0, 1] range
    assert batch["geo"].min() >= -0.1
    assert batch["geo"].max() <= 1.1


def test_smart_batch_loader_with_time_params() -> None:
    """Test SMARTBatchLoader correctly extracts and tensors time_params."""
    dataset = DummyPlasticityDataset(num_samples=2)
    sample_0 = dict(dataset[0])
    sample_0["time_params"] = np.array([0.5], dtype=np.float32)
    sample_1 = dict(dataset[1])
    sample_1["time_params"] = np.array([1.0], dtype=np.float32)

    loader = SMARTBatchLoader()
    batch = loader.loadDataset([sample_0, sample_1])

    assert "params" in batch
    assert "time_params" in batch
    assert batch["params"].shape == (2, 1)
    assert batch["params"][0, 0].item() == pytest.approx(0.5)
    assert batch["params"][1, 0].item() == pytest.approx(1.0)


def test_standard_batch_loader() -> None:
    """Test StandardBatchLoader preserves full grid structure."""
    dataset = DummyPlasticityDataset(num_samples=2)
    loader = StandardBatchLoader()
    batch = loader.loadDataset([dataset[0], dataset[1]])

    assert batch["coords_initial"].shape == (2, 101, 31, 2)
    assert batch["die_boundary"].shape == (2, 101, 2)
    assert batch["targets"]["displacements"].shape == (2, 101, 31, 2)


def test_relative_l2_loss_engine() -> None:
    """Test RelativeL2LossEngine computes correct relative errors."""
    engine = RelativeL2LossEngine()
    pred_u = torch.ones((2, 101, 31, 2), dtype=torch.float32) * 2.0
    true_u = torch.ones((2, 101, 31, 2), dtype=torch.float32)

    predictions = {"displacements": pred_u}
    targets = {"displacements": true_u}
    batch = {}

    loss, components = engine.compute(predictions, targets, batch)
    assert loss.item() == pytest.approx(1.0, rel=1e-3)
    assert "loss_total" in components


def test_composite_physics_loss_engine() -> None:
    """Test CompositePhysicsLossEngine penalizes penetration violations."""
    engine = CompositePhysicsLossEngine(lambda_contact=1.0)
    b, nx, ny = 2, 101, 31

    coords_x0 = torch.zeros((b, nx, ny, 2), dtype=torch.float32)
    coords_x0[:, :, -1, 1] = 15.0  # Initial surface height 15 mm

    # Deformed displacement pushing surface to y = 25 mm
    pred_u = torch.zeros((b, nx, ny, 2), dtype=torch.float32)
    pred_u[:, :, -1, 1] = 10.0     # 15 + 10 = 25 mm

    true_u = torch.zeros((b, nx, ny, 2), dtype=torch.float32)

    # Die is at y = 20 mm -> penetration of 5 mm
    die_raw = torch.zeros((b, nx, 2), dtype=torch.float32)
    die_raw[:, :, 1] = 20.0

    batch = {
        "coords_initial": coords_x0,
        "die_boundary_raw": die_raw,
    }

    loss, components = engine.compute({"displacements": pred_u}, {"displacements": true_u}, batch)
    assert components["loss_contact"] > 0.0
    assert loss.item() > 0.0


def test_optimizer_and_checkpoint_handler() -> None:
    """Test StandardOptimizer and TorchCheckpointHandler state management."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        model = nn.Linear(10, 2)
        optimizer = StandardOptimizer.create_adamw(model.parameters(), lr=1e-3, epochs=5)
        checkpointer = TorchCheckpointHandler(checkpoint_dir=temp_dir, keep_last_n=2)

        assert optimizer.get_learning_rate() == pytest.approx(1e-3)

        # Test optimizer step
        x = torch.randn(4, 10)
        out = model(x)
        loss = out.sum()
        grad_norm = optimizer.step(loss)
        assert grad_norm >= 0.0

        # Test save checkpoint
        ckpt_path = checkpointer.save(epoch=1, model=model, metrics={"loss_total": 0.5}, is_best=True)
        assert ckpt_path.exists()
        assert (temp_dir / "best_model.pt").exists()
        assert (temp_dir / "latest.pt").exists()

        # Test load best
        new_model = nn.Linear(10, 2)
        checkpointer.load_best(new_model)
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_smart_model_wrapper() -> None:
    """Test SMARTModelWrapper initialization, forward pass, and output reconstruction."""
    dataset = DummyPlasticityDataset(num_samples=2)
    loader = SMARTBatchLoader()
    batch = loader.loadDataset([dataset[0], dataset[1]])

    # Lightweight test config
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
    assert isinstance(wrapper.get_torch_module(), nn.Module)

    predictions = wrapper.forward(batch)
    assert "surf_pred" in predictions
    assert "vol_pred" in predictions
    assert "displacements" in predictions

    # Verify reconstructed full mesh shape (B, 101, 31, 2)
    assert predictions["displacements"].shape == (2, 101, 31, 2)


def test_training_engine_end_to_end() -> None:
    """Test full TrainingEngine execution over multiple epochs."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        dataset = DummyPlasticityDataset(num_samples=4)
        loader = SMARTBatchLoader()

        train_dataloader = DataLoader(dataset, batch_size=2, collate_fn=loader.loadDataset)
        val_dataloader = DataLoader(dataset, batch_size=2, collate_fn=loader.loadDataset)

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
        loss_engine = RelativeL2LossEngine()
        optimizer = StandardOptimizer.create_adamw(wrapper.get_torch_module().parameters(), lr=1e-3, epochs=2)
        checkpointer = TorchCheckpointHandler(checkpoint_dir=temp_dir, keep_last_n=2)
        logger = CliProgressMonitor(batch_size=2)

        engine = TrainingEngine(
            model_wrapper=wrapper,
            batch_loader=loader,
            loss_engine=loss_engine,
            optimizer=optimizer,
            checkpointer=checkpointer,
            logger=logger,
        )

        results = engine.train(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            epochs=2,
            device=torch.device("cpu"),
        )

        assert "best_val_loss" in results
        assert results["best_checkpoint"] is not None
        assert results["best_checkpoint"].exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_training_engine_amp_integration() -> None:
    """Test TrainingEngine with automatic mixed precision enabled."""
    temp_dir = Path(tempfile.mkdtemp())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        dataset = DummyPlasticityDataset(num_samples=4)
        loader = SMARTBatchLoader()

        train_dataloader = DataLoader(dataset, batch_size=2, collate_fn=loader.loadDataset)
        val_dataloader = DataLoader(dataset, batch_size=2, collate_fn=loader.loadDataset)

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
        loss_engine = RelativeL2LossEngine()
        optimizer = StandardOptimizer.create_adamw(
            wrapper.get_torch_module().parameters(), lr=1e-3, epochs=2, use_amp=True
        )
        checkpointer = TorchCheckpointHandler(checkpoint_dir=temp_dir, keep_last_n=2)
        logger = CliProgressMonitor(batch_size=2)

        engine = TrainingEngine(
            model_wrapper=wrapper,
            batch_loader=loader,
            loss_engine=loss_engine,
            optimizer=optimizer,
            checkpointer=checkpointer,
            logger=logger,
            use_amp=True,
        )

        results = engine.train(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            epochs=2,
            device=device,
        )

        assert "best_val_loss" in results
        assert results["best_checkpoint"] is not None
        assert results["best_checkpoint"].exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

