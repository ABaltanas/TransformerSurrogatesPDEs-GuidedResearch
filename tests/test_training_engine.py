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
from src.training.optimizers import (
    CompositeKillSwitch,
    EarlyStoppingKillSwitch,
    Lion,
    NaNOrDivergenceKillSwitch,
    OptimizerController,
    OptimizerFactory,
    StandardOptimizer,
)
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


def test_training_engine_resume() -> None:
    """Test resuming TrainingEngine from a saved checkpoint."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_training_resume_"))
    device = torch.device("cpu")
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

        # 1. Initial run: epochs 1-2
        wrapper = SMARTModelWrapper(config=config)
        loss_engine = RelativeL2LossEngine()
        optimizer = StandardOptimizer.create_adamw(
            wrapper.get_torch_module().parameters(), lr=1e-3, epochs=4
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
        )

        results_phase1 = engine.train(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            epochs=2,
            device=device,
        )

        latest_path = checkpointer.get_latest_checkpoint_path()
        assert latest_path is not None
        assert latest_path.exists()
        assert checkpointer.get_best_val_loss() == pytest.approx(results_phase1["best_val_loss"])

        # 2. Resume run: epochs 3-4
        wrapper_resumed = SMARTModelWrapper(config=config)
        loaded_payload = checkpointer.load_checkpoint(latest_path, wrapper_resumed.get_torch_module())
        assert loaded_payload["epoch"] == 2

        optimizer_resumed = StandardOptimizer.create_adamw(
            wrapper_resumed.get_torch_module().parameters(), lr=1e-3, epochs=4
        )
        optimizer_resumed.load_state(loaded_payload, step_scheduler_to_epoch=2)
        # Verify learning rate was stepped to epoch 2
        assert optimizer_resumed.get_learning_rate() < 1e-3

        checkpointer_resumed = TorchCheckpointHandler(checkpoint_dir=temp_dir, keep_last_n=2)
        engine_resumed = TrainingEngine(
            model_wrapper=wrapper_resumed,
            batch_loader=loader,
            loss_engine=loss_engine,
            optimizer=optimizer_resumed,
            checkpointer=checkpointer_resumed,
            logger=logger,
        )

        results_phase2 = engine_resumed.train(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            epochs=4,
            device=device,
            start_epoch=loaded_payload["epoch"] + 1,
            best_val_loss=checkpointer_resumed.get_best_val_loss(),
        )

        final_latest = checkpointer_resumed.get_latest_checkpoint_path()
        assert final_latest is not None
        ckpt_final = torch.load(final_latest, map_location="cpu", weights_only=False)
        assert ckpt_final["epoch"] == 4
        assert "optimizer_state_dict" in ckpt_final
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_optimizer_controller_bridge_and_adamw() -> None:
    """Test OptimizerController Bridge abstraction wrapping AdamW."""
    model = nn.Linear(5, 2)
    controller = OptimizerController.create_adamw(
        model_parameters=model.parameters(),
        lr=1e-3,
        scheduler_type="cosine",
        epochs=10,
    )
    assert isinstance(controller, OptimizerController)
    assert controller.get_learning_rate() == pytest.approx(1e-3)

    x = torch.randn(2, 5)
    controller.zero_grad()
    loss = model(x).sum()
    grad_norm = controller.step(loss, max_grad_norm=1.0)
    assert grad_norm >= 0.0

    # Step learning rate
    new_lr = controller.epoch_step(0.5)
    assert new_lr < 1e-3


def test_optimizer_controller_lion() -> None:
    """Test pure Lion optimizer reducing loss on a toy linear regression problem."""
    torch.manual_seed(42)
    X = torch.randn(100, 5)
    true_w = torch.tensor([[1.5], [-2.0], [0.5], [1.0], [-0.5]])
    y = X @ true_w

    model = nn.Linear(5, 1, bias=False)
    controller = OptimizerController.create_lion(
        model_parameters=model.parameters(),
        lr=0.01,
        weight_decay=1e-4,
        scheduler_type=None,
    )
    assert isinstance(controller.optimizer, Lion)

    initial_loss = torch.mean((model(X) - y) ** 2).item()
    for _ in range(150):
        controller.zero_grad()
        loss = torch.mean((model(X) - y) ** 2)
        controller.step(loss)

    final_loss = torch.mean((model(X) - y) ** 2).item()
    assert final_loss < initial_loss * 0.2


def test_sequential_lr_warmup_cosine() -> None:
    """Test SequentialLR warmup-cosine chaining 10% linear warmup and 90% cosine decay."""
    model = nn.Linear(4, 2)
    controller = OptimizerController.create_adamw(
        model_parameters=model.parameters(),
        lr=1e-3,
        scheduler_type="warmup_cosine",
        epochs=100,
        min_lr=1e-6,
        warmup_ratio=0.1,
    )
    # At initialization (before step), start factor is 0.1 -> lr is 1e-4
    initial_lr = controller.get_learning_rate()
    assert initial_lr == pytest.approx(1e-4)

    # Step through warmup (10 epochs)
    for _ in range(10):
        controller.epoch_step()

    peak_lr = controller.get_learning_rate()
    assert peak_lr == pytest.approx(1e-3, rel=1e-2)

    # Step past warmup into cosine decay
    for _ in range(20):
        controller.epoch_step()

    decayed_lr = controller.get_learning_rate()
    assert decayed_lr < peak_lr


def test_nan_kill_switch_step_abortion() -> None:
    """Test NaNOrDivergenceKillSwitch aborting training upon NaN loss or divergence."""
    model = nn.Linear(2, 2)
    kill_switch = NaNOrDivergenceKillSwitch(max_loss=100.0, max_grad_norm=10.0)
    controller = OptimizerController.create_adamw(
        model_parameters=model.parameters(),
        lr=1e-3,
        kill_switch=kill_switch,
    )

    # Normal step: no failure
    controller.zero_grad()
    loss = model(torch.randn(2, 2)).sum()
    controller.step(loss)

    # Exploded loss: exceeds max_loss
    controller.zero_grad()
    big_loss = model(torch.randn(2, 2)).sum() + 500.0
    with pytest.raises(RuntimeError, match="Training aborted by KillSwitch"):
        controller.step(big_loss)

    # NaN loss
    controller.zero_grad()
    nan_loss = torch.tensor(float("nan"), requires_grad=True)
    with pytest.raises(RuntimeError, match="Training aborted by KillSwitch"):
        controller.step(nan_loss)


def test_early_stopping_kill_switch_epoch() -> None:
    """Test EarlyStoppingKillSwitch raising StopIteration after patience runs out."""
    model = nn.Linear(2, 2)
    kill_switch = EarlyStoppingKillSwitch(patience=3, min_delta=1e-3, mode="min")
    controller = OptimizerController.create_adamw(
        model_parameters=model.parameters(),
        lr=1e-3,
        kill_switch=kill_switch,
    )

    # Epoch 1: initial best = 1.0
    controller.epoch_step(1.0)
    # Epoch 2: plateau 1 (no improvement)
    controller.epoch_step(1.0)
    # Epoch 3: plateau 2
    controller.epoch_step(1.0)
    # Epoch 4: plateau 3 -> patience exceeded, raises StopIteration
    with pytest.raises(StopIteration, match="Early stopping triggered by KillSwitch"):
        controller.epoch_step(1.0)


def test_training_engine_early_stopping_integration() -> None:
    """Test TrainingEngine gracefully stopping early and preserving final state."""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_early_stopping_"))
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
        kill_switch = EarlyStoppingKillSwitch(patience=2, min_delta=10.0, mode="min")
        optimizer = OptimizerController.create_adamw(
            wrapper.get_torch_module().parameters(),
            lr=1e-3,
            epochs=10,
            kill_switch=kill_switch,
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
        )

        results = engine.train(
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            epochs=10,
            device=torch.device("cpu"),
        )

        # Early stopping must have triggered before epoch 10
        assert results.get("early_stopped") is True
        assert results["best_checkpoint"].exists()
        assert (temp_dir / "latest.pt").exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_standard_optimizer_backward_compatibility() -> None:
    """Test StandardOptimizer alias works seamlessly as a drop-in replacement."""
    model = nn.Linear(3, 2)
    opt = StandardOptimizer.create_adamw(model.parameters(), lr=1e-3, epochs=5)
    assert isinstance(opt, OptimizerController)
    assert opt.get_learning_rate() == pytest.approx(1e-3)

    x = torch.randn(2, 3)
    loss = model(x).sum()
    opt.zero_grad()
    norm = opt.step(loss)
    assert norm >= 0.0

    state = opt.get_state_dict()
    assert "optimizer_state_dict" in state


def test_optimizer_factory_typed_constructors() -> None:
    """Test OptimizerFactory explicit create_adamw and create_lion methods."""
    model = nn.Linear(4, 2)

    # Test AdamW creation via factory
    adamw_ctrl = OptimizerFactory.create_adamw(
        model_parameters=model.parameters(),
        lr=1e-3,
        scheduler_type="warmup_cosine",
        epochs=50,
    )
    assert isinstance(adamw_ctrl, OptimizerController)
    assert isinstance(adamw_ctrl.optimizer, torch.optim.AdamW)
    assert adamw_ctrl.get_learning_rate() == pytest.approx(1e-4)  # 0.1 start_factor

    # Test Lion creation via factory
    lion_ctrl = OptimizerFactory.create_lion(
        model_parameters=model.parameters(),
        lr=5e-5,
        betas=(0.95, 0.98),
        scheduler_type="cosine",
        epochs=30,
    )
    assert isinstance(lion_ctrl, OptimizerController)
    assert isinstance(lion_ctrl.optimizer, Lion)
    assert lion_ctrl.get_learning_rate() == pytest.approx(5e-5)


def test_optimizer_factory_dynamic_create() -> None:
    """Test OptimizerFactory.create with string identifiers and error handling."""
    model = nn.Linear(3, 1)

    opt_adamw = OptimizerFactory.create(
        optimizer_type="adamw",
        model_parameters=model.parameters(),
        lr=2e-4,
    )
    assert isinstance(opt_adamw.optimizer, torch.optim.AdamW)

    opt_lion = OptimizerFactory.create(
        optimizer_type="lion",
        model_parameters=model.parameters(),
        lr=1e-4,
    )
    assert isinstance(opt_lion.optimizer, Lion)

    with pytest.raises(ValueError, match="Unsupported optimizer_type"):
        OptimizerFactory.create(
            optimizer_type="unknown_opt",
            model_parameters=model.parameters(),
        )


def test_write_parametros_txt_audit_file() -> None:
    """Test write_parametros_txt generation in temporal training script."""
    import argparse
    from scripts.train_smart_temporal import write_parametros_txt

    temp_dir = Path(tempfile.mkdtemp(prefix="test_parametros_"))
    try:
        fpath = temp_dir / "parametros.txt"
        args = argparse.Namespace(
            data_dir=Path("data/processed/ds_plasticity_temporal"),
            models_dir=temp_dir,
            results_dir=Path("results/TrainingAnalysis"),
            epochs=50,
            batch_size=1,
            optimizer="lion",
            lr=1e-4,
            weight_decay=1e-4,
            scheduler="warmup_cosine",
            early_stopping_patience=None,
            max_grad_norm=1.0,
            use_amp=False,
            device="cpu",
            seed=42,
            parameter_channels=1,
            latent_dim=256,
            latent_geo_points=101,
            subsampled_points=101,
            num_blocks=8,
            num_heads=8,
            pos_scale_factor=1000.0,
            dropout=0.0,
        )
        summary = {"best_val_loss": 0.0421, "best_checkpoint": temp_dir / "best_model.pt"}
        kpis = {"best_epoch": 45, "val_loss_reduction_pct": 85.2}
        write_parametros_txt(fpath, args, summary=summary, kpis=kpis, duration_s=120.5)

        assert fpath.exists()
        text = fpath.read_text(encoding="utf-8")
        assert "SMART SURROGATE MODEL - TRAINING CONFIGURATION AUDIT" in text
        assert "optimizer:                    lion" in text
        assert "scheduler:                    warmup_cosine" in text
        assert "latent_dim:                   256" in text
        assert "latent_geo_points:            101" in text
        assert "num_blocks:                   8" in text
        assert "Best Validation Loss:         4.210000e-02" in text
        assert "Total Val Loss Reduction:     85.2%" in text
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)



