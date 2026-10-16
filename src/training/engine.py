"""Orchestrator for surrogate model training, validation, and optimization."""

from pathlib import Path
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.interfaces import (
    BatchLoader,
    CheckpointHandler,
    LossEngine,
    ModelWrapper,
    Optimizer,
    TelemetryLogger,
)


class TrainingEngine:
    """Coordinates surrogate model training, validation, optimization, and checkpointing.

    Acts as the central orchestrator coordinating decoupled strategy components:
    - ModelWrapper: Executes the forward pass and isolates architecture signatures.
    - BatchLoader: Prepares model-ready batches.
    - LossEngine: Computes scalar backpropagation objectives and component breakdowns.
    - Optimizer: Handles gradient clipping, parameter updates, and LR scheduling.
    - CheckpointHandler: Persists atomic checkpoints to disk.
    - TelemetryLogger: Monitors progress and reports metrics to the CLI and dashboards.

    Parameters
    ----------
    model_wrapper : ModelWrapper
        Wrapper housing and executing the neural network architecture.
    batch_loader : BatchLoader
        Strategy preparing raw partition samples into model-ready tensor batches.
    loss_engine : LossEngine
        Objective calculation strategy.
    optimizer : Optimizer
        Optimizer and learning rate schedule manager.
    checkpointer : CheckpointHandler
        Checkpoint persistence manager.
    logger : TelemetryLogger
        Telemetry and metrics logging observer.
    """

    def __init__(
        self,
        model_wrapper: ModelWrapper,
        batch_loader: BatchLoader,
        loss_engine: LossEngine,
        optimizer: Optimizer,
        checkpointer: CheckpointHandler,
        logger: TelemetryLogger,
    ) -> None:
        self.model_wrapper = model_wrapper
        self.batch_loader = batch_loader
        self.loss_engine = loss_engine
        self.optimizer = optimizer
        self.checkpointer = checkpointer
        self.logger = logger

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        device: torch.device,
        max_grad_norm: Optional[float] = 1.0,
    ) -> Dict[str, Any]:
        """Execute full training and validation schedule.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader wrapping the training partition.
        val_loader : DataLoader
            DataLoader wrapping the validation partition.
        epochs : int
            Total training epochs.
        device : torch.device
            Hardware target device (e.g. cuda:0 or cpu).
        max_grad_norm : Optional[float], default=1.0
            Maximum gradient norm threshold for clipping.

        Returns
        -------
        Dict[str, Any]
            Execution summary containing best validation loss and checkpoint paths.
        """
        best_val_loss = float("inf")
        model = self.model_wrapper.get_torch_module()
        model.to(device)

        for epoch in range(1, epochs + 1):
            self.logger.start_epoch(epoch, epochs, len(train_loader))

            # 1. Training step across all batches in train_loader
            train_metrics = self._train_epoch(train_loader, epoch, epochs, device, max_grad_norm)

            # 2. Validation step across all batches in val_loader
            val_metrics = self._validate_epoch(val_loader, device)

            # 3. Learning rate schedule step
            current_lr = self.optimizer.epoch_step(val_metrics.get("loss_total", None))

            # 4. Telemetry logging and progress report
            self.logger.end_epoch(epoch, epochs, train_metrics, val_metrics, current_lr)

            # 5. Checkpointing
            val_loss = val_metrics.get("loss_total", float("inf"))
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            self.checkpointer.save(
                epoch=epoch,
                model=model,
                metrics=val_metrics,
                is_best=is_best,
            )

        self.logger.close()
        return {
            "best_val_loss": best_val_loss,
            "best_checkpoint": self.checkpointer.get_best_checkpoint_path(),
        }

    def _train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        total_epochs: int,
        device: torch.device,
        max_grad_norm: Optional[float],
    ) -> Dict[str, float]:
        """Execute single training epoch across batches."""
        model = self.model_wrapper.get_torch_module()
        model.train()
        accumulated_losses: Dict[str, float] = {}
        total_batches = len(dataloader)

        for batch_idx, raw_batch in enumerate(dataloader, start=1):
            # Transfer tensor batch to hardware device
            batch = self._transfer_batch_to_device(raw_batch, device)
            self.optimizer.zero_grad()

            # Forward pass executed via ModelWrapper
            predictions = self.model_wrapper.forward(batch)
            loss, loss_components = self.loss_engine.compute(predictions, batch["targets"], batch)

            # Backward pass, gradient clipping, and weight update
            self.optimizer.step(loss, max_grad_norm=max_grad_norm)

            # Accumulate loss metrics for epoch averaging
            for k, v in loss_components.items():
                accumulated_losses[k] = accumulated_losses.get(k, 0.0) + v

            current_lr = self.optimizer.get_learning_rate()
            self.logger.update_batch(batch_idx, loss.item(), current_lr)

        return {k: v / max(1, total_batches) for k, v in accumulated_losses.items()}

    def _validate_epoch(
        self,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Execute validation epoch with gradients disabled."""
        model = self.model_wrapper.get_torch_module()
        model.eval()
        accumulated_losses: Dict[str, float] = {}
        total_batches = len(dataloader)

        with torch.no_grad():
            for raw_batch in dataloader:
                batch = self._transfer_batch_to_device(raw_batch, device)
                predictions = self.model_wrapper.forward(batch)
                _, loss_components = self.loss_engine.compute(predictions, batch["targets"], batch)

                for k, v in loss_components.items():
                    accumulated_losses[k] = accumulated_losses.get(k, 0.0) + v

        return {k: v / max(1, total_batches) for k, v in accumulated_losses.items()}

    def _transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        """Recursively transfer tensors within nested batch dictionaries to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: self._transfer_batch_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._transfer_batch_to_device(v, device) for v in batch]
        return batch
