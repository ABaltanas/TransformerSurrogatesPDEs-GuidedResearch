"""Interfaces and abstract base classes for the surrogate model training subsystem.

This module defines the architectural contracts for batch loading, model wrapping,
loss calculation, optimization, checkpoint persistence, and telemetry logging.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn


class BatchLoader(ABC):
    """Abstract interface for preparing and collating model-specific tensor batches."""

    @abstractmethod
    def loadDataset(self, samples: Sequence[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """Convert a sequence of raw sample dictionaries into a batched tensor dictionary.

        Parameters
        ----------
        samples : Sequence[Dict[str, np.ndarray]]
            List of sample dictionaries retrieved from partition archives (.npz).

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of batched PyTorch tensors formatted for the target architecture.
        """
        pass


class ModelWrapper(ABC):
    """Abstract interface wrapping neural operator architectures.

    Houses and manages the neural network instance, isolating model-specific forward
    signatures and standardizing output tensor representations.
    """

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute the forward pass of the underlying neural network.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batched input tensors prepared by a BatchLoader.

        Returns
        -------
        Dict[str, torch.Tensor]
            Standardized predictions dictionary containing predicted fields (e.g. displacements).
        """
        pass

    @abstractmethod
    def get_torch_module(self) -> nn.Module:
        """Return the underlying PyTorch nn.Module for optimizer parameter registration.

        Returns
        -------
        nn.Module
            The encapsulated PyTorch neural network module.
        """
        pass


class LossEngine(ABC):
    """Abstract interface for calculating training and validation objectives."""

    @abstractmethod
    def compute(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the scalar backpropagation loss and individual component breakdowns.

        Parameters
        ----------
        predictions : Dict[str, torch.Tensor]
            Predicted tensors produced by a ModelWrapper.
        targets : Dict[str, torch.Tensor]
            Ground-truth target tensors from the batch.
        batch : Dict[str, torch.Tensor]
            The full input batch dictionary (providing boundary conditions or geometry if needed).

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, float]]
            - Backpropagatable scalar loss tensor.
            - Dictionary of individual loss components formatted as scalar floats for logging.
        """
        pass


class Optimizer(ABC):
    """Abstract interface managing parameter updates and learning rate schedules."""

    @abstractmethod
    def zero_grad(self) -> None:
        """Reset parameter gradients before the backward pass."""
        pass

    @abstractmethod
    def step(self, loss: torch.Tensor, max_grad_norm: Optional[float] = None) -> float:
        """Execute backward pass, gradient clipping, and parameter weight update.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss tensor to backpropagate.
        max_grad_norm : Optional[float], default=None
            Maximum gradient norm threshold for clipping. If None, no clipping is applied.

        Returns
        -------
        float
            The computed gradient norm before clipping.
        """
        pass

    @abstractmethod
    def epoch_step(self, metric: Optional[float] = None) -> float:
        """Update learning rate schedules after epoch completion.

        Parameters
        ----------
        metric : Optional[float], default=None
            Validation metric for schedulers requiring feedback (e.g., ReduceLROnPlateau).

        Returns
        -------
        float
            The updated active learning rate.
        """
        pass

    @abstractmethod
    def get_learning_rate(self) -> float:
        """Retrieve the current active learning rate.

        Returns
        -------
        float
            Active learning rate from the primary optimizer parameter group.
        """
        pass


class CheckpointHandler(ABC):
    """Abstract interface managing atomic checkpoint persistence and state recovery."""

    @abstractmethod
    def save(
        self,
        epoch: int,
        model: nn.Module,
        metrics: Dict[str, float],
        is_best: bool,
    ) -> Path:
        """Save a model checkpoint atomically to disk.

        Parameters
        ----------
        epoch : int
            Current training epoch index.
        model : nn.Module
            Neural network module containing weights to persist.
        metrics : Dict[str, float]
            Validation and training metrics evaluated at this epoch.
        is_best : bool
            Flag indicating whether this checkpoint achieved the best validation performance.

        Returns
        -------
        Path
            Absolute path to the persisted checkpoint file.
        """
        pass

    @abstractmethod
    def load_best(self, model: nn.Module) -> Dict[str, Any]:
        """Load the best recorded checkpoint weights into the given model.

        Parameters
        ----------
        model : nn.Module
            Target neural network module to populate with saved weights.

        Returns
        -------
        Dict[str, Any]
            Checkpoint metadata dictionary (epoch, metrics, optimizer state).
        """
        pass

    @abstractmethod
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Return the path to the best recorded checkpoint file.

        Returns
        -------
        Optional[Path]
            Path to the best checkpoint file, or None if no checkpoint has been saved.
        """
        pass


class TelemetryLogger(ABC):
    """Abstract interface for training telemetry observers and CLI monitors."""

    @abstractmethod
    def start_epoch(self, epoch: int, total_epochs: int, total_batches: int) -> None:
        """Notify logger that a new training epoch has started.

        Parameters
        ----------
        epoch : int
            Current epoch index (1-indexed).
        total_epochs : int
            Total planned epochs.
        total_batches : int
            Number of batches in the training loader.
        """
        pass

    @abstractmethod
    def update_batch(self, batch_idx: int, loss_val: float, lr: float) -> None:
        """Notify logger that an optimization step has finished for a batch.

        Parameters
        ----------
        batch_idx : int
            Current batch index within the epoch (1-indexed).
        loss_val : float
            Current batch scalar loss value.
        lr : float
            Current active learning rate.
        """
        pass

    @abstractmethod
    def end_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
    ) -> None:
        """Notify logger that an epoch and validation evaluation have completed.

        Parameters
        ----------
        epoch : int
            Completed epoch index.
        total_epochs : int
            Total planned epochs.
        train_metrics : Dict[str, float]
            Averaged metrics across training batches.
        val_metrics : Dict[str, float]
            Metrics evaluated on the validation partition.
        lr : float
            Active learning rate at the end of the epoch.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up logging handlers, close open files, and finalize remote runs."""
        pass
