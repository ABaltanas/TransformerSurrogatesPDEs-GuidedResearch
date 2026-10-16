"""Surrogate model training subsystem.

Provides orchestrators, interfaces, and strategies for neural operator optimization,
batch preparation, model wrapping, and real-time execution telemetry.
"""

from src.training.batch_loaders import SMARTBatchLoader, StandardBatchLoader
from src.training.checkpoints import TorchCheckpointHandler
from src.training.engine import TrainingEngine
from src.training.interfaces import (
    BatchLoader,
    CheckpointHandler,
    LossEngine,
    ModelWrapper,
    Optimizer,
    TelemetryLogger,
)
from src.training.loggers import (
    CliProgressMonitor,
    CompositeTelemetryLogger,
    JsonMetricsLogger,
)
from src.training.losses import (
    CompositePhysicsLossEngine,
    MSELossEngine,
    RelativeL2LossEngine,
)
from src.training.optimizers import StandardOptimizer
from src.training.wrappers.smart_wrapper import SMARTModelWrapper

__all__ = [
    # Interfaces
    "BatchLoader",
    "ModelWrapper",
    "LossEngine",
    "Optimizer",
    "CheckpointHandler",
    "TelemetryLogger",
    # Concrete Loaders
    "SMARTBatchLoader",
    "StandardBatchLoader",
    # Concrete Wrappers
    "SMARTModelWrapper",
    # Concrete Losses
    "RelativeL2LossEngine",
    "MSELossEngine",
    "CompositePhysicsLossEngine",
    # Optimizer
    "StandardOptimizer",
    # Checkpointing
    "TorchCheckpointHandler",
    # Telemetry
    "CliProgressMonitor",
    "JsonMetricsLogger",
    "CompositeTelemetryLogger",
    # Orchestrator
    "TrainingEngine",
]
