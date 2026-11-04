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
    KillSwitch,
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

__all__ = [
    # Interfaces
    "BatchLoader",
    "ModelWrapper",
    "LossEngine",
    "Optimizer",
    "KillSwitch",
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
    # Optimizer & Strategies
    "OptimizerController",
    "OptimizerFactory",
    "StandardOptimizer",
    "Lion",
    "NaNOrDivergenceKillSwitch",
    "EarlyStoppingKillSwitch",
    "CompositeKillSwitch",
    # Checkpointing
    "TorchCheckpointHandler",
    # Telemetry
    "CliProgressMonitor",
    "JsonMetricsLogger",
    "CompositeTelemetryLogger",
    # Orchestrator
    "TrainingEngine",
]
