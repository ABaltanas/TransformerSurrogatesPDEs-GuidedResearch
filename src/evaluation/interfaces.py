"""Interfaces and abstract base classes for the surrogate evaluation subsystem.

Defines the strategy contracts for model inference, physical denormalization,
PDE error metric computation, physical feasibility validation, hardware profiling,
and report generation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader


class InferenceRunner(ABC):
    """Abstract interface for running deterministic batch inference on test partitions."""

    @abstractmethod
    def predict(
        self,
        model: Any,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, np.ndarray]:
        """Execute deterministic inference across all batches in the DataLoader.

        Parameters
        ----------
        model : Any
            The neural operator module (nn.Module) or ModelWrapper instance.
        dataloader : DataLoader
            DataLoader wrapping the test dataset partition.
        device : torch.device
            Hardware target device (e.g. cuda:0 or cpu).

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing aggregated prediction arrays across the entire test partition.
        """
        pass


class Denormalizer(ABC):
    """Abstract interface for rescaling normalized predictions back into physical units."""

    @abstractmethod
    def denormalize(
        self,
        pred_norm: Dict[str, np.ndarray],
        target_norm: Dict[str, np.ndarray],
        stats: Dict[str, Any],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Invert statistical scaling and restore physical engineering units.

        Parameters
        ----------
        pred_norm : Dict[str, np.ndarray]
            Normalized prediction dictionary from InferenceRunner.
        target_norm : Dict[str, np.ndarray]
            Ground truth target dictionary in normalized representation.
        stats : Dict[str, Any]
            Statistical scaling moments from statistical_data.json.

        Returns
        -------
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
            Pair of dictionaries (phys_predictions, phys_targets) in physical units (mm).
        """
        pass


class PhysicalMetricsCalculator(ABC):
    """Abstract interface for computing solid mechanics PDE error metrics."""

    @abstractmethod
    def compute_metrics(
        self,
        pred_phys: Dict[str, np.ndarray],
        target_phys: Dict[str, np.ndarray],
        initial_mesh: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate quantitative physical error metrics between predictions and targets.

        Parameters
        ----------
        pred_phys : Dict[str, np.ndarray]
            Predictions rescaled to physical units.
        target_phys : Dict[str, np.ndarray]
            Ground-truth simulation targets in physical units.
        initial_mesh : np.ndarray
            Reference undeformed coordinate mesh (coords_initial).

        Returns
        -------
        Dict[str, float]
            Dictionary of computed scalar scientific metrics (Rel L2, L_inf, H1, etc.).
        """
        pass


class FeasibilityValidator(ABC):
    """Abstract interface for assessing physical boundary condition feasibility."""

    @abstractmethod
    def check_constraints(
        self,
        pred_coords: np.ndarray,
        boundary_constraints: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate adherence to physical boundary conditions and geometric constraints.

        Parameters
        ----------
        pred_coords : np.ndarray
            Predicted deformed spatial coordinate field, shape (N, nx, ny, 2).
        boundary_constraints : np.ndarray
            Boundary obstacle geometry coordinates (e.g., rigid die contour).

        Returns
        -------
        Dict[str, float]
            Quantitative feasibility violation statistics.
        """
        pass


class BenchmarkProfiler(ABC):
    """Abstract interface for measuring execution speed, memory footprint, and speedup."""

    @abstractmethod
    def profile_inference(
        self,
        model: Any,
        sample_batch: Dict[str, Any],
        device: torch.device,
        fem_reference_time_s: float = 120.0,
    ) -> Dict[str, float]:
        """Profile model latency, memory usage, and numerical acceleration.

        Parameters
        ----------
        model : Any
            The neural network module or ModelWrapper to benchmark.
        sample_batch : Dict[str, Any]
            A representative input batch.
        device : torch.device
            Evaluation hardware device.
        fem_reference_time_s : float, default=120.0
            Abaqus FEM simulation runtime in seconds per sample.

        Returns
        -------
        Dict[str, float]
            Computational metrics (latency_ms, peak_vram_mb, speedup_vs_fem, parameters).
        """
        pass


class ReportExporter(ABC):
    """Abstract interface for formatting and persisting evaluation summaries."""

    @abstractmethod
    def export(self, summary: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
        """Export publication tables and machine-readable metric records to disk.

        Parameters
        ----------
        summary : Dict[str, Any]
            Comprehensive evaluation record.
        output_dir : Path
            Target directory for writing artifacts.

        Returns
        -------
        Dict[str, Path]
            Dictionary mapping artifact names to their persisted file paths.
        """
        pass
