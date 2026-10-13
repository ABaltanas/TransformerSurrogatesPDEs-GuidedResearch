"""Abstract interfaces for the DatasetEngine pipeline."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np


class FormatImporter(ABC):
    """Abstract interface for importing raw simulation files into memory."""

    @abstractmethod
    def load(self, path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Load raw simulation data from disk into memory.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the raw simulation file.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of raw arrays extracted from the simulation file.
        """
        pass


class PhysicalSetup(ABC):
    """Abstract interface for cleaning and restructuring raw simulation arrays."""

    @abstractmethod
    def setup(self, raw_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Clean, align, and structure raw arrays into physical tensors.

        Parameters
        ----------
        raw_dict : Dict[str, np.ndarray]
            Raw arrays loaded by FormatImporter.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of structured physical fields (meshes, boundaries, states).
        """
        pass


class DatasetModeler(ABC):
    """Abstract interface for task-specific dataset modifications."""

    @abstractmethod
    def modify(self, fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply task-specific modifications or reductions to physical fields.

        Parameters
        ----------
        fields : Dict[str, np.ndarray]
            Dictionary of physical fields structured by PhysicalSetup.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of modified fields configured for the target ML task.
        """
        pass


class FormatExporter(ABC):
    """Abstract interface for serializing partitioned datasets and statistics."""

    @abstractmethod
    def save(
        self,
        splits: Dict[str, Dict[str, np.ndarray]],
        statisticalData: Dict[str, Any],
        output_dir: Union[str, Path],
    ) -> List[Path]:
        """Serialize partitioned datasets and scaling parameters to disk.

        Parameters
        ----------
        splits : Dict[str, Dict[str, np.ndarray]]
            Nested dictionary of partitioned arrays (train, val, test).
        statisticalData : Dict[str, Any]
            Statistical scaling moments computed from the training partition.
        output_dir : Union[str, Path]
            Target directory for output files.

        Returns
        -------
        List[Path]
            List of file paths created on disk.
        """
        pass
