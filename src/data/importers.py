"""Concrete format importers for simulation datasets."""

from pathlib import Path
from typing import Dict, Union

import numpy as np
import scipy.io as sio

from src.data.interfaces import FormatImporter


class MatlabImporter(FormatImporter):
    """Format importer specialized for MATLAB .mat simulation files."""

    def load(self, path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Load MATLAB .mat file into a dictionary of NumPy arrays.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the .mat file or directory containing it.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing simulation arrays (e.g., 'input', 'output').

        Raises
        ------
        FileNotFoundError
            If the file cannot be found.
        ValueError
            If the file does not contain valid simulation arrays.
        """
        resolved_path = Path(path).expanduser().resolve()

        if resolved_path.is_dir():
            mat_files = list(resolved_path.glob("*.mat"))
            if not mat_files:
                raise FileNotFoundError(f"No .mat files found in directory: {resolved_path}")
            # Prefer plas_N987_T20.mat if present
            preferred = [f for f in mat_files if "plas" in f.name]
            resolved_path = preferred[0] if preferred else mat_files[0]

        if not resolved_path.exists():
            raise FileNotFoundError(f"MATLAB simulation file does not exist: {resolved_path}")

        mat_dict = sio.loadmat(str(resolved_path))

        # Filter out MATLAB internal metadata keys
        cleaned_dict: Dict[str, np.ndarray] = {
            k: v for k, v in mat_dict.items() if not k.startswith("__")
        }

        if "input" not in cleaned_dict or "output" not in cleaned_dict:
            available = list(cleaned_dict.keys())
            raise ValueError(
                f"Expected 'input' and 'output' in {resolved_path.name}, but found keys: {available}"
            )

        return cleaned_dict
