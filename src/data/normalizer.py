"""Statistical normalization and moment calculation components."""

from typing import Any, Dict

import numpy as np


class DatasetNormalizer:
    """Computes statistical scaling parameters strictly on training partitions."""

    def compute_statistical_data(
        self, train_arrays: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate statistical moments from training arrays.

        Parameters
        ----------
        train_arrays : Dict[str, np.ndarray]
            Training partition arrays.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary containing global and per-channel statistics
            (mean, std, min, max) for arrays in the training partition.
        """
        statistical_data: Dict[str, Dict[str, Any]] = {}

        for key, arr in train_arrays.items():
            if not np.issubdtype(arr.dtype, np.number):
                continue

            arr_float = arr.astype(np.float64)

            stats_entry: Dict[str, Any] = {
                "mean": float(np.mean(arr_float)),
                "std": float(np.std(arr_float)),
                "min": float(np.min(arr_float)),
                "max": float(np.max(arr_float)),
            }

            # If multi-channel tensor (ndim >= 2), compute per-channel statistics along the last axis
            if arr.ndim >= 2 and arr.shape[-1] <= 16:
                num_channels = arr.shape[-1]
                stats_entry["per_channel"] = {
                    "mean": [float(np.mean(arr_float[..., c])) for c in range(num_channels)],
                    "std": [float(np.std(arr_float[..., c])) for c in range(num_channels)],
                    "min": [float(np.min(arr_float[..., c])) for c in range(num_channels)],
                    "max": [float(np.max(arr_float[..., c])) for c in range(num_channels)],
                }

            statistical_data[key] = stats_entry

        return statistical_data
