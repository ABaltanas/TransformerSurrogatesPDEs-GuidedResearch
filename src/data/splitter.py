"""Dataset splitting component for partitioning simulations into train, validation, and test subsets."""

from typing import Dict

import numpy as np


class DatasetSplitter:
    """Partitions simulation arrays into train, validation, and test subsets."""

    def split(
        self,
        arrays: Dict[str, np.ndarray],
        train_n: int,
        val_n: int,
        test_n: int,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Partition datasets along the simulation sample dimension.

        Parameters
        ----------
        arrays : Dict[str, np.ndarray]
            Dictionary of arrays produced by PhysicalSetup.
        train_n : int
            Number of training samples.
        val_n : int
            Number of validation samples.
        test_n : int
            Number of testing samples.

        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            Nested dictionary containing 'train', 'val', and 'test' partitions.

        Raises
        ------
        ValueError
            If requested counts exceed available simulations or are invalid.
        """
        if train_n <= 0 or val_n <= 0 or test_n <= 0:
            raise ValueError(f"Partition counts must be positive: train={train_n}, val={val_n}, test={test_n}")

        # Find total samples from the first array with length matching sample dimension
        sample_lengths = [
            arr.shape[0] for k, arr in arrays.items() if k != "time_steps" and arr.ndim > 1
        ]
        if not sample_lengths:
            raise ValueError("No sample-indexed arrays found in input dictionary.")

        total_samples = sample_lengths[0]
        requested_total = train_n + val_n + test_n
        if requested_total > total_samples:
            raise ValueError(
                f"Requested total samples ({requested_total}) exceeds available samples ({total_samples})."
            )

        splits: Dict[str, Dict[str, np.ndarray]] = {
            "train": {},
            "val": {},
            "test": {},
        }

        train_slice = slice(0, train_n)
        val_slice = slice(train_n, train_n + val_n)
        test_slice = slice(train_n + val_n, train_n + val_n + test_n)

        for key, arr in arrays.items():
            if arr.ndim > 1 and arr.shape[0] == total_samples:
                splits["train"][key] = arr[train_slice].copy()
                splits["val"][key] = arr[val_slice].copy()
                splits["test"][key] = arr[test_slice].copy()
            else:
                # Global invariant arrays (e.g. time_steps) are shared across all partitions
                splits["train"][key] = arr.copy()
                splits["val"][key] = arr.copy()
                splits["test"][key] = arr.copy()

        return splits
