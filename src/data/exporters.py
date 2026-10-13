"""Format exporters for serializing simulation arrays and statistics to disk."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from src.data.interfaces import FormatExporter


class NpzExporter(FormatExporter):
    """Exports partitioned datasets to compressed NumPy .npz files and JSON statistics."""

    def save(
        self,
        splits: Dict[str, Dict[str, np.ndarray]],
        statisticalData: Dict[str, Any],
        output_dir: Union[str, Path],
    ) -> List[Path]:
        """Serialize partitioned arrays and statistical data to disk.

        Parameters
        ----------
        splits : Dict[str, Dict[str, np.ndarray]]
            Dictionary containing 'train', 'val', and 'test' partitions.
        statisticalData : Dict[str, Any]
            Statistical scaling parameters computed from training partition.
        output_dir : Union[str, Path]
            Target directory for output files.

        Returns
        -------
        List[Path]
            List of generated file paths.

        Raises
        ------
        ValueError
            If arrays contain NaN or Inf values.
        """
        out_path = Path(output_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        saved_paths: List[Path] = []

        # 1. Save partitioned npz archives
        for split_name, arrays in splits.items():
            # Validate no corruption before saving
            for arr_name, arr in arrays.items():
                if np.issubdtype(arr.dtype, np.number):
                    if np.isnan(arr).any():
                        raise ValueError(f"Corrupted array: NaN detected in {split_name}['{arr_name}']")
                    if np.isinf(arr).any():
                        raise ValueError(f"Corrupted array: Inf detected in {split_name}['{arr_name}']")

            target_file = out_path / f"{split_name}.npz"
            np.savez_compressed(str(target_file), **arrays)
            saved_paths.append(target_file)

        # 2. Save statistical data JSON
        stats_file = out_path / "statistical_data.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(statisticalData, f, indent=2)
        saved_paths.append(stats_file)

        # 3. Verify non-zero file sizes
        for file_path in saved_paths:
            if not file_path.exists() or file_path.stat().st_size == 0:
                raise IOError(f"Verification failed: exported file is empty or missing: {file_path}")

        return saved_paths
