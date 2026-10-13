"""DatasetEngine orchestrating the modular preprocessing pipeline."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.data.interfaces import (
    DatasetModeler,
    FormatExporter,
    FormatImporter,
    PhysicalSetup,
)
from src.data.normalizer import DatasetNormalizer
from src.data.splitter import DatasetSplitter


class DatasetEngine:
    """Orchestrator coordinating the decoupled preprocessing strategies."""

    def __init__(
        self,
        importer: FormatImporter,
        cleaner: PhysicalSetup,
        splitter: DatasetSplitter,
        normalizer: DatasetNormalizer,
        exporter: FormatExporter,
        modeler: Optional[DatasetModeler] = None,
    ) -> None:
        """Initialize DatasetEngine with plug-and-play strategy components.

        Parameters
        ----------
        importer : FormatImporter
            Strategy for loading raw simulation files into memory.
        cleaner : PhysicalSetup
            Strategy for cleaning, coordinate alignment, and tensor structuring.
        splitter : DatasetSplitter
            Component for partitioning into train, val, and test subsets.
        normalizer : DatasetNormalizer
            Component for calculating zero-leakage statistical moments.
        exporter : FormatExporter
            Strategy for writing formatted archives and metadata to disk.
        modeler : Optional[DatasetModeler], optional
            Strategy or composite chain for task-specific dataset formulation.
        """
        self.importer = importer
        self.cleaner = cleaner
        self.splitter = splitter
        self.normalizer = normalizer
        self.exporter = exporter
        self.modeler = modeler

    def process(
        self,
        raw_path: Union[str, Path],
        output_dir: Union[str, Path],
        split_ratios: Tuple[int, int, int],
    ) -> Dict[str, Path]:
        """Execute the end-to-end dataset preprocessing pipeline.

        Parameters
        ----------
        raw_path : Union[str, Path]
            Path to the raw simulation file.
        output_dir : Union[str, Path]
            Destination directory for processed dataset files.
        split_ratios : Tuple[int, int, int]
            Counts for (train_n, val_n, test_n).

        Returns
        -------
        Dict[str, Path]
            Dictionary mapping artifact names to their written file paths.
        """
        train_n, val_n, test_n = split_ratios

        # 1. Ingest raw format into memory
        raw_data = self.importer.load(raw_path)

        # 2. Extract and structure physical tensors
        fields = self.cleaner.setup(raw_data)

        # 3. Apply dataset modification (single modeler or CompositeModeler chain)
        if self.modeler is not None:
            fields = self.modeler.modify(fields)

        # 4. Partition into Train / Validation / Test
        splits = self.splitter.split(fields, train_n, val_n, test_n)

        # 5. Compute normalization moments (zero data leakage)
        statistical_data = self.normalizer.compute_statistical_data(splits["train"])

        # 6. Export partitions and scaling parameters to disk
        saved_paths: List[Path] = self.exporter.save(splits, statistical_data, output_dir)

        # Return mapping of artifact names to paths
        return {p.stem: p for p in saved_paths}
