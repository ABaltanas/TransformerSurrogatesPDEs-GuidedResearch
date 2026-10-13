"""Command-line script to preprocess raw simulation datasets using DatasetEngine.

Usage:
------
python scripts/preprocess_dataset.py \
    --raw_path data/raw/plasticity-V1/plasticity/plas_N987_T20.mat \
    --output_dir data/processed/plasticity \
    --train_count 900 \
    --val_count 7 \
    --test_count 80
"""

import argparse
from pathlib import Path
import sys
from timeit import default_timer

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data import (
    ConsiderOnlyFinalStepModeler,
    DatasetEngine,
    DatasetNormalizer,
    DatasetSplitter,
    MatlabImporter,
    NpzExporter,
    PlasticitySetup,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for dataset preprocessing."""
    parser = argparse.ArgumentParser(
        description="Dataset preprocessing engine runner using DatasetEngine."
    )
    parser.add_argument(
        "--raw_path",
        type=str,
        default="data/raw/plasticity-V1/plasticity/plas_N987_T20.mat",
        help="Path to raw source file or directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/plasticity",
        help="Directory to save clean processed dataset files.",
    )
    parser.add_argument(
        "--train_count",
        type=int,
        default=900,
        help="Number of training samples (default: 900).",
    )
    parser.add_argument(
        "--val_count",
        type=int,
        default=7,
        help="Number of validation samples (default: 7).",
    )
    parser.add_argument(
        "--test_count",
        type=int,
        default=80,
        help="Number of testing samples (default: 80).",
    )
    parser.add_argument(
        "--final_step_only",
        action="store_true",
        help="Model static steady state at final increment (t = -1), dropping time dimension.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute preprocessing pipeline and print execution summary."""
    args = parse_arguments()

    print("=" * 70)
    print("           DATASET PREPROCESSING ENGINE RUNNER           ")
    print("=" * 70)
    print(f"Raw Source Path:  {args.raw_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Split Allocation: Train={args.train_count}, Val={args.val_count}, Test={args.test_count}")
    print(f"Final Step Only:  {args.final_step_only}")
    print("-" * 70)

    start_time: float = default_timer()

    try:
        # Instantiate plug-and-play strategies
        importer = MatlabImporter()
        setup = PlasticitySetup()
        splitter = DatasetSplitter()
        normalizer = DatasetNormalizer()
        exporter = NpzExporter()
        modeler = ConsiderOnlyFinalStepModeler() if args.final_step_only else None

        # Build DatasetEngine orchestrator
        engine = DatasetEngine(
            importer=importer,
            cleaner=setup,
            splitter=splitter,
            normalizer=normalizer,
            exporter=exporter,
            modeler=modeler,
        )

        print("[*] Running DatasetEngine pipeline...")
        saved_artifacts = engine.process(
            raw_path=args.raw_path,
            output_dir=args.output_dir,
            split_ratios=(args.train_count, args.val_count, args.test_count),
        )

        elapsed: float = default_timer() - start_time
        print("=" * 70)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print(f"Total Time:       {elapsed:.2f} seconds")
        print("Generated Artifacts:")
        for key, path in saved_artifacts.items():
            file_size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  - {key:<20}: {str(path)} ({file_size_mb:.2f} MB)")
        print("=" * 70)

    except Exception as exc:
        print(f"ERROR: Dataset preprocessing failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
