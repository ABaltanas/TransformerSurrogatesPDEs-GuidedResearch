"""Training script for the time-conditioned SMART surrogate model on ds_plasticity_temporal.

Utilizes the modular TrainingEngine architecture and component strategies from src.training:
- SMARTBatchLoader for geometry, surface/volume partitioning, and time parameter conditioning
- SMARTModelWrapper housing the SMART architecture with parameter_channels=1
- RelativeL2LossEngine for relative Frobenius L2 error minimization
- StandardOptimizer managing AdamW and CosineAnnealing scheduling
- TorchCheckpointHandler for atomic disk persistence
- CliProgressMonitor and JsonMetricsLogger for real-time progress and telemetry
- TrainingTelemetryAnalyzer for post-training telemetry analysis and KPI curves
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from src.training import (
    CliProgressMonitor,
    CompositeTelemetryLogger,
    JsonMetricsLogger,
    RelativeL2LossEngine,
    SMARTBatchLoader,
    SMARTModelWrapper,
    StandardOptimizer,
    TorchCheckpointHandler,
    TrainingEngine,
)


class PlasticityPartitionDataset(Dataset):
    """PyTorch Dataset loading simulation samples from a preprocessed .npz archive.

    Parameters
    ----------
    npz_path : Path
        Path to train.npz, val.npz, or test.npz.
    """

    def __init__(self, npz_path: Path) -> None:
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Partition archive not found at: {self.npz_path}")

        data = np.load(self.npz_path)
        self.die_boundary: np.ndarray = data["die_boundary"]
        self.coords_initial: np.ndarray = data["coords_initial"]
        self.displacements: np.ndarray = data["displacements"]
        self.time_params: Optional[np.ndarray] = (
            data["time_params"] if "time_params" in data else None
        )
        self.num_samples: int = self.die_boundary.shape[0]

    def __len__(self) -> int:
        """Return the number of simulation samples in this partition."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Retrieve a single simulation sample dictionary.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing 'die_boundary', 'coords_initial', 'displacements',
            and optional 'time_params'.
        """
        item = {
            "die_boundary": self.die_boundary[idx],
            "coords_initial": self.coords_initial[idx],
            "displacements": self.displacements[idx],
        }
        if self.time_params is not None:
            item["time_params"] = self.time_params[idx]
        return item


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for temporal SMART training."""
    parser = argparse.ArgumentParser(
        description="Train time-conditioned SMART surrogate on the 2D plasticity trajectory benchmark."
    )
    # Data paths
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ds_plasticity_temporal",
        help="Directory containing train.npz, val.npz, and statistical_data.json.",
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "smart_temporal_experiment_1",
        help="Parent directory where checkpoints and training logs will be saved.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "TrainingAnalysis",
        help="Base directory where training analysis and KPI charts will be saved.",
    )
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per training step.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau", "none"],
        help="LR scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm threshold."
    )
    # SMART Architecture hyperparameters
    parser.add_argument("--parameter_channels", type=int, default=1, help="Conditioning parameter channels (time t).")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent feature dimension.")
    parser.add_argument("--latent_geo_points", type=int, default=64, help="Points in coarse latent geometry.")
    parser.add_argument("--subsampled_points", type=int, default=101, help="Points in subsampled geometry attention.")
    parser.add_argument("--num_blocks", type=int, default=6, help="Number of encoder-decoder transformer blocks.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument(
        "--use_amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic mixed precision (AMP FP16) for GPU training.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest.pt",
        default=None,
        help="Resume training from checkpoint (e.g. latest.pt, checkpoint_epoch_0047.pt).",
    )
    # Execution setup
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target device."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    return parser.parse_args()


def main() -> None:
    """Execute time-conditioned surrogate model training workflow."""
    args = parse_args()

    # 1. Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    print("=" * 80)
    print("SMART SURROGATE MODEL TRAINING: 2D PLASTICITY TEMPORAL TRAJECTORY")
    print(f"Target Hardware Device: {device}")
    print(f"Data Directory:         {args.data_dir}")
    print(f"Models Save Directory:  {args.models_dir}")
    print(f"Conditioning Params:    parameter_channels={args.parameter_channels} (time t in [0, 1])")
    print(f"Epochs:                 {args.epochs} (Batch Size: {args.batch_size})")
    print(f"Mixed Precision (AMP):  {args.use_amp}")
    print("=" * 80)

    # 2. Ingest Normalization Moments
    stats_path = args.data_dir / "statistical_data.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing statistical_data.json at {stats_path}")
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    # 3. Create Datasets and DataLoaders
    train_npz = args.data_dir / "train.npz"
    val_npz = args.data_dir / "val.npz"

    train_dataset = PlasticityPartitionDataset(train_npz)
    val_dataset = PlasticityPartitionDataset(val_npz)

    batch_loader = SMARTBatchLoader(stats=stats)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=batch_loader.loadDataset,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=batch_loader.loadDataset,
        num_workers=0,
    )

    print(f"Loaded Train Partition: {len(train_dataset)} instances ({len(train_loader)} batches)")
    print(f"Loaded Val Partition:   {len(val_dataset)} instances ({len(val_loader)} batches)")

    # 4. Instantiate SMART ModelWrapper with parameter_channels=1
    smart_config = {
        "spatial_dim": 2,
        "surface_channels": 2,
        "volume_channels": 2,
        "parameter_channels": args.parameter_channels,
        "latent_dim": args.latent_dim,
        "latent_geometry_points": args.latent_geo_points,
        "subsampled_geometry_points": args.subsampled_points,
        "num_encoder_decoder_blocks": args.num_blocks,
        "num_heads": args.num_heads,
        "pos_scale_factor": 1000.0,
        "dropout": 0.0,
    }

    model_wrapper = SMARTModelWrapper(config=smart_config)
    torch_module = model_wrapper.get_torch_module()
    total_params = sum(p.numel() for p in torch_module.parameters())
    print(f"Initialized SMART Model: {total_params:,} parameters (parameter_channels={args.parameter_channels})")

    # 5. Build Training Components
    loss_engine = RelativeL2LossEngine()

    scheduler_type = None if args.scheduler == "none" else args.scheduler
    optimizer = StandardOptimizer.create_adamw(
        model_parameters=torch_module.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_type=scheduler_type,
        epochs=args.epochs,
        use_amp=args.use_amp,
    )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = TorchCheckpointHandler(checkpoint_dir=args.models_dir, keep_last_n=3)

    # State restoration if resuming
    start_epoch = 1
    best_val_loss = None

    if args.resume is not None:
        ckpt_target = Path(args.resume)
        if not ckpt_target.is_absolute() and not ckpt_target.exists():
            ckpt_target = args.models_dir / ckpt_target

        if not ckpt_target.exists():
            latest_found = checkpointer.get_latest_checkpoint_path()
            if latest_found is not None and latest_found.exists():
                ckpt_target = latest_found
            else:
                raise FileNotFoundError(f"Resume checkpoint not found at: {ckpt_target}")

        print(f"\nResuming training from checkpoint: {ckpt_target}")
        loaded_payload = checkpointer.load_checkpoint(ckpt_target, torch_module)
        resumed_epoch = loaded_payload.get("epoch", 0)
        start_epoch = resumed_epoch + 1

        optimizer.load_state(loaded_payload, step_scheduler_to_epoch=resumed_epoch)
        best_val_loss = checkpointer.get_best_val_loss()

        print(f"Resumed at Starting Epoch: {start_epoch} / {args.epochs}")
        print(f"Active Learning Rate:     {optimizer.get_learning_rate():.4e}")
        print(f"Historical Best Val Loss: {best_val_loss:.4e}\n")

        if start_epoch > args.epochs:
            print(
                f"Checkpoint is already at epoch {resumed_epoch}, which meets or exceeds "
                f"target epochs {args.epochs}. Nothing to train."
            )
            return

    cli_logger = CliProgressMonitor(batch_size=args.batch_size)
    json_logger = JsonMetricsLogger(log_file=args.models_dir / "training_metrics.jsonl")
    composite_logger = CompositeTelemetryLogger([cli_logger, json_logger])

    # 6. Initialize Orchestrator
    engine = TrainingEngine(
        model_wrapper=model_wrapper,
        batch_loader=batch_loader,
        loss_engine=loss_engine,
        optimizer=optimizer,
        checkpointer=checkpointer,
        logger=composite_logger,
        use_amp=args.use_amp,
    )

    # 7. Execute Training Loop
    summary = engine.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        device=device,
        max_grad_norm=args.max_grad_norm,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
    )

    print("\nTraining completed successfully!")
    print(f"Best Validation Loss: {summary['best_val_loss']:.4e}")
    print(f"Best Checkpoint:      {summary['best_checkpoint']}")

    # 8. Post-Training Telemetry Analysis & KPI Chart Generation
    from src.evaluation.training_history import TrainingTelemetryAnalyzer

    raw_lr_str = f"{args.lr:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    lr_str = raw_lr_str.replace(".0e", "e") if ".0e" in raw_lr_str else raw_lr_str
    training_params = f"epochs_{args.epochs}_batch_{args.batch_size}_lr_{lr_str}"
    dataset_name = args.data_dir.name
    experiment_name = args.models_dir.name

    analysis_output_dir = (
        args.results_dir
        / dataset_name
        / "SMART"
        / training_params
        / experiment_name
    )
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = args.models_dir / "training_metrics.jsonl"
    if metrics_file.exists():
        print(f"\nAnalyzing training dynamics and generating KPI charts in {analysis_output_dir}...")
        analyzer = TrainingTelemetryAnalyzer()
        analysis_result = analyzer.analyze_and_export(
            metrics_file=metrics_file,
            output_dir=analysis_output_dir,
        )
        kpis = analysis_result["training_kpis"]
        print(f"Optimal Checkpoint Epoch: Epoch {kpis['best_epoch']} (Val Rel-L2: {kpis['best_val_loss']:.4e})")
        print(f"Total Val Loss Reduction: {kpis['val_loss_reduction_pct']:.1f}%")
        print(f"Training Artifacts Saved: {analysis_output_dir}")


if __name__ == "__main__":
    main()
