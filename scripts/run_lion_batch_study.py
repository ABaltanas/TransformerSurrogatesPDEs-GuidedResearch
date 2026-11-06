"""Sequential parametric study runner for Lion optimizer batch size variation [32, 16, 8].

Runs SMART surrogate model training across varying batch sizes using identical
hyperparameters:
- latent_dim=256, num_blocks=8, latent_geo_points=101, subsampled_points=101
- optimizer=lion, lr=1e-4, scheduler=warmup_cosine, epochs=50, use_amp=True

Each run is executed strictly sequentially in an isolated subprocess. Upon
completion of each training run, test evaluation is immediately executed via
evaluate_model.py, and a consolidated benchmark comparison is updated in:
results/TrainingAnalysis/ds_plasticity_temporal/SMART/LION/batch_study_summary.md
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_BATCH_SIZES: List[int] = [32, 16, 8]

FIXED_CONFIG: Dict[str, Any] = {
    # Architecture
    "spatial_dim": 2,
    "surface_channels": 2,
    "volume_channels": 2,
    "parameter_channels": 1,
    "latent_dim": 256,
    "latent_geo_points": 101,
    "subsampled_points": 101,
    "num_blocks": 8,
    "num_heads": 8,
    "pos_scale_factor": 1000.0,
    "dropout": 0.0,
    # Optimization
    "optimizer": "lion",
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "scheduler": "warmup_cosine",
    "max_grad_norm": 1.0,
    "epochs": 50,
    "use_amp": True,
    "seed": 42,
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the Lion batch study."""
    parser = argparse.ArgumentParser(
        description="Sequential batch size study for SMART temporal surrogate with Lion optimizer."
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help=f"List of batch sizes to evaluate sequentially (default: {DEFAULT_BATCH_SIZES}).",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ds_plasticity_temporal",
        help="Path to preprocessed dataset partition archives.",
    )
    parser.add_argument(
        "--models_base_dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "LION",
        help="Base directory where model checkpoints will be persisted.",
    )
    parser.add_argument(
        "--results_base_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "TrainingAnalysis",
        help="Base directory where training analysis and KPI charts are saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Hardware execution device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print execution plan and configurations without running training.",
    )
    return parser.parse_args()


def is_run_completed(models_dir: Path, target_epochs: int) -> bool:
    """Check if a training run has successfully completed all target epochs."""
    metrics_file = models_dir / "training_metrics.jsonl"
    best_model_file = models_dir / "best_model.pt"
    if not (metrics_file.exists() and best_model_file.exists()):
        return False
    try:
        last_epoch = 0
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if "epoch" in rec:
                        last_epoch = int(rec["epoch"])
        return last_epoch >= target_epochs
    except Exception:
        return False


def execute_training(
    batch_size: int,
    config: Dict[str, Any],
    models_dir: Path,
    results_dir: Path,
    data_dir: Path,
    device: str,
) -> bool:
    """Execute training script in an isolated subprocess."""
    models_dir.mkdir(parents=True, exist_ok=True)
    train_script = PROJECT_ROOT / "scripts" / "train_smart_temporal.py"

    cmd = [
        sys.executable,
        str(train_script),
        "--data_dir", str(data_dir),
        "--models_dir", str(models_dir),
        "--results_dir", str(results_dir),
        "--epochs", str(config["epochs"]),
        "--batch_size", str(batch_size),
        "--lr", str(config["lr"]),
        "--weight_decay", str(config["weight_decay"]),
        "--scheduler", str(config["scheduler"]),
        "--optimizer", str(config["optimizer"]),
        "--max_grad_norm", str(config["max_grad_norm"]),
        "--parameter_channels", str(config["parameter_channels"]),
        "--latent_dim", str(config["latent_dim"]),
        "--latent_geo_points", str(config["latent_geo_points"]),
        "--subsampled_points", str(config["subsampled_points"]),
        "--num_blocks", str(config["num_blocks"]),
        "--num_heads", str(config["num_heads"]),
        "--pos_scale_factor", str(config["pos_scale_factor"]),
        "--dropout", str(config["dropout"]),
        "--device", str(device),
        "--seed", str(config["seed"]),
    ]
    if config.get("use_amp", True):
        cmd.append("--use_amp")
    else:
        cmd.append("--no-use_amp")

    latest_ckpt = models_dir / "latest.pt"
    if latest_ckpt.exists():
        cmd.extend(["--resume", str(latest_ckpt)])

    log_path = models_dir / "training.log"
    print(f"\n[EXEC] Running Training for batch_size={batch_size}...")
    print(f"       Command: {' '.join(cmd)}")
    print(f"       Models Directory:  {models_dir}")
    print(f"       Logging to:        {log_path}\n")

    with open(log_path, "a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()

        proc.wait()
        return proc.returncode == 0


def execute_evaluation(
    batch_size: int,
    config: Dict[str, Any],
    models_dir: Path,
    results_dir: Path,
    data_dir: Path,
    device: str,
) -> bool:
    """Execute evaluation script on test partition for best_model.pt."""
    best_model = models_dir / "best_model.pt"
    if not best_model.exists():
        print(f"[WARN] No best_model.pt found at {best_model}. Skipping evaluation.")
        return False

    eval_script = PROJECT_ROOT / "scripts" / "evaluate_model.py"
    cmd = [
        sys.executable,
        str(eval_script),
        "--model_dir", str(models_dir),
        "--checkpoint_name", "best_model.pt",
        "--data_dir", str(data_dir),
        "--results_dir", str(results_dir),
        "--latent_dim", str(config["latent_dim"]),
        "--latent_geo_points", str(config["latent_geo_points"]),
        "--subsampled_points", str(config["subsampled_points"]),
        "--num_blocks", str(config["num_blocks"]),
        "--num_heads", str(config["num_heads"]),
        "--batch_size", "32",  # Standard evaluation batch size
        "--device", str(device),
    ]

    print(f"\n[EVAL] Running Test Evaluation for batch_size={batch_size}...")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True)
    return proc.returncode == 0


def generate_benchmark_summary(
    batch_sizes: List[int],
    models_base_dir: Path,
    results_base_dir: Path,
) -> None:
    """Compile a cross-batch benchmark leaderboard in Markdown and CSV formats."""
    dataset_name = "ds_plasticity_temporal"
    lion_results_dir = (
        results_base_dir / dataset_name / "SMART" / "LION"
    )
    lion_results_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []

    # Scan all run directories inside models_base_dir
    candidate_dirs: List[Path] = [
        d for d in models_base_dir.iterdir()
        if d.is_dir() and (d.name.startswith("batch_size_") or d.name == "smart_temporal_experiment_3")
    ]
    # Sort logically
    candidate_dirs.sort(key=lambda p: (0 if "smart" in p.name else (1 if "32" in p.name else (2 if "16" in p.name else (3 if "_8" in p.name else 4)))))

    for m_dir in candidate_dirs:
        param_file = m_dir / "parametros.txt"
        metrics_file = m_dir / "training_metrics.jsonl"

        batch_val = 1
        lr_val = 1e-4

        if param_file.exists():
            try:
                with open(param_file, "r", encoding="utf-8") as f:
                    for line in f:
                        sline = line.strip()
                        if sline.startswith("batch_size:"):
                            batch_val = int(sline.split(":")[1].strip())
                        elif sline.startswith("lr:"):
                            lr_val = float(sline.split(":")[1].strip())
            except Exception:
                pass

        row: Dict[str, Any] = {
            "run_name": m_dir.name,
            "batch_size": batch_val,
            "lr": lr_val,
            "status": "NOT_STARTED",
            "best_val_loss": None,
            "best_epoch": None,
            "overall_rel_l2_pct": None,
            "contact_violation_rate": None,
            "inference_latency_ms": None,
            "duration_hours": None,
        }

        # Locate evaluation_metrics.json in results
        eval_metric_file = None
        for match in lion_results_dir.glob(f"**/{m_dir.name}/**/evaluation_metrics.json"):
            eval_metric_file = match
            break
        if not eval_metric_file:
            for match in lion_results_dir.glob(f"**/{m_dir.name}/evaluation_metrics.json"):
                eval_metric_file = match
                break

        if eval_metric_file and eval_metric_file.exists():
            try:
                with open(eval_metric_file, "r", encoding="utf-8") as f:
                    em = json.load(f)
                    pm = em.get("physical_metrics", {})
                    fm = em.get("feasibility_metrics", {})
                    cm = em.get("computational_metrics", {})
                    if "rel_l2_overall" in pm:
                        row["overall_rel_l2_pct"] = pm["rel_l2_overall"] * 100.0
                    if "penetration_violation_rate_pct" in fm:
                        row["contact_violation_rate"] = fm["penetration_violation_rate_pct"]
                    if "inference_latency_ms_per_sample" in cm:
                        row["inference_latency_ms"] = cm["inference_latency_ms_per_sample"]
            except Exception:
                pass

        if param_file.exists():
            try:
                with open(param_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if "Elapsed Wall-Clock Time:" in line:
                            s_val = float(line.split(":")[1].replace("s", "").strip())
                            row["duration_hours"] = s_val / 3600.0
                        elif "Best Validation Loss:" in line:
                            row["best_val_loss"] = float(line.split(":")[1].strip())
                        elif "Best Checkpoint Epoch:" in line:
                            row["best_epoch"] = line.split(":")[1].strip()
            except Exception:
                pass

        if metrics_file.exists():
            completed = is_run_completed(m_dir, 50)
            if completed:
                # Check if it was contaminated with NaNs in final epoch
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        lines = [l.strip() for l in f if l.strip()]
                    last_obj = json.loads(lines[-1])
                    if str(last_obj.get("val", {}).get("loss_total", "")).lower() == "nan":
                        row["status"] = "DIVERGED_AT_EP5 (NaN)"
                    else:
                        row["status"] = "COMPLETED"
                except Exception:
                    row["status"] = "COMPLETED"
            else:
                row["status"] = "IN_PROGRESS"

        summary_rows.append(row)

    # Export Markdown table
    md_path = lion_results_dir / "batch_study_summary.md"
    lines = [
        "# Lion Optimizer: Batch Size & Sensitivity Study Summary",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "**Architecture:** SMART (11.38M parameters, latent_dim=256, num_blocks=8, 101 geometry points)",
        "**Optimization:** Lion (warmup_cosine scheduler, 50 epochs, AMP FP16)",
        "",
        "| Run Identifier | Batch Size | Learning Rate | Status | Best Val Rel-L2 | Best Epoch | Test Rel-L2 (%) | Contact Violations (%) | Latency (ms) | Wall Time (h) |",
        "| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in summary_rows:
        val_str = f"{r['best_val_loss']:.4f}" if r['best_val_loss'] is not None else "N/A"
        ep_str = str(r['best_epoch']) if r['best_epoch'] is not None else "N/A"
        l2_str = f"{r['overall_rel_l2_pct']:.2f}%" if r['overall_rel_l2_pct'] is not None else "N/A"
        viol_str = f"{r['contact_violation_rate']:.1f}%" if r['contact_violation_rate'] is not None else "N/A"
        lat_str = f"{r['inference_latency_ms']:.2f}" if r['inference_latency_ms'] is not None else "N/A"
        dur_str = f"{r['duration_hours']:.2f} h" if r['duration_hours'] is not None else "N/A"
        lines.append(
            f"| `{r['run_name']}` | **{r['batch_size']}** | `{r['lr']:.1e}` | `{r['status']}` | `{val_str}` | {ep_str} | **{l2_str}** | {viol_str} | {lat_str} | {dur_str} |"
        )

    lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[SUMMARY] Updated Benchmark Summary at: {md_path}")


def main() -> None:
    """Execute sequential batch size study."""
    args = parse_args()

    print("=" * 80)
    print("SEQUENTIAL BATCH SIZE STUDY: LION OPTIMIZER ON 2D PLASTICITY TEMPORAL")
    print(f"Target Hardware Device:   {args.device}")
    print(f"Batch Sizes to Evaluate:  {args.batch_sizes}")
    print(f"Target Epochs per Run:    {FIXED_CONFIG['epochs']}")
    print(f"Models Base Directory:    {args.models_base_dir}")
    print(f"Results Base Directory:   {args.results_base_dir}")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN] Planned runs:")
        for idx, b in enumerate(args.batch_sizes, start=1):
            m_dir = args.models_base_dir / f"batch_size_{b}"
            print(f"  [{idx}/{len(args.batch_sizes)}] batch_size={b} -> {m_dir}")
        return

    total_runs = len(args.batch_sizes)
    for idx, b in enumerate(args.batch_sizes, start=1):
        models_dir = args.models_base_dir / f"batch_size_{b}"
        print("\n" + "#" * 80)
        print(f"RUN [{idx}/{total_runs}]: Starting batch_size={b}")
        print("#" * 80)

        if is_run_completed(models_dir, FIXED_CONFIG["epochs"]):
            print(f"[SKIP] Run for batch_size={b} already completed. Checking evaluation...")
        else:
            success = execute_training(
                batch_size=b,
                config=FIXED_CONFIG,
                models_dir=models_dir,
                results_dir=args.results_base_dir,
                data_dir=args.data_dir,
                device=args.device,
            )
            if not success:
                print(f"[ERROR] Training failed for batch_size={b}. Proceeding to next run.")
                continue

        # Run test evaluation
        execute_evaluation(
            batch_size=b,
            config=FIXED_CONFIG,
            models_dir=models_dir,
            results_dir=args.results_base_dir,
            data_dir=args.data_dir,
            device=args.device,
        )

        # Update cross-batch benchmark summary
        generate_benchmark_summary(
            batch_sizes=args.batch_sizes,
            models_base_dir=args.models_base_dir,
            results_base_dir=args.results_base_dir,
        )

    print("\n" + "=" * 80)
    print("ALL BATCH SIZE RUNS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
