"""Runner script for the FP32 Fine-Tuning experiment (+50 epochs, completing 200).

Loads model weights from the pristine Epoch 149 checkpoint (best_model.pt) of the 150-epoch LION run
(models/LION/batch_size_16_fp32_swap_test). Executes fine-tuning in a dedicated folder
(models/LION/batch_size_16_fp32_ep200_finetune) with:
- Optimizer: LION (pure FP32, --no-use_amp)
- Batch size: 16
- Learning Rate: 1.0e-5 constant (default) or cosine annealing
- Historical continuity: seeds training_metrics.jsonl with epochs 1-150 and copies initial best_model.pt benchmark
- Evaluation: automatically benchmarks best_model.pt on the test partition upon completion
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_lion_batch_study import generate_benchmark_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FP32 Fine-Tuning (+50 epochs -> 200 total).")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5).")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "cosine"],
        help="Scheduler type ('none' for constant LR, 'cosine' for annealing).",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate if using cosine scheduler (default: 1e-6).",
    )
    parser.add_argument(
        "--models_dir_name",
        type=str,
        default="batch_size_16_fp32_ep200_finetune",
        help="Subdirectory name under models/LION/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    src_dir = PROJECT_ROOT / "models" / "LION" / "batch_size_16_fp32_swap_test"
    target_dir = PROJECT_ROOT / "models" / "LION" / args.models_dir_name
    results_dir = PROJECT_ROOT / "results" / "TrainingAnalysis"
    data_dir = PROJECT_ROOT / "data" / "processed" / "ds_plasticity_temporal"

    target_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SMART SURROGATE MODEL: FINE-TUNING (+50 EPOCHS -> 200 TOTAL)")
    print(f"Source Model Directory (Baseline Ep 149): {src_dir}")
    print(f"Target Model Directory (Isolated Run):    {target_dir}")
    print(f"Learning Rate Configuration:              LR={args.lr:.2e} (Scheduler: {args.scheduler})")
    print("=" * 80)

    src_best = src_dir / "best_model.pt"
    if not src_best.exists():
        raise FileNotFoundError(f"Source checkpoint best_model.pt not found at: {src_best}")

    target_best = target_dir / "best_model.pt"
    target_latest = target_dir / "latest.pt"

    # Seed target directory with initial best_model.pt if not present
    if not target_best.exists():
        print(f"[SETUP] Copying initial baseline {src_best} -> {target_best}...")
        shutil.copy2(src_best, target_best)

    # Seed training_metrics.jsonl with clean lines 1..150 if not already present
    target_metrics = target_dir / "training_metrics.jsonl"
    if not target_metrics.exists():
        src_metrics = src_dir / "training_metrics.jsonl"
        if src_metrics.exists():
            print(f"[SETUP] Seeding {target_metrics} with epochs 1-150 historical metrics...")
            shutil.copy2(src_metrics, target_metrics)

    train_script = PROJECT_ROOT / "scripts" / "train_smart_temporal.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--data_dir", str(data_dir),
        "--models_dir", str(target_dir),
        "--results_dir", str(results_dir),
        "--epochs", "200",
        "--batch_size", "16",
        "--lr", str(args.lr),
        "--min_lr", str(args.min_lr),
        "--weight_decay", "1e-4",
        "--scheduler", args.scheduler,
        "--optimizer", "lion",
        "--max_grad_norm", "1.0",
        "--parameter_channels", "1",
        "--latent_dim", "256",
        "--latent_geo_points", "101",
        "--subsampled_points", "101",
        "--num_blocks", "8",
        "--num_heads", "8",
        "--pos_scale_factor", "1000.0",
        "--dropout", "0.0",
        "--device", "cuda",
        "--seed", "42",
        "--no-use_amp",
    ]

    # If already partially trained in target_dir, resume from latest.pt
    if target_latest.exists():
        print(f"[RESUME] Found existing fine-tuning checkpoint at {target_latest}; resuming...")
        cmd.extend(["--resume", str(target_latest)])
    else:
        cmd.extend([
            "--pretrained_weights", str(src_best),
            "--start_epoch", "151",
        ])

    log_path = target_dir / "training.log"
    print(f"\n[EXEC] Running Fine-Tuning in pure FP32 (no-use_amp)...")
    print(f"       Command: {' '.join(cmd)}\n")

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
        if proc.returncode != 0:
            print(f"[ERROR] Fine-tuning failed with return code {proc.returncode}")
            sys.exit(proc.returncode)

    # Evaluation on test benchmark
    best_model = target_dir / "best_model.pt"
    if best_model.exists():
        print(f"\n[EVAL] Running Test Partition Benchmark Evaluation for {target_dir.name}...")
        eval_script = PROJECT_ROOT / "scripts" / "evaluate_model.py"
        eval_cmd = [
            sys.executable,
            str(eval_script),
            "--model_dir", str(target_dir),
            "--checkpoint_name", "best_model.pt",
            "--data_dir", str(data_dir),
            "--results_dir", str(results_dir),
            "--latent_dim", "256",
            "--latent_geo_points", "101",
            "--subsampled_points", "101",
            "--num_blocks", "8",
            "--num_heads", "8",
            "--batch_size", "32",
            "--device", "cuda",
        ]
        res = subprocess.run(eval_cmd, cwd=str(PROJECT_ROOT), text=True)
        if res.returncode == 0:
            print(f"[EVAL] Benchmark evaluation completed successfully!")
        else:
            print(f"[EVAL-ERROR] Evaluation failed with return code {res.returncode}")

    # Leaderboard summary update
    models_base_dir = PROJECT_ROOT / "models" / "LION"
    try:
        generate_benchmark_summary([32, 16, 8, 1], models_base_dir, results_dir)
    except Exception as e:
        print(f"[WARN] Failed to update benchmark summary leaderboard: {e}")

    print("\n" + "=" * 80)
    print("FINE-TUNING ANNEALING EXPERIMENT (+50 EPOCHS) COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
