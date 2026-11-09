"""Runner script for the extended Lion experiment: batch_size=16 with epochs=150 and lr=1e-4.

Executes the 150-epoch training run in models/LION/batch_size_16_epochs_150, evaluates on the
test partition upon completion, and updates the cross-experiment benchmark summary.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_lion_batch_study import generate_benchmark_summary


def main() -> None:
    models_dir = PROJECT_ROOT / "models" / "LION" / "batch_size_16_epochs_150"
    results_dir = PROJECT_ROOT / "results" / "TrainingAnalysis"
    data_dir = PROJECT_ROOT / "data" / "processed" / "ds_plasticity_temporal"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXTENDED EXPERIMENT: SMART TEMPORAL WITH BATCH_SIZE=16, EPOCHS=150, LR=1e-4")
    print(f"Target Models Directory:  {models_dir}")
    print(f"Target Results Directory: {results_dir}")
    print("=" * 80)

    train_script = PROJECT_ROOT / "scripts" / "train_smart_temporal.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--data_dir", str(data_dir),
        "--models_dir", str(models_dir),
        "--results_dir", str(results_dir),
        "--epochs", "150",
        "--batch_size", "16",
        "--lr", "0.0001",
        "--weight_decay", "0.0001",
        "--scheduler", "warmup_cosine",
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
        "--use_amp",
    ]

    latest_ckpt = models_dir / "latest.pt"
    if latest_ckpt.exists():
        cmd.extend(["--resume", str(latest_ckpt)])

    log_path = models_dir / "training.log"
    print(f"\n[EXEC] Running Training for batch_size=16, epochs=150, lr=1e-4...")
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
            print(f"[ERROR] Training failed with return code {proc.returncode}")
            sys.exit(proc.returncode)

    # Evaluation
    best_model = models_dir / "best_model.pt"
    if best_model.exists():
        print(f"\n[EVAL] Running Test Evaluation for batch_size=16, epochs=150, lr=1e-4...")
        eval_script = PROJECT_ROOT / "scripts" / "evaluate_model.py"
        eval_cmd = [
            sys.executable,
            str(eval_script),
            "--model_dir", str(models_dir),
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
        subprocess.run(eval_cmd, cwd=str(PROJECT_ROOT), text=True)

    # Update Summary Leaderboard
    models_base_dir = PROJECT_ROOT / "models" / "LION"
    generate_benchmark_summary([32, 16, 8, 1], models_base_dir, results_dir)
    print("\n" + "=" * 80)
    print("EXTENDED EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
