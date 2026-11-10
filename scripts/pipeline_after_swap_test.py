"""Automated experiment queue pipeline.

1. Monitors the active 150-epoch FP32 training in models/LION/batch_size_16_fp32_swap_test.
2. Upon its completion (~06:15 AM), executes full test benchmark evaluation via evaluate_model.py.
3. Automatically launches the user-requested batch_size=1 FP32 experiment:
   - batch_size=1
   - lr=1e-4
   - scheduler=warmup_cosine
   - optimizer=lion
   - epochs=25
   - no-use_amp (pure FP32)
   - models_dir=models/LION/batch_size_1_fp32_ep50
4. Upon completion of that run, executes evaluate_model.py and updates the benchmark summary.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_lion_batch_study import generate_benchmark_summary


def is_active_training_running(target_str: str) -> bool:
    """Check if a training process with target_str in its arguments is currently active."""
    try:
        output = subprocess.check_output(["pgrep", "-f", target_str], text=True)
        # Exclude this script itself if it matches
        my_pid = os.getpid()
        pids = [int(p) for p in output.strip().split() if p.isdigit() and int(p) != my_pid]
        return len(pids) > 0
    except subprocess.CalledProcessError:
        return False


def run_evaluation(model_dir: Path, data_dir: Path, results_dir: Path) -> None:
    """Execute evaluate_model.py for best_model.pt in model_dir."""
    best_model = model_dir / "best_model.pt"
    if not best_model.exists():
        print(f"[WARN] No best_model.pt found in {model_dir}; skipping evaluation.")
        return

    print(f"\n[EVAL] Running Benchmark Evaluation for {model_dir.name}...")
    eval_script = PROJECT_ROOT / "scripts" / "evaluate_model.py"
    eval_cmd = [
        sys.executable,
        str(eval_script),
        "--model_dir", str(model_dir),
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
        print(f"[EVAL] Evaluation completed successfully for {model_dir.name}!")
    else:
        print(f"[EVAL-ERROR] Evaluation failed with return code {res.returncode}")


def main() -> None:
    swap_model_dir = PROJECT_ROOT / "models" / "LION" / "batch_size_16_fp32_swap_test"
    next_model_dir = PROJECT_ROOT / "models" / "LION" / "batch_size_1_fp32_ep50"
    results_dir = PROJECT_ROOT / "results" / "TrainingAnalysis"
    data_dir = PROJECT_ROOT / "data" / "processed" / "ds_plasticity_temporal"

    pipeline_log = PROJECT_ROOT / "models" / "LION" / "pipeline_queue.log"
    print(f"[QUEUE] Automated Pipeline Active at {datetime.now(timezone.utc).isoformat()}")
    print(f"[QUEUE] Monitoring: {swap_model_dir.name}")
    print(f"[QUEUE] Next queued job: {next_model_dir.name} (batch_size=1, epochs=25, lr=1e-4, no-use_amp)\n")

    # 1. Wait for current FP32 training to finish
    check_interval = 30  # seconds
    target_match_str = "batch_size_16_fp32_swap_test"

    while is_active_training_running(target_match_str):
        time.sleep(check_interval)

    print(f"\n[QUEUE] Current training ({swap_model_dir.name}) has terminated!")
    time.sleep(5)  # grace period for filesystem flushes

    # 2. Evaluate completed 150-epoch FP32 model
    run_evaluation(swap_model_dir, data_dir, results_dir)

    # Update leaderboard
    models_base_dir = PROJECT_ROOT / "models" / "LION"
    try:
        generate_benchmark_summary([32, 16, 8, 1], models_base_dir, results_dir)
    except Exception as e:
        print(f"[WARN] Failed to update summary leaderboard: {e}")

    # 3. Launch next user-requested training
    next_model_dir.mkdir(parents=True, exist_ok=True)
    train_script = PROJECT_ROOT / "scripts" / "train_smart_temporal.py"

    next_cmd = [
        sys.executable,
        "-u",
        str(train_script),
        "--data_dir", str(data_dir),
        "--models_dir", str(next_model_dir),
        "--results_dir", str(results_dir),
        "--epochs", "25",
        "--batch_size", "1",
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
        "--no-use_amp",
    ]

    print(f"\n[QUEUE] Launching Next Experiment: batch_size=1, epochs=25, lr=1e-4, no-use_amp...")
    print(f"        Command: {' '.join(next_cmd)}\n")

    log_path = next_model_dir / "training.log"
    with open(log_path, "a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            next_cmd,
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
            print(f"[ERROR] Queued training failed with return code {proc.returncode}")
            sys.exit(proc.returncode)

    print(f"\n[QUEUE] Next training ({next_model_dir.name}) completed successfully!")

    # 4. Evaluate new model
    run_evaluation(next_model_dir, data_dir, results_dir)

    # Update leaderboard again
    try:
        generate_benchmark_summary([32, 16, 8, 1], models_base_dir, results_dir)
    except Exception as e:
        print(f"[WARN] Failed to update summary leaderboard: {e}")

    print("\n" + "=" * 80)
    print("ALL QUEUED EXPERIMENTS & BENCHMARKS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
