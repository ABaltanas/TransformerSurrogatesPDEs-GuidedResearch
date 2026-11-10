"""Runner script for the FP32 Precision Swap Test.

Resumes from the pristine Epoch 72 checkpoint (best_model.pt) of the 150-epoch Lion run,
executing in pure FP32 (--no-use_amp) to definitively verify whether the Epoch 75-78
divergence was caused by the FP16 dynamic range ceiling (65,504) or an algorithmic instability.
"""

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


def main() -> None:
    src_dir = PROJECT_ROOT / "models" / "LION" / "batch_size_16_epochs_150"
    target_dir = PROJECT_ROOT / "models" / "LION" / "batch_size_16_fp32_swap_test"
    results_dir = PROJECT_ROOT / "results" / "TrainingAnalysis"
    data_dir = PROJECT_ROOT / "data" / "processed" / "ds_plasticity_temporal"

    target_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FP32 PRECISION SWAP TEST: VERIFYING NUMERICAL RANGE VS. DIVERGENCE")
    print(f"Source Model Dir:  {src_dir}")
    print(f"Target Model Dir:  {target_dir}")
    print("=" * 80)

    src_best = src_dir / "best_model.pt"
    if not src_best.exists():
        raise FileNotFoundError(f"Source best_model.pt not found at: {src_best}")

    target_latest = target_dir / "latest.pt"
    target_best = target_dir / "best_model.pt"

    if not target_latest.exists():
        print(f"[SETUP] Copying {src_best} -> {target_latest} (Starting from clean Ep 72)...")
        shutil.copy2(src_best, target_latest)

    if not target_best.exists():
        print(f"[SETUP] Copying {src_best} -> {target_best}...")
        shutil.copy2(src_best, target_best)

    # Initialize training_metrics.jsonl with clean lines 1..72 if not already present
    target_metrics = target_dir / "training_metrics.jsonl"
    if not target_metrics.exists():
        src_metrics = src_dir / "training_metrics.jsonl"
        if src_metrics.exists():
            print(f"[SETUP] Seeding {target_metrics} with epochs 1-72 history...")
            lines_to_copy = []
            with open(src_metrics, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        d = json.loads(line)
                        if d.get("epoch", 0) <= 72:
                            lines_to_copy.append(line)
                    except Exception:
                        pass
            with open(target_metrics, "w", encoding="utf-8") as f:
                f.writelines(lines_to_copy)

    train_script = PROJECT_ROOT / "scripts" / "train_smart_temporal.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--data_dir", str(data_dir),
        "--models_dir", str(target_dir),
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
        "--no-use_amp",
        "--resume", str(target_latest),
    ]

    log_path = target_dir / "training.log"
    print(f"\n[EXEC] Running Training in FP32 (no-use_amp)...")
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

    print("\n" + "=" * 80)
    print("FP32 TEST RUN COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
