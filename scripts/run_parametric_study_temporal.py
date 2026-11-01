"""Automated sequential parametric study runner for the SMART temporal surrogate model.

Allows defining all tuneable architecture and optimization hyperparameters in a centralized
configuration dictionary. Any parameter specified as a list or tuple `[...]` triggers a
combinatorial grid sweep (Cartesian product) across the specified values.

Each training run is executed strictly sequentially in an isolated subprocess to guarantee
CUDA memory release and prevent GPU thrashing. Results and checkpoints are saved into
dynamically named directories based on the swept parameter values:
    models/<experiment_name>/<swept_param_1>_<val_1>_<swept_param_2>_<val_2>/

Every run directory includes a dedicated `parametros.txt` audit file detailing the exact
hyperparameters utilized for that execution. Upon completion of all runs, a cross-experiment
benchmark leaderboard (`parametric_study_summary.md` and `.csv`) is generated.

Usage:
------
# 1. Edit PARAMETRIC_CONFIG below to specify desired scalar values or sweep lists `[...]`, then:
python scripts/run_parametric_study_temporal.py

# 2. Or override settings directly via command line:
python scripts/run_parametric_study_temporal.py --experiment_name experiment1 --dry_run
"""

import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==============================================================================
# DEFAULT PARAMETRIC STUDY CONFIGURATION
# ==============================================================================
# Define preset defaults for all tuneable parameters.
# Any parameter set to a list or tuple (e.g., [128, 256]) will be swept in a loop.
# Any parameter set to a scalar (e.g., 64) will remain fixed across all runs.
# ==============================================================================

DEFAULT_EXPERIMENT_NAME: str = "experiment1"

PARAMETRIC_CONFIG: Dict[str, Union[Any, List[Any], Tuple[Any, ...]]] = {
    # ─── ARCHITECTURE HYPERPARAMETERS ─────────────────────────────────────────
    # Spatial and physical dimensions (fixed by problem formulation)
    "spatial_dim": 2,                  # 2D plane strain coordinates (x, y)
    "surface_channels": 2,             # Contact boundary displacement vector (u_x, u_y)
    "volume_channels": 2,              # Continuum interior displacement vector (u_x, u_y)
    "parameter_channels": 1,           # Normalized time conditioning t in [0, 1]

    # Expressive capacity hyperparameters (tuneable)
    "latent_dim": [128, 256],          # Latent feature dimension d (sweep example: [128, 256])
    "latent_geo_points": 64,           # Number of latent geometry scaffold points K
    "subsampled_points": 101,          # Geometry points sampled in cross-attention (die has 101 nodes)
    "num_blocks": [4, 6, 8],           # Number of encoder-decoder Transformer blocks L
    "num_heads": 8,                    # Attention heads (must divide latent_dim)
    "pos_scale_factor": 1000.0,        # Multiplicative scaling factor for positional embeddings
    "dropout": 0.0,                    # Attention and feed-forward dropout probability

    # ─── OPTIMIZATION & TRAINING HYPERPARAMETERS ──────────────────────────────
    "epochs": 30,                      # Number of training epochs per run
    "batch_size": 64,                  # Training batch size
    "lr": 1e-4,                        # Initial learning rate
    "weight_decay": 1e-4,              # AdamW weight decay regularizer
    "scheduler": "cosine",             # LR schedule: "cosine", "plateau", or "none"
    "max_grad_norm": 1.0,              # Gradient clipping threshold norm
    "use_amp": True,                   # Automatic mixed precision (AMP FP16)

    # ─── EXECUTION & ENVIRONMENT ──────────────────────────────────────────────
    "seed": 42,                        # Reproducibility seed
}


def parse_cli_arguments() -> argparse.Namespace:
    """Parse optional command-line overrides for the parametric study runner."""
    parser = argparse.ArgumentParser(
        description="Sequential parametric study runner for SMART temporal surrogate."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help=f"Parent experiment folder name (default: '{DEFAULT_EXPERIMENT_NAME}').",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ds_plasticity_temporal",
        help="Path to directory containing train.npz, val.npz, and statistical_data.json.",
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Base parent directory where experiment models will be saved.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "TrainingAnalysis",
        help="Base directory where analysis and charts will be stored.",
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
        help="Print planned parameter combinations and folder structures without training.",
    )
    return parser.parse_args()


# ==============================================================================
# PARAMETRIC COMBINATIONS & DIRECTORY NAMING
# ==============================================================================

def format_value_for_slug(value: Any) -> str:
    """Format a hyperparameter value into a clean, filesystem-safe string representation.

    Parameters
    ----------
    value : Any
        Hyperparameter value (float, int, str, bool).

    Returns
    -------
    str
        Clean string representation for directory naming.
    """
    if isinstance(value, float):
        if value == 0.0:
            return "0.0"
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            s = f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+")
            return s.replace(".0e", "e") if ".0e" in s else s
        return f"{value:g}"
    return str(value)


def identify_swept_parameters(
    config: Dict[str, Any]
) -> Tuple[List[str], Dict[str, List[Any]], Dict[str, Any]]:
    """Partition the configuration into swept parameters and fixed parameters.

    Parameters
    ----------
    config : Dict[str, Any]
        Base configuration mapping parameter keys to values or lists of values.

    Returns
    -------
    Tuple[List[str], Dict[str, List[Any]], Dict[str, Any]]
        - List of keys that are swept (defined as list/tuple with length > 1).
        - Dictionary of swept parameter keys to their candidate values.
        - Dictionary of fixed parameter keys to their scalar value.
    """
    swept_keys: List[str] = []
    swept_params: Dict[str, List[Any]] = {}
    fixed_params: Dict[str, Any] = {}

    for key, value in config.items():
        if isinstance(value, (list, tuple)):
            if len(value) > 1:
                swept_keys.append(key)
                swept_params[key] = list(value)
            elif len(value) == 1:
                fixed_params[key] = value[0]
            else:
                raise ValueError(f"Parameter '{key}' has an empty list/tuple.")
        else:
            fixed_params[key] = value

    return swept_keys, swept_params, fixed_params


def generate_run_slug(swept_keys: List[str], run_config: Dict[str, Any]) -> str:
    """Construct a meaningful folder name derived exclusively from swept parameter values.

    Parameters
    ----------
    swept_keys : List[str]
        List of parameter keys that were varied across the parametric study.
    run_config : Dict[str, Any]
        Active configuration dictionary for this specific training run.

    Returns
    -------
    str
        Folder slug (e.g., 'latent_dim_128_num_blocks_4' or 'single_run').
    """
    if not swept_keys:
        return "single_run"

    parts: List[str] = []
    for key in swept_keys:
        val_str = format_value_for_slug(run_config[key])
        parts.append(f"{key}_{val_str}")
    return "_".join(parts)


# ==============================================================================
# AUDIT TRAIL LOGGING (parametros.txt)
# ==============================================================================

def write_parametros_txt(
    file_path: Path,
    run_config: Dict[str, Any],
    swept_keys: List[str],
    experiment_name: str,
    run_slug: str,
    run_index: int,
    total_runs: int,
    results_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Create or update a detailed parametros.txt audit file for a single run.

    Parameters
    ----------
    file_path : Path
        Absolute path to the parametros.txt file.
    run_config : Dict[str, Any]
        Active configuration values for this run.
    swept_keys : List[str]
        List of keys that were varied in the parametric study.
    experiment_name : str
        Parent experiment group name.
    run_slug : str
        Run identifier subfolder name.
    run_index : int
        Current run index (1-based).
    total_runs : int
        Total number of runs in the parametric study.
    results_summary : Optional[Dict[str, Any]], default=None
        Telemetry metrics achieved after training completes.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "=" * 80,
        "SMART SURROGATE MODEL - PARAMETRIC STUDY CONFIGURATION AUDIT",
        "=" * 80,
        f"Experiment Group:      {experiment_name}",
        f"Run Identifier:        {run_slug} (Run {run_index} of {total_runs})",
        f"Timestamp:             {now_str}",
        f"Execution Mode:        Strictly Sequential",
    ]

    # Highlight swept parameters
    lines.append("-" * 80)
    lines.append("SWEPT PARAMETERS (VARIED IN THIS STUDY):")
    if swept_keys:
        for k in swept_keys:
            lines.append(f"  * {k}: {run_config[k]}")
    else:
        lines.append("  (None - Single configuration run)")

    # Architecture configuration
    lines.append("-" * 80)
    lines.append("ARCHITECTURE CONFIGURATION:")
    lines.append(f"  spatial_dim:                  {run_config.get('spatial_dim', 2)}")
    lines.append(f"  surface_channels:             {run_config.get('surface_channels', 2)}")
    lines.append(f"  volume_channels:              {run_config.get('volume_channels', 2)}")
    lines.append(f"  parameter_channels:           {run_config.get('parameter_channels', 1)}")
    lines.append(f"  latent_dim:                   {run_config.get('latent_dim')}")
    lines.append(f"  latent_geo_points:            {run_config.get('latent_geo_points')}")
    lines.append(f"  subsampled_points:            {run_config.get('subsampled_points')}")
    lines.append(f"  num_blocks:                   {run_config.get('num_blocks')}")
    lines.append(f"  num_heads:                    {run_config.get('num_heads')}")
    lines.append(f"  pos_scale_factor:             {run_config.get('pos_scale_factor')}")
    lines.append(f"  dropout:                      {run_config.get('dropout')}")

    # Optimization configuration
    lines.append("-" * 80)
    lines.append("OPTIMIZATION & TRAINING CONFIGURATION:")
    lines.append(f"  epochs:                       {run_config.get('epochs')}")
    lines.append(f"  batch_size:                   {run_config.get('batch_size')}")
    lines.append(f"  lr:                           {run_config.get('lr')}")
    lines.append(f"  weight_decay:                 {run_config.get('weight_decay')}")
    lines.append(f"  scheduler:                    {run_config.get('scheduler')}")
    lines.append(f"  max_grad_norm:                {run_config.get('max_grad_norm')}")
    lines.append(f"  use_amp:                      {run_config.get('use_amp')}")

    # Environment configuration
    lines.append("-" * 80)
    lines.append("ENVIRONMENT & PATHS:")
    lines.append(f"  device:                       {run_config.get('device')}")
    lines.append(f"  seed:                         {run_config.get('seed')}")
    lines.append(f"  data_dir:                     {run_config.get('data_dir')}")
    lines.append(f"  models_dir:                   {run_config.get('models_dir')}")
    lines.append(f"  results_dir:                  {run_config.get('results_dir')}")

    # Post-training telemetry if available
    if results_summary is not None:
        lines.append("-" * 80)
        lines.append("TRAINING OUTCOME & TELEMETRY:")
        lines.append(f"  Status:                       {results_summary.get('status', 'COMPLETED')}")
        if "best_val_loss" in results_summary:
            lines.append(f"  Best Validation Loss:         {results_summary['best_val_loss']:.6e}")
        if "best_epoch" in results_summary:
            lines.append(f"  Best Checkpoint Epoch:        {results_summary['best_epoch']} / {run_config.get('epochs')}")
        if "final_train_loss" in results_summary:
            lines.append(f"  Final Training Loss:          {results_summary['final_train_loss']:.6e}")
        if "final_val_loss" in results_summary:
            lines.append(f"  Final Validation Loss:        {results_summary['final_val_loss']:.6e}")
        if "duration_seconds" in results_summary:
            lines.append(f"  Elapsed Wall-Clock Time:      {results_summary['duration_seconds']:.1f} s")
        if "error_message" in results_summary and results_summary["error_message"]:
            lines.append(f"  Error Message:                {results_summary['error_message']}")

    lines.append("=" * 80 + "\n")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==============================================================================
# SUBPROCESS EXECUTION & METRICS EXTRACTION
# ==============================================================================

def execute_single_training_run(
    run_config: Dict[str, Any],
    swept_keys: List[str],
    experiment_name: str,
    run_slug: str,
    run_index: int,
    total_runs: int,
) -> Dict[str, Any]:
    """Execute a single training run strictly sequentially via an isolated subprocess.

    Parameters
    ----------
    run_config : Dict[str, Any]
        Complete configuration dictionary for this run.
    swept_keys : List[str]
        List of swept parameter keys.
    experiment_name : str
        Parent experiment group name.
    run_slug : str
        Slug identifying the active combination.
    run_index : int
        Current run index.
    total_runs : int
        Total run count.

    Returns
    -------
    Dict[str, Any]
        Dictionary with telemetry metrics and execution status.
    """
    models_dir = Path(run_config["models_dir"])
    results_dir = Path(run_config["results_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write initial parametros.txt before training starts
    param_txt_path = models_dir / "parametros.txt"
    write_parametros_txt(
        file_path=param_txt_path,
        run_config=run_config,
        swept_keys=swept_keys,
        experiment_name=experiment_name,
        run_slug=run_slug,
        run_index=run_index,
        total_runs=total_runs,
    )

    # 2. Build CLI arguments for scripts/train_smart_temporal.py
    train_script = PROJECT_ROOT / "scripts" / "train_smart_temporal.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--data_dir", str(run_config["data_dir"]),
        "--models_dir", str(models_dir),
        "--results_dir", str(results_dir),
        "--epochs", str(run_config["epochs"]),
        "--batch_size", str(run_config["batch_size"]),
        "--lr", str(run_config["lr"]),
        "--weight_decay", str(run_config["weight_decay"]),
        "--scheduler", str(run_config["scheduler"]),
        "--max_grad_norm", str(run_config["max_grad_norm"]),
        "--parameter_channels", str(run_config["parameter_channels"]),
        "--latent_dim", str(run_config["latent_dim"]),
        "--latent_geo_points", str(run_config["latent_geo_points"]),
        "--subsampled_points", str(run_config["subsampled_points"]),
        "--num_blocks", str(run_config["num_blocks"]),
        "--num_heads", str(run_config["num_heads"]),
        "--pos_scale_factor", str(run_config["pos_scale_factor"]),
        "--dropout", str(run_config["dropout"]),
        "--device", str(run_config["device"]),
        "--seed", str(run_config["seed"]),
    ]
    if run_config.get("use_amp", True):
        cmd.append("--use_amp")
    else:
        cmd.append("--no-use_amp")

    print("\n" + "=" * 80)
    print(f"[{run_index}/{total_runs}] EXECUTING RUN: {run_slug}")
    print(f"Model Directory:   {models_dir}")
    print(f"Parameters Audit:  {param_txt_path}")
    print("=" * 80)

    start_time = time.time()
    status = "SUCCESS"
    error_msg = None

    try:
        # Strictly sequential execution: check=True waits for completion
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        status = "FAILED"
        error_msg = f"Training subprocess exited with non-zero returncode {e.returncode}."
        print(f"\n[ERROR] Run {run_slug} failed: {error_msg}")
    except Exception as e:
        status = "FAILED"
        error_msg = str(e)
        print(f"\n[ERROR] Unexpected exception during run {run_slug}: {error_msg}")

    elapsed_s = time.time() - start_time

    # 3. Extract metrics from output logs if training succeeded
    run_summary: Dict[str, Any] = {
        "run_slug": run_slug,
        "status": status,
        "duration_seconds": elapsed_s,
        "error_message": error_msg,
    }

    metrics_file = models_dir / "training_metrics.jsonl"
    if metrics_file.exists():
        try:
            records = []
            with open(metrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            if records:
                val_losses = [r.get("val_loss", float("inf")) for r in records]
                min_val = min(val_losses)
                best_ep = val_losses.index(min_val) + 1
                run_summary["best_val_loss"] = min_val
                run_summary["best_epoch"] = best_ep
                run_summary["final_train_loss"] = records[-1].get("train_loss", None)
                run_summary["final_val_loss"] = records[-1].get("val_loss", None)
        except Exception as ex:
            print(f"[WARNING] Failed to parse metrics file {metrics_file}: {ex}")

    # 4. Update parametros.txt with final telemetry
    write_parametros_txt(
        file_path=param_txt_path,
        run_config=run_config,
        swept_keys=swept_keys,
        experiment_name=experiment_name,
        run_slug=run_slug,
        run_index=run_index,
        total_runs=total_runs,
        results_summary=run_summary,
    )

    return run_summary


# ==============================================================================
# CROSS-RUN LEADERBOARD & SYNTHESIS
# ==============================================================================

def generate_study_summary(
    experiment_dir: Path,
    experiment_name: str,
    all_results: List[Dict[str, Any]],
    swept_keys: List[str],
) -> None:
    """Generate markdown and CSV synthesis files comparing all runs in the study.

    Parameters
    ----------
    experiment_dir : Path
        Parent experiment directory (e.g., models/experiment1).
    experiment_name : str
        Experiment group name.
    all_results : List[Dict[str, Any]]
        List of result dictionaries for each completed run.
    swept_keys : List[str]
        Keys that were varied across runs.
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)
    summary_md_path = experiment_dir / "parametric_study_summary.md"
    summary_csv_path = experiment_dir / "parametric_study_summary.csv"

    # Sort results by best validation loss (ascending)
    sorted_results = sorted(
        all_results,
        key=lambda x: x.get("best_val_loss", float("inf")) if x.get("best_val_loss") is not None else float("inf"),
    )

    # 1. Write CSV summary
    csv_headers = ["rank", "run_identifier", "status"] + swept_keys + [
        "best_val_loss", "best_epoch", "final_train_loss", "duration_seconds"
    ]
    csv_rows = [",".join(csv_headers)]
    for rank, res in enumerate(sorted_results, start=1):
        row = [
            str(rank),
            res["run_slug"],
            res["status"],
        ]
        for k in swept_keys:
            row.append(str(res.get(k, "")))
        b_val = f"{res['best_val_loss']:.6e}" if res.get("best_val_loss") is not None else "N/A"
        b_ep = str(res.get("best_epoch", "N/A"))
        f_tr = f"{res['final_train_loss']:.6e}" if res.get("final_train_loss") is not None else "N/A"
        dur = f"{res.get('duration_seconds', 0.0):.1f}"
        row.extend([b_val, b_ep, f_tr, dur])
        csv_rows.append(",".join(row))

    with open(summary_csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_rows) + "\n")

    # 2. Write Markdown summary
    md_lines = [
        f"# Parametric Study Leaderboard: `{experiment_name}`",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Total Sequential Runs:** {len(all_results)}  ",
        f"**Swept Parameters:** `{', '.join(swept_keys)}`" if swept_keys else "**Swept Parameters:** None (Single Run)",
        "",
        "---",
        "",
        "### Ranked Results (Sorted by Best Validation Loss)",
        "",
    ]

    # Markdown table header
    table_header = ["Rank", "Run Identifier", "Status"] + [f"`{k}`" for k in swept_keys] + [
        "Best Val Loss", "Best Epoch", "Train Loss", "Duration (s)"
    ]
    md_lines.append("| " + " | ".join(table_header) + " |")
    md_lines.append("| " + " | ".join([":---:" if i == 0 or i == 2 else ":---" for i in range(len(table_header))]) + " |")

    for rank, res in enumerate(sorted_results, start=1):
        b_val = f"**{res['best_val_loss']:.4e}**" if res.get("best_val_loss") is not None else "N/A"
        b_ep = f"Epoch {res.get('best_epoch')}" if res.get("best_epoch") is not None else "N/A"
        f_tr = f"{res['final_train_loss']:.4e}" if res.get("final_train_loss") is not None else "N/A"
        dur = f"{res.get('duration_seconds', 0.0):.1f}s"
        status_badge = "✅ PASS" if res["status"] == "SUCCESS" else "❌ FAIL"

        row = [str(rank), f"[`{res['run_slug']}`]({res['run_slug']}/parametros.txt)", status_badge]
        for k in swept_keys:
            row.append(str(res.get(k, "")))
        row.extend([b_val, b_ep, f_tr, dur])
        md_lines.append("| " + " | ".join(row) + " |")

    md_lines.append("")
    md_lines.append("---")
    md_lines.append(f"*Artifacts persisted at `{experiment_dir}`*")
    md_lines.append("")

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n" + "=" * 80)
    print("PARAMETRIC STUDY COMPLETED SUCCESSFULLY")
    print(f"Summary Markdown: {summary_md_path}")
    print(f"Summary CSV:      {summary_csv_path}")
    print("=" * 80 + "\n")
    print("\n".join(md_lines))


# ==============================================================================
# MAIN ORCHESTRATION PIPELINE
# ==============================================================================

def main() -> None:
    """Orchestrate sequential parametric study execution."""
    cli_args = parse_cli_arguments()
    experiment_name = cli_args.experiment_name

    # 1. Identify swept vs fixed parameters from PARAMETRIC_CONFIG
    swept_keys, swept_params, fixed_params = identify_swept_parameters(PARAMETRIC_CONFIG)

    # 2. Build Cartesian product of all swept values
    if swept_keys:
        value_lists = [swept_params[k] for k in swept_keys]
        combinations = list(itertools.product(*value_lists))
    else:
        combinations = [()]

    total_runs = len(combinations)
    print("=" * 80)
    print("SEQUENTIAL PARAMETRIC STUDY ORCHESTRATOR")
    print("=" * 80)
    print(f"Experiment Group Name:   {experiment_name}")
    print(f"Total Sequential Runs:   {total_runs}")
    print(f"Swept Parameters ({len(swept_keys)}):  {swept_keys if swept_keys else 'None'}")
    if swept_keys:
        for k in swept_keys:
            print(f"  * {k}: {swept_params[k]}")
    print("Execution Mode:          Strictly Sequential (1 model at a time)")
    print("=" * 80)

    # 3. Dry-run inspection
    if cli_args.dry_run:
        print("\n[DRY RUN MODE] Inspecting planned run configurations:")
        for idx, combo in enumerate(combinations, start=1):
            run_cfg = dict(fixed_params)
            for k, val in zip(swept_keys, combo):
                run_cfg[k] = val
            run_slug = generate_run_slug(swept_keys, run_cfg)
            target_model_dir = cli_args.models_dir / experiment_name / run_slug
            target_res_dir = cli_args.results_dir / "ds_plasticity_temporal" / "SMART" / experiment_name / run_slug
            print(f"  [{idx:02d}/{total_runs:02d}] Slug: {run_slug}")
            print(f"         Models:  {target_model_dir}")
            print(f"         Results: {target_res_dir}")
            print(f"         Swept Values: { {k: run_cfg[k] for k in swept_keys} }")
        print("\nDry run complete. No models were trained.")
        return

    # 4. Strictly sequential execution loop
    all_run_results: List[Dict[str, Any]] = []
    base_experiment_model_dir = cli_args.models_dir / experiment_name

    for idx, combo in enumerate(combinations, start=1):
        # Build specific run configuration
        run_cfg = dict(fixed_params)
        for k, val in zip(swept_keys, combo):
            run_cfg[k] = val

        run_slug = generate_run_slug(swept_keys, run_cfg)
        target_model_dir = base_experiment_model_dir / run_slug
        target_results_dir = (
            cli_args.results_dir
            / "ds_plasticity_temporal"
            / "SMART"
            / experiment_name
            / run_slug
        )

        run_cfg["models_dir"] = target_model_dir
        run_cfg["results_dir"] = target_results_dir
        run_cfg["data_dir"] = cli_args.data_dir
        run_cfg["device"] = cli_args.device

        # Execute training strictly sequentially
        result = execute_single_training_run(
            run_config=run_cfg,
            swept_keys=swept_keys,
            experiment_name=experiment_name,
            run_slug=run_slug,
            run_index=idx,
            total_runs=total_runs,
        )

        # Store swept parameter values for summary reporting
        for k in swept_keys:
            result[k] = run_cfg[k]
        all_run_results.append(result)

    # 5. Compile global comparative leaderboard
    generate_study_summary(
        experiment_dir=base_experiment_model_dir,
        experiment_name=experiment_name,
        all_results=all_run_results,
        swept_keys=swept_keys,
    )


if __name__ == "__main__":
    main()
