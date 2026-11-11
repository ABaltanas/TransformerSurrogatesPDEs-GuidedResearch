#!/usr/bin/env python3
"""Post-processing script to compute Joint 4-Channel Relative L2 Error.

Iterates over trained surrogate model evaluations in results/TrainingAnalysis,
performs deterministic test-split inference to compute both:
1. Pure Displacement Relative L2 Error (repo standard)
2. Joint 4-Channel Relative L2 Error (literature standard: Geo-FNO, Transolver, PLT)

Updates existing evaluation_metrics.json and evaluation_summary.md files
with the new joint metric without requiring model retraining.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_model import PlasticityPartitionDataset
from src.evaluation.denormalizer import StatisticalDenormalizer
from src.evaluation.exporters import CompositeReportExporter
from src.evaluation.metrics import PlasticityMetricsCalculator
from src.evaluation.runner import BatchedInferenceRunner
from src.training.batch_loaders import SMARTBatchLoader
from src.training.wrappers.smart_wrapper import SMARTModelWrapper


def deduce_smart_config(checkpoint_path: Path) -> Dict[str, Any]:
    """Deduce SMART architecture hyperparameters from checkpoint state dict and audit files."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt["model_state_dict"]

    block_nums = set()
    for k in sd:
        if "encoder_blocks." in k:
            block_nums.add(int(k.split("encoder_blocks.")[1].split(".")[0]))
    num_blocks = len(block_nums) if block_nums else 6

    latent_dim = 128
    for k, v in sd.items():
        if "encoder_blocks.0.geo_attn.q.weight" in k:
            latent_dim = v.shape[0]
            break

    param_channels = 0
    for k, v in sd.items():
        if "modulator.mlp.0.weight" in k:
            param_channels = v.shape[1]
            break

    pdir = checkpoint_path.parent
    pfile = pdir / "parametros.txt"
    cfg = {
        "spatial_dim": 2,
        "surface_channels": 2,
        "volume_channels": 2,
        "parameter_channels": param_channels,
        "latent_dim": latent_dim,
        "latent_geometry_points": 64,
        "subsampled_geometry_points": 101,
        "num_encoder_decoder_blocks": num_blocks,
        "num_heads": 8,
        "pos_scale_factor": 1000.0,
        "dropout": 0.0,
    }

    if pfile.exists():
        try:
            with open(pfile, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) == 2:
                        k, val = parts[0].strip(), parts[1].strip()
                        key_map = {
                            "latent_geo_points": "latent_geometry_points",
                            "subsampled_points": "subsampled_geometry_points",
                            "num_blocks": "num_encoder_decoder_blocks",
                        }
                        dest_key = key_map.get(k, k)
                        if dest_key in cfg:
                            cfg[dest_key] = int(val)
        except Exception:
            pass

    return cfg


def resolve_checkpoint_path(raw_path: str, eval_dir: Path) -> Optional[Path]:
    """Locate the checkpoint file on disk."""
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p

    cand = PROJECT_ROOT / raw_path
    if cand.exists():
        return cand

    # Check alternative in models/LION or models
    name = p.name
    parent_name = p.parent.name
    cand_lion = PROJECT_ROOT / "models" / "LION" / parent_name / name
    if cand_lion.exists():
        return cand_lion

    cand_models = PROJECT_ROOT / "models" / parent_name / name
    if cand_models.exists():
        return cand_models

    return None


def resolve_dataset_dir(eval_json_path: Path, results_dir: Path, data_root: Path) -> Optional[Path]:
    """Infer dataset directory from the results hierarchy."""
    try:
        rel = eval_json_path.relative_to(results_dir)
        dataset_name = rel.parts[0]
        data_dir = data_root / dataset_name
        if (data_dir / "test.npz").exists() and (data_dir / "statistical_data.json").exists():
            return data_dir
    except Exception:
        pass

    # Fallback keyword checks
    path_str = str(eval_json_path)
    for ds_name in ["ds_plasticity_temporal", "ds_plasticity_onlyFinalStep", "ds_plasticity_nochange"]:
        if ds_name in path_str:
            data_dir = data_root / ds_name
            if (data_dir / "test.npz").exists() and (data_dir / "statistical_data.json").exists():
                return data_dir

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Joint 4-Channel Relative L2 Error and update evaluation records."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "TrainingAnalysis",
        help="Path to results/TrainingAnalysis directory.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Path to data/processed directory.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target execution hardware (cuda or cpu).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview metric calculations without modifying json/md files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    data_root = args.data_root.resolve()
    device = torch.device(args.device)

    print("=" * 90)
    print("JOINT 4-CHANNEL RELATIVE L2 ERROR POST-PROCESSING PIPELINE")
    print(f"Results Directory: {results_dir}")
    print(f"Data Root:         {data_root}")
    print(f"Execution Device:  {device}")
    print(f"Dry Run Mode:      {args.dry_run}")
    print("=" * 90)

    eval_files = sorted(results_dir.rglob("evaluation_metrics.json"))
    if not eval_files:
        print("No evaluation_metrics.json files found.")
        return

    print(f"Found {len(eval_files)} evaluation record(s) to process.\n")

    runner = BatchedInferenceRunner()
    denormalizer = StatisticalDenormalizer()
    metrics_calc = PlasticityMetricsCalculator()
    exporter = CompositeReportExporter()

    # Cache loaded test partitions and data loaders
    cache_datasets: Dict[str, Tuple[PlasticityPartitionDataset, Dict[str, Any], np.ndarray]] = {}

    summary_records: List[Dict[str, Any]] = []

    for idx, eval_file in enumerate(eval_files, 1):
        eval_dir = eval_file.parent
        rel_eval_dir = eval_dir.relative_to(results_dir)

        with open(eval_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        raw_ckpt = summary.get("checkpoint", "")
        ckpt_path = resolve_checkpoint_path(raw_ckpt, eval_dir)
        if ckpt_path is None or not ckpt_path.exists():
            print(f"[{idx}/{len(eval_files)}] SKIP: Checkpoint not found for {rel_eval_dir} (raw: {raw_ckpt})")
            continue

        dataset_dir = resolve_dataset_dir(eval_file, results_dir, data_root)
        if dataset_dir is None:
            print(f"[{idx}/{len(eval_files)}] SKIP: Dataset directory not found for {rel_eval_dir}")
            continue

        ds_key = str(dataset_dir)
        if ds_key not in cache_datasets:
            test_npz = dataset_dir / "test.npz"
            stats_json = dataset_dir / "statistical_data.json"
            with open(stats_json, "r", encoding="utf-8") as sf:
                stats = json.load(sf)
            ds = PlasticityPartitionDataset(test_npz)
            data_npz = np.load(test_npz)
            initial_mesh = data_npz["coords_initial"]
            cache_datasets[ds_key] = (ds, stats, initial_mesh)

        test_dataset, stats, initial_mesh = cache_datasets[ds_key]

        # Deduce and instantiate SMART model
        cfg = deduce_smart_config(ckpt_path)
        model = SMARTModelWrapper(config=cfg)
        sd = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
        model.get_torch_module().load_state_dict(sd, strict=True)
        model.get_torch_module().to(device)

        # DataLoader
        b_loader = SMARTBatchLoader(stats=stats)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=b_loader.loadDataset,
            num_workers=0,
        )

        # Ingest and predict
        raw_predictions = runner.predict(model, test_loader, device=device)
        targets_dict = {
            "displacements": test_dataset.displacements,
            "coords_initial": test_dataset.coords_initial,
        }
        phys_preds, phys_targets = denormalizer.denormalize(
            pred_norm=raw_predictions,
            target_norm=targets_dict,
            stats=stats,
        )

        pred_u = phys_preds["displacements"]
        true_u = phys_targets["displacements"]

        pure_rel_l2 = metrics_calc.compute_relative_l2(pred_u, true_u)
        joint_4ch_l2 = metrics_calc.compute_joint_4channel_relative_l2(
            pred_u=pred_u,
            true_u=true_u,
            initial_mesh=initial_mesh,
        )

        # Ensure physical_metrics is updated
        if "physical_metrics" not in summary:
            summary["physical_metrics"] = {}

        # Preserve original rel_l2_overall if already evaluated, else set it
        if "rel_l2_overall" in summary["physical_metrics"]:
            pure_rel_l2 = summary["physical_metrics"]["rel_l2_overall"]
        else:
            summary["physical_metrics"]["rel_l2_overall"] = pure_rel_l2

        summary["physical_metrics"]["joint_4channel_rel_l2"] = joint_4ch_l2

        # Record for summary table
        pure_pct = pure_rel_l2 * 100.0
        joint_pct = joint_4ch_l2 * 100.0
        ratio = pure_pct / (joint_pct + 1e-12)

        summary_records.append({
            "experiment": str(rel_eval_dir),
            "pure_l2_pct": pure_pct,
            "joint_l2_pct": joint_pct,
            "reduction_factor": ratio,
        })

        print(
            f"[{idx:02d}/{len(eval_files):02d}] {rel_eval_dir}:\n"
            f"       Pure Displacement Rel L2: {pure_pct:6.2f}%\n"
            f"       Joint 4-Channel Rel L2:   {joint_pct:6.2f}% ({ratio:.1f}x lower)"
        )

        if not args.dry_run:
            exporter.export(summary, eval_dir)

    print("\n" + "=" * 90)
    print("SUMMARY COMPARISON ACROSS ALL BENCHMARK EXPERIMENTS")
    print("=" * 90)
    print(f"{'Experiment Path':<60} | {'Pure L2':>8} | {'Joint 4-Ch':>10} | {'Ratio':>7}")
    print("-" * 90)
    for rec in summary_records:
        print(
            f"{rec['experiment']:<60} | {rec['pure_l2_pct']:>7.2f}% | "
            f"{rec['joint_l2_pct']:>9.2f}% | {rec['reduction_factor']:>6.1f}x"
        )
    print("=" * 90)
    if args.dry_run:
        print("[Note] Dry run enabled. No files were modified.")
    else:
        print("[Success] All evaluation_metrics.json and evaluation_summary.md files successfully updated.")


if __name__ == "__main__":
    main()
