"""Unit tests for the sequential parametric study runner."""

import csv
import json
from pathlib import Path
import tempfile
import pytest

from scripts.run_parametric_study_temporal import (
    format_value_for_slug,
    generate_run_slug,
    generate_study_summary,
    identify_swept_parameters,
    write_parametros_txt,
)


def test_format_value_for_slug() -> None:
    """Test formatting numbers and values into filesystem-safe slugs."""
    assert format_value_for_slug(128) == "128"
    assert format_value_for_slug(0.0) == "0.0"
    assert format_value_for_slug(1e-4) == "1e-4"
    assert format_value_for_slug(1.5e-4) == "1.5e-4"
    assert format_value_for_slug(True) == "True"
    assert format_value_for_slug("cosine") == "cosine"


def test_identify_swept_parameters() -> None:
    """Test separating swept parameters from fixed ones."""
    config = {
        "epochs": 30,
        "latent_dim": [128, 256],
        "num_blocks": (4, 6, 8),
        "lr": [1e-4],  # Single-element list should be treated as fixed
        "dropout": 0.0,
    }
    swept_keys, swept_params, fixed_params = identify_swept_parameters(config)

    assert swept_keys == ["latent_dim", "num_blocks"]
    assert swept_params == {
        "latent_dim": [128, 256],
        "num_blocks": [4, 6, 8],
    }
    assert fixed_params == {
        "epochs": 30,
        "lr": 1e-4,
        "dropout": 0.0,
    }


def test_generate_run_slug() -> None:
    """Test generating dynamic folder slugs based exclusively on swept parameters."""
    swept_keys = ["latent_dim", "num_blocks"]
    run_config = {
        "latent_dim": 256,
        "num_blocks": 6,
        "epochs": 30,
        "lr": 1e-4,
    }
    slug = generate_run_slug(swept_keys, run_config)
    assert slug == "latent_dim_256_num_blocks_6"

    # Single run fallback when no parameters are swept
    assert generate_run_slug([], run_config) == "single_run"


def test_write_parametros_txt() -> None:
    """Test generation of parametros.txt audit file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "models" / "experiment1" / "latent_dim_128" / "parametros.txt"
        run_config = {
            "spatial_dim": 2,
            "surface_channels": 2,
            "volume_channels": 2,
            "parameter_channels": 1,
            "latent_dim": 128,
            "latent_geo_points": 64,
            "subsampled_points": 101,
            "num_blocks": 6,
            "num_heads": 8,
            "pos_scale_factor": 1000.0,
            "dropout": 0.0,
            "epochs": 30,
            "batch_size": 64,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "max_grad_norm": 1.0,
            "use_amp": True,
            "device": "cpu",
            "seed": 42,
            "data_dir": "data/processed/ds_plasticity_temporal",
            "models_dir": str(tmp_path.parent),
            "results_dir": "results/TrainingAnalysis",
        }
        swept_keys = ["latent_dim"]

        write_parametros_txt(
            file_path=tmp_path,
            run_config=run_config,
            swept_keys=swept_keys,
            experiment_name="experiment1",
            run_slug="latent_dim_128",
            run_index=1,
            total_runs=2,
            results_summary={
                "status": "SUCCESS",
                "best_val_loss": 0.0598,
                "best_epoch": 28,
                "final_train_loss": 0.0543,
                "duration_seconds": 120.5,
            },
        )

        assert tmp_path.exists()
        content = tmp_path.read_text(encoding="utf-8")
        assert "Experiment Group:      experiment1" in content
        assert "Run Identifier:        latent_dim_128 (Run 1 of 2)" in content
        assert "* latent_dim: 128" in content
        assert "latent_dim:                   128" in content
        assert "Best Validation Loss:         5.980000e-02" in content
        assert "Best Checkpoint Epoch:        28 / 30" in content


def test_generate_study_summary() -> None:
    """Test multi-run leaderboard generation (Markdown and CSV)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / "models" / "experiment1"
        exp_dir.mkdir(parents=True)

        results = [
            {
                "run_slug": "latent_dim_128_num_blocks_4",
                "status": "SUCCESS",
                "latent_dim": 128,
                "num_blocks": 4,
                "best_val_loss": 0.0725,
                "best_epoch": 25,
                "final_train_loss": 0.0680,
                "duration_seconds": 95.0,
            },
            {
                "run_slug": "latent_dim_256_num_blocks_6",
                "status": "SUCCESS",
                "latent_dim": 256,
                "num_blocks": 6,
                "best_val_loss": 0.0489,
                "best_epoch": 29,
                "final_train_loss": 0.0410,
                "duration_seconds": 150.0,
            },
        ]
        swept_keys = ["latent_dim", "num_blocks"]

        generate_study_summary(
            experiment_dir=exp_dir,
            experiment_name="experiment1",
            all_results=results,
            swept_keys=swept_keys,
        )

        md_path = exp_dir / "parametric_study_summary.md"
        csv_path = exp_dir / "parametric_study_summary.csv"
        assert md_path.exists()
        assert csv_path.exists()

        md_text = md_path.read_text(encoding="utf-8")
        assert "Parametric Study Leaderboard: `experiment1`" in md_text
        # Best model (latent_dim=256) should be ranked 1
        lines = [line for line in md_text.splitlines() if line.startswith("| 1 |")]
        assert len(lines) == 1
        assert "latent_dim_256_num_blocks_6" in lines[0]

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 2
            assert reader[0]["rank"] == "1"
            assert reader[0]["run_identifier"] == "latent_dim_256_num_blocks_6"
            assert float(reader[0]["best_val_loss"]) == pytest.approx(0.0489)
