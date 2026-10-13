"""Unit tests for the DatasetEngine modular architecture.

Verifies MatlabImporter, PlasticitySetup, DatasetSplitter, DatasetNormalizer,
NpzExporter, and the end-to-end DatasetEngine orchestrator.
"""

import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
import scipy.io as sio

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (
    CompositeModeler,
    ConsiderOnlyFinalStepModeler,
    DatasetEngine,
    DatasetModeler,
    DatasetNormalizer,
    DatasetSplitter,
    MatlabImporter,
    NpzExporter,
    PlasticitySetup,
)


def test_matlab_importer() -> None:
    """Verify MatlabImporter loads .mat files and filters internal metadata."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        mat_file = tmp_path / "test_sim.mat"

        dummy_data = {
            "input": np.ones((10, 101), dtype=np.float32),
            "output": np.zeros((10, 101, 31, 20, 4), dtype=np.float32),
        }
        sio.savemat(str(mat_file), dummy_data)

        importer = MatlabImporter()
        loaded = importer.load(mat_file)

        assert "input" in loaded
        assert "output" in loaded
        assert loaded["input"].shape == (10, 101)
        assert loaded["output"].shape == (10, 101, 31, 20, 4)


def test_plasticity_setup_and_kinematic_consistency() -> None:
    """Verify PlasticitySetup extracts correct 5D, 4D, 3D, and 1D tensors and aligns axes."""
    n_samples, nx, ny, nt = 5, 101, 31, 20
    synthetic_input = np.ones((n_samples, nx), dtype=np.float32) * 12.0
    synthetic_output = np.zeros((n_samples, nx, ny, nt, 4), dtype=np.float32)

    # Reference coordinates (t = 0)
    synthetic_output[:, :, :, 0, 0] = np.linspace(-50, 0, nx)[:, None]
    synthetic_output[:, :, :, 0, 1] = np.linspace(0, 15, ny)[None, :]

    # Displacements at t = 3
    synthetic_output[:, :, :, 3, 2] = 0.75  # u_x
    synthetic_output[:, :, :, 3, 3] = -1.5  # u_y
    synthetic_output[:, :, :, 3, 0] = synthetic_output[:, :, :, 0, 0] + 0.75  # deformed X
    synthetic_output[:, :, :, 3, 1] = synthetic_output[:, :, :, 0, 1] - 1.5  # deformed Y

    setup = PlasticitySetup()
    structured = setup.setup({"input": synthetic_input, "output": synthetic_output})

    # Verify tensor shapes
    assert structured["die_boundary"].shape == (n_samples, nx, 2)            # 3D
    assert structured["coords_initial"].shape == (n_samples, nx, ny, 2)       # 4D
    assert structured["coords"].shape == (n_samples, nx, ny, nt, 2)           # 5D
    assert structured["displacements"].shape == (n_samples, nx, ny, nt, 2)    # 5D
    assert structured["target_all"].shape == (n_samples, nx, ny, nt, 4)       # 5D
    assert structured["time_steps"].shape == (nt,)                           # 1D

    # Verify kinematic consistency: deformed X = initial X + u_x
    x_init = structured["coords_initial"][:, :, :, 0]
    u_x = structured["displacements"][:, :, :, 3, 0]
    x_def = structured["coords"][:, :, :, 3, 0]
    assert np.allclose(x_def, x_init + u_x, atol=1e-5)


def test_dataset_splitter() -> None:
    """Verify DatasetSplitter partitions sample arrays and propagates global time steps."""
    n_samples, nx, ny, nt = 10, 101, 31, 20
    dummy_fields = {
        "coords": np.ones((n_samples, nx, ny, nt, 2), dtype=np.float32),
        "die_boundary": np.ones((n_samples, nx, 2), dtype=np.float32),
        "time_steps": np.linspace(0, 1, nt, dtype=np.float32),
    }

    splitter = DatasetSplitter()
    splits = splitter.split(dummy_fields, train_n=7, val_n=1, test_n=2)

    assert splits["train"]["coords"].shape[0] == 7
    assert splits["val"]["coords"].shape[0] == 1
    assert splits["test"]["coords"].shape[0] == 2

    # Global time_steps must remain length 20 in all partitions
    assert splits["train"]["time_steps"].shape == (20,)
    assert splits["val"]["time_steps"].shape == (20,)
    assert splits["test"]["time_steps"].shape == (20,)


def test_dataset_normalizer() -> None:
    """Verify DatasetNormalizer computes mean, std, and per-channel statistics."""
    # Synthetic 2-channel 2D data: 4 samples
    data = np.array([
        [[1.0, 10.0]],
        [[2.0, 20.0]],
        [[3.0, 30.0]],
        [[4.0, 40.0]],
    ], dtype=np.float32)

    normalizer = DatasetNormalizer()
    stats = normalizer.compute_statistical_data({"targets": data})

    assert "targets" in stats
    assert math.isclose(stats["targets"]["mean"], 13.75, abs_tol=1e-5)
    assert "per_channel" in stats["targets"]
    assert math.isclose(stats["targets"]["per_channel"]["mean"][0], 2.5, abs_tol=1e-5)
    assert math.isclose(stats["targets"]["per_channel"]["mean"][1], 25.0, abs_tol=1e-5)


def test_npz_exporter_and_validation() -> None:
    """Verify NpzExporter writes valid .npz and statistical_data.json and catches NaNs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        dummy_splits = {
            "train": {"arr": np.array([1.0, 2.0], dtype=np.float32)},
            "val": {"arr": np.array([3.0], dtype=np.float32)},
            "test": {"arr": np.array([4.0], dtype=np.float32)},
        }
        dummy_stats = {"targets": {"mean": 2.5, "std": 1.0}}

        exporter = NpzExporter()
        saved = exporter.save(dummy_splits, dummy_stats, tmp_path)

        assert len(saved) == 4
        assert (tmp_path / "train.npz").exists()
        assert (tmp_path / "val.npz").exists()
        assert (tmp_path / "test.npz").exists()
        assert (tmp_path / "statistical_data.json").exists()

        # Check content of JSON
        with open(tmp_path / "statistical_data.json", "r") as f:
            loaded_json = json.load(f)
        assert loaded_json["targets"]["mean"] == 2.5

        # Verify corruption check detects NaN
        corrupted_splits = {
            "train": {"arr": np.array([1.0, np.nan], dtype=np.float32)},
        }
        try:
            exporter.save(corrupted_splits, dummy_stats, tmp_path / "corrupt")
            assert False, "Should have raised ValueError on NaN"
        except ValueError as err:
            assert "NaN detected" in str(err)


def test_dataset_engine_end_to_end() -> None:
    """Verify DatasetEngine orchestrates all 5 components in sequence."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_file = tmp_path / "test_raw.mat"
        out_dir = tmp_path / "processed"

        n_samples, nx, ny, nt = 12, 101, 31, 20
        raw_mat = {
            "input": np.ones((n_samples, nx), dtype=np.float32) * 15.0,
            "output": np.zeros((n_samples, nx, ny, nt, 4), dtype=np.float32),
        }
        sio.savemat(str(raw_file), raw_mat)

        engine = DatasetEngine(
            importer=MatlabImporter(),
            cleaner=PlasticitySetup(),
            splitter=DatasetSplitter(),
            normalizer=DatasetNormalizer(),
            exporter=NpzExporter(),
        )

        artifacts = engine.process(
            raw_path=raw_file,
            output_dir=out_dir,
            split_ratios=(8, 1, 3),
        )

        assert "train" in artifacts
        assert "val" in artifacts
        assert "test" in artifacts
        assert "statistical_data" in artifacts

        # Load back train.npz and verify shape
        train_npz = np.load(str(artifacts["train"]))
        assert train_npz["coords"].shape == (8, nx, ny, nt, 2)
        assert train_npz["die_boundary"].shape == (8, nx, 2)
        assert train_npz["time_steps"].shape == (nt,)


def test_consider_only_final_step_modeler() -> None:
    """Verify ConsiderOnlyFinalStepModeler slices 5D to 4D and removes time_steps."""
    n_samples, nx, ny, nt = 4, 101, 31, 20
    original_coords = np.arange(
        n_samples * nx * ny * nt * 2, dtype=np.float32
    ).reshape((n_samples, nx, ny, nt, 2))
    original_displacements = np.ones(
        (n_samples, nx, ny, nt, 2), dtype=np.float32
    ) * 5.0
    original_target = np.ones(
        (n_samples, nx, ny, nt, 4), dtype=np.float32
    ) * 3.0
    original_initial = np.zeros((n_samples, nx, ny, 2), dtype=np.float32)
    original_boundary = np.zeros((n_samples, nx, 2), dtype=np.float32)
    original_time = np.linspace(0.0, 1.0, nt, dtype=np.float32)

    fields = {
        "coords": original_coords,
        "displacements": original_displacements,
        "target_all": original_target,
        "coords_initial": original_initial,
        "die_boundary": original_boundary,
        "time_steps": original_time,
    }

    modeler = ConsiderOnlyFinalStepModeler()
    assert isinstance(modeler, DatasetModeler)

    modified = modeler.modify(fields)

    # Verify tensor shapes have reduced to 4D
    assert modified["coords"].shape == (n_samples, nx, ny, 2)
    assert modified["displacements"].shape == (n_samples, nx, ny, 2)
    assert modified["target_all"].shape == (n_samples, nx, ny, 4)
    assert np.array_equal(modified["coords"], original_coords[:, :, :, -1, :])

    # Invariant tensors remain untouched
    assert modified["coords_initial"].shape == (n_samples, nx, ny, 2)
    assert modified["die_boundary"].shape == (n_samples, nx, 2)

    # time_steps must be eliminated in static formulation (Option A)
    assert "time_steps" not in modified

    # Non-destructive: original input dictionary unchanged
    assert "time_steps" in fields
    assert fields["coords"].ndim == 5


def test_composite_modeler() -> None:
    """Verify CompositeModeler chains modelers sequentially and supports identity."""

    class DummyScaleModeler(DatasetModeler):
        def modify(self, fields: dict) -> dict:
            res = dict(fields)
            res["val"] = res["val"] * 2
            return res

    class DummyOffsetModeler(DatasetModeler):
        def modify(self, fields: dict) -> dict:
            res = dict(fields)
            res["val"] = res["val"] + 10
            return res

    data = {"val": np.array([5.0], dtype=np.float32)}

    # Empty composite acts as identity
    empty_composite = CompositeModeler([])
    assert np.array_equal(empty_composite.modify(data)["val"], np.array([5.0]))

    # Chained execution: (5 * 2) + 10 = 20
    chained = CompositeModeler([DummyScaleModeler(), DummyOffsetModeler()])
    result = chained.modify(data)
    assert math.isclose(float(result["val"][0]), 20.0, abs_tol=1e-5)


def test_dataset_engine_with_final_step_modeler() -> None:
    """Verify DatasetEngine end-to-end integration with ConsiderOnlyFinalStepModeler."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_file = tmp_path / "test_raw.mat"
        out_dir = tmp_path / "processed_static"

        n_samples, nx, ny, nt = 12, 101, 31, 20
        raw_mat = {
            "input": np.ones((n_samples, nx), dtype=np.float32) * 15.0,
            "output": np.zeros((n_samples, nx, ny, nt, 4), dtype=np.float32),
        }
        sio.savemat(str(raw_file), raw_mat)

        engine = DatasetEngine(
            importer=MatlabImporter(),
            cleaner=PlasticitySetup(),
            splitter=DatasetSplitter(),
            normalizer=DatasetNormalizer(),
            exporter=NpzExporter(),
            modeler=ConsiderOnlyFinalStepModeler(),
        )

        artifacts = engine.process(
            raw_path=raw_file,
            output_dir=out_dir,
            split_ratios=(8, 1, 3),
        )

        assert "train" in artifacts
        assert "val" in artifacts
        assert "test" in artifacts
        assert "statistical_data" in artifacts

        # Load back train.npz and verify 4D shape and absence of time_steps
        train_npz = np.load(str(artifacts["train"]))
        assert train_npz["coords"].shape == (8, nx, ny, 2)
        assert train_npz["displacements"].shape == (8, nx, ny, 2)
        assert train_npz["die_boundary"].shape == (8, nx, 2)
        assert "time_steps" not in train_npz.files

        # Verify statistical data JSON contains statistics for 4D coords
        with open(str(artifacts["statistical_data"]), "r") as f:
            stats = json.load(f)
        assert "coords" in stats
        assert "per_channel" in stats["coords"]
        assert len(stats["coords"]["per_channel"]["mean"]) == 2


if __name__ == "__main__":
    tests = [
        test_matlab_importer,
        test_plasticity_setup_and_kinematic_consistency,
        test_dataset_splitter,
        test_dataset_normalizer,
        test_npz_exporter_and_validation,
        test_dataset_engine_end_to_end,
        test_consider_only_final_step_modeler,
        test_composite_modeler,
        test_dataset_engine_with_final_step_modeler,
    ]
    for test_fn in tests:
        print(f"Running {test_fn.__name__}...", end=" ")
        test_fn()
        print("PASSED")
    print(f"\nAll {len(tests)} tests passed successfully!")
