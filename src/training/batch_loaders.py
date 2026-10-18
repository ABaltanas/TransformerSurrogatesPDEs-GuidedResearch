"""Concrete BatchLoader implementations for surrogate model training pipelines.

This module provides data collation strategies that bridge clean simulation arrays
saved in partition files (.npz) into the exact tensor signatures required by
different neural operator architectures.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import torch

from src.training.interfaces import BatchLoader


class SMARTBatchLoader(BatchLoader):
    """BatchLoader formatting 2D elastoplastic forging data for the SMART model.

    SMART expects three geometric sets:
    1. geo: Rigid boundary geometry point cloud (die_boundary), shape (B, N_geo, 2).
    2. surf_query_pos: Surface query coordinates (top boundary workpiece nodes), shape (B, N_surf, 2).
    3. vol_query_pos: Volume query coordinates (interior workpiece nodes), shape (B, N_vol, 2).

    Parameters
    ----------
    stats : Optional[Dict[str, Any]], default=None
        Statistical moments from statistical_data.json used for standardizing target fields.
        If None, raw unstandardized physical displacements are utilized.
    min_pos : Optional[Sequence[float]], default=None
        Lower coordinate bounds [x_min, y_min] for scaling positions to [0, 1].
        Defaults to [-50.0, 0.0] mm.
    max_pos : Optional[Sequence[float]], default=None
        Upper coordinate bounds [x_max, y_max] for scaling positions to [0, 1].
        Defaults to [1.0, 27.0] mm.
    surface_channels : int, default=2
        Number of physical channels to extract for the surface boundary (1 or 2).
    """

    def __init__(
        self,
        stats: Optional[Dict[str, Any]] = None,
        min_pos: Optional[Sequence[float]] = None,
        max_pos: Optional[Sequence[float]] = None,
        surface_channels: int = 2,
    ) -> None:
        self.stats = stats
        self.min_pos = np.array(min_pos if min_pos is not None else [-50.0, 0.0], dtype=np.float32)
        self.max_pos = np.array(max_pos if max_pos is not None else [1.0, 27.0], dtype=np.float32)
        self.pos_span = self.max_pos - self.min_pos
        self.surface_channels = int(surface_channels)

    def loadDataset(self, samples: Sequence[Dict[str, Union[np.ndarray, Any]]]) -> Dict[str, torch.Tensor]:
        """Collate simulation sample dictionaries into SMART-formatted batch tensors.

        Parameters
        ----------
        samples : Sequence[Dict[str, Union[np.ndarray, Any]]]
            List of sample dictionaries containing 'die_boundary', 'coords_initial',
            and 'displacements'.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'geo': (B, 101, 2) float32 normalized die contour.
            - 'surf_query_pos': (B, 101, 2) float32 normalized top surface query coordinates.
            - 'vol_query_pos': (B, 3030, 2) float32 normalized continuum query coordinates.
            - 'targets': Sub-dictionary containing 'surf_data' (B, 101, C_surf), 'vol_data' (B, 3030, 2),
              and 'displacements' (B, 101, 31, 2).
        """
        batch_size = len(samples)

        # 1. Stack raw arrays across samples
        raw_die_list = [np.asarray(s["die_boundary"], dtype=np.float32) for s in samples]
        raw_x0_list = [np.asarray(s["coords_initial"], dtype=np.float32) for s in samples]
        raw_u_list = [np.asarray(s["displacements"], dtype=np.float32) for s in samples]

        stacked_die = np.stack(raw_die_list, axis=0)  # (B, 101, 2)
        stacked_x0 = np.stack(raw_x0_list, axis=0)    # (B, 101, 31, 2)
        stacked_u = np.stack(raw_u_list, axis=0)      # (B, 101, 31, 2)

        # 2. Normalize spatial coordinates to [0, 1] using global bounding box
        norm_die = (stacked_die - self.min_pos) / self.pos_span

        # Surface partition: Top boundary nodes (y ≈ 15mm, index y = -1)
        # Shape: (B, 101, 2)
        raw_surf_x0 = stacked_x0[:, :, -1, :]
        norm_surf_x0 = (raw_surf_x0 - self.min_pos) / self.pos_span

        # Volume partition: Remaining interior continuum nodes (indices y < -1)
        # Shape: (B, 101, 30, 2) -> flatten to (B, 3030, 2)
        raw_vol_x0 = stacked_x0[:, :, :-1, :].reshape(batch_size, -1, 2)
        norm_vol_x0 = (raw_vol_x0 - self.min_pos) / self.pos_span

        # 3. Extract and standardize targets
        # Surface target:
        if self.surface_channels == 1:
            # Vertical displacement u_y only along contacting boundary (B, 101, 1)
            raw_surf_u = stacked_u[:, :, -1, 1:2]
        else:
            # Full displacement vector (u_x, u_y) along contacting boundary (B, 101, 2)
            raw_surf_u = stacked_u[:, :, -1, :]

        # Volume target: Full displacement vector (u_x, u_y) in the interior (B, 3030, 2)
        raw_vol_u = stacked_u[:, :, :-1, :].reshape(batch_size, -1, 2)

        if self.stats is not None and "displacements" in self.stats:
            u_stats = self.stats["displacements"]["per_channel"]
            u_mean = np.array(u_stats["mean"], dtype=np.float32)
            u_std = np.array(u_stats["std"], dtype=np.float32)

            if self.surface_channels == 1:
                std_surf_u = (raw_surf_u - u_mean[1]) / (u_std[1] + 1e-8)
            else:
                std_surf_u = (raw_surf_u - u_mean) / (u_std + 1e-8)
            std_vol_u = (raw_vol_u - u_mean) / (u_std + 1e-8)
        else:
            std_surf_u = raw_surf_u
            std_vol_u = raw_vol_u

        # 4. Convert all prepared arrays to PyTorch tensors
        batch_dict: Dict[str, Any] = {
            "geo": torch.from_numpy(norm_die.astype(np.float32)),
            "surf_query_pos": torch.from_numpy(norm_surf_x0.astype(np.float32)),
            "vol_query_pos": torch.from_numpy(norm_vol_x0.astype(np.float32)),
            "targets": {
                "surf_data": torch.from_numpy(std_surf_u.astype(np.float32)),
                "vol_data": torch.from_numpy(std_vol_u.astype(np.float32)),
                "displacements": torch.from_numpy(stacked_u.astype(np.float32)),
            },
            "coords_initial": torch.from_numpy(stacked_x0.astype(np.float32)),
            "die_boundary_raw": torch.from_numpy(stacked_die.astype(np.float32)),
        }

        # 5. Extract optional conditioning parameter vector (e.g. normalized time t in [0, 1])
        if len(samples) > 0 and "time_params" in samples[0]:
            raw_params_list = [np.asarray(s["time_params"], dtype=np.float32) for s in samples]
            stacked_params = np.stack(raw_params_list, axis=0)
            if stacked_params.ndim == 1:
                stacked_params = stacked_params[:, None]
            batch_dict["params"] = torch.from_numpy(stacked_params.astype(np.float32))
            batch_dict["time_params"] = batch_dict["params"]
        elif len(samples) > 0 and "params" in samples[0]:
            raw_params_list = [np.asarray(s["params"], dtype=np.float32) for s in samples]
            stacked_params = np.stack(raw_params_list, axis=0)
            if stacked_params.ndim == 1:
                stacked_params = stacked_params[:, None]
            batch_dict["params"] = torch.from_numpy(stacked_params.astype(np.float32))

        return batch_dict


class StandardBatchLoader(BatchLoader):
    """BatchLoader for grid-based neural operators (e.g. FNO, UNet).

    Maintains the 2D structured mesh dimensions (B, nx, ny, C) without partitioning
    nodes into surface and volume subsets.
    """

    def __init__(self, stats: Optional[Dict[str, Any]] = None) -> None:
        self.stats = stats

    def loadDataset(self, samples: Sequence[Dict[str, Union[np.ndarray, Any]]]) -> Dict[str, torch.Tensor]:
        """Collate simulation samples into structured 2D grid tensors.

        Parameters
        ----------
        samples : Sequence[Dict[str, Union[np.ndarray, Any]]]
            List of sample dictionaries.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing 'inputs' and 'targets' formatted as 4D tensors.
        """
        raw_x0 = np.stack([np.asarray(s["coords_initial"], dtype=np.float32) for s in samples], axis=0)
        raw_u = np.stack([np.asarray(s["displacements"], dtype=np.float32) for s in samples], axis=0)
        raw_die = np.stack([np.asarray(s["die_boundary"], dtype=np.float32) for s in samples], axis=0)

        norm_u = raw_u
        if self.stats is not None and "displacements" in self.stats:
            u_stats = self.stats["displacements"]["per_channel"]
            u_mean = np.array(u_stats["mean"], dtype=np.float32)
            u_std = np.array(u_stats["std"], dtype=np.float32)
            norm_u = (raw_u - u_mean) / (u_std + 1e-8)

        return {
            "coords_initial": torch.from_numpy(raw_x0),
            "die_boundary": torch.from_numpy(raw_die),
            "targets": {
                "displacements": torch.from_numpy(norm_u),
                "raw_displacements": torch.from_numpy(raw_u),
            },
        }
