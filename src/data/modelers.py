"""Dataset modeler implementations for ML task formulation and modification."""

from typing import Dict, Sequence

import numpy as np

from src.data.interfaces import DatasetModeler


class ConsiderOnlyFinalStepModeler(DatasetModeler):
    """Retains only the final time step of simulation trajectories.

    Transforms 5D trajectory tensors of shape (N, nx, ny, nt, C) into 4D static
    prediction tensors of shape (N, nx, ny, C) at the final time increment t = -1.
    Removes the time_steps vector to produce a purely static dataset formulation.
    """

    def modify(self, fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Slice 5D temporal fields down to the 4D final-step state.

        Parameters
        ----------
        fields : Dict[str, np.ndarray]
            Dictionary of physical fields produced by PhysicalSetup.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing 4D final-step fields with time_steps removed.
        """
        modified: Dict[str, np.ndarray] = dict(fields)

        if "coords" in modified and modified["coords"].ndim == 5:
            modified["coords"] = modified["coords"][:, :, :, -1, :]

        if "displacements" in modified and modified["displacements"].ndim == 5:
            modified["displacements"] = modified["displacements"][:, :, :, -1, :]

        if "target_all" in modified and modified["target_all"].ndim == 5:
            modified["target_all"] = modified["target_all"][:, :, :, -1, :]

        if "time_steps" in modified:
            del modified["time_steps"]

        return modified


class CompositeModeler(DatasetModeler):
    """Chains multiple DatasetModeler modifications in sequence (Composite Pattern).

    Executes a list of child modelers sequentially, passing the output of each
    as the input to the next:
        fields_k = modeler_k.modify(fields_{k-1})
    """

    def __init__(self, modelers: Sequence[DatasetModeler]) -> None:
        """Initialize CompositeModeler with an ordered sequence of child modelers.

        Parameters
        ----------
        modelers : Sequence[DatasetModeler]
            Sequence of DatasetModeler instances to execute in order.
        """
        self.modelers = list(modelers)

    def modify(self, fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply all chained modelers in sequential order.

        Parameters
        ----------
        fields : Dict[str, np.ndarray]
            Input fields dictionary.

        Returns
        -------
        Dict[str, np.ndarray]
            Fields dictionary resulting from applying all chained modelers.
        """
        current_fields: Dict[str, np.ndarray] = fields
        for modeler in self.modelers:
            current_fields = modeler.modify(current_fields)
        return current_fields


class TemporalTrajectoryModeler(DatasetModeler):
    """Unrolls temporal simulation trajectories into parameter-conditioned static instances.

    Transforms 5D trajectory tensors of shape (N, nx, ny, nt, C) into unrolled 4D tensors
    of shape (N * nt, nx, ny, C), while replicating invariant boundary and initial conditions
    and generating an explicit scalar conditioning parameter tensor 'time_params' of shape
    (N * nt, 1) containing normalized physical time t in [0, 1].

    Preserves simulation and time step identifiers to allow downstream grouping:
    - 'simulation_idx': shape (N * nt, 1), integer identifier of the source simulation run.
    - 'time_step_idx': shape (N * nt, 1), integer index of the time increment (0 to nt - 1).

    Parameters
    ----------
    include_initial_step : bool, default=True
        Whether to include t = 0 (undeformed reference state with zero displacement).
        If False, starts from t = 1 (active deformation).
    """

    def __init__(self, include_initial_step: bool = True) -> None:
        self.include_initial_step = include_initial_step

    def modify(self, fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Unroll temporal simulation trajectories along the sample dimension.

        Parameters
        ----------
        fields : Dict[str, np.ndarray]
            Dictionary of physical fields produced by PhysicalSetup.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing unrolled 4D fields and 2D conditioning parameter tensors.
        """
        modified: Dict[str, np.ndarray] = dict(fields)

        # Detect total simulations and temporal steps
        sample_lengths = [
            arr.shape[0] for k, arr in modified.items() if k != "time_steps" and arr.ndim > 1
        ]
        if not sample_lengths:
            raise ValueError("No sample-indexed arrays found in input fields dictionary.")

        num_simulations = sample_lengths[0]

        # Extract normalized time steps
        if "time_steps" in modified:
            time_steps_raw = modified["time_steps"]
        elif "coords" in modified and modified["coords"].ndim == 5:
            num_t = modified["coords"].shape[3]
            time_steps_raw = np.linspace(0.0, 1.0, num_t, dtype=np.float32)
        else:
            raise ValueError("Cannot determine time dimension: missing 'time_steps' or 5D tensor.")

        total_timesteps = len(time_steps_raw)
        step_slice = slice(0 if self.include_initial_step else 1, None)
        active_times = time_steps_raw[step_slice]
        num_active_t = len(active_times)
        total_unrolled = num_simulations * num_active_t

        # 1. Unroll 5D temporal tensors: (N, nx, ny, nt, C) -> (N * nt, nx, ny, C)
        for key in ["coords", "displacements", "target_all"]:
            if key in modified and modified[key].ndim == 5:
                arr = modified[key][:, :, :, step_slice, :]
                modified[key] = arr.transpose(0, 3, 1, 2, 4).reshape(
                    total_unrolled, arr.shape[1], arr.shape[2], arr.shape[-1]
                )

        # 2. Replicate static inputs: (N, ...) -> (N * nt, ...)
        if "die_boundary" in modified:
            modified["die_boundary"] = np.repeat(
                modified["die_boundary"], num_active_t, axis=0
            )

        if "coords_initial" in modified:
            modified["coords_initial"] = np.repeat(
                modified["coords_initial"], num_active_t, axis=0
            )

        if "die_profile" in modified:
            modified["die_profile"] = np.repeat(
                modified["die_profile"], num_active_t, axis=0
            )

        # 3. Create scalar conditioning parameters: (N * nt, 1)
        modified["time_params"] = np.tile(
            active_times.reshape(-1, 1), (num_simulations, 1)
        ).astype(np.float32)

        # 4. Create metadata indexing arrays for tracking: (N * nt, 1)
        modified["simulation_idx"] = np.repeat(
            np.arange(num_simulations, dtype=np.int32).reshape(-1, 1),
            num_active_t,
            axis=0,
        )

        step_indices = np.arange(
            0 if self.include_initial_step else 1, total_timesteps, dtype=np.int32
        ).reshape(-1, 1)
        modified["time_step_idx"] = np.tile(step_indices, (num_simulations, 1))

        return modified

