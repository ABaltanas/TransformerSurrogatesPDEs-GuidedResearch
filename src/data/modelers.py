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
