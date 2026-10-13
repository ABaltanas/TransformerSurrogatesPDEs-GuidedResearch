"""Physical setup and restructuring components for scientific simulation data."""

from typing import Dict

import numpy as np

from src.data.interfaces import PhysicalSetup


class PlasticitySetup(PhysicalSetup):
    """Physical setup strategy for the 2D elastoplastic forging benchmark."""

    def setup(self, raw_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Align coordinates and structure physical tensors.

        Parameters
        ----------
        raw_dict : Dict[str, np.ndarray]
            Raw dictionary containing 'input' (N, 101) and 'output' (N, 101, 31, 20, 4).

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of structured physical fields:
            - 'die_boundary': (N, 101, 2) 3D top die boundary contour [x, S_d(x)]
            - 'coords_initial': (N, 101, 31, 2) 4D undeformed reference mesh (X_0, Y_0) at t=0
            - 'coords': (N, 101, 31, 20, 2) 5D deformed mesh coordinates (X, Y) over time
            - 'displacements': (N, 101, 31, 20, 2) 5D displacement fields (u_x, u_y) over time
            - 'target_all': (N, 101, 31, 20, 4) 5D all 4 solution channels [X, Y, u_x, u_y]
            - 'time_steps': (20,) 1D normalized time grid t in [0, 1]
        """
        inp: np.ndarray = raw_dict["input"]
        out: np.ndarray = raw_dict["output"]

        num_timesteps: int = out.shape[3]

        # In raw ABAQUS output, horizontal axis 1 is indexed from +0.35 down to -49.65.
        # Flip along axis 1 if descending so both inputs and outputs are monotonically increasing.
        if out.shape[1] > 1 and out[0, 0, 0, 0, 0] > out[0, -1, 0, 0, 0]:
            out = np.flip(out, axis=1).copy()

        # 4D Reference initial mesh at t = 0: (N, 101, 31, 2)
        coords_initial: np.ndarray = out[:, :, :, 0, :2].copy()

        # 3D Top boundary die profile: (N, 101, 2)
        x_top_boundary: np.ndarray = coords_initial[:, :, 0, 0]
        die_boundary: np.ndarray = np.stack(
            [x_top_boundary, inp], axis=-1
        ).astype(np.float32)

        # 5D Deformed coordinates (X, Y) over time: (N, 101, 31, 20, 2)
        coords: np.ndarray = out[:, :, :, :, :2].copy()

        # 5D Displacements (u_x, u_y) over time: (N, 101, 31, 20, 2)
        displacements: np.ndarray = out[:, :, :, :, 2:4].copy()

        # 5D Full 4-channel target: (N, 101, 31, 20, 4)
        target_all: np.ndarray = out.copy()

        # 1D Normalized time increments: (20,)
        time_steps: np.ndarray = np.linspace(
            0.0, 1.0, num_timesteps, dtype=np.float32
        )

        return {
            "die_boundary": die_boundary,
            "coords_initial": coords_initial,
            "coords": coords,
            "displacements": displacements,
            "target_all": target_all,
            "time_steps": time_steps,
        }
