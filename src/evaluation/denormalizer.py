"""Physical scaling and denormalization strategies for surrogate evaluation."""

from typing import Any, Dict, Tuple
import numpy as np

from src.evaluation.interfaces import Denormalizer


class StatisticalDenormalizer(Denormalizer):
    """Restores physical engineering units (millimeters) using training scaling moments.

    Applies the mathematical inversion of standardization:
    u_phys = u_norm * std + mean
    x_phys = coords_initial + u_phys
    """

    def denormalize(
        self,
        pred_norm: Dict[str, np.ndarray],
        target_norm: Dict[str, np.ndarray],
        stats: Dict[str, Any],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Invert normalization to obtain true millimeter fields.

        Parameters
        ----------
        pred_norm : Dict[str, np.ndarray]
            Predictions from the model. Must include 'displacements'.
        target_norm : Dict[str, np.ndarray]
            Ground truth target fields from the test dataset.
        stats : Dict[str, Any]
            Scaling parameters from statistical_data.json.

        Returns
        -------
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
            Pair of physical prediction and target dictionaries.
        """
        phys_preds: Dict[str, np.ndarray] = dict(pred_norm)
        phys_targets: Dict[str, np.ndarray] = dict(target_norm)

        # Retrieve displacement scaling moments
        if "displacements" in stats and "per_channel" in stats["displacements"]:
            u_stats = stats["displacements"]["per_channel"]
            u_mean = np.array(u_stats["mean"], dtype=np.float64)
            u_std = np.array(u_stats["std"], dtype=np.float64)

            # Invert predictions
            pred_u_norm = pred_norm["displacements"].astype(np.float64)
            pred_u_phys = pred_u_norm * u_std + u_mean
            phys_preds["displacements"] = pred_u_phys

            # Check if targets were already in physical scale or normalized
            if "raw_displacements" in target_norm:
                phys_targets["displacements"] = target_norm["raw_displacements"].astype(np.float64)
            elif "displacements" in target_norm:
                target_u = target_norm["displacements"].astype(np.float64)
                # If target is normalized (variance close to 1), denormalize it
                if np.abs(np.std(target_u) - 1.0) < 0.2 and np.abs(np.mean(target_u)) < 0.2:
                    phys_targets["displacements"] = target_u * u_std + u_mean
                else:
                    phys_targets["displacements"] = target_u

        # Reconstruct physical coordinates: x_phys = x0 + u
        if "coords_initial" in target_norm:
            x0 = target_norm["coords_initial"].astype(np.float64)
            phys_preds["coords"] = x0 + phys_preds["displacements"]
            if "coords" in target_norm:
                phys_targets["coords"] = target_norm["coords"].astype(np.float64)
            else:
                phys_targets["coords"] = x0 + phys_targets["displacements"]

        return phys_preds, phys_targets
