"""Solid mechanics physical feasibility and contact boundary validators."""

from typing import Dict
import numpy as np

from src.evaluation.interfaces import FeasibilityValidator


class DiePenetrationValidator(FeasibilityValidator):
    """Verifies that predicted workpiece deformation does not penetrate the rigid upper die.

    Parameters
    ----------
    tolerance_mm : float, default=0.05
        Allowable penetration threshold in millimeters for passing feasibility.
    """

    def __init__(self, tolerance_mm: float = 0.05) -> None:
        self.tolerance_mm = tolerance_mm

    def check_constraints(
        self,
        pred_coords: np.ndarray,
        boundary_constraints: np.ndarray,
    ) -> Dict[str, float]:
        """Quantify contact penetration violations.

        Parameters
        ----------
        pred_coords : np.ndarray
            Predicted deformed coordinates, shape (N, 101, 31, 2).
        boundary_constraints : np.ndarray
            Rigid die boundary coordinates, shape (N, 101, 2).

        Returns
        -------
        Dict[str, float]
            Feasibility violation statistics.
        """
        num_samples = pred_coords.shape[0]

        # Workpiece top surface coordinates (select boundary with higher elevation along axis 2)
        top_idx = 0 if np.mean(pred_coords[:, :, 0, 1]) > np.mean(pred_coords[:, :, -1, 1]) else -1
        surf_y = pred_coords[:, :, top_idx, 1].astype(np.float64)  # (N, 101)
        die_y = boundary_constraints[:, :, 1].astype(np.float64)  # (N, 101)

        # Penetration depth: Δ = max(0, surf_y - die_y)
        penetration = np.maximum(0.0, surf_y - die_y)  # (N, 101)

        max_penetration = float(np.max(penetration))

        violating_mask = penetration > 1e-4
        if np.any(violating_mask):
            mean_penetration = float(np.mean(penetration[violating_mask]))
        else:
            mean_penetration = 0.0

        total_boundary_nodes = float(penetration.size)
        violating_nodes = float(np.sum(violating_mask))
        violation_rate = (violating_nodes / total_boundary_nodes) * 100.0

        is_pass = 1.0 if max_penetration <= self.tolerance_mm else 0.0

        return {
            "max_die_penetration_mm": max_penetration,
            "mean_die_penetration_mm": mean_penetration,
            "penetration_violation_rate_pct": violation_rate,
            "contact_feasibility_pass": is_pass,
        }
