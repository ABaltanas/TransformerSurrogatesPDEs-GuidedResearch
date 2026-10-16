"""Solid mechanics PDE error metrics calculator."""

from typing import Dict
import numpy as np

from src.evaluation.interfaces import PhysicalMetricsCalculator


class PlasticityMetricsCalculator(PhysicalMetricsCalculator):
    """Calculates quantitative scientific error metrics on physical displacement fields.

    Parameters
    ----------
    eps : float, default=1e-8
        Numerical stability constant for division.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def compute_metrics(
        self,
        pred_phys: Dict[str, np.ndarray],
        target_phys: Dict[str, np.ndarray],
        initial_mesh: np.ndarray,
    ) -> Dict[str, float]:
        """Compute comprehensive suite of physical accuracy metrics.

        Parameters
        ----------
        pred_phys : Dict[str, np.ndarray]
            Physical predictions dictionary containing 'displacements' (N, 101, 31, 2).
        target_phys : Dict[str, np.ndarray]
            Physical ground-truth dictionary containing 'displacements' (N, 101, 31, 2).
        initial_mesh : np.ndarray
            Reference undeformed grid coordinates (N, 101, 31, 2).

        Returns
        -------
        Dict[str, float]
            Evaluated metrics dictionary.
        """
        pred_u = pred_phys["displacements"].astype(np.float64)
        true_u = target_phys["displacements"].astype(np.float64)

        rel_l2 = self.compute_relative_l2(pred_u, true_u)
        l_inf = self.compute_l_infinity(pred_u, true_u)
        surf_vol_dict = self.compute_surface_vs_volume(pred_u, true_u)
        h1_norm = self.compute_sobolev_h1(pred_u, true_u)

        return {
            "rel_l2_overall": rel_l2,
            "l_inf_max_mm": l_inf,
            "surface_rel_l2": surf_vol_dict["surface_rel_l2"],
            "volume_rel_l2": surf_vol_dict["volume_rel_l2"],
            "sobolev_h1_semi_norm": h1_norm,
        }

    def compute_relative_l2(self, pred_u: np.ndarray, true_u: np.ndarray) -> float:
        """Compute mean relative L2 Frobenius error across samples."""
        num_samples = pred_u.shape[0]
        sample_errors = []

        for i in range(num_samples):
            diff = np.linalg.norm(pred_u[i] - true_u[i])
            norm = np.linalg.norm(true_u[i]) + self.eps
            sample_errors.append(diff / norm)

        return float(np.mean(sample_errors))

    def compute_l_infinity(self, pred_u: np.ndarray, true_u: np.ndarray) -> float:
        """Compute the maximum pointwise absolute error in millimeters."""
        diff = np.abs(pred_u - true_u)
        return float(np.max(diff))

    def compute_surface_vs_volume(self, pred_u: np.ndarray, true_u: np.ndarray) -> Dict[str, float]:
        """Compute partitioned relative L2 errors on contact surface vs. continuum volume."""
        num_samples = pred_u.shape[0]

        surf_errors = []
        vol_errors = []

        for i in range(num_samples):
            # Surface row (y = -1, y ≈ 15mm)
            s_diff = np.linalg.norm(pred_u[i, :, -1, :] - true_u[i, :, -1, :])
            s_norm = np.linalg.norm(true_u[i, :, -1, :]) + self.eps
            surf_errors.append(s_diff / s_norm)

            # Volume interior (y < -1)
            v_diff = np.linalg.norm(pred_u[i, :, :-1, :] - true_u[i, :, :-1, :])
            v_norm = np.linalg.norm(true_u[i, :, :-1, :]) + self.eps
            vol_errors.append(v_diff / v_norm)

        return {
            "surface_rel_l2": float(np.mean(surf_errors)),
            "volume_rel_l2": float(np.mean(vol_errors)),
        }

    def compute_sobolev_h1(self, pred_u: np.ndarray, true_u: np.ndarray) -> float:
        """Compute Sobolev H1 semi-norm error evaluating finite difference deformation gradients."""
        num_samples = pred_u.shape[0]
        h1_errors = []

        for i in range(num_samples):
            # Gradient along mesh axis 0 (xi) and axis 1 (eta)
            pred_grad_x, pred_grad_y = np.gradient(pred_u[i], axis=(0, 1))
            true_grad_x, true_grad_y = np.gradient(true_u[i], axis=(0, 1))

            diff_grad = np.sqrt(
                np.sum((pred_grad_x - true_grad_x) ** 2) + np.sum((pred_grad_y - true_grad_y) ** 2)
            )
            norm_grad = np.sqrt(
                np.sum(true_grad_x ** 2) + np.sum(true_grad_y ** 2)
            ) + self.eps

            h1_errors.append(diff_grad / norm_grad)

        return float(np.mean(h1_errors))
