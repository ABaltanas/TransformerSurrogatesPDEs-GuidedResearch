"""Loss engines for surrogate model training and physical constraint enforcement."""

from typing import Dict, Tuple
import torch
import torch.nn as nn

from src.training.interfaces import LossEngine


class RelativeL2LossEngine(LossEngine):
    """Calculates relative Frobenius L2 loss across predicted physical fields.

    Parameters
    ----------
    eps : float, default=1e-7
        Epsilon constant for numerical stability in division.
    """

    def __init__(self, eps: float = 1e-7) -> None:
        self.eps = eps

    def compute(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute relative L2 loss for predictions against targets.

        Supports partitioned outputs (surface and volume) as well as full displacement
        tensors.

        Parameters
        ----------
        predictions : Dict[str, torch.Tensor]
            Dictionary containing model predictions ('surf_pred', 'vol_pred', 'displacements').
        targets : Dict[str, torch.Tensor]
            Dictionary containing ground-truth targets ('surf_data', 'vol_data', 'displacements').
        batch : Dict[str, torch.Tensor]
            Complete input batch dictionary.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, float]]
            Backpropagatable total loss and breakdown dictionary.
        """
        loss_components: Dict[str, float] = {}

        # 1. Partitioned evaluation if model produced surf and vol predictions
        if "surf_pred" in predictions and "surf_data" in targets and "vol_pred" in predictions and "vol_data" in targets:
            surf_pred = predictions["surf_pred"]
            surf_true = targets["surf_data"]
            vol_pred = predictions["vol_pred"]
            vol_true = targets["vol_data"]

            # Surface relative L2
            surf_diff = torch.norm(surf_pred - surf_true, p=2, dim=-1)
            surf_norm = torch.norm(surf_true, p=2, dim=-1) + self.eps
            loss_surf = torch.mean(surf_diff / surf_norm)

            # Volume relative L2
            vol_diff = torch.norm(vol_pred - vol_true, p=2, dim=-1)
            vol_norm = torch.norm(vol_true, p=2, dim=-1) + self.eps
            loss_vol = torch.mean(vol_diff / vol_norm)

            total_loss = loss_surf + loss_vol
            loss_components["loss_surf"] = float(loss_surf.item())
            loss_components["loss_vol"] = float(loss_vol.item())
            loss_components["loss_total"] = float(total_loss.item())
            return total_loss, loss_components

        # 2. Unified displacement field evaluation
        pred_u = predictions["displacements"]
        true_u = targets["displacements"]

        diff = torch.norm(pred_u - true_u, p=2, dim=-1)
        norm = torch.norm(true_u, p=2, dim=-1) + self.eps
        total_loss = torch.mean(diff / norm)

        loss_components["loss_rel_l2"] = float(total_loss.item())
        loss_components["loss_total"] = float(total_loss.item())
        return total_loss, loss_components


class MSELossEngine(LossEngine):
    """Calculates standard Mean Squared Error (MSE) loss across target fields."""

    def __init__(self) -> None:
        self.criterion = nn.MSELoss()

    def compute(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute mean squared error loss."""
        pred_u = predictions["displacements"]
        true_u = targets["displacements"]

        total_loss = self.criterion(pred_u, true_u)
        return total_loss, {
            "loss_mse": float(total_loss.item()),
            "loss_total": float(total_loss.item()),
        }


class CompositePhysicsLossEngine(LossEngine):
    """Combines relative L2 empirical loss with physical boundary contact penalties.

    Penalizes non-penetration constraint violations where the predicted workpiece
    surface coordinates exceed the rigid die boundary:
    Δ = max(0, y_workpiece - y_die)

    Parameters
    ----------
    lambda_contact : float, default=0.1
        Weight multiplier for the contact non-penetration penalty term.
    eps : float, default=1e-7
        Numerical stability constant for division in relative L2 norm.
    """

    def __init__(self, lambda_contact: float = 0.1, eps: float = 1e-7) -> None:
        self.base_engine = RelativeL2LossEngine(eps=eps)
        self.lambda_contact = lambda_contact

    def compute(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined empirical and physical penalty loss."""
        base_loss, components = self.base_engine.compute(predictions, targets, batch)

        # Check if coordinates and die contour are present for penetration calculation
        if "coords_initial" in batch and "die_boundary_raw" in batch:
            coords_x0 = batch["coords_initial"]        # (B, 101, 31, 2)
            die_raw = batch["die_boundary_raw"]        # (B, 101, 2)
            pred_u = predictions["displacements"]      # (B, 101, 31, 2)

            # Top surface deformed coordinates (y = -1)
            surf_x0 = coords_x0[:, :, -1, :]           # (B, 101, 2)
            surf_u = pred_u[:, :, -1, :]               # (B, 101, 2)
            surf_y_pred = surf_x0[:, :, 1] + surf_u[:, :, 1]
            die_y = die_raw[:, :, 1]

            # Penalize penetration: y_pred > die_y
            penetration = torch.relu(surf_y_pred - die_y)
            contact_loss = torch.mean(penetration ** 2)

            total_loss = base_loss + self.lambda_contact * contact_loss
            components["loss_contact"] = float(contact_loss.item())
            components["loss_total"] = float(total_loss.item())
            return total_loss, components

        return base_loss, components
