"""ModelWrapper implementation for the SMART surrogate architecture."""

from pathlib import Path
import sys
from typing import Any, Dict, Optional
import torch
import torch.nn as nn

# Ensure external/smart repository is discoverable on python path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SMART_REPO = _PROJECT_ROOT / "external" / "smart"
if _SMART_REPO.exists() and str(_SMART_REPO) not in sys.path:
    sys.path.insert(0, str(_SMART_REPO))

try:
    from smart.models.smart.smart import SMART
except ImportError:
    SMART = None  # Fallback if external submodule is uninitialized

from src.training.interfaces import ModelWrapper


class SMARTModelWrapper(ModelWrapper):
    """Wrapper encapsulating the SMART architecture for 2D elastoplastic forging.

    Initializes the SMART model, handles keyword argument unpacking during the forward
    pass, and reconstructs the full 2D displacement field tensor from partitioned
    surface and volume predictions.

    Parameters
    ----------
    config : Optional[Dict[str, Any]], default=None
        Hyperparameter configuration dictionary for SMART. Default values align with
        the 2D plasticity benchmark (spatial_dim=2, latent_dim=128, etc.).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if SMART is None:
            raise ImportError(
                "Failed to import SMART from external/smart. "
                "Ensure git submodules are initialized: git submodule update --init --recursive"
            )

        cfg = config or {}

        self.spatial_dim = int(cfg.get("spatial_dim", 2))
        self.surface_channels = int(cfg.get("surface_channels", 2))
        self.volume_channels = int(cfg.get("volume_channels", 2))
        self.parameter_channels = int(cfg.get("parameter_channels", 0))
        self.latent_dim = int(cfg.get("latent_dim", 128))
        self.latent_geometry_points = int(cfg.get("latent_geometry_points", 64))
        self.subsampled_geometry_points = int(cfg.get("subsampled_geometry_points", 101))
        self.num_encoder_decoder_blocks = int(cfg.get("num_encoder_decoder_blocks", 6))
        self.num_heads = int(cfg.get("num_heads", 8))
        self.pos_scale_factor = float(cfg.get("pos_scale_factor", 1000.0))
        self.dropout = float(cfg.get("dropout", 0.0))

        # 1. Instantiate the deep learning model instance
        self.model = SMART(
            spatial_dim=self.spatial_dim,
            surface_channels=self.surface_channels,
            volume_channels=self.volume_channels,
            parameter_channels=self.parameter_channels,
            latent_dim=self.latent_dim,
            latent_geometry_points=self.latent_geometry_points,
            subsampled_geometry_points=self.subsampled_geometry_points,
            num_encoder_decoder_blocks=self.num_encoder_decoder_blocks,
            num_heads=self.num_heads,
            pos_scale_factor=self.pos_scale_factor,
            dropout=self.dropout,
        )
        self.model.initialize_weights()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Execute forward pass through SMART and reconstruct full displacement field.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Batch dictionary containing 'geo', 'surf_query_pos', 'vol_query_pos',
            and optional 'params'.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'surf_pred': (B, 101, C_surf) surface predictions.
            - 'vol_pred': (B, 3030, C_vol) volume predictions.
            - 'displacements': (B, 101, 31, 2) reconstructed full 2D displacement field.
        """
        geo = batch["geo"]
        surf_query_pos = batch["surf_query_pos"]
        vol_query_pos = batch["vol_query_pos"]
        params = batch.get("params", None)

        # Execute SMART forward pass
        y_hat_surf, y_hat_vol = self.model(
            geo=geo,
            surf_query_pos=surf_query_pos,
            vol_query_pos=vol_query_pos,
            params=params,
        )

        # Reconstruct the full (B, 101, 31, 2) mesh displacement tensor
        full_displacements = self._reconstruct_displacement_field(
            surf_pred=y_hat_surf,
            vol_pred=y_hat_vol,
            batch_size=geo.shape[0],
            device=geo.device,
        )

        return {
            "surf_pred": y_hat_surf,
            "vol_pred": y_hat_vol,
            "displacements": full_displacements,
        }

    def _reconstruct_displacement_field(
        self,
        surf_pred: torch.Tensor,
        vol_pred: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Stitch partitioned surface and volume predictions into unified mesh shape.

        Parameters
        ----------
        surf_pred : torch.Tensor
            Predictions at top boundary nodes, shape (B, 101, C_surf).
        vol_pred : torch.Tensor
            Predictions at interior nodes, shape (B, 3030, C_vol).
        batch_size : int
            Active batch size B.
        device : torch.device
            Target tensor execution device.

        Returns
        -------
        torch.Tensor
            Unified displacement field with shape (B, 101, 31, 2).
        """
        # Reshape interior volume predictions to (B, 101, 30, 2)
        vol_mesh = vol_pred.reshape(batch_size, 101, 30, -1)

        if self.surface_channels == 2:
            surf_mesh = surf_pred.unsqueeze(2)  # (B, 101, 1, 2)
        elif self.surface_channels == 1:
            # If surface predicted only u_y, zero-fill u_x or take horizontal deformation from adjacent nodes
            surf_ux = torch.zeros((batch_size, 101, 1, 1), device=device, dtype=surf_pred.dtype)
            surf_uy = surf_pred.unsqueeze(2)  # (B, 101, 1, 1)
            surf_mesh = torch.cat([surf_ux, surf_uy], dim=-1)  # (B, 101, 1, 2)
        else:
            surf_mesh = surf_pred[:, :, :2].unsqueeze(2)

        # Concatenate along the y-coordinate dimension (30 interior + 1 top = 31)
        full_u = torch.cat([vol_mesh[:, :, :, :2], surf_mesh[:, :, :, :2]], dim=2)
        return full_u

    def get_torch_module(self) -> nn.Module:
        """Return encapsulated SMART PyTorch module for optimizer registration.

        Returns
        -------
        nn.Module
            The internal SMART instance.
        """
        return self.model
