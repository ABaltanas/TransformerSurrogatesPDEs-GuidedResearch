"""Checkpoint persistence and state restoration handlers."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from src.training.interfaces import CheckpointHandler


class TorchCheckpointHandler(CheckpointHandler):
    """Manages atomic checkpoint saving, best model tracking, and checkpoint pruning.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory where checkpoint bundles will be saved.
    keep_last_n : int, default=3
        Number of most recent intermediate checkpoints to retain. If <= 0, retains all.
    """

    def __init__(self, checkpoint_dir: Path, keep_last_n: int = 3) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.saved_checkpoints: List[Path] = []
        self.best_checkpoint_path: Optional[Path] = None

    def save(
        self,
        epoch: int,
        model: nn.Module,
        metrics: Dict[str, float],
        is_best: bool,
    ) -> Path:
        """Atomically persist a training checkpoint.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        model : nn.Module
            Neural network module whose weights are to be stored.
        metrics : Dict[str, float]
            Evaluated training and validation metrics.
        is_best : bool
            Flag indicating if this checkpoint achieved the best validation score.

        Returns
        -------
        Path
            Path to the saved checkpoint file.
        """
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "is_best": is_best,
        }

        epoch_filename = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        self._atomic_save(payload, epoch_filename)
        self.saved_checkpoints.append(epoch_filename)

        # Update latest checkpoint pointer
        latest_filename = self.checkpoint_dir / "latest.pt"
        self._atomic_save(payload, latest_filename)

        # Update best model checkpoint if requested
        if is_best:
            best_filename = self.checkpoint_dir / "best_model.pt"
            self._atomic_save(payload, best_filename)
            self.best_checkpoint_path = best_filename

        # Prune old intermediate checkpoints
        self._prune_old_checkpoints()

        return epoch_filename

    def _atomic_save(self, payload: Dict[str, Any], target_path: Path) -> None:
        """Write checkpoint to temporary file first and replace atomically."""
        temp_path = target_path.with_suffix(".pt.tmp")
        torch.save(payload, temp_path)
        os.replace(temp_path, target_path)

    def _prune_old_checkpoints(self) -> None:
        """Remove older intermediate checkpoints exceeding keep_last_n threshold."""
        if self.keep_last_n > 0:
            while len(self.saved_checkpoints) > self.keep_last_n:
                oldest = self.saved_checkpoints.pop(0)
                if oldest.exists() and oldest != self.best_checkpoint_path:
                    try:
                        oldest.unlink()
                    except OSError:
                        pass

    def load_best(self, model: nn.Module) -> Dict[str, Any]:
        """Load the best saved weights into the target model module.

        Parameters
        ----------
        model : nn.Module
            Model module to populate with weights.

        Returns
        -------
        Dict[str, Any]
            Loaded checkpoint payload.

        Raises
        ------
        FileNotFoundError
            If no best_model.pt exists in the checkpoint directory.
        """
        best_path = self.get_best_checkpoint_path()
        if best_path is None or not best_path.exists():
            raise FileNotFoundError(f"No best checkpoint found in {self.checkpoint_dir}")

        checkpoint = torch.load(best_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Return the path to the best recorded checkpoint file.

        Returns
        -------
        Optional[Path]
            Path to best_model.pt, or None if not present.
        """
        expected = self.checkpoint_dir / "best_model.pt"
        if expected.exists():
            return expected
        return self.best_checkpoint_path
