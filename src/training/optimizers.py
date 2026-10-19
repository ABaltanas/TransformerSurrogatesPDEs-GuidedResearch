"""Optimization and learning rate schedule managers."""

from typing import Any, Iterable, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer as TorchOptimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
)

from src.training.interfaces import Optimizer


class StandardOptimizer(Optimizer):
    """Encapsulates PyTorch optimization, gradient clipping, and learning rate scheduling.

    Parameters
    ----------
    optimizer : TorchOptimizer
        PyTorch optimizer instance (e.g. AdamW).
    scheduler : Optional[Any], default=None
        PyTorch learning rate scheduler instance.
    parameters : Optional[Iterable[nn.Parameter]], default=None
        Model parameters to clip during optimization.
    """

    def __init__(
        self,
        optimizer: TorchOptimizer,
        scheduler: Optional[Any] = None,
        parameters: Optional[Iterable[nn.Parameter]] = None,
        use_amp: bool = False,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = list(parameters) if parameters is not None else None
        self.use_amp = use_amp
        if scaler is not None:
            self.scaler = scaler
        elif use_amp:
            self.scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
        else:
            self.scaler = None

    @classmethod
    def create_adamw(
        cls,
        model_parameters: Iterable[nn.Parameter],
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        scheduler_type: Optional[str] = "cosine",
        epochs: int = 100,
        min_lr: float = 1e-6,
        use_amp: bool = False,
    ) -> "StandardOptimizer":
        """Factory constructor instantiating AdamW and standard schedulers.

        Parameters
        ----------
        model_parameters : Iterable[nn.Parameter]
            Parameters of the neural network module.
        lr : float, default=1e-4
            Initial learning rate.
        weight_decay : float, default=1e-4
            Weight decay coefficient.
        scheduler_type : Optional[str], default='cosine'
            Type of scheduler: 'cosine', 'plateau', or None.
        epochs : int, default=100
            Total planned training epochs.
        min_lr : float, default=1e-6
            Minimum learning rate for cosine annealing.

        Returns
        -------
        StandardOptimizer
            Configured optimizer instance.
        """
        params_list = list(model_parameters)
        optimizer = AdamW(params_list, lr=lr, weight_decay=weight_decay)

        scheduler = None
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        elif scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=min_lr)

        return cls(
            optimizer=optimizer,
            scheduler=scheduler,
            parameters=params_list,
            use_amp=use_amp,
        )

    def zero_grad(self) -> None:
        """Reset parameter gradients."""
        self.optimizer.zero_grad()

    def step(self, loss: torch.Tensor, max_grad_norm: Optional[float] = None) -> float:
        """Execute backward pass, gradient clipping, and parameter update.

        Parameters
        ----------
        loss : torch.Tensor
            Scalar loss tensor to backpropagate.
        max_grad_norm : Optional[float], default=None
            Threshold for gradient norm clipping.

        Returns
        -------
        float
            Gradient norm before clipping (or 0.0 if not computed).
        """
        grad_norm = 0.0
        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            if max_grad_norm is not None and self.parameters is not None:
                self.scaler.unscale_(self.optimizer)
                norm_tensor = nn.utils.clip_grad_norm_(self.parameters, max_grad_norm)
                grad_norm = float(norm_tensor.item())
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None and self.parameters is not None:
                norm_tensor = nn.utils.clip_grad_norm_(self.parameters, max_grad_norm)
                grad_norm = float(norm_tensor.item())
            self.optimizer.step()

        return grad_norm

    def epoch_step(self, metric: Optional[float] = None) -> float:
        """Step the learning rate scheduler and return the updated learning rate.

        Parameters
        ----------
        metric : Optional[float], default=None
            Validation metric for ReduceLROnPlateau.

        Returns
        -------
        float
            Updated active learning rate.
        """
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()

        return self.get_learning_rate()

    def get_learning_rate(self) -> float:
        """Retrieve the current learning rate.

        Returns
        -------
        float
            Learning rate from the first parameter group.
        """
        return float(self.optimizer.param_groups[0]["lr"])

    def get_state_dict(self) -> Dict[str, Any]:
        """Retrieve state dictionaries for persistence.

        Returns
        -------
        Dict[str, Any]
            State dictionary containing optimizer, scheduler, and scaler states.
        """
        states: Dict[str, Any] = {
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            states["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None and hasattr(self.scaler, "state_dict"):
            states["scaler_state_dict"] = self.scaler.state_dict()
        return states

    def load_state(
        self,
        checkpoint: Dict[str, Any],
        step_scheduler_to_epoch: Optional[int] = None,
    ) -> None:
        """Load optimizer, scaler, and scheduler states from a checkpoint.

        Parameters
        ----------
        checkpoint : Dict[str, Any]
            Loaded checkpoint payload.
        step_scheduler_to_epoch : Optional[int], default=None
            If scheduler_state_dict is missing from checkpoint, fast-forward the scheduler
            by this number of epochs to reach the proper learning rate.
        """
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if (
            "scaler_state_dict" in checkpoint
            and checkpoint["scaler_state_dict"] is not None
            and self.scaler is not None
        ):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if (
            "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"] is not None
            and self.scheduler is not None
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        elif step_scheduler_to_epoch is not None and step_scheduler_to_epoch > 0 and self.scheduler is not None:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(step_scheduler_to_epoch):
                    self.scheduler.step()
