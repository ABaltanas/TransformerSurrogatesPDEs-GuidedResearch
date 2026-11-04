"""Optimization and learning rate schedule managers utilizing Bridge and Strategy patterns."""

import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer as TorchOptimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    ReduceLROnPlateau,
    SequentialLR,
)

from src.training.interfaces import KillSwitch, Optimizer


class Lion(TorchOptimizer):
    """Lion Optimizer (EvoLved Sign Momentum).

    Chen et al. (Google Brain, 2023): 'Symbolic Discovery of Optimization Algorithms'.
    Adheres strictly to the PyTorch Optimizer contract without external dependencies.

    Parameters
    ----------
    params : Iterable[nn.Parameter]
        Iterable of parameters to optimize or dicts defining parameter groups.
    lr : float, default=1e-4
        Learning rate.
    betas : Tuple[float, float], default=(0.9, 0.99)
        Coefficients for computing sign update and running average of gradients.
    weight_decay : float, default=0.0
        Decoupled weight decay coefficient.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        """Execute a single optimization step.

        Parameters
        ----------
        closure : Optional[Callable], default=None
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        Optional[float]
            The evaluated loss if closure was provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]

                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Update = sign(beta1 * exp_avg + (1 - beta1) * grad)
                update = exp_avg.clone().mul_(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                p.add_(update, alpha=-lr)

                # Momentum tracking = beta2 * exp_avg + (1 - beta2) * grad
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


class NaNOrDivergenceKillSwitch(KillSwitch):
    """Health & safety strategy detecting NaN/Inf values and loss or gradient explosions.

    Parameters
    ----------
    max_loss : float, default=1e6
        Upper loss threshold above which training is immediately aborted.
    max_grad_norm : Optional[float], default=1e6
        Upper gradient norm threshold above which training is immediately aborted.
    """

    def __init__(self, max_loss: float = 1e6, max_grad_norm: Optional[float] = 1e6) -> None:
        self.max_loss = max_loss
        self.max_grad_norm = max_grad_norm

    def check_step(self, loss_val: float, grad_norm: float) -> bool:
        """Evaluate step metrics; aborts immediately with RuntimeError if divergence detected.

        Parameters
        ----------
        loss_val : float
            Current batch loss value.
        grad_norm : float
            Computed gradient norm before clipping.

        Returns
        -------
        bool
            Always False if healthy; raises RuntimeError on failure.
        """
        if math.isnan(loss_val) or math.isinf(loss_val):
            raise RuntimeError(f"Training aborted by KillSwitch: Loss is NaN or Inf ({loss_val}).")
        if loss_val > self.max_loss:
            raise RuntimeError(
                f"Training aborted by KillSwitch: Loss {loss_val:.4e} exceeded threshold {self.max_loss:.4e}."
            )
        if self.max_grad_norm is not None:
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                raise RuntimeError(f"Training aborted by KillSwitch: Grad norm is NaN or Inf ({grad_norm}).")
            if grad_norm > self.max_grad_norm:
                raise RuntimeError(
                    f"Training aborted by KillSwitch: Grad norm {grad_norm:.4e} exceeded threshold {self.max_grad_norm:.4e}."
                )
        return False

    def check_epoch(self, metric_val: Optional[float]) -> bool:
        """Evaluate epoch validation metric; aborts with RuntimeError if NaN/Inf/diverged.

        Parameters
        ----------
        metric_val : Optional[float]
            Validation metric for the completed epoch.

        Returns
        -------
        bool
            Always False if healthy; raises RuntimeError on failure.
        """
        if metric_val is not None:
            if math.isnan(metric_val) or math.isinf(metric_val):
                raise RuntimeError(f"Training aborted by KillSwitch: Validation metric is NaN or Inf ({metric_val}).")
            if metric_val > self.max_loss:
                raise RuntimeError(
                    f"Training aborted by KillSwitch: Validation metric {metric_val:.4e} exceeded threshold {self.max_loss:.4e}."
                )
        return False


class EarlyStoppingKillSwitch(KillSwitch):
    """Health & safety strategy monitoring validation metric improvement across a patience window.

    Parameters
    ----------
    patience : int, default=10
        Number of epochs without metric improvement before early stopping triggers.
    min_delta : float, default=1e-4
        Minimum change in metric to qualify as an improvement.
    mode : str, default='min'
        'min' if smaller metric is better (e.g. loss); 'max' if larger metric is better.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"Mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric: Optional[float] = None
        self.wait_count: int = 0

    def check_step(self, loss_val: float, grad_norm: float) -> bool:
        """Step check: early stopping only inspects epoch-level metrics."""
        return False

    def check_epoch(self, metric_val: Optional[float]) -> bool:
        """Evaluate epoch validation metric; raises StopIteration when patience is exceeded.

        Parameters
        ----------
        metric_val : Optional[float]
            Validation metric for the completed epoch.

        Returns
        -------
        bool
            Always False if healthy; raises StopIteration when patience exhausted.
        """
        if metric_val is None:
            return False

        if self.best_metric is None:
            self.best_metric = metric_val
            self.wait_count = 0
            return False

        if self.mode == "min":
            improved = metric_val < (self.best_metric - self.min_delta)
        else:
            improved = metric_val > (self.best_metric + self.min_delta)

        if improved:
            self.best_metric = metric_val
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                raise StopIteration(
                    f"Early stopping triggered by KillSwitch: No improvement for {self.wait_count} epochs."
                )
        return False


class CompositeKillSwitch(KillSwitch):
    """Composite safety strategy evaluating multiple KillSwitch strategies sequentially.

    Parameters
    ----------
    kill_switches : Sequence[KillSwitch]
        Sequence of KillSwitch strategy instances to execute.
    """

    def __init__(self, kill_switches: Sequence[KillSwitch]) -> None:
        self.kill_switches = list(kill_switches)

    def check_step(self, loss_val: float, grad_norm: float) -> bool:
        """Evaluate check_step across all registered kill switches."""
        for ks in self.kill_switches:
            ks.check_step(loss_val, grad_norm)
        return False

    def check_epoch(self, metric_val: Optional[float]) -> bool:
        """Evaluate check_epoch across all registered kill switches."""
        for ks in self.kill_switches:
            ks.check_epoch(metric_val)
        return False


class OptimizerController(Optimizer):
    """Bridge Abstraction managing parameter updates, schedules, and health strategies.

    Decouples the high-level optimization lifecycle from the underlying PyTorch
    optimizer implementor (e.g. AdamW, Lion), while delegating learning rate schedules
    and health & safety guards to Strategy implementations.

    Parameters
    ----------
    optimizer : TorchOptimizer
        Underlying PyTorch optimizer instance (e.g. AdamW, Lion).
    scheduler : Optional[Any], default=None
        Learning rate scheduler strategy. If None, learning rate remains constant.
    kill_switch : Optional[KillSwitch], default=None
        Health and safety strategy (divergence check, early stopping). If None, training runs uninhibited.
    parameters : Optional[Iterable[nn.Parameter]], default=None
        Model parameters to clip during optimization.
    use_amp : bool, default=False
        Whether automatic mixed precision scaling is active.
    scaler : Optional[torch.amp.GradScaler], default=None
        Custom GradScaler instance if provided.
    """

    def __init__(
        self,
        optimizer: TorchOptimizer,
        scheduler: Optional[Any] = None,
        kill_switch: Optional[KillSwitch] = None,
        parameters: Optional[Iterable[nn.Parameter]] = None,
        use_amp: bool = False,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> None:
        super().__init__(optimizer)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.kill_switch = kill_switch
        self.parameters = list(parameters) if parameters is not None else None
        self.use_amp = use_amp
        if scaler is not None:
            self.scaler = scaler
        elif use_amp:
            self.scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
        else:
            self.scaler = None

    @staticmethod
    def _build_scheduler(
        optimizer: TorchOptimizer,
        scheduler_type: Optional[str],
        epochs: int,
        min_lr: float,
        warmup_ratio: float = 0.1,
    ) -> Optional[Any]:
        """Construct learning rate scheduler strategy.

        Parameters
        ----------
        optimizer : TorchOptimizer
            Configured PyTorch optimizer instance.
        scheduler_type : Optional[str]
            Scheduler type: 'cosine', 'plateau', 'warmup_cosine', or None.
        epochs : int
            Total training epochs.
        min_lr : float
            Minimum learning rate floor.
        warmup_ratio : float, default=0.1
            Fraction of total epochs dedicated to linear warmup in 'warmup_cosine'.

        Returns
        -------
        Optional[Any]
            The configured scheduler instance or None.
        """
        if scheduler_type is None:
            return None
        elif scheduler_type == "cosine":
            return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        elif scheduler_type == "plateau":
            return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=min_lr)
        elif scheduler_type == "warmup_cosine":
            warmup_epochs = max(1, int(epochs * warmup_ratio))
            if warmup_epochs >= epochs:
                warmup_epochs = max(1, epochs - 1)
            cosine_epochs = max(1, epochs - warmup_epochs)

            warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
            cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=min_lr)
            return SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
        else:
            raise ValueError(
                f"Unsupported scheduler_type '{scheduler_type}'. Expected 'cosine', 'plateau', 'warmup_cosine', or None."
            )

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
        kill_switch: Optional[KillSwitch] = None,
        warmup_ratio: float = 0.1,
    ) -> "OptimizerController":
        """Factory constructor instantiating AdamW, scheduler strategy, and safety guards.

        Parameters
        ----------
        model_parameters : Iterable[nn.Parameter]
            Trainable parameters of the neural network module.
        lr : float, default=1e-4
            Initial learning rate.
        weight_decay : float, default=1e-4
            AdamW weight decay coefficient.
        scheduler_type : Optional[str], default='cosine'
            Type of scheduler: 'cosine', 'plateau', 'warmup_cosine', or None.
        epochs : int, default=100
            Total planned training epochs.
        min_lr : float, default=1e-6
            Minimum learning rate floor.
        use_amp : bool, default=False
            Whether to initialize automatic mixed precision GradScaler.
        kill_switch : Optional[KillSwitch], default=None
            Pluggable safety strategy (NaN divergence, early stopping).
        warmup_ratio : float, default=0.1
            Warmup epoch ratio for 'warmup_cosine'.

        Returns
        -------
        OptimizerController
            Configured optimizer controller instance.
        """
        params_list = list(model_parameters)
        optimizer = AdamW(params_list, lr=lr, weight_decay=weight_decay)
        scheduler = cls._build_scheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            epochs=epochs,
            min_lr=min_lr,
            warmup_ratio=warmup_ratio,
        )
        return cls(
            optimizer=optimizer,
            scheduler=scheduler,
            kill_switch=kill_switch,
            parameters=params_list,
            use_amp=use_amp,
        )

    @classmethod
    def create_lion(
        cls,
        model_parameters: Iterable[nn.Parameter],
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        scheduler_type: Optional[str] = "cosine",
        epochs: int = 100,
        min_lr: float = 1e-6,
        use_amp: bool = False,
        kill_switch: Optional[KillSwitch] = None,
        warmup_ratio: float = 0.1,
    ) -> "OptimizerController":
        """Factory constructor instantiating Lion, scheduler strategy, and safety guards.

        Parameters
        ----------
        model_parameters : Iterable[nn.Parameter]
            Trainable parameters of the neural network module.
        lr : float, default=1e-4
            Initial learning rate.
        weight_decay : float, default=1e-4
            Decoupled weight decay coefficient.
        betas : Tuple[float, float], default=(0.9, 0.99)
        scheduler_type : Optional[str], default='cosine'
            Type of scheduler: 'cosine', 'plateau', 'warmup_cosine', or None.
        epochs : int, default=100
            Total planned training epochs.
        min_lr : float, default=1e-6
            Minimum learning rate floor.
        use_amp : bool, default=False
            Whether to initialize automatic mixed precision GradScaler.
        kill_switch : Optional[KillSwitch], default=None
            Pluggable safety strategy (NaN divergence, early stopping).
        warmup_ratio : float, default=0.1
            Warmup epoch ratio for 'warmup_cosine'.

        Returns
        -------
        OptimizerController
            Configured optimizer controller instance.
        """
        params_list = list(model_parameters)
        optimizer = Lion(params_list, lr=lr, betas=betas, weight_decay=weight_decay)
        scheduler = cls._build_scheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            epochs=epochs,
            min_lr=min_lr,
            warmup_ratio=warmup_ratio,
        )
        return cls(
            optimizer=optimizer,
            scheduler=scheduler,
            kill_switch=kill_switch,
            parameters=params_list,
            use_amp=use_amp,
        )

    def zero_grad(self) -> None:
        """Reset parameter gradients."""
        self._optimizer.zero_grad()

    def step(self, loss: torch.Tensor, max_grad_norm: Optional[float] = None) -> float:
        """Execute backward pass, gradient clipping, parameter update, and step health check.

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
        loss_val = float(loss.item()) if torch.is_tensor(loss) else float(loss)
        grad_norm = 0.0

        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            if max_grad_norm is not None and self.parameters is not None:
                self.scaler.unscale_(self._optimizer)
                norm_tensor = nn.utils.clip_grad_norm_(self.parameters, max_grad_norm)
                grad_norm = float(norm_tensor.item())
            elif self.parameters is not None:
                self.scaler.unscale_(self._optimizer)
                norm_tensor = nn.utils.clip_grad_norm_(self.parameters, float("inf"))
                grad_norm = float(norm_tensor.item())

            if self.kill_switch is not None:
                self.kill_switch.check_step(loss_val=loss_val, grad_norm=grad_norm)

            self.scaler.step(self._optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None and self.parameters is not None:
                norm_tensor = nn.utils.clip_grad_norm_(self.parameters, max_grad_norm)
                grad_norm = float(norm_tensor.item())
            elif self.parameters is not None:
                norm_tensor = nn.utils.clip_grad_norm_(self.parameters, float("inf"))
                grad_norm = float(norm_tensor.item())

            if self.kill_switch is not None:
                self.kill_switch.check_step(loss_val=loss_val, grad_norm=grad_norm)

            self._optimizer.step()

        return grad_norm

    def epoch_step(self, metric: Optional[float] = None) -> float:
        """Step the learning rate scheduler and evaluate epoch health strategies.

        Parameters
        ----------
        metric : Optional[float], default=None
            Validation metric for ReduceLROnPlateau or epoch health check.

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

        if self.kill_switch is not None:
            self.kill_switch.check_epoch(metric)

        return self.get_learning_rate()

    def get_learning_rate(self) -> float:
        """Retrieve the current learning rate from the primary parameter group.

        Returns
        -------
        float
            Learning rate from the first parameter group.
        """
        return float(self._optimizer.param_groups[0]["lr"])

    def get_state_dict(self) -> Dict[str, Any]:
        """Retrieve state dictionaries for persistence.

        Returns
        -------
        Dict[str, Any]
            State dictionary containing optimizer, scheduler, and scaler states.
        """
        states: Dict[str, Any] = {
            "optimizer_state_dict": self._optimizer.state_dict(),
        }
        if self.scheduler is not None:
            states["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None and hasattr(self.scaler, "state_dict"):
            states["scaler_state_dict"] = self.scaler.state_dict()
        return states

    def load_state(
        self,
        checkpoint: Dict[str, Any],
        device: Optional[torch.device] = None,
        step_scheduler_to_epoch: Optional[int] = None,
    ) -> None:
        """Load optimizer, scaler, and scheduler states from a checkpoint payload.

        Parameters
        ----------
        checkpoint : Dict[str, Any]
            Loaded checkpoint payload.
        device : Optional[torch.device], default=None
            Target device to move optimizer tensor buffers (e.g. CUDA).
        step_scheduler_to_epoch : Optional[int], default=None
            If scheduler_state_dict is missing from checkpoint, fast-forward the scheduler
            by this number of epochs to reach the proper learning rate.
        """
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if device is not None:
                self.to(device)

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

    def to(self, device: torch.device) -> None:
        """Move optimizer state tensors to the specified hardware device."""
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)


# Drop-in backward compatibility alias
StandardOptimizer = OptimizerController
