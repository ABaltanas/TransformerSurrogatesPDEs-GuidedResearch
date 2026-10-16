"""Telemetry and progress monitoring observers for surrogate model training."""

import json
from pathlib import Path
import time
from typing import Dict, List, Optional
import torch

from src.training.interfaces import TelemetryLogger


class CliProgressMonitor(TelemetryLogger):
    """Real-time terminal monitor providing percentage progress, throughput, and dual ETA.

    Parameters
    ----------
    batch_size : int, default=1
        Batch size to calculate per-second sample throughput.
    bar_length : int, default=25
        Length of the ANSI progress bar in characters.
    """

    def __init__(self, batch_size: int = 1, bar_length: int = 25) -> None:
        self.batch_size = batch_size
        self.bar_length = bar_length
        self.job_start_time: float = 0.0
        self.epoch_start_time: float = 0.0
        self.total_batches: int = 0
        self.total_epochs: int = 0
        self.epoch_batch_durations: List[float] = []
        self.smoothed_loss: float = 0.0

    def start_epoch(self, epoch: int, total_epochs: int, total_batches: int) -> None:
        """Initialize timing structures for a new epoch."""
        if self.job_start_time == 0.0:
            self.job_start_time = time.time()
        self.epoch_start_time = time.time()
        self.total_batches = max(1, total_batches)
        self.total_epochs = total_epochs
        self.epoch_batch_durations.clear()
        self.smoothed_loss = 0.0

    def update_batch(self, batch_idx: int, loss_val: float, lr: float) -> None:
        """Update live terminal display after an optimization step."""
        now = time.time()
        elapsed_epoch = now - self.epoch_start_time
        batch_duration = elapsed_epoch / max(1, batch_idx)
        self.epoch_batch_durations.append(batch_duration)

        # Exponential moving average loss
        if self.smoothed_loss == 0.0:
            self.smoothed_loss = loss_val
        else:
            self.smoothed_loss = 0.9 * self.smoothed_loss + 0.1 * loss_val

        # Calculate epoch ETA
        remaining_batches = self.total_batches - batch_idx
        epoch_eta_s = remaining_batches * batch_duration

        # Throughput
        batches_per_sec = batch_idx / max(1e-5, elapsed_epoch)
        samples_per_sec = batches_per_sec * self.batch_size

        # Render progress bar
        pct = (batch_idx / self.total_batches) * 100.0
        filled = int(self.bar_length * (batch_idx / self.total_batches))
        bar = "=" * filled + ">" + "." * max(0, self.bar_length - filled - 1)
        if filled >= self.bar_length:
            bar = "=" * self.bar_length

        # GPU VRAM if available
        vram_str = ""
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            vram_str = f" | VRAM: {vram_mb:.0f}MB"

        eta_min = int(epoch_eta_s // 60)
        eta_sec = int(epoch_eta_s % 60)

        print(
            f"\rEpoch Batch [{batch_idx:02d}/{self.total_batches:02d}] "
            f"[{bar}] {pct:5.1f}% | "
            f"Epoch ETA: {eta_min:02d}:{eta_sec:02d} | "
            f"Speed: {samples_per_sec:.1f} smp/s | "
            f"Loss: {loss_val:.4e} (Smooth: {self.smoothed_loss:.4e}) | "
            f"LR: {lr:.2e}{vram_str}",
            end="",
            flush=True,
        )

    def end_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
    ) -> None:
        """Render end-of-epoch summary with overall training ETA."""
        epoch_duration = time.time() - self.epoch_start_time
        total_elapsed = time.time() - self.job_start_time

        # Calculate total job ETA
        avg_epoch_time = total_elapsed / max(1, epoch)
        remaining_epochs = total_epochs - epoch
        total_eta_s = remaining_epochs * avg_epoch_time

        eta_hours = int(total_eta_s // 3600)
        eta_mins = int((total_eta_s % 3600) // 60)

        train_l2 = train_metrics.get("loss_total", 0.0)
        val_l2 = val_metrics.get("loss_total", 0.0)

        print(
            f"\n✔ Epoch [{epoch:03d}/{total_epochs:03d}] Finished in {epoch_duration:.1f}s | "
            f"Total Run ETA: {eta_hours:02d}h {eta_mins:02d}m | "
            f"Train RelL2: {train_l2:.4e} | Val RelL2: {val_l2:.4e} | LR: {lr:.2e}"
        )

    def close(self) -> None:
        """Render final execution summary banner."""
        if self.job_start_time > 0.0:
            total_duration = time.time() - self.job_start_time
            h = int(total_duration // 3600)
            m = int((total_duration % 3600) // 60)
            s = int(total_duration % 60)
            print("=" * 80)
            print(f"[TrainingEngine] Training Completed in {h:02d}h {m:02d}m {s:02d}s")
            print("=" * 80)


class JsonMetricsLogger(TelemetryLogger):
    """Persists metrics as JSON Lines (JSONL) records to a file for persistent logging.

    Parameters
    ----------
    log_file : Path
        Target file path where metrics will be appended.
    """

    def __init__(self, log_file: Path) -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def start_epoch(self, epoch: int, total_epochs: int, total_batches: int) -> None:
        pass

    def update_batch(self, batch_idx: int, loss_val: float, lr: float) -> None:
        pass

    def end_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
    ) -> None:
        """Write an epoch record to disk."""
        record = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "timestamp": time.time(),
            "train": train_metrics,
            "val": val_metrics,
            "lr": lr,
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def close(self) -> None:
        pass


class CompositeTelemetryLogger(TelemetryLogger):
    """Composite observer broadcasting telemetry events to multiple child loggers.

    Parameters
    ----------
    loggers : List[TelemetryLogger]
        List of concrete logger observers to execute in sequence.
    """

    def __init__(self, loggers: List[TelemetryLogger]) -> None:
        self.loggers = list(loggers)

    def start_epoch(self, epoch: int, total_epochs: int, total_batches: int) -> None:
        for logger in self.loggers:
            logger.start_epoch(epoch, total_epochs, total_batches)

    def update_batch(self, batch_idx: int, loss_val: float, lr: float) -> None:
        for logger in self.loggers:
            logger.update_batch(batch_idx, loss_val, lr)

    def end_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
    ) -> None:
        for logger in self.loggers:
            logger.end_epoch(epoch, total_epochs, train_metrics, val_metrics, lr)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()
