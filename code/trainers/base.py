"""
Base trainer with template-method pattern.

Subclasses override only `configure_optimizers`, `train_step`,
and optionally `validate_step` / `evaluate`.  Everything else —
epoch loop, progress bar, checkpoint management, early stopping,
best-model reload — lives here once.
"""

from abc import ABC, abstractmethod

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from utils import AverageMeter, EarlyStopping


class BaseTrainer(ABC):
    label: str = "Train"
    early_stop_mode: str = "min"  # "min" for loss, "max" for accuracy

    def __init__(self, model, config, *, ckpt_name: str = "best.pt"):
        self.model = model
        self.config = config
        self.device = config.device
        self._ckpt_path = f"{config.save_dir}/{ckpt_name}"
        self.optimizer, self.scheduler = self.configure_optimizers()
        self.early_stop = EarlyStopping(
            patience=self._patience, mode=self.early_stop_mode,
        )

    @property
    def _epochs(self) -> int:
        raise NotImplementedError

    @property
    def _patience(self) -> int:
        return self.config.patience

    # ── hooks subclasses must / may override ──

    @abstractmethod
    def configure_optimizers(self) -> tuple[Optimizer, LRScheduler]:
        ...

    @abstractmethod
    def train_step(self, batch) -> dict:
        """Run one training step.

        Must return a dict with at least ``{"loss": Tensor}``.
        Extra keys (e.g. ``"acc"``) are tracked and printed automatically.
        """

    def validate(self, val_loader) -> float:
        """Return a scalar used for early-stopping / checkpoint selection.

        Default: average training loss of the epoch (overridden by Stage 2/3).
        """
        return None  # signals "use train loss"

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        """Optional full evaluation (used by Stage 3)."""
        raise NotImplementedError

    # ── main loop ──

    def train(self, train_loader, val_loader=None):
        self.model.to(self.device)
        best_metric = float("inf") if self.early_stop_mode == "min" else 0.0
        metric_improved = (
            (lambda new, old: new < old)
            if self.early_stop_mode == "min"
            else (lambda new, old: new > old)
        )

        for epoch in range(1, self._epochs + 1):
            self.model.train()
            meters: dict[str, AverageMeter] = {}

            pbar = tqdm(train_loader, desc=f"[{self.label}] Epoch {epoch}", leave=False)
            for batch in pbar:
                result = self.train_step(batch)

                loss = result["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                bs = self._batch_size(batch)
                for k, v in result.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    meters.setdefault(k, AverageMeter()).update(val, bs)
                pbar.set_postfix({k: f"{m.avg:.4f}" for k, m in meters.items()})

            self.scheduler.step()

            val_metric = self.validate(val_loader)
            if val_metric is None:
                val_metric = meters["loss"].avg

            log_parts = [f"{self.label} Epoch {epoch:3d}"]
            for k, m in meters.items():
                log_parts.append(f"{k} {m.avg:.4f}")
            log_parts.append(f"val-metric {val_metric:.4f}")
            print("  " + "  |  ".join(log_parts))

            if metric_improved(val_metric, best_metric):
                best_metric = val_metric
                torch.save(self.model.state_dict(), self._ckpt_path)

            self._periodic_checkpoint(epoch)

            if self.early_stop.step(val_metric):
                print(f"  {self.label} early stopping at epoch {epoch}")
                break

        self._reload_best()
        print(f"  {self.label} done — best metric {best_metric:.4f}")

    # ── helpers ──

    def _periodic_checkpoint(self, epoch: int):
        every = self.config.ckpt_every
        if every > 0 and epoch % every == 0:
            stem = self._ckpt_path.rsplit(".", 1)[0]
            torch.save(self.model.state_dict(), f"{stem}_epoch_{epoch}.pt")

    def _reload_best(self):
        ckpt = torch.load(self._ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt)

    @staticmethod
    def _batch_size(batch: dict) -> int:
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v.size(0)
        return 1
