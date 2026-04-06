"""
Stage 1 — Cross-Subject Contrastive Pre-training (CLISA-style InfoNCE).

Each batch contains paired segments from two subjects at the same
epoch positions.  Positive pair = same position, different subjects.
All other positions are negatives.
"""

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from losses import InfoNCELoss
from .base import BaseTrainer


class Stage1Trainer(BaseTrainer):
    label = "Stage1"
    early_stop_mode = "min"

    def __init__(self, model, config):
        super().__init__(model, config, ckpt_name="stage1_best.pt")
        self.criterion = InfoNCELoss(temperature=config.temperature)

    @property
    def _epochs(self):
        return self.config.stage1_epochs

    def configure_optimizers(self):
        opt = Adam(
            self.model.cross_encoder.parameters(),
            lr=self.config.stage1_lr,
            weight_decay=self.config.stage1_weight_decay,
        )
        T_0 = max(1, self.config.stage1_epochs // self.config.stage1_restart_times)
        sched = CosineAnnealingWarmRestarts(opt, T_0=T_0, eta_min=0)
        return opt, sched

    def train_step(self, batch):
        eeg_a = batch["eeg_a"].to(self.device)
        eeg_b = batch["eeg_b"].to(self.device)
        cond_labels = batch["cond_labels"].to(self.device)
        n_per = eeg_a.size(0)

        z_a = self.model.forward_cross_subject(eeg_a, n_per_subject=n_per)
        z_b = self.model.forward_cross_subject(eeg_b, n_per_subject=n_per)

        loss, acc = self.criterion(z_a, z_b, cond_labels=cond_labels)
        return {"loss": loss, "acc": acc}
