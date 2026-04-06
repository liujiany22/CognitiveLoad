"""
Stage 2 — Language-Guided Alignment Pre-training.

EEG encoder + projector learn to align EEG representations with frozen
text embeddings (one per class label) via bidirectional CLIP-style
contrastive loss.
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from losses import CLIPLoss
from utils import AverageMeter
from .base import BaseTrainer


class Stage2Trainer(BaseTrainer):
    label = "Stage2"
    early_stop_mode = "min"

    def __init__(self, model, config):
        super().__init__(model, config, ckpt_name="stage2_best.pt")
        self.criterion = CLIPLoss()

    @property
    def _epochs(self):
        return self.config.stage2_epochs

    def configure_optimizers(self):
        params = (
            list(self.model.align_encoder.parameters())
            + list(self.model.align_projector.parameters())
            + [self.model.logit_scale]
        )
        opt = AdamW(params, lr=self.config.stage2_lr,
                    weight_decay=self.config.stage2_weight_decay)
        sched = CosineAnnealingLR(opt, T_max=self.config.stage2_epochs)
        return opt, sched

    def train_step(self, batch):
        eeg = batch["eeg"].to(self.device)
        labels = batch["label"].to(self.device)

        eeg_proj, text_emb, scale = self.model.forward_alignment(eeg, labels)
        loss, acc = self.criterion(eeg_proj, text_emb, scale)
        return {"loss": loss, "acc": acc}

    @torch.no_grad()
    def validate(self, val_loader) -> float | None:
        if val_loader is None:
            return None
        self.model.eval()
        meter = AverageMeter()
        for batch in val_loader:
            eeg = batch["eeg"].to(self.device)
            labels = batch["label"].to(self.device)
            eeg_proj, text_emb, scale = self.model.forward_alignment(eeg, labels)
            loss, _ = self.criterion(eeg_proj, text_emb, scale)
            meter.update(loss.item(), eeg.size(0))
        return meter.avg
