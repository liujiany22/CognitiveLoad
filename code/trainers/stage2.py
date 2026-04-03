"""
Stage 2 — Language-Guided Alignment Pre-training.

EEG encoder + projector learn to align EEG representations with frozen
text embeddings (one per class label) via bidirectional CLIP-style
contrastive loss.

The text embedding layer is frozen (not trained); it provides fixed
anchor points in the projection space that the EEG branch learns to
match.
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from losses import CLIPLoss
from utils import AverageMeter, EarlyStopping


class Stage2Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device

        self.criterion = CLIPLoss()

        self.optimizer = AdamW(
            list(model.align_encoder.parameters()) +
            list(model.align_projector.parameters()) +
            [model.logit_scale],
            lr=config.stage2_lr,
            weight_decay=config.stage2_weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.stage2_epochs,
        )
        self.early_stop = EarlyStopping(patience=config.patience, mode="min")

    def train(self, train_loader, val_loader=None):
        self.model.to(self.device)
        best_loss = float("inf")

        for epoch in range(1, self.config.stage2_epochs + 1):
            self.model.train()
            loss_m = AverageMeter()
            acc_m = AverageMeter()

            pbar = tqdm(train_loader, desc=f"[Stage2] Epoch {epoch}", leave=False)
            for batch in pbar:
                eeg = batch["eeg"].to(self.device)
                labels = batch["label"].to(self.device)
                B = eeg.size(0)

                self.optimizer.zero_grad()
                eeg_proj, text_emb, scale = self.model.forward_eeg_phase(eeg, labels)
                loss, acc = self.criterion(eeg_proj, text_emb, scale)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.align_encoder.parameters()) +
                    list(self.model.align_projector.parameters()),
                    1.0,
                )
                self.optimizer.step()

                loss_m.update(loss.item(), B)
                acc_m.update(acc.item(), B)
                pbar.set_postfix(
                    loss=f"{loss_m.avg:.3f}",
                    acc=f"{acc_m.avg:.3f}",
                )

            self.scheduler.step()

            val_loss = self._validate(val_loader) if val_loader else loss_m.avg

            print(
                f"  Stage2 Epoch {epoch:3d}  |  "
                f"loss {loss_m.avg:.4f}  |  "
                f"align-acc {acc_m.avg:.4f}  |  "
                f"val-loss {val_loss:.4f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    f"{self.config.save_dir}/stage2_best.pt",
                )

            if self.early_stop.step(val_loss):
                print(f"  Stage2 early stopping at epoch {epoch}")
                break

        ckpt = torch.load(
            f"{self.config.save_dir}/stage2_best.pt",
            map_location=self.device, weights_only=True,
        )
        self.model.load_state_dict(ckpt)
        print(f"  Stage2 done — best val-loss {best_loss:.4f}")

    @torch.no_grad()
    def _validate(self, loader):
        if loader is None:
            return float("inf")
        self.model.eval()
        meter = AverageMeter()
        for batch in loader:
            eeg = batch["eeg"].to(self.device)
            labels = batch["label"].to(self.device)
            eeg_proj, text_emb, scale = self.model.forward_eeg_phase(eeg, labels)
            loss, _ = self.criterion(eeg_proj, text_emb, scale)
            meter.update(loss.item(), eeg.size(0))
        return meter.avg
