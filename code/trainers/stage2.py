"""
Stage 2 — Language-Guided Alignment Pre-training (NICE++ inspired).

Two-phase alternating optimisation within each mini-batch, mirroring
the NICE++ (NICE-LLM) training loop:

  Phase A — Text projection shaping (like NICE++ Image↔Text loss):
      TextProjector is updated via a supervised contrastive loss so that
      projected text embeddings of the same condition cluster together
      while different conditions are pushed apart.
      Only optimizer_text is stepped; EEG path is untouched.

  Phase B — EEG ↔ refined-text alignment (like NICE++ EEG↔Image loss):
      Text projections are DETACHED (frozen for this step).
      EEG encoder + projector learn to match the semantically shaped
      text representations via bidirectional CLIP-style contrastive loss.
      Only optimizer_eeg is stepped; TextProjector is untouched.

This decoupled design ensures that language semantics flow into the EEG
representation space indirectly — text shapes the target space, and EEG
learns to land in it.
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from losses import SupConLoss, CLIPLoss
from utils import AverageMeter, EarlyStopping


class Stage2Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device

        self.criterion_text = SupConLoss(temperature=config.temperature)
        self.criterion_eeg = CLIPLoss()

        # separate optimisers — exactly like NICE++
        self.optimizer_text = AdamW(
            list(model.text_projector.parameters()) + [model.logit_scale],
            lr=config.stage2_lr,
            weight_decay=config.stage2_weight_decay,
        )
        self.optimizer_eeg = AdamW(
            list(model.align_encoder.parameters()) +
            list(model.align_projector.parameters()),
            lr=config.stage2_lr,
            weight_decay=config.stage2_weight_decay,
        )

        self.scheduler_text = CosineAnnealingLR(
            self.optimizer_text, T_max=config.stage2_epochs,
        )
        self.scheduler_eeg = CosineAnnealingLR(
            self.optimizer_eeg, T_max=config.stage2_epochs,
        )
        self.early_stop = EarlyStopping(patience=config.patience, mode="min")

    def train(self, train_loader, val_loader=None):
        self.model.to(self.device)
        best_loss = float("inf")

        for epoch in range(1, self.config.stage2_epochs + 1):
            self.model.train()
            loss_text_m = AverageMeter()
            loss_eeg_m = AverageMeter()
            acc_m = AverageMeter()

            pbar = tqdm(train_loader, desc=f"[Stage2] Epoch {epoch}", leave=False)
            for batch in pbar:
                eeg = batch["eeg"].to(self.device)
                labels = batch["condition_id"].to(self.device)
                tf = batch["task_features"].to(self.device)
                B = eeg.size(0)

                # ── Phase A: update TextProjector (NICE++ Image↔Text) ──
                self.optimizer_text.zero_grad()
                text_proj = self.model.forward_text_phase(tf)
                loss_text = self.criterion_text(text_proj, labels)
                loss_text.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.text_projector.parameters(), 1.0,
                )
                self.optimizer_text.step()

                # ── Phase B: update EEG encoder + projector (NICE++ EEG↔Image) ──
                self.optimizer_eeg.zero_grad()
                eeg_proj, text_proj_det, scale = self.model.forward_eeg_phase(eeg, tf)
                loss_eeg, acc = self.criterion_eeg(eeg_proj, text_proj_det, scale)
                loss_eeg.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.align_encoder.parameters()) +
                    list(self.model.align_projector.parameters()),
                    1.0,
                )
                self.optimizer_eeg.step()

                loss_text_m.update(loss_text.item(), B)
                loss_eeg_m.update(loss_eeg.item(), B)
                acc_m.update(acc.item(), B)
                pbar.set_postfix(
                    lt=f"{loss_text_m.avg:.3f}",
                    le=f"{loss_eeg_m.avg:.3f}",
                    acc=f"{acc_m.avg:.3f}",
                )

            self.scheduler_text.step()
            self.scheduler_eeg.step()

            val_loss = self._validate(val_loader) if val_loader else loss_eeg_m.avg

            print(
                f"  Stage2 Epoch {epoch:3d}  |  "
                f"text-loss {loss_text_m.avg:.4f}  |  "
                f"eeg-loss {loss_eeg_m.avg:.4f}  |  "
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
            tf = batch["task_features"].to(self.device)
            eeg_proj, text_proj, scale = self.model.forward_eeg_phase(eeg, tf)
            loss, _ = self.criterion_eeg(eeg_proj, text_proj, scale)
            meter.update(loss.item(), eeg.size(0))
        return meter.avg
