"""
Stage 1 — Cross-Subject Contrastive Pre-training (CLISA-style InfoNCE).

Each batch contains paired segments from two subjects at the same
epoch positions.  Positive pair = same position, different subjects.
All other positions are negatives.
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from losses import InfoNCELoss
from utils import AverageMeter, EarlyStopping


class Stage1Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.criterion = InfoNCELoss(temperature=config.temperature)

        self.optimizer = AdamW(
            list(model.cross_encoder.parameters()) +
            list(model.cross_projector.parameters()),
            lr=config.stage1_lr,
            weight_decay=config.stage1_weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.stage1_epochs,
        )
        self.early_stop = EarlyStopping(patience=config.patience, mode="min")

    def train(self, pair_loader):
        self.model.to(self.device)
        best_loss = float("inf")

        for epoch in range(1, self.config.stage1_epochs + 1):
            self.model.train()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()

            pbar = tqdm(pair_loader, desc=f"[Stage1] Epoch {epoch}", leave=False)
            for batch in pbar:
                eeg_a = batch["eeg_a"].to(self.device)
                eeg_b = batch["eeg_b"].to(self.device)
                n_per = eeg_a.size(0)

                z_a = self.model.forward_cross_subject(eeg_a, n_per_subject=n_per)
                z_b = self.model.forward_cross_subject(eeg_b, n_per_subject=n_per)

                loss, acc = self.criterion(z_a, z_b)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                loss_meter.update(loss.item(), n_per)
                acc_meter.update(acc.item(), n_per)
                pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

            self.scheduler.step()

            print(
                f"  Stage1 Epoch {epoch:3d}  |  "
                f"loss {loss_meter.avg:.4f}  |  "
                f"nn-acc {acc_meter.avg:.4f}"
            )

            if loss_meter.avg < best_loss:
                best_loss = loss_meter.avg
                torch.save(
                    self.model.state_dict(),
                    f"{self.config.save_dir}/stage1_best.pt",
                )

            every = self.config.ckpt_every
            if every > 0 and epoch % every == 0:
                torch.save(
                    self.model.state_dict(),
                    f"{self.config.save_dir}/stage1_epoch_{epoch}.pt",
                )

            if self.early_stop.step(loss_meter.avg):
                print(f"  Stage1 early stopping at epoch {epoch}")
                break

        ckpt = torch.load(
            f"{self.config.save_dir}/stage1_best.pt",
            map_location=self.device, weights_only=True,
        )
        self.model.load_state_dict(ckpt)
        print(f"  Stage1 done — best loss {best_loss:.4f}")
