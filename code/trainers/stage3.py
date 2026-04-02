"""
Stage 3 — Dual-Branch Classification Fine-tuning.

Both encoders (cross-subject & stimulus-aligned) are combined: their
embeddings are concatenated, fused through an MLP, and classified into
cognitive-load levels (low / mid / high).

The encoder weights can optionally be frozen or fine-tuned end-to-end.
"""

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from losses import CogLoadLoss
from utils import AverageMeter, EarlyStopping, compute_metrics


class Stage3Trainer:
    def __init__(self, model, config, freeze_encoders: bool = False):
        self.model = model
        self.config = config
        self.device = config.device
        self.criterion = CogLoadLoss(n_classes=config.n_classes)

        if freeze_encoders:
            for p in model.cross_encoder.parameters():
                p.requires_grad = False
            for p in model.align_encoder.parameters():
                p.requires_grad = False

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable, lr=config.stage3_lr, weight_decay=config.stage3_weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.stage3_epochs,
        )
        self.early_stop = EarlyStopping(patience=config.patience, mode="max")

    def train(self, train_loader, val_loader):
        self.model.to(self.device)
        best_acc = 0.0

        for epoch in range(1, self.config.stage3_epochs + 1):
            self.model.train()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()

            pbar = tqdm(train_loader, desc=f"[Stage3] Epoch {epoch}", leave=False)
            for batch in pbar:
                eeg = batch["eeg"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(eeg)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean().item()

                loss_meter.update(loss.item(), eeg.size(0))
                acc_meter.update(acc, eeg.size(0))
                pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

            self.scheduler.step()

            val_metrics = self.evaluate(val_loader)
            val_acc = val_metrics["accuracy"]

            print(
                f"  Stage3 Epoch {epoch:3d}  |  "
                f"train-loss {loss_meter.avg:.4f}  |  "
                f"train-acc {acc_meter.avg:.4f}  |  "
                f"val-acc {val_acc:.4f}  |  "
                f"val-f1 {val_metrics['f1_macro']:.4f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    self.model.state_dict(),
                    f"{self.config.save_dir}/stage3_best.pt",
                )

            if self.early_stop.step(val_acc):
                print(f"  Stage3 early stopping at epoch {epoch}")
                break

        ckpt = torch.load(
            f"{self.config.save_dir}/stage3_best.pt",
            map_location=self.device, weights_only=True,
        )
        self.model.load_state_dict(ckpt)
        print(f"  Stage3 done — best val-acc {best_acc:.4f}")

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []

        for batch in loader:
            eeg = batch["eeg"].to(self.device)
            labels = batch["label"]
            logits = self.model(eeg)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        return compute_metrics(y_true, y_pred)
