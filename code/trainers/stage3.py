"""
Stage 3 — Classification (CLISA-style).

Full pipeline (mirrors CLISA):
  1. Sub-segment 5 s epochs → 1 s windows  (de_extract_sec)
  2. Frozen encoder → intermediate features → Differential Entropy
  3. normTrain  — z-score DE features using training-set mean / var
  4. LDS smooth — Kalman filter along consecutive temporal windows
  5. Train a shallow 3-layer MLP on processed DE features
"""

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.dataset import DEFeatureDataset
from losses import ClassificationLoss
from utils import compute_metrics, lds_smooth
from .base import BaseTrainer


class Stage3Trainer(BaseTrainer):
    label = "Stage3"
    early_stop_mode = "max"

    def __init__(self, model, config):
        tag = f"_{config.ablation}" if config.ablation else ""
        ckpt_name = f"stage3_best{tag}.pt"

        for p in model.cross_encoder.parameters():
            p.requires_grad = False
        for p in model.align_encoder.parameters():
            p.requires_grad = False

        super().__init__(model, config, ckpt_name=ckpt_name)

        self.criterion = ClassificationLoss(
            n_classes=config.n_classes,
            label_smoothing=0.0,
        )

    @property
    def _epochs(self):
        return self.config.stage3_epochs

    @property
    def _patience(self):
        return self.config.stage3_patience

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        opt = Adam(trainable, lr=self.config.stage3_lr,
                   weight_decay=self.config.stage3_weight_decay)
        sched = StepLR(opt, step_size=self.config.stage3_epochs, gamma=0.8)
        return opt, sched

    # ── DE pre-extraction pipeline ──────────────────────────────────────

    def prepare_de_loaders(self, data, train_subs, val_subs, test_subs):
        """Sub-segment → extract DE → normTrain → LDS → DataLoaders.

        Returns
        -------
        dict with keys ``"train"``, ``"val"``, ``"test"`` → DataLoader
            Each DataLoader yields ``{"feat": Tensor, "label": Tensor}``.
        """
        cfg = self.config
        de_sec = cfg.de_extract_sec
        srate = cfg.sampling_rate
        step_sec = cfg.epoch_step_sec
        samples_per_sub = int(de_sec * srate)
        n_subs_per_epoch = int(cfg.epoch_sec / de_sec)
        step_subs = int(step_sec / de_sec)

        all_subs = sorted(set(train_subs) | set(val_subs) | set(test_subs))

        # 1. Sub-segment into de_extract_sec windows; de-duplicate by position
        positions = data.get("positions")
        if positions is None:
            positions = np.arange(len(data["eeg"]), dtype=np.int64)

        sub_eegs, sub_labels, sub_subjects, sub_positions = [], [], [], []
        seen: set = set()

        for i in range(len(data["eeg"])):
            epoch = data["eeg"][i]                # (C, T)
            sid = int(data["subject_ids"][i])
            lbl = int(data["labels"][i])
            pos = int(positions[i])

            for j in range(n_subs_per_epoch):
                sub_pos = pos * step_subs + j
                key = (sid, lbl, sub_pos)
                if key in seen:
                    continue
                seen.add(key)

                start = j * samples_per_sub
                end = start + samples_per_sub
                if end > epoch.shape[1]:
                    break
                sub_eegs.append(epoch[:, start:end])
                sub_labels.append(lbl)
                sub_subjects.append(sid)
                sub_positions.append(sub_pos)

        sub_eeg = np.stack(sub_eegs, dtype=np.float32)
        sub_labels_arr = np.array(sub_labels, dtype=np.int64)
        sub_subjects_arr = np.array(sub_subjects, dtype=np.int64)
        sub_positions_arr = np.array(sub_positions, dtype=np.int64)
        n_total = len(sub_eeg)

        print(f"  DE extraction: {n_total} unique {de_sec}s windows "
              f"(from {len(data['eeg'])} epochs)")

        # 2. Extract DE features through frozen encoder
        self.model.to(self.device)
        self.model.eval()
        eeg_t = torch.from_numpy(sub_eeg).unsqueeze(1).float()   # (M, 1, C, T_sub)

        de_parts: list[np.ndarray] = []
        bs = 256
        with torch.no_grad():
            for i in range(0, n_total, bs):
                batch = eeg_t[i:i + bs].to(self.device)
                de_parts.append(self.model.extract_de(batch).cpu().numpy())

        de_features = np.concatenate(de_parts, axis=0)            # (M, de_dim)
        print(f"  DE features shape: {de_features.shape}")

        # 3. normTrain — z-score using training-set statistics
        train_mask = np.isin(sub_subjects_arr, train_subs)
        train_feats = de_features[train_mask]
        de_mean = train_feats.mean(axis=0)
        de_std = np.sqrt(train_feats.var(axis=0) + 1e-5)
        de_features = (de_features - de_mean) / de_std
        print(f"  normTrain: z-scored with training-set stats "
              f"(n_train={int(train_mask.sum())})")

        # 4. LDS smoothing per (subject, condition) group
        unique_labels = sorted(set(sub_labels_arr))
        for sid in all_subs:
            for lbl in unique_labels:
                mask = (sub_subjects_arr == sid) & (sub_labels_arr == lbl)
                if mask.sum() < 2:
                    continue
                idx = np.where(mask)[0]
                order = np.argsort(sub_positions_arr[idx])
                idx_sorted = idx[order]
                de_features[idx_sorted] = lds_smooth(de_features[idx_sorted])

        print("  LDS smoothing applied")

        # 5. Build DataLoaders
        loaders: dict[str, DataLoader] = {}
        for name, subs in [("train", train_subs),
                           ("val", val_subs),
                           ("test", test_subs)]:
            mask = np.isin(sub_subjects_arr, subs)
            ds = DEFeatureDataset(de_features[mask], sub_labels_arr[mask])

            if name == "train":
                lbl_np = sub_labels_arr[mask]
                counts = np.bincount(
                    lbl_np, minlength=cfg.n_classes,
                ).astype(np.float64)
                weights = 1.0 / counts[lbl_np]
                sampler = WeightedRandomSampler(
                    weights=weights, num_samples=len(ds), replacement=True,
                )
                loaders[name] = DataLoader(
                    ds, batch_size=cfg.stage3_batch_size,
                    sampler=sampler, num_workers=cfg.num_workers,
                    drop_last=True,
                )
            else:
                loaders[name] = DataLoader(
                    ds, batch_size=cfg.stage3_batch_size,
                    shuffle=False, num_workers=cfg.num_workers,
                )

        for name, dl in loaders.items():
            print(f"  {name:5s} loader: {len(dl.dataset)} samples")
        return loaders

    # ── training / evaluation on pre-extracted features ────────────────

    def train_step(self, batch):
        feat = batch["feat"].to(self.device)
        labels = batch["label"].to(self.device)

        logits = self.model.forward_from_de(feat)
        loss = self.criterion(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return {"loss": loss, "acc": acc}

    @torch.no_grad()
    def validate(self, val_loader) -> float | None:
        if val_loader is None:
            return None
        metrics = self.evaluate(val_loader)
        return metrics["balanced_accuracy"]

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []

        for batch in loader:
            feat = batch["feat"].to(self.device)
            labels = batch["label"]
            logits = self.model.forward_from_de(feat)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels)

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        return compute_metrics(y_true, y_pred)
