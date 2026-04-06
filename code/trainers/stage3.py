"""
Stage 3 — Classification.

# <de> before: CLISA-style pipeline
#   1. Sub-segment 5 s epochs → 1 s windows  (de_extract_sec)
#   2. Frozen encoder → intermediate features → Differential Entropy
#   3. normTrain  — z-score DE features using training-set mean / var
#   4. LDS smooth — Kalman filter along consecutive temporal windows
#   5. Train a shallow 3-layer MLP on processed DE features
# <de> after: full encoder pipeline
#   1. Frozen full CrossEncoder on original 5 s epochs
#   2. normTrain  — z-score features using training-set mean / var
#   3. LDS smooth — Kalman filter per (subject, condition) sequence
#   4. Train a shallow 3-layer MLP on processed features
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

    # ── Feature pre-extraction pipeline ─────────────────────────────────
    # <de> before: sub-segmented 5s→1s, then forward_intermediate + DE → (M, 256)
    # <de> after:  full CrossEncoder on original 5s epochs → (N, out_dim)

    def prepare_de_loaders(self, data, train_subs, val_subs, test_subs):
        """Extract features → normTrain → LDS → DataLoaders.

        Returns
        -------
        dict with keys ``"train"``, ``"val"``, ``"test"`` → DataLoader
            Each DataLoader yields ``{"feat": Tensor, "label": Tensor}``.
        """
        cfg = self.config
        all_subs = sorted(set(train_subs) | set(val_subs) | set(test_subs))

        eeg_all = data["eeg"]                                     # (N, C, T)
        labels_arr = data["labels"].astype(np.int64)
        subjects_arr = data["subject_ids"].astype(np.int64)
        positions_arr = data.get("positions")
        if positions_arr is None:
            positions_arr = np.arange(len(eeg_all), dtype=np.int64)
        else:
            positions_arr = positions_arr.astype(np.int64)
        n_total = len(eeg_all)

        print(f"  Feature extraction: {n_total} epochs")

        # 1. Extract features through full frozen CrossEncoder
        self.model.to(self.device)
        self.model.eval()
        eeg_t = torch.from_numpy(eeg_all).unsqueeze(1).float()   # (N, 1, C, T)

        feat_parts: list[np.ndarray] = []
        bs = 256
        with torch.no_grad():
            for i in range(0, n_total, bs):
                batch = eeg_t[i:i + bs].to(self.device)
                feat_parts.append(self.model.extract_features(batch).cpu().numpy())

        features = np.concatenate(feat_parts, axis=0)             # (N, feat_dim)
        print(f"  Features shape: {features.shape}")

        # 2. normTrain — z-score using training-set statistics
        train_mask = np.isin(subjects_arr, train_subs)
        train_feats = features[train_mask]
        feat_mean = train_feats.mean(axis=0)
        feat_std = np.sqrt(train_feats.var(axis=0) + 1e-5)
        features = (features - feat_mean) / feat_std
        print(f"  normTrain: z-scored with training-set stats "
              f"(n_train={int(train_mask.sum())})")

        # 3. LDS smoothing per (subject, condition) group
        unique_labels = sorted(set(labels_arr))
        for sid in all_subs:
            for lbl in unique_labels:
                mask = (subjects_arr == sid) & (labels_arr == lbl)
                if mask.sum() < 2:
                    continue
                idx = np.where(mask)[0]
                order = np.argsort(positions_arr[idx])
                idx_sorted = idx[order]
                features[idx_sorted] = lds_smooth(features[idx_sorted])

        print("  LDS smoothing applied")

        # 4. Build DataLoaders
        loaders: dict[str, DataLoader] = {}
        for name, subs in [("train", train_subs),
                           ("val", val_subs),
                           ("test", test_subs)]:
            mask = np.isin(subjects_arr, subs)
            ds = DEFeatureDataset(features[mask], labels_arr[mask])

            if name == "train":
                lbl_np = labels_arr[mask]
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
