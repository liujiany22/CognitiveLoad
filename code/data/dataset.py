"""PyTorch datasets and data-loader builders."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from itertools import combinations


class EEGDataset(Dataset):
    """Standard dataset that yields (eeg, label, subject_id).  Used by Stage 2."""

    def __init__(self, data: Dict[str, np.ndarray], subject_ids: Optional[List[int]] = None):
        mask = np.ones(len(data["eeg"]), dtype=bool)
        if subject_ids is not None:
            mask = np.isin(data["subject_ids"], subject_ids)

        self.eeg = torch.from_numpy(data["eeg"][mask])
        self.labels = torch.from_numpy(data["labels"][mask])
        self.subject_ids = torch.from_numpy(data["subject_ids"][mask])

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return {
            "eeg": self.eeg[idx].unsqueeze(0),        # (1, C, T)
            "label": self.labels[idx],
            "subject_id": self.subject_ids[idx],
        }


class CrossSubjectPairDataset(Dataset):
    """CLISA-style cross-subject pair dataset for Stage 1 contrastive learning.

    For each subject pair (A, B) and each condition, the *same* epoch-
    position indices are selected for both subjects, mirroring CLISA's
    shared-stimulus alignment.

    Positive pair for InfoNCE: position i ↔ position i (different subject).
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        subject_ids: List[int],
        segs_per_cond: int = 14,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.segs_per_cond = segs_per_cond

        mask = np.isin(data["subject_ids"], subject_ids)
        self.eeg = data["eeg"][mask]
        self.labels = data["labels"][mask]
        self.subs = data["subject_ids"][mask]
        self.positions = data["positions"][mask]

        self.unique_labels = sorted(set(self.labels))

        self._build_index()

        self.pairs = list(combinations(sorted(set(subject_ids)), 2))
        self.rng.shuffle(self.pairs)

    def _build_index(self):
        self.pos_to_idx: Dict[Tuple[int, int, int], int] = {}
        self.available_pos: Dict[Tuple[int, int], List[int]] = {}
        for i in range(len(self.eeg)):
            s, l, p = int(self.subs[i]), int(self.labels[i]), int(self.positions[i])
            self.pos_to_idx[(s, l, p)] = i
            self.available_pos.setdefault((s, l), []).append(p)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sub_a, sub_b = self.pairs[idx % len(self.pairs)]
        eeg_a_list, eeg_b_list, cond_list = [], [], []

        for lbl in self.unique_labels:
            common = sorted(
                set(self.available_pos.get((sub_a, lbl), []))
                & set(self.available_pos.get((sub_b, lbl), []))
            )
            if not common:
                continue

            k = min(self.segs_per_cond, len(common))
            chosen = self.rng.choice(common, k, replace=False)

            for p in chosen:
                eeg_a_list.append(self.eeg[self.pos_to_idx[(sub_a, lbl, p)]])
                eeg_b_list.append(self.eeg[self.pos_to_idx[(sub_b, lbl, p)]])
                cond_list.append(lbl)

        if not eeg_a_list:
            return self.__getitem__((idx + 1) % len(self.pairs))

        return {
            "eeg_a": torch.from_numpy(np.stack(eeg_a_list)).unsqueeze(1).float(),
            "eeg_b": torch.from_numpy(np.stack(eeg_b_list)).unsqueeze(1).float(),
            "cond_labels": torch.tensor(cond_list, dtype=torch.long),
        }


class DEFeatureDataset(Dataset):
    """Dataset wrapping pre-extracted & post-processed DE feature vectors."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {"feat": self.features[idx], "label": self.labels[idx]}


def _collate_pair(batch):
    """Collate pair batches by concatenation."""
    return {
        "eeg_a": torch.cat([b["eeg_a"] for b in batch], dim=0),
        "eeg_b": torch.cat([b["eeg_b"] for b in batch], dim=0),
        "cond_labels": torch.cat([b["cond_labels"] for b in batch], dim=0),
    }


def build_dataloaders(
    data: Dict[str, np.ndarray],
    train_subs: List[int],
    val_subs: List[int],
    test_subs: List[int],
    config,
) -> Dict[str, DataLoader]:
    """Build data-loaders for Stage 1 and Stage 2.

    Stage 3 has its own pre-extraction pipeline and builds DE-feature
    loaders via :meth:`Stage3Trainer.prepare_de_loaders`.

    Strict three-way split by subject to prevent test-set peeking:
      - train: model training (Stage 1/2)
      - val:   early-stopping & checkpoint selection (Stage 2)
      - test:  held out for final evaluation only
    """

    # Stage 1 — cross-subject pairs (train subs only)
    pair_ds = CrossSubjectPairDataset(
        data, train_subs,
        segs_per_cond=config.stage1_segs_per_cond,
        seed=config.seed,
    )
    pair_loader = DataLoader(
        pair_ds, batch_size=1, shuffle=True, collate_fn=_collate_pair,
        num_workers=config.num_workers,
    )

    # Stage 2 — standard EEG loaders
    train_ds = EEGDataset(data, train_subs)
    val_ds = EEGDataset(data, val_subs)
    test_ds = EEGDataset(data, test_subs)

    train_loader = DataLoader(
        train_ds, batch_size=config.stage2_batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.stage3_batch_size, shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.stage3_batch_size, shuffle=False,
        num_workers=config.num_workers,
    )

    return {
        "pair": pair_loader,
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
