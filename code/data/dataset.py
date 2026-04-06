"""
PyTorch datasets and data-loader builders for DualAlign-CogLoad.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Tuple, Optional
from itertools import combinations


class CogLoadDataset(Dataset):
    """Standard dataset that yields (eeg, label, subject_id).  Used by Stage 2 / 3."""

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
        eeg_a_list, eeg_b_list = [], []

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

        if not eeg_a_list:
            return self.__getitem__((idx + 1) % len(self.pairs))

        return {
            "eeg_a": torch.from_numpy(np.stack(eeg_a_list)).unsqueeze(1).float(),
            "eeg_b": torch.from_numpy(np.stack(eeg_b_list)).unsqueeze(1).float(),
        }


def _collate_pair(batch):
    """Collate pair batches by concatenation."""
    return {
        "eeg_a": torch.cat([b["eeg_a"] for b in batch], dim=0),
        "eeg_b": torch.cat([b["eeg_b"] for b in batch], dim=0),
    }


def build_dataloaders(
    data: Dict[str, np.ndarray],
    train_subs: List[int],
    val_subs: List[int],
    test_subs: List[int],
    config,
) -> Dict[str, DataLoader]:
    """Build all data-loaders needed by the three stages.

    Strict three-way split by subject to prevent test-set peeking:
      - train: model training (Stage 1/2/3)
      - val:   early-stopping & checkpoint selection (Stage 2/3)
      - test:  final one-shot evaluation only
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

    # Stage 2 & 3 — standard loaders (unchanged)
    train_ds = CogLoadDataset(data, train_subs)
    val_ds = CogLoadDataset(data, val_subs)
    test_ds = CogLoadDataset(data, test_subs)

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

    # Inverse-frequency sampling so each batch has balanced class distribution
    train_labels = train_ds.labels.numpy()
    class_counts = np.bincount(train_labels, minlength=config.n_classes).astype(np.float64)
    sample_weights = 1.0 / class_counts[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    finetune_loader = DataLoader(
        train_ds, batch_size=config.stage3_batch_size, sampler=sampler,
        num_workers=config.num_workers, drop_last=True,
    )

    return {
        "pair": pair_loader,
        "train": train_loader,
        "val": val_loader,
        "finetune": finetune_loader,
        "test": test_loader,
    }
