"""
PyTorch datasets and data-loader builders for DualAlign-CogLoad.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Tuple, Optional
from itertools import combinations


class CogLoadDataset(Dataset):
    """Standard dataset that yields (eeg, task_features, condition_id, label)."""

    def __init__(self, data: Dict[str, np.ndarray], subject_ids: Optional[List[int]] = None):
        mask = np.ones(len(data["eeg"]), dtype=bool)
        if subject_ids is not None:
            mask = np.isin(data["subject_ids"], subject_ids)

        self.eeg = torch.from_numpy(data["eeg"][mask])
        self.labels = torch.from_numpy(data["labels"][mask])
        self.subject_ids = torch.from_numpy(data["subject_ids"][mask])
        self.condition_ids = torch.from_numpy(data["condition_ids"][mask])
        self.task_features = torch.from_numpy(data["task_features"][mask])

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return {
            "eeg": self.eeg[idx].unsqueeze(0),        # (1, C, T)
            "label": self.labels[idx],
            "subject_id": self.subject_ids[idx],
            "condition_id": self.condition_ids[idx],
            "task_features": self.task_features[idx],
        }


class CrossSubjectPairDataset(Dataset):
    """
    Yields matched pairs from two different subjects under the same
    fine-grained condition (9-class: task × time-segment × performance).

    For Stage-1 cross-subject contrastive pre-training (CL-SSTER style).
    Positive pair = same condition, different subjects.
    Different conditions within the same batch serve as hard negatives.

    Note: good-performer subjects (conds 3-5) and bad-performer subjects
    (conds 6-8) share only rest conditions (0-2).  The arithmetic
    conditions will only produce pairs between subjects of the same
    performance group — this is intentional.
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        subject_ids: List[int],
        samples_per_pair: int = 64,
        seed: int = 42,
    ):
        self.rng = np.random.RandomState(seed)
        self.samples_per_pair = samples_per_pair

        mask = np.isin(data["subject_ids"], subject_ids)
        self.eeg = data["eeg"][mask]
        self.labels = data["labels"][mask]
        self.subs = data["subject_ids"][mask]
        self.conds = data["condition_ids"][mask]
        self.task_features = data["task_features"][mask]

        self.unique_subs = sorted(set(subject_ids))
        self.unique_conds = sorted(set(self.conds))
        self.n_conds = len(self.unique_conds)

        self._build_index()

        self.pairs = list(combinations(self.unique_subs, 2))
        self.rng.shuffle(self.pairs)

    def _build_index(self):
        """Map (subject, condition_id) → list of sample indices."""
        self.index: Dict[Tuple[int, int], List[int]] = {}
        for i in range(len(self.eeg)):
            key = (int(self.subs[i]), int(self.conds[i]))
            self.index.setdefault(key, []).append(i)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sub_a, sub_b = self.pairs[idx % len(self.pairs)]
        eeg_a_list, eeg_b_list, labels_list, tf_list = [], [], [], []

        per_cond = max(1, self.samples_per_pair // self.n_conds)

        for cond in self.unique_conds:
            pool_a = self.index.get((sub_a, cond), [])
            pool_b = self.index.get((sub_b, cond), [])
            if not pool_a or not pool_b:
                continue

            k = min(per_cond, len(pool_a), len(pool_b))
            sel_a = self.rng.choice(pool_a, k, replace=len(pool_a) < k)
            sel_b = self.rng.choice(pool_b, k, replace=len(pool_b) < k)

            eeg_a_list.append(self.eeg[sel_a])
            eeg_b_list.append(self.eeg[sel_b])
            labels_list.append(self.labels[sel_a])
            tf_list.append(self.task_features[sel_a])

        if not eeg_a_list:
            return self.__getitem__((idx + 1) % len(self.pairs))

        eeg_a = np.concatenate(eeg_a_list, axis=0)
        eeg_b = np.concatenate(eeg_b_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        tf = np.concatenate(tf_list, axis=0)

        return {
            "eeg_a": torch.from_numpy(eeg_a).unsqueeze(1).float(),  # (K, 1, C, T)
            "eeg_b": torch.from_numpy(eeg_b).unsqueeze(1).float(),
            "labels": torch.from_numpy(labels).long(),
            "task_features": torch.from_numpy(tf).float(),
        }


def _collate_pair(batch):
    """Collate variable-size pair batches by concatenation."""
    eeg_a = torch.cat([b["eeg_a"] for b in batch], dim=0)
    eeg_b = torch.cat([b["eeg_b"] for b in batch], dim=0)
    labels = torch.cat([b["labels"] for b in batch], dim=0)
    tf = torch.cat([b["task_features"] for b in batch], dim=0)
    return {"eeg_a": eeg_a, "eeg_b": eeg_b, "labels": labels, "task_features": tf}


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
        samples_per_pair=config.stage1_batch_size,
        seed=config.seed,
    )
    pair_loader = DataLoader(
        pair_ds, batch_size=1, shuffle=True, collate_fn=_collate_pair,
        num_workers=config.num_workers,
    )

    # Stage 2 & 3 — standard loaders
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

    finetune_loader = DataLoader(
        train_ds, batch_size=config.stage3_batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True,
    )

    return {
        "pair": pair_loader,
        "train": train_loader,
        "val": val_loader,
        "finetune": finetune_loader,
        "test": test_loader,
    }
