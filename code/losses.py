"""
Loss functions for the three training stages.

Stage 1  — InfoNCELoss: cross-subject contrastive (CLISA-style)
Stage 2  — CLIPLoss:    EEG ↔ frozen-text bidirectional alignment
Stage 3  — CogLoadLoss: classification CE with label smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    NT-Xent loss for cross-subject contrastive learning (CL-SSTER style).

    Positive pair: (z_a[i], z_b[i]) — same condition, different subjects.
    Negatives:     all other samples in the batch.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)
        B = z_a.size(0)

        features = torch.cat([z_a, z_b], dim=0)
        sim = features @ features.T / self.temperature
        sim.fill_diagonal_(-1e9)

        labels = torch.cat([
            torch.arange(B, 2 * B, device=z_a.device),
            torch.arange(0, B, device=z_a.device),
        ])

        loss = F.cross_entropy(sim, labels)

        with torch.no_grad():
            preds = sim.argmax(dim=1)
            acc = (preds == labels).float().mean()

        return loss, acc


class SupConLoss(nn.Module):
    """
    Supervised contrastive loss (Khosla et al., 2020).

    All samples sharing the same label form positive pairs; the rest
    are negatives.  Returns (loss, nn_acc) where nn_acc is the
    nearest-neighbour retrieval accuracy.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        features = F.normalize(features, dim=1)
        B = features.size(0)
        device = features.device

        sim = features @ features.T / self.temperature

        # positive mask: same label, exclude self
        label_col = labels.unsqueeze(1)
        pos_mask = (label_col == label_col.T).float()
        pos_mask.fill_diagonal_(0)

        n_pos = pos_mask.sum(dim=1)
        has_pos = n_pos > 0

        # log-softmax over all non-self entries
        self_mask = torch.eye(B, device=device)
        logits = sim - 1e9 * self_mask
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)
        loss = -mean_log_prob[has_pos].mean()

        with torch.no_grad():
            sim_nn = sim.clone()
            sim_nn.fill_diagonal_(-1e9)
            acc = (labels[sim_nn.argmax(dim=1)] == labels).float().mean()

        return loss, acc


class CLIPLoss(nn.Module):
    """
    Bidirectional contrastive loss for EEG ↔ text alignment
    (Stage 2, Phase B — same structure as NICE/NICE++).
    """

    def forward(
        self,
        eeg_feat: torch.Tensor,
        text_feat: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        eeg_feat = F.normalize(eeg_feat, dim=1)
        text_feat = F.normalize(text_feat, dim=1)

        logits_eeg = logit_scale * eeg_feat @ text_feat.T
        logits_text = logits_eeg.T

        labels = torch.arange(len(eeg_feat), device=eeg_feat.device)
        loss = (F.cross_entropy(logits_eeg, labels) +
                F.cross_entropy(logits_text, labels)) / 2

        with torch.no_grad():
            acc = (logits_eeg.argmax(1) == labels).float().mean()

        return loss, acc


class CogLoadLoss(nn.Module):
    """Stage-3 loss: classification CE with label smoothing and optional class balancing."""

    def __init__(self, n_classes: int = 2, label_smoothing: float = 0.05,
                 class_weights: list[float] | None = None):
        super().__init__()
        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels, weight=self.weight,
                               label_smoothing=self.label_smoothing)
