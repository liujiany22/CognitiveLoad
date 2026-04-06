"""
Loss functions for the three training stages.

Stage 1  — InfoNCELoss: cross-subject contrastive (CLISA-style)
Stage 2  — CLIPLoss:    EEG ↔ frozen-text bidirectional alignment
Stage 3  — ClassificationLoss: CE with label smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    NT-Xent loss for cross-subject contrastive learning (CL-SSTER style).

    Positive pair: (z_a[i], z_b[i]) — same position, different subjects.
    Negatives:     only *different-condition* samples.

    When ``cond_labels`` is provided, same-condition pairs (excluding the
    positive) are masked out of the denominator so they act as neither
    positives nor negatives.  This avoids pushing apart samples that
    share the same cognitive state but differ only in temporal position.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor,
                cond_labels: torch.Tensor | None = None) -> torch.Tensor:
        z_a = F.normalize(z_a, dim=1)
        z_b = F.normalize(z_b, dim=1)
        B = z_a.size(0)

        features = torch.cat([z_a, z_b], dim=0)           # (2B, D)
        sim = features @ features.T / self.temperature     # (2B, 2B)
        sim.fill_diagonal_(-1e9)

        if cond_labels is not None:
            # cond_labels: (B,) — same order for z_a and z_b
            cond_all = cond_labels.repeat(2)               # (2B,)
            same_cond = cond_all.unsqueeze(0) == cond_all.unsqueeze(1)

            pos_mask = torch.zeros(2 * B, 2 * B, dtype=torch.bool,
                                   device=z_a.device)
            idx = torch.arange(B, device=z_a.device)
            pos_mask[idx, idx + B] = True
            pos_mask[idx + B, idx] = True

            # Exclude same-condition non-positive pairs from the denominator
            exclude = same_cond & ~pos_mask
            sim = sim.masked_fill(exclude, -1e9)

        labels = torch.cat([
            torch.arange(B, 2 * B, device=z_a.device),
            torch.arange(0, B, device=z_a.device),
        ])

        loss = F.cross_entropy(sim, labels)

        with torch.no_grad():
            preds = sim.argmax(dim=1)
            acc = (preds == labels).float().mean()

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


class ClassificationLoss(nn.Module):
    """Classification CE with label smoothing and optional class balancing."""

    def __init__(self, n_classes: int = 2, label_smoothing: float = 0.05,
                 class_weights: list[float] | None = None):
        super().__init__()
        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels, weight=self.weight,
                               label_smoothing=self.label_smoothing)
