"""
DualAlign-CogLoad: full model with three operating modes.

  Stage 1 — forward_cross_subject   : cross-subject contrastive
  Stage 2 — forward_eeg_phase       : EEG ↔ frozen-text-embedding alignment
  Stage 3 — forward (default)       : dual-branch fusion → classification

Text features come from a frozen nn.Embedding layer indexed by label.
No training is applied to the text side — only the EEG branch aligns
towards these fixed anchor points in the projection space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import EEGEncoder


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ContrastiveProjector(nn.Module):
    """MLP projection head for InfoNCE (Stage 1)."""

    def __init__(self, embed_dim: int, proj_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x):
        return self.proj(x)


class AlignmentProjector(nn.Module):
    """Proj_eeg: EEG embedding → alignment space."""

    def __init__(self, embed_dim: int, proj_dim: int, dropout: float = 0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            ResidualMLP(proj_dim, dropout),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        return self.proj(x)


class DualAlignModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        enc_kwargs = dict(
            n_channels=config.n_channels,
            n_temporal_filters=config.n_temporal_filters,
            n_spatial_filters=config.n_spatial_filters,
            temporal_kernel=config.temporal_kernel,
            pool_kernel=config.pool_kernel,
            pool_stride=config.pool_stride,
            n_temporal_out=config.n_temporal_out,
            embed_dim=config.embed_dim,
            dropout=config.encoder_dropout,
            use_channel_attention=config.use_channel_attention,
        )

        # ── Branch 1: Cross-subject encoder (CL-SSTER inspired) ──
        self.cross_encoder = EEGEncoder(**enc_kwargs)
        self.cross_projector = ContrastiveProjector(config.embed_dim, config.proj_dim)

        # ── Branch 2: Stimulus-aligned encoder ──
        self.align_encoder = EEGEncoder(**enc_kwargs)
        self.align_projector = AlignmentProjector(
            config.embed_dim, config.proj_dim, dropout=0.5,
        )

        # Frozen text embedding: each class label maps to a fixed vector
        self.text_embedding = nn.Embedding(config.n_classes, config.proj_dim)
        self.text_embedding.requires_grad_(False)

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / config.temperature)))

        # ── Fusion + classifier (Stage 3) ──
        self.fusion = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.fusion_dim, config.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.classifier_dropout),
        )
        self.classifier = nn.Linear(config.fusion_dim, config.n_classes)

        self._config = config

    # ── Stage 1 ──
    def forward_cross_subject(self, eeg: torch.Tensor) -> torch.Tensor:
        feat = self.cross_encoder(eeg)
        return self.cross_projector(feat)

    # ── Stage 2: align EEG with frozen text embeddings ──
    def forward_eeg_phase(self, eeg: torch.Tensor, labels: torch.Tensor):
        """
        Returns (eeg_proj, text_emb, logit_scale).
        text_emb is from a frozen embedding — no gradient flows to it.
        """
        eeg_proj = self.align_projector(self.align_encoder(eeg))
        text_emb = self.text_embedding(labels)  # already frozen
        return eeg_proj, text_emb, self.logit_scale.exp()

    # ── Stage 3 / inference ──
    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        cross_feat = self.cross_encoder(eeg)
        align_feat = self.align_encoder(eeg)
        combined = torch.cat([cross_feat, align_feat], dim=-1)
        fused = self.fusion(combined)
        return self.classifier(fused)

    def init_align_from_cross(self):
        """Initialise the alignment encoder from the cross-subject encoder."""
        self.align_encoder.load_state_dict(self.cross_encoder.state_dict())
