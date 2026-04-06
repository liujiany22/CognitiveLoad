"""
EEG encoder backbone — hybrid of CL-SSTER and NICE architectures.

Architecture:
    [B, 1, C, T]
    → ChannelAttention      (SE-style, optional)
    → TemporalConv          (band-pass filtering across time)
    → BatchNorm + ELU
    → AvgPool               (temporal down-sampling)
    → StratifiedLayerNorm   (per-subject norm, Stage 1 only; identity otherwise)
    → SpatialConv           (channel mixing)
    → BatchNorm + ELU + Dropout
    → AdaptiveAvgPool       (fixed temporal output)
    → Flatten → Linear      → embed_dim
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention (NICE-SA inspired)."""

    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(n_channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T)
        weights = x.squeeze(1).mean(dim=-1)                        # (B, C)
        weights = self.fc(weights).unsqueeze(1).unsqueeze(-1)      # (B, 1, C, 1)
        return x * weights


class StratifiedLayerNorm(nn.Module):
    """Per-subject layer normalisation for 4-D feature maps (CLISA-style).

    When *n_per_subject* is given the batch is assumed to be laid out as
    ``[sub_0 samples, sub_1 samples, ...]``.  Each subject slice is
    normalised independently across the flattened feature dimensions,
    removing cross-subject amplitude / offset differences.

    When *n_per_subject* is ``None`` (Stage 2 / 3) the module is an
    identity — no computation, no side-effects on other stages.
    """

    EPS = 1e-5

    def forward(self, x: torch.Tensor, n_per_subject: int | None = None) -> torch.Tensor:
        if n_per_subject is None:
            return x
        B = x.shape[0]
        n_subs = B // n_per_subject
        out = x.clone()
        for i in range(n_subs):
            sl = slice(n_per_subject * i, n_per_subject * (i + 1))
            chunk = x[sl]
            flat = chunk.reshape(chunk.shape[0], -1)
            mean = flat.mean(dim=0, keepdim=True)
            std = flat.std(dim=0, keepdim=True) + self.EPS
            out[sl] = ((flat - mean) / std).reshape(chunk.shape)
        return out


class EEGEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int = 32,
        n_temporal_filters: int = 40,
        n_spatial_filters: int = 40,
        temporal_kernel: int = 25,
        pool_kernel: int = 8,
        pool_stride: int = 4,
        n_temporal_out: int = 16,
        embed_dim: int = 256,
        dropout: float = 0.5,
        use_channel_attention: bool = True,
    ):
        super().__init__()

        self.ch_attn = (
            ChannelAttention(n_channels) if use_channel_attention else None
        )

        self.temporal_block = nn.Sequential(
            nn.Conv2d(1, n_temporal_filters, (1, temporal_kernel)),
            nn.BatchNorm2d(n_temporal_filters),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, pool_kernel), stride=(1, pool_stride)),
        )

        self.stratified_norm = StratifiedLayerNorm()

        self.spatial_block = nn.Sequential(
            nn.Conv2d(n_temporal_filters, n_spatial_filters, (n_channels, 1)),
            nn.BatchNorm2d(n_spatial_filters),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, n_temporal_out))

        self.flatten_dim = n_spatial_filters * n_temporal_out
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, embed_dim),
        )

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, n_per_subject: int | None = None) -> torch.Tensor:
        # x: (B, 1, C, T)
        if self.ch_attn is not None:
            x = self.ch_attn(x)
        x = self.temporal_block(x)                  # (B, F_t, C, T')
        x = self.stratified_norm(x, n_per_subject)  # per-subject norm (Stage 1 only)
        x = self.spatial_block(x)                   # (B, F_s, 1, T')
        x = self.adaptive_pool(x)                   # (B, F_s, 1, n_out)
        x = self.projection(x)                      # (B, embed_dim)
        return x
