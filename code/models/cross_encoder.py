"""
CLISA-style ConvNet encoder for Stage 1 cross-subject contrastive learning.

Architecture (faithful to CLISA ConvNet_baseNonlinearHead):
    [B, 1, C, T]
    → StratifiedLayerNorm  ('initial', optional)
    → SpatialConv           Conv2d(1, n_sf, (C, 1))
    → Permute(0,2,1,3)
    → TemporalConv          Conv2d(1, n_tf, (1, k), same-padded)
    → ELU
    → AvgPool               (1, pool_len)
    → StratifiedLayerNorm  ('middle1', optional)
    → DepthSpatialConv      Conv2d(n_tf, n_tf·mf, (n_sf, 1), groups=n_tf)
    → ELU
    → DepthTemporalConv     Conv2d(n_tf·mf, n_tf·mf², (1, 6), groups=n_tf·mf)
    → ELU
    → StratifiedLayerNorm  ('middle2', optional)
    → Flatten               → out_dim

For CLISA-style classification the encoder also exposes
``forward_intermediate`` which returns the TemporalConv output
*before ELU* — shape (B, n_tf, n_sf, T').  A companion
``DifferentialEntropy`` module converts that map to a flat
(B, n_tf·n_sf) feature vector, mirroring CLISA extract_pretrainFeat.py.

No projection head — output features go directly to contrastive loss.

Reference:
    Shen et al., "Contrastive Learning of Subject-Invariant EEG
    Representations for Cross-Subject Emotion Recognition",
    IEEE Trans. Affective Computing, 2021.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StratifiedLayerNorm(nn.Module):
    """Per-subject layer normalisation for 4-D feature maps (CLISA-faithful).

    For each subject slice the input (n, C1, C2, T) is reshaped to
    (n·T, C1·C2) and normalised along dim-0, treating every time-point
    of every sample as an independent observation.  This preserves
    per-channel statistics while removing cross-subject amplitude bias.

    When *n_per_subject* is ``None`` the module is an identity.
    """

    EPS = 1e-3

    def forward(self, x: torch.Tensor, n_per_subject: int | None = None) -> torch.Tensor:
        if n_per_subject is None:
            return x
        B = x.shape[0]
        n_subs = B // n_per_subject
        out = x.clone()
        for i in range(n_subs):
            sl = slice(n_per_subject * i, n_per_subject * (i + 1))
            chunk = x[sl]                                          # (n, C1, C2, T)
            n, T = chunk.shape[0], chunk.shape[-1]
            flat = chunk.reshape(n, -1, T).permute(0, 2, 1)       # (n, T, C1·C2)
            flat = flat.reshape(n * T, -1)                         # (n·T, C1·C2)
            normed = (flat - flat.mean(dim=0)) / (flat.std(dim=0) + self.EPS)
            out[sl] = (
                normed
                .reshape(n, T, -1)
                .permute(0, 2, 1)
                .reshape(chunk.shape)
            )
        return out


class CrossEncoder(nn.Module):
    """ConvNet for cross-subject contrastive learning (CLISA-faithful).

    Key design choices:
      - Spatial → Temporal convolution order
      - Depthwise (grouped) convolutions in second block
      - StratifiedLayerNorm at configurable positions (no BatchNorm)
      - No channel attention, no dropout, no linear projection
      - Output is flattened features — no projection head
    """

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        n_spatial_filters: int = 16,
        n_time_filters: int = 16,
        time_filter_len: int = 60,
        avg_pool_len: int = 30,
        multi_fact: int = 2,
        stratified: str = "middle1",
    ):
        super().__init__()
        positions = set(stratified.split(",")) if stratified else set()
        mf = multi_fact

        # Block 1: Spatial → Temporal → Pool
        self.spatial_conv = nn.Conv2d(1, n_spatial_filters, (n_channels, 1))
        self.time_conv = nn.Conv2d(
            1, n_time_filters, (1, time_filter_len),
            padding=(0, (time_filter_len - 1) // 2),
        )
        self.avg_pool = nn.AvgPool2d((1, avg_pool_len))

        # Block 2: Depthwise spatial → Depthwise temporal
        self.depth_spatial = nn.Conv2d(
            n_time_filters, n_time_filters * mf,
            (n_spatial_filters, 1), groups=n_time_filters,
        )
        self.depth_temporal = nn.Conv2d(
            n_time_filters * mf, n_time_filters * mf * mf,
            (1, 6), groups=n_time_filters * mf,
        )

        # StratifiedLayerNorm at configurable positions
        self.sln_initial = StratifiedLayerNorm() if "initial" in positions else None
        self.sln_middle1 = StratifiedLayerNorm() if "middle1" in positions else None
        self.sln_middle2 = StratifiedLayerNorm() if "middle2" in positions else None

        # Compute output dimension via dry run (shape-only, no gradients)
        with torch.no_grad():
            dummy = torch.zeros(2, 1, n_channels, n_timepoints)
            self.out_dim = self._features(dummy).shape[1]

    def _features(self, x: torch.Tensor, n_per_subject: int | None = None) -> torch.Tensor:
        if self.sln_initial is not None:
            x = self.sln_initial(x, n_per_subject)

        x = self.spatial_conv(x)           # (B, n_sf, 1, T)
        x = x.permute(0, 2, 1, 3)         # (B, 1, n_sf, T)
        x = self.time_conv(x)             # (B, n_tf, n_sf, T')
        x = F.elu(x)
        x = self.avg_pool(x)              # (B, n_tf, n_sf, T'')

        if self.sln_middle1 is not None:
            x = self.sln_middle1(x, n_per_subject)

        x = F.elu(self.depth_spatial(x))   # (B, n_tf·mf, 1, T'')
        x = F.elu(self.depth_temporal(x))  # (B, n_tf·mf², 1, T''')

        if self.sln_middle2 is not None:
            x = self.sln_middle2(x, n_per_subject)

        return x.reshape(x.shape[0], -1)

    def forward(self, x: torch.Tensor, n_per_subject: int | None = None) -> torch.Tensor:
        return self._features(x, n_per_subject)

    def forward_intermediate(self, x: torch.Tensor) -> torch.Tensor:
        """Return TemporalConv output **before ELU** — (B, n_tf, n_sf, T').

        Corresponds to ``out1`` in CLISA ``extract_pretrainFeat.py``:
            out1 = self.timeConv(out)   # captured before ELU
        StratifiedLayerNorm is skipped (matches CLISA which sets
        ``stratified=[]`` during feature extraction).
        """
        x = self.spatial_conv(x)           # (B, n_sf, 1, T)
        x = x.permute(0, 2, 1, 3)         # (B, 1, n_sf, T)
        x = self.time_conv(x)             # (B, n_tf, n_sf, T')
        return x


class DifferentialEntropy(nn.Module):
    """Differential Entropy along the time dimension (CLISA-style).

    Mirrors CLISA ``extract_pretrainFeat.py``::

        de = 0.5 * np.log(2 * np.pi * np.exp(1) * np.var(out, 3))

    Input:  (B, n_tf, n_sf, T)   — TemporalConv feature map
    Output: (B, n_tf * n_sf)     — flattened DE feature vector
    """

    LOG_2PIE = math.log(2.0 * math.pi * math.e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.var(dim=3)                              # (B, n_tf, n_sf)
        de = 0.5 * (self.LOG_2PIE + torch.log(var + 1e-8))
        return de.reshape(x.shape[0], -1)               # (B, n_tf * n_sf)
