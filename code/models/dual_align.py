"""
DualAlign: full model with three operating modes.

  Stage 1 — forward_cross_subject  : cross-subject contrastive (CrossEncoder)
  Stage 2 — forward_alignment      : EEG ↔ frozen-text-embedding alignment (AlignEncoder)
  Stage 3 — extract_de + forward_from_de : frozen encoder → DE → classifier

Stage 3 pipeline (CLISA-style, handled by Stage3Trainer):
  1. Re-segment 5 s epochs → 1 s sub-windows
  2. extract_de()       : frozen CrossEncoder intermediate → DifferentialEntropy
  3. normTrain          : z-score DE features with training-set statistics
  4. LDS smoothing      : Kalman filter per (subject, condition) temporal sequence
  5. forward_from_de()  : shallow 3-layer MLP classifier on processed DE features
"""

import torch
import torch.nn as nn

# <de> before: from .cross_encoder import CrossEncoder, DifferentialEntropy
from .cross_encoder import CrossEncoder
from .align_encoder import AlignEncoder


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


class AlignmentProjector(nn.Module):
    """Proj_eeg: EEG embedding → alignment space (Stage 2)."""

    def __init__(self, embed_dim: int, proj_dim: int, dropout: float = 0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            ResidualMLP(proj_dim, dropout),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        return self.proj(x)


class DualAlign(nn.Module):

    def __init__(self, config):
        super().__init__()

        # ── Branch 1: cross-subject contrastive ──
        self.cross_encoder = CrossEncoder(
            n_channels=config.n_channels,
            n_timepoints=config.n_timepoints,
            n_spatial_filters=config.cross_n_spatial_filters,
            n_time_filters=config.cross_n_time_filters,
            time_filter_len=config.cross_time_filter_len,
            avg_pool_len=config.cross_avg_pool_len,
            multi_fact=config.cross_multi_fact,
            stratified=config.cross_stratified,
        )

        # ── Branch 2: stimulus-aligned ──
        self.align_encoder = AlignEncoder(
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
        self.align_projector = AlignmentProjector(
            config.embed_dim, config.proj_dim, dropout=0.5,
        )

        # Frozen text embedding: each class label maps to a fixed vector
        self.text_embedding = nn.Embedding(config.n_classes, config.proj_dim)
        self.text_embedding.requires_grad_(False)

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / config.temperature)))

        # ── Stage 3 classifier ──
        # <de> before: used DifferentialEntropy on intermediate features (dim=256)
        # self.de_layer = DifferentialEntropy()
        # de_dim = config.cross_n_time_filters * config.cross_n_spatial_filters
        # <de> after: use full CrossEncoder output (dim=cross_encoder.out_dim)
        feat_dim = self.cross_encoder.out_dim
        h = config.classifier_hidden
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, config.n_classes),
        )

        self._config = config

    # ── Stage 1: cross-subject contrastive ──
    def forward_cross_subject(self, eeg: torch.Tensor,
                              n_per_subject: int | None = None) -> torch.Tensor:
        return self.cross_encoder(eeg, n_per_subject=n_per_subject)

    # ── Stage 2: align EEG with frozen text embeddings ──
    def forward_alignment(self, eeg: torch.Tensor, labels: torch.Tensor):
        """Returns (eeg_proj, text_emb, logit_scale)."""
        eeg_proj = self.align_projector(self.align_encoder(eeg))
        text_emb = self.text_embedding(labels)
        return eeg_proj, text_emb, self.logit_scale.exp()

    # ── Stage 3 / inference ──
    # <de> before: extract_de used forward_intermediate + DifferentialEntropy
    # @torch.no_grad()
    # def extract_de(self, eeg):
    #     intermediate = self.cross_encoder.forward_intermediate(eeg)
    #     return self.de_layer(intermediate)
    # <de> after: extract_features uses full CrossEncoder forward
    @torch.no_grad()
    def extract_features(self, eeg: torch.Tensor) -> torch.Tensor:
        """Extract features using the full CrossEncoder."""
        return self.cross_encoder(eeg)

    def forward_from_de(self, de_feat: torch.Tensor) -> torch.Tensor:
        """Run the classifier on pre-extracted features."""
        return self.classifier(de_feat)

    # <de> before: forward used forward_intermediate + DE
    # def forward(self, eeg):
    #     intermediate = self.cross_encoder.forward_intermediate(eeg)
    #     feat = self.de_layer(intermediate)
    #     return self.classifier(feat)
    # <de> after: forward uses full CrossEncoder
    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        feat = self.cross_encoder(eeg)
        return self.classifier(feat)

    def load_compatible_state_dict(self, state_dict: dict):
        """Load weights, silently skipping keys whose shapes don't match."""
        own = self.state_dict()
        compatible = {
            k: v for k, v in state_dict.items()
            if k in own and v.shape == own[k].shape
        }
        self.load_state_dict(compatible, strict=False)
