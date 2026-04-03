from dataclasses import dataclass
import os
import torch


@dataclass
class Config:
    # ── Data ──
    n_subjects: int = 20
    n_channels: int = 32
    n_timepoints: int = 512          # 2 s @ 256 Hz
    sampling_rate: int = 256
    n_classes: int = 2
    test_subject_ratio: float = 0.2  # fraction of subjects held out for final evaluation
    val_subject_ratio: float = 0.15  # fraction of subjects for validation (early-stopping)

    # ── EEG Encoder ──
    n_temporal_filters: int = 40
    n_spatial_filters: int = 40
    temporal_kernel: int = 25
    pool_kernel: int = 8
    pool_stride: int = 4
    n_temporal_out: int = 16
    embed_dim: int = 256
    encoder_dropout: float = 0.5
    use_channel_attention: bool = True

    # ── Contrastive heads ──
    proj_dim: int = 128
    temperature: float = 0.07

    # ── Stage 1  Cross-Subject Pre-training ──
    stage1_epochs: int = 100
    stage1_lr: float = 1e-3
    stage1_weight_decay: float = 1e-2
    stage1_batch_size: int = 64

    # ── Stage 2  Stimulus-Task Alignment ──
    stage2_epochs: int = 80
    stage2_lr: float = 5e-4
    stage2_weight_decay: float = 1e-2
    stage2_batch_size: int = 128

    # ── Stage 3  Classification Fine-tuning ──
    stage3_epochs: int = 60
    stage3_lr: float = 1e-4
    stage3_weight_decay: float = 1e-2
    stage3_batch_size: int = 64
    fusion_dim: int = 256
    classifier_dropout: float = 0.3

    # ── General ──
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    data_dir: str = "data_cache"
    num_workers: int = 0
    patience: int = 15

    # ── Data source ──
    data_source: str = "eegmat"      # registered loader name
    data_path: str = ""              # root directory of the dataset
    epoch_sec: float = 2.0           # epoch length (seconds) for epoching

    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
