from dataclasses import dataclass
import os
from typing import List, Optional
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

    # ── Stage 1 Encoder (CLISA-style ConvNet) ──
    cross_n_spatial_filters: int = 16
    cross_n_time_filters: int = 16
    cross_time_filter_len: int = 60
    cross_avg_pool_len: int = 30
    cross_multi_fact: int = 2
    cross_stratified: str = "initial,middle1,middle2"  # comma-separated: initial,middle1,middle2

    # ── Stage 2 Encoder (AlignEncoder) ──
    n_temporal_filters: int = 40
    n_spatial_filters: int = 40
    temporal_kernel: int = 25
    pool_kernel: int = 8
    pool_stride: int = 4
    n_temporal_out: int = 16
    embed_dim: int = 256
    encoder_dropout: float = 0.5
    use_channel_attention: bool = True

    # ── Stage 2 Projection / Alignment ──
    proj_dim: int = 128
    temperature: float = 0.07

    # ── Stage 1  Cross-Subject Pre-training ──
    stage1_epochs: int = 80
    stage1_lr: float = 7e-4
    stage1_weight_decay: float = 0.015
    stage1_batch_size: int = 28
    stage1_segs_per_cond: int = 14   # segments sampled per condition per subject pair
    stage1_restart_times: int = 3    # cosine warm-restart cycles (CLISA default)

    # ── Stage 2  Stimulus-Task Alignment ──
    stage2_epochs: int = 80
    stage2_lr: float = 5e-4
    stage2_weight_decay: float = 1e-2
    stage2_batch_size: int = 128

    # ── Stage 3  Classification (CLISA-style: frozen encoder → DE → MLP) ──
    stage3_epochs: int = 100
    stage3_lr: float = 5e-4
    stage3_weight_decay: float = 0.05
    stage3_batch_size: int = 270
    stage3_patience: int = 50
    classifier_hidden: int = 30
    de_extract_sec: float = 1.0      # DE feature extraction window (CLISA uses 1 s)

    # ── Ablation ──
    ablation: str = ""               # "" or "cross_only" (skip Stage 2)

    # ── Checkpointing ──
    ckpt_every: int = 20             # save checkpoint every N epochs (0 = off)

    # ── General ──
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    data_dir: str = "data_cache"
    num_workers: int = 0
    patience: int = 30

    # ── Data source ──
    data_source: str = "eegmat"      # registered loader name
    data_path: str = "datasets/eegmat"  # root directory of the dataset
    epoch_sec: float = 5.0          # epoch length (seconds) for epoching
    epoch_step_sec: float = 2.0      # sliding-window step; set < epoch_sec for overlap

    # ── Computed at runtime ──
    class_weights: Optional[List[float]] = None  # inverse-frequency weights per class

    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
