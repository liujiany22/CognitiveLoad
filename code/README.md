# DualAlign-CogLoad

**Dual-Alignment Network for EEG-Based Cognitive Load Monitoring**

A three-stage deep learning framework that combines **cross-subject contrastive alignment** (inspired by [CL-SSTER](https://github.com/SherazKhan/cl-sster)) and **EEG–text embedding alignment** (inspired by [NICE-EEG](https://github.com/eeyhsong/NICE-EEG)) to classify cognitive load states (rest vs. mental arithmetic) from EEG signals.

---

## Motivation

| Challenge | Solution |
|-----------|----------|
| EEG varies across individuals | **Stage 1** learns a shared cross-subject representation via contrastive learning (CL-SSTER style) |
| Need to capture task-specific brain responses | **Stage 2** aligns EEG embeddings with frozen text anchors via CLIP-style contrastive loss |
| Must predict cognitive load robustly | **Stage 3** freezes encoders, extracts DE features from Stage-1 encoder, trains a shallow MLP classifier (CLISA-style) |

## Architecture

```
                     ┌──────────────────────┐
                     │  Raw EEG (B,1,C,T)   │
                     └──────────┬───────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
   ┌────────────────────┐          ┌────────────────────┐
   │  CrossEncoder       │          │  AlignEncoder       │
   │  (Stage 1: InfoNCE) │          │  (Stage 2: CLIP)    │
   └────────┬───────────┘          └────────────────────┘
            │
            │  Stage 3 uses CrossEncoder only:
            │  5s epoch → 5×1s sub-windows → encoder → DE
            ▼
   ┌──────────────────────┐
   │  Differential Entropy │
   │  var(dim=time) → log  │
   │  → (B, n_tf × n_sf)  │
   └────────┬──────────────┘
            ▼
   ┌──────────────────────┐
   │  normTrain (z-score)  │
   │  + LDS Kalman smooth  │
   └────────┬──────────────┘
            ▼
   ┌──────────────────────┐
   │  3-layer MLP          │
   │  256 → 30 → 30 → C   │
   └──────────────────────┘
```

### CrossEncoder (Stage 1 — CLISA-style)

```
StratifiedLayerNorm (optional, 'initial')
  → SpatialConv: Conv2d(1, 16, (C, 1))
  → Permute → TemporalConv: Conv2d(1, 16, (1, 60), same-pad)
  → ELU → AvgPool(1, 30)
  → StratifiedLayerNorm (optional, 'middle1')
  → DepthSpatialConv (grouped) → ELU
  → DepthTemporalConv (grouped) → ELU
  → StratifiedLayerNorm (optional, 'middle2')
  → Flatten → out_dim
```

### AlignEncoder (Stage 2 — EEGNet/ShallowConvNet-style)

```
ChannelAttention (SE-style, optional)
  → TemporalConv (1→40 filters, k=25)
  → BatchNorm → ELU → AvgPool
  → SpatialConv (40→40, collapses channels)
  → BatchNorm → ELU → Dropout
  → AdaptiveAvgPool → Flatten → Linear → 256-d
```

## Training Stages

### Stage 1 — Cross-Subject Contrastive Pre-training

- **From CL-SSTER**: pairs EEG from different subjects performing the same task condition
- **Loss**: InfoNCE (NT-Xent) — same condition across subjects = positive pair
- **Goal**: learn universal cognitive-load features that generalise across individuals

### Stage 2 — EEG–Text Alignment

- Aligns EEG embeddings with frozen text anchors (one per class label, from `nn.Embedding`)
- **Loss**: bidirectional CLIP-style contrastive loss
- **Goal**: learn task-discriminative features by aligning EEG to fixed label anchors
- Initialised from Stage 1 weights (transfer learning)
- Text embedding is **not trained** — only the EEG branch is updated

### Stage 3 — Classification (CLISA-style)

- Freezes both encoders; only trains a shallow MLP classifier
- **DE extraction window**: re-segments each 5 s epoch into 1 s sub-windows (`de_extract_sec`), de-duplicates overlapping segments, then runs the frozen CrossEncoder on each 1 s window to extract intermediate features (TemporalConv output before ELU)
- Computes **Differential Entropy** (DE) along time: `0.5 * log(2πe * var)`
- **normTrain**: z-score normalises the DE feature matrix using training-set mean/var (per-feature dimension), matching CLISA's `normTrain` step
- **LDS smoothing**: applies a scalar Kalman filter (Linear Dynamical System) per-feature along consecutive 1 s windows within each (subject, condition) group, matching CLISA's `smooth_lds.py`
- DE features (n_tf × n_sf = 256-d) feed a 3-layer MLP (256 → 30 → 30 → C)
- **Loss**: plain cross-entropy (no label smoothing)
- **Optimizer**: Adam + StepLR (matches CLISA)

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

### Training

```bash
# Full pipeline (stages 1 → 2 → 3)
python train.py --stage all

# Or run each stage individually:
python train.py --stage 1                                      # creates run dir
python train.py --stage 2 --run_dir checkpoints/20260404_1523  # needs stage1_best.pt
python train.py --stage 3 --run_dir checkpoints/20260404_1523  # needs stage2_best.pt

# Evaluate on held-out test set
python evaluate.py --run_dir checkpoints/20260404_1523
```

### Common options

```bash
# Custom dataset location / epoch length / fewer epochs
python train.py --stage 1 --data_path /my/data --epoch_sec 4.0 --stage1_epochs 30

# Stage 3 always freezes encoders (CLISA-style); just run:
python train.py --stage 3 --run_dir checkpoints/20260404_1523
```

### Ablation: skip Stage 2

```bash
# Skip Stage 2 entirely — go directly from Stage 1 to Stage 3
python train.py --stage 3 --run_dir checkpoints/20260404_1523 --ablation cross_only

# Evaluate the cross_only variant
python evaluate.py --run_dir checkpoints/20260404_1523 --ablation cross_only
```

## Adding a New Dataset

The framework uses a **class-based registry** pattern. To add a new EEG dataset:

### 1. Create a loader file

Create a new Python file in `data/loaders/` (e.g. `data/loaders/my_dataset.py`):

```python
import numpy as np
from data.base_loader import BaseDatasetLoader


class MyDatasetLoader(BaseDatasetLoader):
    name = "my_dataset"                      # unique identifier
    n_classes = 3                            # number of classes
    label_names = {0: "low", 1: "mid", 2: "high"}

    def cache_tag(self, cfg) -> str:
        """Optional: custom cache filename to avoid collisions."""
        return f"my_dataset_{cfg.epoch_sec}s"

    def load_raw(self, cfg) -> dict:
        """Load raw data and return the standardised dict."""
        root = cfg.data_path                 # --data_path from CLI
        # ... your loading logic ...
        return {
            "eeg":           eeg,            # (n_trials, n_channels, n_timepoints) float32
            "labels":        labels,         # (n_trials,) int64
            "subject_ids":   subject_ids,    # (n_trials,) int64
            "positions":     positions,      # (n_trials,) int64 — epoch index per trial
        }
```

### 2. Run

```bash
python train.py --stage all --data_source my_dataset --data_path /path/to/data
```

The loader is auto-discovered — no manual registration code needed. Just drop the file into `data/loaders/` and it will be available.

### Standard data dict format

All loaders must return a dict with these keys:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `eeg` | `(N, C, T)` | float32 | EEG epochs |
| `labels` | `(N,)` | int64 | Class labels (0-indexed) |
| `subject_ids` | `(N,)` | int64 | Subject identifiers |
| `positions` | `(N,)` | int64 | Temporal position index within each (subject, condition) group — used by Stage 1 cross-subject pairing and Stage 3 DE sub-segmentation / LDS ordering |

Additionally, set the `n_classes` and `label_names` class attributes on the loader.

## Built-in Datasets

### EEGMAT (PhysioNet)
- 36 subjects, 19 channels, 500 Hz (resampled to 256 Hz)
- 2 classes: rest vs. mental arithmetic
- [PhysioNet page](https://physionet.org/content/eegmat/1.0.0/)

### Recommended additional datasets

| Dataset | Subjects | Channels | Classes | Link |
|---------|----------|----------|---------|------|
| HHU N-back | 10 | 56 | 4 (0/1/2/3-back) | [IEEE DataPort](https://ieee-dataport.org/documents/hhu-n-back-task-eeg-dataset) |
| STEW | 48 | 14 | continuous (bin to 3) | [IEEE DataPort](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset) |

## Project Structure

```
code/
├── train.py               # Unified training (--stage 1/2/3/all)
├── evaluate.py            # Test-set evaluation
├── cli.py                 # Shared setup (config, data, splits)
├── config.py              # All hyperparameters
├── losses.py              # InfoNCE, CLIP, CE losses
├── utils.py               # Metrics, LDS smoothing, seeding, helpers
├── data/
│   ├── base_loader.py     # BaseDatasetLoader ABC + registry
│   ├── preprocessing.py   # Bandpass, normalise, MVNN
│   ├── dataset.py         # PyTorch datasets (EEG, cross-pair, DE-feature) & loaders
│   └── loaders/           # ← drop new dataset loaders here
│       ├── __init__.py    # Auto-discovery of loader modules
│       └── eegmat.py      # EEGMAT dataset loader
├── models/
│   ├── cross_encoder.py   # CrossEncoder + StratifiedLayerNorm + DifferentialEntropy
│   ├── align_encoder.py   # AlignEncoder + channel attention
│   └── dual_align.py      # DualAlign model (Stage 1/2/3 forward paths)
├── trainers/
│   ├── base.py            # BaseTrainer (template-method pattern)
│   ├── stage1.py          # Cross-subject pre-training
│   ├── stage2.py          # Stimulus alignment
│   └── stage3.py          # Classification fine-tuning
├── checkpoints/           # Timestamped run directories
│   └── 20260404_1523/     #   stage1_best.pt, stage2_best.pt, ...
└── logs/                  # Evaluation results
```

## Key Differences from Original Works

| Aspect | CL-SSTER | NICE-EEG | DualAlign-CogLoad (Ours) |
|--------|----------|----------|--------------------------|
| **Task** | Shared representation learning | Visual object recognition | Cognitive load classification |
| **Stimulus** | Continuous video/speech | Static images | Task labels (rest / arithmetic) |
| **Alignment** | EEG ↔ EEG (cross-subject) | EEG ↔ DNN features | Both: EEG↔EEG + EEG↔frozen text embedding |
| **Labels** | Unsupervised | Zero-shot | Supervised (2 classes) |
| **Encoder** | Spatial→Temporal CNN | Temporal→Spatial CNN | Hybrid with channel attention |

## Citation

If you use this code, please cite the original works:

```bibtex
@article{cl_sster,
  title={Contrastive Learning of Shared Spatiotemporal EEG Representations Across Individuals for Naturalistic Neuroscience},
  author={...},
  year={2024}
}

@inproceedings{nice_eeg,
  title={Decoding Natural Images from EEG for Object Recognition},
  author={Song, Yonghao and others},
  booktitle={ICLR},
  year={2024}
}
```
