# DualAlign-CogLoad

**Dual-Alignment Network for EEG-Based Cognitive Load Monitoring**

A three-stage deep learning framework that combines **cross-subject contrastive alignment** (inspired by [CL-SSTER](https://github.com/SherazKhan/cl-sster)) and **EEG–text embedding alignment** (inspired by [NICE-EEG](https://github.com/eeyhsong/NICE-EEG)) to classify cognitive load states (rest vs. mental arithmetic) from EEG signals.

---

## Motivation

| Challenge | Solution |
|-----------|----------|
| EEG varies across individuals | **Stage 1** learns a shared cross-subject representation via contrastive learning (CL-SSTER style) |
| Need to capture task-specific brain responses | **Stage 2** aligns EEG embeddings with frozen text anchors via CLIP-style contrastive loss |
| Must predict cognitive load robustly | **Stage 3** fuses both representations and fine-tunes a classifier |

## Architecture

```
                     ┌──────────────────────┐
                     │  Raw EEG (B,1,C,T)   │
                     └──────────┬───────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
   ┌────────────────────┐          ┌────────────────────┐
   │  Cross-Subject     │          │  Stimulus-Aligned   │
   │  Encoder (Stage 1) │          │  Encoder (Stage 2)  │
   │  + InfoNCE loss    │          │  + CLIP loss         │
   └────────┬───────────┘          └────────┬───────────┘
            │ embed_dim                     │ embed_dim
            └───────────┬───────────────────┘
                        ▼
              ┌──────────────────┐
              │  Fusion MLP      │
              │  concat → dense  │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  Classifier      │
              │  → 2 classes     │
              └──────────────────┘
```

### EEG Encoder (shared architecture, separate weights)

```
ChannelAttention (SE-style)
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

### Stage 3 — Classification Fine-tuning

- Concatenates embeddings from both branches
- MLP fusion → 2-class softmax
- **Loss**: cross-entropy with label smoothing
- No text features at inference — pure EEG classification

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

### Training (step by step)

Each stage is a standalone script.  Run them in order:

```bash
# Step 1 — Cross-subject contrastive pre-training
#   Creates a timestamped run directory (e.g. checkpoints/20260404_1523/)
python train_stage1.py

# Step 2 — Stimulus-task alignment (pass the run dir from step 1)
python train_stage2.py --run_dir checkpoints/20260404_1523

# Step 3 — Classification fine-tuning
python train_stage3.py --run_dir checkpoints/20260404_1523

# Evaluate on held-out test set
python evaluate.py --run_dir checkpoints/20260404_1523
```

### Common options

```bash
# Custom dataset location / epoch length / fewer epochs
python train_stage1.py --data_path /my/data --epoch_sec 4.0 --stage1_epochs 30

# Freeze encoders during Stage 3
python train_stage3.py --run_dir checkpoints/20260404_1523 --freeze_encoders
```

### Ablation experiments

```bash
# Stage-1 features only (automatically skips Stage-2 weights)
python train_stage3.py --run_dir checkpoints/20260404_1523 --ablation cross_only

# Stage-2 features only
python train_stage3.py --run_dir checkpoints/20260404_1523 --ablation align_only

# Evaluate an ablation variant
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
        }
```

### 2. Run

```bash
python main.py --data_source my_dataset --data_path /path/to/data
```

The loader is auto-discovered — no manual registration code needed. Just drop the file into `data/loaders/` and it will be available.

### Standard data dict format

All loaders must return a dict with these keys:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `eeg` | `(N, C, T)` | float32 | EEG epochs |
| `labels` | `(N,)` | int64 | Class labels (0-indexed) |
| `subject_ids` | `(N,)` | int64 | Subject identifiers |

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
├── train_stage1.py        # Entry point — Stage 1 pre-training
├── train_stage2.py        # Entry point — Stage 2 alignment
├── train_stage3.py        # Entry point — Stage 3 fine-tuning
├── evaluate.py            # Entry point — test-set evaluation
├── pipeline.py            # Shared setup (config, data, splits)
├── config.py              # All hyperparameters
├── losses.py              # InfoNCE, CLIP, CE losses
├── utils.py               # Metrics, seeding, helpers
├── data/
│   ├── base_loader.py     # BaseDatasetLoader ABC + registry
│   ├── preprocessing.py   # Bandpass, normalise, MVNN
│   ├── text_embeddings.py # (placeholder — text logic lives in loaders)
│   ├── dataset.py         # PyTorch datasets & loaders
│   └── loaders/           # ← drop new dataset loaders here
│       ├── __init__.py    # Auto-discovery of loader modules
│       └── eegmat.py      # EEGMAT dataset loader
├── models/
│   ├── encoder.py         # EEG encoder + channel attention
│   └── dual_align.py      # Full DualAlign model
├── trainers/
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
