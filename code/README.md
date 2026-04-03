# DualAlign-CogLoad

**Dual-Alignment Network for EEG-Based Cognitive Load Monitoring**

A three-stage deep learning framework that combines **cross-subject contrastive alignment** (inspired by [CL-SSTER](https://github.com/SherazKhan/cl-sster)) and **EEG–stimulus feature alignment** (inspired by [NICE-EEG](https://github.com/eeyhsong/NICE-EEG)) to predict cognitive load levels (low / medium / high) from EEG signals.

---

## Motivation

| Challenge | Solution |
|-----------|----------|
| EEG varies across individuals | **Stage 1** learns a shared cross-subject representation via contrastive learning (CL-SSTER style) |
| Need to capture task-specific brain responses | **Stage 2** aligns individual EEG with task-condition embeddings via CLIP-style contrastive loss (NICE style) |
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
              │  → 3 classes     │
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

### Stage 2 — Stimulus-Task Alignment

- **From NICE**: aligns EEG embeddings with task-condition feature vectors
- **Loss**: bidirectional CLIP-style contrastive loss
- **Goal**: learn task-discriminative features that capture how each person responds to different load levels
- Initialised from Stage 1 weights (transfer learning)

### Stage 3 — Classification Fine-tuning

- Concatenates embeddings from both branches
- MLP fusion → 3-class softmax
- **Loss**: cross-entropy with label smoothing

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

### Run with EEGMAT dataset

```bash
# Full pipeline (all 3 stages)
python main.py --data_source eegmat --data_path datasets/eegmat

# Run only Stage 3 (loads pre-trained checkpoints)
python main.py --data_source eegmat --data_path datasets/eegmat --stage 3

# Freeze encoders during fine-tuning
python main.py --data_source eegmat --data_path datasets/eegmat --freeze_encoders

# Custom epoch length and fewer training epochs
python main.py --data_source eegmat --data_path datasets/eegmat \
    --epoch_sec 4.0 --stage1_epochs 10 --stage2_epochs 10 --stage3_epochs 10
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
            "condition_ids": condition_ids,  # (n_trials,) int64
            "task_features": task_features,  # (n_trials, feat_dim) float32
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
| `condition_ids` | `(N,)` | int64 | Fine-grained condition IDs for contrastive pairing |
| `task_features` | `(N, D)` | float32 | Task-condition feature vectors (e.g. text embeddings) |

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
├── config.py              # All hyperparameters
├── main.py                # Training pipeline entry point
├── losses.py              # InfoNCE, CLIP, CE losses
├── utils.py               # Metrics, seeding, helpers
├── data/
│   ├── base_loader.py     # BaseDatasetLoader ABC + registry
│   ├── preprocessing.py   # Bandpass, normalise, MVNN
│   ├── text_embeddings.py # Text-based task feature encoder
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
├── checkpoints/           # Saved model weights
└── logs/                  # Training logs & results
```

## Key Differences from Original Works

| Aspect | CL-SSTER | NICE-EEG | DualAlign-CogLoad (Ours) |
|--------|----------|----------|--------------------------|
| **Task** | Shared representation learning | Visual object recognition | Cognitive load classification |
| **Stimulus** | Continuous video/speech | Static images | Task conditions (n-back levels) |
| **Alignment** | EEG ↔ EEG (cross-subject) | EEG ↔ DNN features | Both: EEG↔EEG + EEG↔task |
| **Labels** | Unsupervised | Zero-shot | Supervised (3 classes) |
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
