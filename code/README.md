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

### Run with synthetic data

```bash
# Full pipeline (all 3 stages)
python main.py

# Quick test with fewer subjects and epochs
python main.py --n_subjects 8 --stage1_epochs 10 --stage2_epochs 10 --stage3_epochs 10

# Run only Stage 3 (loads pre-trained checkpoints)
python main.py --stage 3

# Freeze encoders during fine-tuning
python main.py --freeze_encoders
```

## Using Real Datasets

The framework is designed to work with any EEG cognitive load dataset. Two recommended public datasets:

### HHU N-back EEG Dataset
- 10 subjects, 56 channels, 1000 Hz
- 4 difficulty levels (0/1/2/3-back) — use 0/1/2-back for 3 classes
- [IEEE DataPort](https://ieee-dataport.org/documents/hhu-n-back-task-eeg-dataset)

### STEW Dataset
- 48 subjects, 14 channels, 128 Hz
- Workload ratings 1–9 (bin into low/mid/high)
- [IEEE DataPort](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset)

To use a real dataset, implement a loader in `data/` that returns the same dictionary format as `generate_cognitive_load_data()`:

```python
{
    "eeg":           np.ndarray  # (n_trials, n_channels, n_timepoints)
    "labels":        np.ndarray  # (n_trials,)  — 0/1/2
    "subject_ids":   np.ndarray  # (n_trials,)
    "condition_ids": np.ndarray  # (n_trials,)
    "task_features": np.ndarray  # (n_trials, task_feature_dim)
}
```

## Project Structure

```
code/
├── config.py              # All hyperparameters
├── main.py                # Training pipeline entry point
├── losses.py              # InfoNCE, CLIP, CE losses
├── utils.py               # Metrics, seeding, helpers
├── data/
│   ├── simulate.py        # Synthetic EEG generator
│   ├── preprocessing.py   # Bandpass, normalise, MVNN
│   └── dataset.py         # PyTorch datasets & loaders
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
