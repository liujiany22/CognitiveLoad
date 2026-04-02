"""
DualAlign-CogLoad — main training pipeline.

Three-stage training:
  1. Cross-subject contrastive pre-training  (CL-SSTER inspired)
  2. Stimulus-task alignment pre-training    (NICE inspired)
  3. Dual-branch fusion classification fine-tuning

Usage:
    python main.py                          # synthetic data, full pipeline
    python main.py --stage 3                # run only stage 3
    python main.py --n_subjects 10 --stage1_epochs 50
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from config import Config
from utils import set_seed, compute_metrics
from data import (
    generate_cognitive_load_data, preprocess_eeg,
    build_dataloaders, load_eegmat,
)
from models import DualAlignModel
from trainers import Stage1Trainer, Stage2Trainer, Stage3Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="DualAlign-CogLoad")
    parser.add_argument("--stage", type=int, default=0,
                        help="Run only this stage (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_subjects", type=int, default=20)
    parser.add_argument("--n_channels", type=int, default=32)
    parser.add_argument("--n_timepoints", type=int, default=512)
    parser.add_argument("--n_trials_per_level", type=int, default=40)
    parser.add_argument("--stage1_epochs", type=int, default=100)
    parser.add_argument("--stage2_epochs", type=int, default=80)
    parser.add_argument("--stage3_epochs", type=int, default=60)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--freeze_encoders", action="store_true",
                        help="Freeze encoder weights in Stage 3")
    parser.add_argument("--data_source", type=str, default="simulate",
                        choices=["simulate", "eegmat"],
                        help="Data source: simulate | eegmat")
    parser.add_argument("--data_path", type=str, default="datasets/eegmat",
                        help="Path to real dataset root")
    parser.add_argument("--epoch_sec", type=float, default=2.0,
                        help="Epoch length in seconds (for real data)")
    return parser.parse_args()


def build_config(args) -> Config:
    cfg = Config()
    for k, v in vars(args).items():
        if hasattr(cfg, k) and v not in (None, ""):
            setattr(cfg, k, v)
    if args.device:
        cfg.device = args.device
    if args.data_source:
        cfg.data_source = args.data_source
    if args.data_path:
        cfg.real_data_path = args.data_path
    if args.epoch_sec:
        cfg.epoch_sec = args.epoch_sec
    return cfg


def prepare_data_simulate(cfg):
    tag = (f"s{cfg.n_subjects}_c{cfg.n_channels}_t{cfg.n_timepoints}"
           f"_tr{cfg.n_trials_per_level}_cl{cfg.n_classes}")
    cache_path = os.path.join(cfg.data_dir, f"synthetic_{tag}.npz")

    if os.path.exists(cache_path):
        print("Loading cached synthetic data …")
        loaded = np.load(cache_path, allow_pickle=False)
        data = {k: loaded[k] for k in loaded.files}
    else:
        print("Generating synthetic cognitive-load EEG data …")
        data = generate_cognitive_load_data(
            n_subjects=cfg.n_subjects,
            n_channels=cfg.n_channels,
            n_timepoints=cfg.n_timepoints,
            sampling_rate=cfg.sampling_rate,
            n_trials_per_level=cfg.n_trials_per_level,
            n_levels=cfg.n_classes,
            task_feature_dim=cfg.task_feature_dim,
            seed=cfg.seed,
        )
        np.savez_compressed(cache_path, **data)
        print(f"  cached to {cache_path}")

    print("Preprocessing EEG …")
    data["eeg"] = preprocess_eeg(
        data["eeg"], fs=cfg.sampling_rate, bandpass=True, normalize="zscore",
    )
    return data


def prepare_data_eegmat(cfg):
    cache_path = os.path.join(
        cfg.data_dir, f"eegmat_ep{cfg.epoch_sec}s_{cfg.sampling_rate}hz_textemb.npz",
    )

    if os.path.exists(cache_path):
        print("Loading cached EEGMAT data …")
        loaded = np.load(cache_path, allow_pickle=False)
        data = {k: loaded[k] for k in loaded.files}
    else:
        print(f"Loading EEGMAT from {cfg.real_data_path} …")
        data = load_eegmat(
            root_dir=cfg.real_data_path,
            epoch_sec=cfg.epoch_sec,
            target_sfreq=float(cfg.sampling_rate),
            cache_dir=cfg.data_dir,
        )
        np.savez_compressed(cache_path, **data)
        print(f"  cached to {cache_path}")

    print("Preprocessing EEG …")
    data["eeg"] = preprocess_eeg(
        data["eeg"], fs=cfg.sampling_rate, bandpass=True, normalize="zscore",
    )
    return data


def prepare_data(cfg):
    if cfg.data_source == "eegmat":
        data = prepare_data_eegmat(cfg)
    else:
        data = prepare_data_simulate(cfg)

    print(f"  EEG shape : {data['eeg'].shape}")
    print(f"  Labels    : {np.bincount(data['labels'])}")

    unique_subs = np.unique(data["subject_ids"])
    cfg.n_subjects = len(unique_subs)
    cfg.n_channels = data["eeg"].shape[1]
    cfg.n_timepoints = data["eeg"].shape[2]
    cfg.n_classes = len(np.unique(data["labels"]))
    cfg.task_feature_dim = data["task_features"].shape[1]
    print(f"  Subjects  : {cfg.n_subjects}")
    print(f"  Channels  : {cfg.n_channels}")
    print(f"  Timepoints: {cfg.n_timepoints}")
    print(f"  Classes   : {cfg.n_classes}")
    print(f"  Task feat : {cfg.task_feature_dim}-d text embeddings")
    return data


def split_subjects(cfg, data):
    unique_subs = sorted(np.unique(data["subject_ids"]).tolist())
    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(unique_subs)
    n_total = len(unique_subs)
    n_test = max(1, int(n_total * cfg.test_subject_ratio))
    n_val = max(1, int(n_total * cfg.val_subject_ratio))
    test_subs = sorted(unique_subs[:n_test])
    val_subs = sorted(unique_subs[n_test:n_test + n_val])
    train_subs = sorted(unique_subs[n_test + n_val:])
    print(f"  Train subjects ({len(train_subs)}): {train_subs}")
    print(f"  Val   subjects ({len(val_subs)}): {val_subs}")
    print(f"  Test  subjects ({len(test_subs)}): {test_subs}")
    return train_subs, val_subs, test_subs


def run_stage1(model, loaders, cfg):
    print("\n" + "=" * 60)
    print("STAGE 1 — Cross-Subject Contrastive Pre-training")
    print("=" * 60)
    trainer = Stage1Trainer(model, cfg)
    trainer.train(loaders["pair"])


def run_stage2(model, loaders, cfg):
    print("\n" + "=" * 60)
    print("STAGE 2 — Stimulus-Task Alignment Pre-training")
    print("=" * 60)
    model.init_align_from_cross()
    trainer = Stage2Trainer(model, cfg)
    trainer.train(loaders["train"], val_loader=loaders["val"])


def run_stage3(model, loaders, cfg, freeze_encoders=False):
    print("\n" + "=" * 60)
    print("STAGE 3 — Dual-Branch Classification Fine-tuning")
    print("=" * 60)
    trainer = Stage3Trainer(model, cfg, freeze_encoders=freeze_encoders)
    trainer.train(loaders["finetune"], loaders["val"])
    return trainer


def final_evaluation(trainer, loaders, cfg):
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    metrics = trainer.evaluate(loaders["test"])
    print(metrics["report"])
    print(f"Overall accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro F1         : {metrics['f1_macro']:.4f}")
    print(f"Per-class F1     : {metrics['f1_per_class']}")
    print(f"Confusion matrix :\n{metrics['confusion_matrix']}")

    results_path = os.path.join(cfg.log_dir, "results.json")
    serialisable = {
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "f1_per_class": metrics["f1_per_class"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    with open(results_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved to {results_path}")


def main():
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)

    print("DualAlign-CogLoad")
    print(f"  device : {cfg.device}")
    print(f"  seed   : {cfg.seed}")

    data = prepare_data(cfg)
    train_subs, val_subs, test_subs = split_subjects(cfg, data)
    loaders = build_dataloaders(data, train_subs, val_subs, test_subs, cfg)

    model = DualAlignModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params : {n_params:,}")

    stage = args.stage

    if stage in (0, 1):
        run_stage1(model, loaders, cfg)
    elif stage > 1:
        ckpt_path = f"{cfg.save_dir}/stage1_best.pt"
        if os.path.exists(ckpt_path):
            model.load_state_dict(
                torch.load(ckpt_path, map_location=cfg.device, weights_only=True),
            )
            print(f"  Loaded Stage-1 checkpoint from {ckpt_path}")

    if stage in (0, 2):
        run_stage2(model, loaders, cfg)
    elif stage > 2:
        ckpt_path = f"{cfg.save_dir}/stage2_best.pt"
        if os.path.exists(ckpt_path):
            model.load_state_dict(
                torch.load(ckpt_path, map_location=cfg.device, weights_only=True),
            )
            print(f"  Loaded Stage-2 checkpoint from {ckpt_path}")

    if stage in (0, 3):
        trainer = run_stage3(
            model, loaders, cfg, freeze_encoders=args.freeze_encoders,
        )
        final_evaluation(trainer, loaders, cfg)


if __name__ == "__main__":
    main()
