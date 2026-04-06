"""Shared utilities for the DualAlign-CogLoad training pipeline.

Every stage script (train_stage1/2/3, evaluate) imports from here
to avoid duplicating data-loading and config-building logic.
"""

import os
from datetime import datetime

import numpy as np

from config import Config
from utils import set_seed
from data import BaseDatasetLoader, build_dataloaders


def add_common_args(parser):
    """Register CLI arguments shared across all scripts.

    Defaults are intentionally ``None`` so that :func:`build_config`
    only overrides :class:`Config` values that the user explicitly set.
    """
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data_source", type=str, default=None,
                        help="Dataset loader "
                             f"(available: {BaseDatasetLoader.available_datasets()})")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Root directory of the dataset")
    parser.add_argument("--epoch_sec", type=float, default=None,
                        help="Epoch length in seconds")
    parser.add_argument("--ckpt_every", type=int, default=None,
                        help="Save checkpoint every N epochs (0 = off)")


def build_config(args) -> Config:
    """Create a Config and overlay any matching CLI arguments.

    Only values that are not ``None`` are applied, so CLI defaults
    should be ``None`` to let :class:`Config` own the real defaults.
    """
    cfg = Config()
    for k, v in vars(args).items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    return cfg


def setup(args):
    """Build config, set seed, load data, build dataloaders.

    Returns
    -------
    cfg : Config
    loaders : dict[str, DataLoader]
    """
    cfg = build_config(args)
    set_seed(cfg.seed)

    print("DualAlign-CogLoad")
    print(f"  device : {cfg.device}")
    print(f"  seed   : {cfg.seed}")

    loader = BaseDatasetLoader.get_loader(cfg.data_source)
    data = loader.load(cfg)

    print(f"  EEG      : {data['eeg'].shape}")
    print(f"  Labels   : {np.bincount(data['labels'])}")

    unique_subs = np.unique(data["subject_ids"])
    cfg.n_subjects = len(unique_subs)
    cfg.n_channels = data["eeg"].shape[1]
    cfg.n_timepoints = data["eeg"].shape[2]
    cfg.n_classes = loader.n_classes
    cfg.label_names = loader.label_names
    print(f"  Subjects : {cfg.n_subjects}  |  Channels : {cfg.n_channels}  |  "
          f"T : {cfg.n_timepoints}  |  Classes : {cfg.n_classes}  {cfg.label_names}")

    all_subs = sorted(unique_subs.tolist())
    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(all_subs)
    n_test = max(1, int(len(all_subs) * cfg.test_subject_ratio))
    n_val = max(1, int(len(all_subs) * cfg.val_subject_ratio))
    test_subs = sorted(all_subs[:n_test])
    val_subs = sorted(all_subs[n_test:n_test + n_val])
    train_subs = sorted(all_subs[n_test + n_val:])
    print(f"  Split    : train {train_subs}  |  val {val_subs}  |  test {test_subs}")

    # Compute inverse-frequency class weights from training split
    train_mask = np.isin(data["subject_ids"], train_subs)
    train_labels = data["labels"][train_mask]
    counts = np.bincount(train_labels, minlength=cfg.n_classes).astype(np.float64)
    cfg.class_weights = (counts.sum() / (cfg.n_classes * counts)).tolist()
    print(f"  Weights  : {[f'{w:.3f}' for w in cfg.class_weights]}")

    loaders = build_dataloaders(data, train_subs, val_subs, test_subs, cfg)
    return cfg, loaders


def create_run_dir(base_dir: str) -> str:
    """Create a timestamped sub-directory under *base_dir*."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(base_dir, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
