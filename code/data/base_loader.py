"""
Base class and registry for dataset loaders.

To add a new dataset
--------------------
1. Create a new file in ``data/loaders/``  (e.g. ``my_dataset.py``).
2. Subclass :class:`BaseDatasetLoader`.
3. Set the ``name`` class attribute to a unique identifier string.
4. Set ``n_classes`` and ``label_names`` for classification metadata.
5. Implement :meth:`load_raw` (and optionally override :meth:`cache_tag`).

The loader is registered automatically upon import — no manual
registration is needed.

Minimal example::

    from data.base_loader import BaseDatasetLoader

    class MyDatasetLoader(BaseDatasetLoader):
        name = "my_dataset"
        n_classes = 2
        label_names = {0: "class_a", 1: "class_b"}

        def load_raw(self, cfg) -> dict:
            eeg = ...          # (n_trials, n_channels, n_timepoints)
            labels = ...       # (n_trials,)
            subject_ids = ...  # (n_trials,)
            return {
                "eeg": eeg, "labels": labels,
                "subject_ids": subject_ids,
            }
"""

import os
from abc import ABC, abstractmethod

import numpy as np

from .preprocessing import preprocess_eeg


class BaseDatasetLoader(ABC):
    """Abstract base class for EEG dataset loaders.

    Subclasses must set :attr:`name`, :attr:`n_classes`,
    :attr:`label_names` and implement :meth:`load_raw`.
    Caching and preprocessing are handled by :meth:`load`.
    """

    name: str = ""
    n_classes: int = 2
    label_names: dict = {}

    _registry: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", ""):
            BaseDatasetLoader._registry[cls.name] = cls

    # ── registry helpers ──

    @classmethod
    def get_loader(cls, name: str) -> "BaseDatasetLoader":
        """Instantiate the loader registered under *name*."""
        if name not in cls._registry:
            avail = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown dataset '{name}'. Available: {avail}"
            )
        return cls._registry[name]()

    @classmethod
    def available_datasets(cls) -> list:
        """Return sorted list of registered dataset names."""
        return sorted(cls._registry.keys())

    # ── interface ──

    @abstractmethod
    def load_raw(self, cfg) -> dict:
        """Load the dataset and return a standardised dict.

        Required keys
        -------------
        eeg           : np.ndarray  (n_trials, n_channels, n_timepoints) float32
        labels        : np.ndarray  (n_trials,) int64
        subject_ids   : np.ndarray  (n_trials,) int64
        """

    def cache_tag(self, cfg) -> str:
        """Return a unique string used as the cache filename stem.

        Override to include dataset-specific parameters (e.g. epoch
        length, sampling rate) so that different configurations produce
        separate cache files.
        """
        return self.name

    # ── template method ──

    def load(self, cfg) -> dict:
        """Load the dataset with transparent caching and preprocessing."""
        tag = self.cache_tag(cfg)
        cache_path = os.path.join(cfg.data_dir, f"{tag}.npz")

        if os.path.exists(cache_path):
            print(f"Loading cached {self.name} data …")
            loaded = np.load(cache_path, allow_pickle=False)
            data = {k: loaded[k] for k in loaded.files}
        else:
            data = self.load_raw(cfg)
            os.makedirs(cfg.data_dir, exist_ok=True)
            np.savez_compressed(cache_path, **data)
            print(f"  cached → {cache_path}")

        print("Preprocessing EEG …")
        data["eeg"] = preprocess_eeg(
            data["eeg"], fs=cfg.sampling_rate, bandpass=True, normalize="zscore",
        )
        return data
