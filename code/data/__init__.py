from .preprocessing import preprocess_eeg
from .dataset import (
    CogLoadDataset,
    CrossSubjectPairDataset,
    build_dataloaders,
)
from .base_loader import BaseDatasetLoader
from .text_embeddings import build_text_embeddings

from . import loaders  # noqa: F401  — triggers auto-registration
