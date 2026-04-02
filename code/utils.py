import random
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, y_pred, label_names=None):
    if label_names is None:
        label_names = ["low", "mid", "high"]
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_per = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=label_names, digits=4,
        zero_division=0,
    )
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_per_class": dict(zip(label_names, f1_per.tolist())),
        "confusion_matrix": cm,
        "report": report,
    }


class EarlyStopping:
    def __init__(self, patience: int = 15, mode: str = "min", delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best = None
        self.counter = 0
        self.stopped = False

    def step(self, value) -> bool:
        if self.best is None:
            self.best = value
            return False
        improved = (
            value < self.best - self.delta
            if self.mode == "min"
            else value > self.best + self.delta
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
        return self.stopped


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)
