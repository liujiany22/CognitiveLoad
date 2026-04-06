import random
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score,
    confusion_matrix, classification_report, cohen_kappa_score,
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
    labels = sorted(set(y_true) | set(y_pred))
    if label_names is None:
        label_names = [str(l) for l in labels]

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    f1_macro = f1_score(y_true, y_pred, average="macro", labels=labels)
    f1_per = f1_score(y_true, y_pred, average=None, labels=labels)
    prec_per = precision_score(y_true, y_pred, average=None, labels=labels,
                               zero_division=0)
    rec_per = recall_score(y_true, y_pred, average=None, labels=labels,
                           zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, target_names=label_names, labels=labels,
        digits=4, zero_division=0,
    )
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "kappa": kappa,
        "f1_macro": f1_macro,
        "f1_per_class": dict(zip(label_names, f1_per.tolist())),
        "precision_per_class": dict(zip(label_names, prec_per.tolist())),
        "recall_per_class": dict(zip(label_names, rec_per.tolist())),
        "confusion_matrix": cm,
        "report": report,
    }


def lds_smooth(sequence: np.ndarray) -> np.ndarray:
    """Linear Dynamical System (Kalman filter) smoothing (CLISA-faithful).

    Smooths a time-series of feature vectors along the temporal axis using
    a scalar Kalman filter applied independently per feature dimension.

    Args:
        sequence: (n_timesteps, n_features) — consecutive feature vectors.
    Returns:
        Smoothed array with the same shape.
    """
    ave = np.mean(sequence, axis=0)
    u0 = ave
    X = sequence.T                            # (n_features, n_timesteps)

    V0 = 0.01
    A = 1
    T_noise = 0.0001
    C = 1
    sigma = 1

    m, n = X.shape
    P = np.zeros((m, n))
    u = np.zeros((m, n))
    V = np.zeros((m, n))
    K = np.zeros((m, n))

    K[:, 0] = (V0 * C / (C * V0 * C + sigma)) * np.ones(m)
    u[:, 0] = u0 + K[:, 0] * (X[:, 0] - C * u0)
    V[:, 0] = (np.ones(m) - K[:, 0] * C) * V0

    for i in range(1, n):
        P[:, i - 1] = A * V[:, i - 1] * A + T_noise
        K[:, i] = P[:, i - 1] * C / (C * P[:, i - 1] * C + sigma)
        u[:, i] = A * u[:, i - 1] + K[:, i] * (X[:, i] - C * A * u[:, i - 1])
        V[:, i] = (np.ones(m) - K[:, i] * C) * P[:, i - 1]

    return u.T                                 # (n_timesteps, n_features)


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
