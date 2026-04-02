"""
EEG preprocessing utilities: bandpass filtering, normalisation, MVNN whitening.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    data: np.ndarray,
    low: float = 1.0,
    high: float = 45.0,
    fs: int = 256,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    data : (..., n_timepoints)
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


def zscore_normalize(data: np.ndarray, axis: int = -1) -> np.ndarray:
    mean = data.mean(axis=axis, keepdims=True)
    std = data.std(axis=axis, keepdims=True) + 1e-8
    return ((data - mean) / std).astype(np.float32)


def robust_scale(data: np.ndarray, axis: int = -1) -> np.ndarray:
    median = np.median(data, axis=axis, keepdims=True)
    q75 = np.percentile(data, 75, axis=axis, keepdims=True)
    q25 = np.percentile(data, 25, axis=axis, keepdims=True)
    iqr = q75 - q25 + 1e-8
    return ((data - median) / iqr).astype(np.float32)


def mvnn_whitening(data: np.ndarray) -> np.ndarray:
    """
    Multivariate Noise Normalisation (from NICE-EEG).

    Parameters
    ----------
    data : (n_trials, n_channels, n_timepoints)

    Returns
    -------
    whitened : same shape
    """
    n_trials, n_ch, n_t = data.shape
    data_2d = data.transpose(0, 2, 1).reshape(-1, n_ch)  # (n_trials*n_t, n_ch)
    cov = np.cov(data_2d, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    whitened = np.einsum("ijk,lk->ijl", data, W)
    return whitened.astype(np.float32)


def preprocess_eeg(
    data: np.ndarray,
    fs: int = 256,
    bandpass: bool = True,
    normalize: str = "zscore",
    whiten: bool = False,
) -> np.ndarray:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    data : (n_trials, n_channels, n_timepoints)
    """
    if bandpass:
        data = bandpass_filter(data, low=1.0, high=45.0, fs=fs)

    if whiten:
        data = mvnn_whitening(data)

    if normalize == "zscore":
        data = zscore_normalize(data, axis=-1)
    elif normalize == "robust":
        data = robust_scale(data, axis=-1)

    return data
