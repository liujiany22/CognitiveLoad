"""
Synthetic cognitive-load EEG generator.

Produces multi-subject, multi-condition data whose spectral profiles shift
with cognitive load level (alpha suppression + frontal-theta enhancement),
mimicking an n-back or arithmetic-based paradigm.
"""

import numpy as np
from typing import Dict, Tuple


REGION_MAP_32 = {
    "frontal": list(range(0, 8)),
    "central": list(range(8, 14)),
    "parietal": list(range(14, 20)),
    "temporal": list(range(20, 26)),
    "occipital": list(range(26, 32)),
}

BAND_REGION_WEIGHTS = {
    "theta":   {"frontal": 1.0, "central": 0.7, "parietal": 0.5, "temporal": 0.4, "occipital": 0.3},
    "alpha":   {"frontal": 0.4, "central": 0.6, "parietal": 0.9, "temporal": 0.5, "occipital": 1.0},
    "beta":    {"frontal": 0.7, "central": 0.8, "parietal": 0.5, "temporal": 0.6, "occipital": 0.4},
}


def _pink_noise(n: int, rng: np.random.RandomState) -> np.ndarray:
    white = rng.randn(n)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0
    spectrum = np.fft.rfft(white) / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n=n)
    return pink / (pink.std() + 1e-10)


def _channel_region(ch: int, n_channels: int) -> str:
    if n_channels == 32:
        for region, indices in REGION_MAP_32.items():
            if ch in indices:
                return region
    frac = ch / n_channels
    if frac < 0.25:
        return "frontal"
    elif frac < 0.44:
        return "central"
    elif frac < 0.63:
        return "parietal"
    elif frac < 0.81:
        return "temporal"
    return "occipital"


def _generate_epoch(
    n_channels: int,
    n_timepoints: int,
    fs: int,
    alpha_peak: float,
    alpha_power: float,
    theta_power: float,
    beta_power: float,
    noise_level: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    t = np.arange(n_timepoints) / fs
    eeg = np.zeros((n_channels, n_timepoints))

    for ch in range(n_channels):
        region = _channel_region(ch, n_channels)
        signal = np.zeros(n_timepoints)

        # theta (4-7 Hz)
        w = BAND_REGION_WEIGHTS["theta"][region]
        freq = rng.uniform(4.0, 7.0)
        phase = rng.uniform(0, 2 * np.pi)
        signal += w * np.sqrt(theta_power) * np.sin(2 * np.pi * freq * t + phase)

        # alpha (8-13 Hz)
        w = BAND_REGION_WEIGHTS["alpha"][region]
        freq = alpha_peak + rng.normal(0, 0.3)
        phase = rng.uniform(0, 2 * np.pi)
        signal += w * np.sqrt(alpha_power) * np.sin(2 * np.pi * freq * t + phase)

        # beta (13-30 Hz)
        w = BAND_REGION_WEIGHTS["beta"][region]
        freq = rng.uniform(13, 25)
        phase = rng.uniform(0, 2 * np.pi)
        signal += w * np.sqrt(beta_power) * np.sin(2 * np.pi * freq * t + phase)

        signal += noise_level * _pink_noise(n_timepoints, rng)
        eeg[ch] = signal

    return eeg


def _generate_task_features(
    level: int,
    n_levels: int,
    dim: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    feat = np.zeros(dim)
    difficulty = level / max(n_levels - 1, 1)
    feat[0] = difficulty
    feat[1] = difficulty ** 2
    feat[2] = np.sin(np.pi * difficulty)

    cond_rng = np.random.RandomState(level * 9999)
    feat[3:] = cond_rng.randn(dim - 3) * 0.5
    feat += rng.randn(dim) * 0.05
    return feat


def generate_cognitive_load_data(
    n_subjects: int = 20,
    n_channels: int = 32,
    n_timepoints: int = 512,
    sampling_rate: int = 256,
    n_trials_per_level: int = 40,
    n_levels: int = 3,
    task_feature_dim: int = 16,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Returns
    -------
    dict with keys:
        eeg            : (total_trials, n_channels, n_timepoints)
        labels         : (total_trials,)          0 / 1 / 2
        subject_ids    : (total_trials,)
        condition_ids  : (total_trials,)          same as labels for n-back
        task_features  : (total_trials, task_feature_dim)
    """
    rng = np.random.RandomState(seed)

    all_eeg, all_labels, all_subs, all_conds, all_feats = [], [], [], [], []

    for sub in range(n_subjects):
        alpha_peak = rng.normal(10.0, 0.5)
        alpha_base = rng.uniform(8.0, 15.0)
        theta_base = rng.uniform(3.0, 6.0)
        beta_base = rng.uniform(1.5, 3.5)
        theta_react = rng.uniform(0.5, 2.0)
        alpha_suppress = rng.uniform(0.3, 0.7)
        noise_level = rng.uniform(1.0, 3.0)

        for level in range(n_levels):
            difficulty = level / max(n_levels - 1, 1)
            alpha_pow = alpha_base * (1 - alpha_suppress * difficulty)
            theta_pow = theta_base * (1 + theta_react * difficulty)
            beta_pow = beta_base * (1 + 0.3 * difficulty)

            for _ in range(n_trials_per_level):
                epoch = _generate_epoch(
                    n_channels, n_timepoints, sampling_rate,
                    alpha_peak, alpha_pow, theta_pow, beta_pow,
                    noise_level, rng,
                )
                tf = _generate_task_features(level, n_levels, task_feature_dim, rng)

                all_eeg.append(epoch)
                all_labels.append(level)
                all_subs.append(sub)
                all_conds.append(level)
                all_feats.append(tf)

    return {
        "eeg": np.stack(all_eeg).astype(np.float32),
        "labels": np.array(all_labels, dtype=np.int64),
        "subject_ids": np.array(all_subs, dtype=np.int64),
        "condition_ids": np.array(all_conds, dtype=np.int64),
        "task_features": np.stack(all_feats).astype(np.float32),
    }
