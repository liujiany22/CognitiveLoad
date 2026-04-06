"""
Loader for the PhysioNet EEGMAT dataset (EEG During Mental Arithmetic Tasks).

Reference:
    Zyma et al. (2019). "Electroencephalograms during Mental Arithmetic
    Task Performance." Data, 4(1):14.

Expected layout (flat, all files in one directory):
    <data_path>/
        Subject00_1.edf   (rest)
        Subject00_2.edf   (arithmetic)
        ...
        Subject35_1.edf
        Subject35_2.edf
        subject-info.csv

Binary classification: 休息 (rest) vs 心算 (mental arithmetic).
"""

import os
import re
import csv
import glob

import numpy as np

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError:
    mne = None

from data.base_loader import BaseDatasetLoader

LABEL_REST = 0
LABEL_ARITH = 1

LABEL_NAMES = {0: "休息", 1: "心算"}
N_CLASSES = 2


def _read_edf(path: str) -> tuple:
    """Return (data [n_channels, n_samples], ch_names, sfreq)."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.pick(picks="eeg", exclude="bads")
    return raw.get_data(), raw.ch_names, raw.info["sfreq"]


def _segment_epochs(data: np.ndarray, sfreq: float,
                    win_sec: float, step_sec: float) -> np.ndarray:
    """Sliding-window segmentation.  Returns (n_epochs, n_ch, win_samples)."""
    n_ch, n_total = data.shape
    win = int(win_sec * sfreq)
    step = int(step_sec * sfreq)
    starts = np.arange(0, n_total - win + 1, step)
    return np.stack([data[:, s:s + win] for s in starts], axis=0)


def _read_subject_info(root_dir: str) -> dict:
    """Parse subject-info.csv → {subject_id: {quality, n_sub}}."""
    info = {}
    csv_path = os.path.join(root_dir, "subject-info.csv")
    if not os.path.isfile(csv_path):
        return info
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["Subject"].replace("Subject", ""))
            info[sid] = {
                "quality": int(row["Count quality"]),
                "n_sub": float(row["Number of subtractions"]),
            }
    return info


class EEGMATLoader(BaseDatasetLoader):
    """PhysioNet EEGMAT dataset loader — 2-class (rest / mental arithmetic)."""

    name = "eegmat"
    n_classes = N_CLASSES
    label_names = LABEL_NAMES

    def cache_tag(self, cfg) -> str:
        return (f"eegmat_ep{cfg.epoch_sec}s"
                f"_step{cfg.epoch_step_sec}s"
                f"_{cfg.sampling_rate}hz")

    def load_raw(self, cfg) -> dict:
        if mne is None:
            raise ImportError("mne is required — pip install mne")

        root_dir = cfg.data_path
        epoch_sec = cfg.epoch_sec
        epoch_step = cfg.epoch_step_sec
        target_sfreq = float(cfg.sampling_rate)

        edf_files = sorted(glob.glob(os.path.join(root_dir, "Subject*_*.edf")))
        if not edf_files:
            raise FileNotFoundError(
                f"No Subject*_*.edf files found in {root_dir}"
            )

        file_map: dict = {}
        pattern = re.compile(r"Subject(\d+)_(\d+)\.edf$")
        for fp in edf_files:
            m = pattern.search(os.path.basename(fp))
            if m:
                sid, suf = int(m.group(1)), int(m.group(2))
                file_map[(sid, suf)] = fp

        subject_ids = sorted({k[0] for k in file_map})
        print(f"  EEGMAT: found {len(subject_ids)} subjects, "
              f"{len(edf_files)} EDF files")

        all_eeg, all_labels, all_subs, all_positions = [], [], [], []
        n_channels_ref = None

        for sid in subject_ids:
            for suffix, label in [(1, LABEL_REST), (2, LABEL_ARITH)]:
                edf_path = file_map.get((sid, suffix))
                if edf_path is None:
                    print(f"  [skip] Subject{sid:02d}_{suffix}.edf not found")
                    continue

                data, ch_names, sfreq = _read_edf(edf_path)

                if target_sfreq and abs(sfreq - target_sfreq) > 1:
                    raw_tmp = mne.io.RawArray(
                        data,
                        mne.create_info(ch_names, sfreq, ch_types="eeg"),
                        verbose=False,
                    )
                    raw_tmp.resample(target_sfreq, verbose=False)
                    data = raw_tmp.get_data()
                    sfreq = target_sfreq

                if n_channels_ref is None:
                    n_channels_ref = data.shape[0]
                    print(f"  Channels: {n_channels_ref}  "
                          f"({', '.join(ch_names[:5])}, …)")
                    print(f"  Sampling rate: {sfreq} Hz")
                if data.shape[0] != n_channels_ref:
                    print(f"  [skip] Subject{sid:02d}_{suffix}: "
                          f"{data.shape[0]} ch (expected {n_channels_ref})")
                    continue

                epochs = _segment_epochs(data, sfreq, epoch_sec, epoch_step)

                for i in range(epochs.shape[0]):
                    all_eeg.append(epochs[i])
                    all_labels.append(label)
                    all_subs.append(sid)
                    all_positions.append(i)

        labels_arr = np.array(all_labels, dtype=np.int64)
        data_out = {
            "eeg": np.stack(all_eeg).astype(np.float32),
            "labels": labels_arr,
            "subject_ids": np.array(all_subs, dtype=np.int64),
            "positions": np.array(all_positions, dtype=np.int64),
        }
        print(f"  Total epochs: {len(all_eeg)}  "
              f"(rest={int((labels_arr == 0).sum())}, "
              f"arith={int((labels_arr == 1).sum())})")
        return data_out
