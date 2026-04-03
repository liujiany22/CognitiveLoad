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
from data.text_embeddings import (
    build_text_embeddings, get_description_key, get_condition_id,
    N_CONDITIONS,
)

LABEL_REST = 0
LABEL_ARITH = 1


def _read_edf(path: str) -> tuple:
    """Return (data [n_channels, n_samples], ch_names, sfreq)."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.pick(picks="eeg", exclude="bads")
    return raw.get_data(), raw.ch_names, raw.info["sfreq"]


def _segment_epochs(data: np.ndarray, sfreq: float,
                    epoch_sec: float = 2.0) -> np.ndarray:
    """Non-overlapping sliding window.  Returns (n_epochs, n_ch, n_timepoints)."""
    n_ch, n_total = data.shape
    win = int(epoch_sec * sfreq)
    n_epochs = n_total // win
    trimmed = data[:, : n_epochs * win]
    return trimmed.reshape(n_ch, n_epochs, win).transpose(1, 0, 2)


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
    """PhysioNet EEGMAT dataset loader."""

    name = "eegmat"

    def cache_tag(self, cfg) -> str:
        return f"eegmat_ep{cfg.epoch_sec}s_{cfg.sampling_rate}hz_textemb"

    def load_raw(self, cfg) -> dict:
        if mne is None:
            raise ImportError("mne is required — pip install mne")

        root_dir = cfg.data_path
        epoch_sec = cfg.epoch_sec
        target_sfreq = float(cfg.sampling_rate)

        text_emb = build_text_embeddings(lang="en", cache_dir=cfg.data_dir)
        embed_dim = next(iter(text_emb.values())).shape[0]

        subj_info = _read_subject_info(root_dir)

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

        all_eeg, all_labels, all_subs, all_conds, all_feats = [], [], [], [], []
        n_channels_ref = None

        for sid in subject_ids:
            info = subj_info.get(sid, {"quality": 1, "n_sub": 15.0})
            perf = info["quality"]

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

                epochs = _segment_epochs(data, sfreq, epoch_sec)
                n_ep = epochs.shape[0]

                for i in range(n_ep):
                    key = get_description_key(label, i, n_ep, perf)
                    feat = text_emb.get(key)
                    if feat is None:
                        feat = np.zeros(embed_dim, dtype=np.float32)
                    cond_id = get_condition_id(label, i, n_ep, perf)

                    all_eeg.append(epochs[i])
                    all_labels.append(label)
                    all_subs.append(sid)
                    all_conds.append(cond_id)
                    all_feats.append(feat)

        data_out = {
            "eeg": np.stack(all_eeg).astype(np.float32),
            "labels": np.array(all_labels, dtype=np.int64),
            "subject_ids": np.array(all_subs, dtype=np.int64),
            "condition_ids": np.array(all_conds, dtype=np.int64),
            "task_features": np.stack(all_feats).astype(np.float32),
        }
        print(f"  Total epochs: {len(all_eeg)}  "
              f"(rest={sum(l == 0 for l in all_labels)}, "
              f"arith={sum(l == 1 for l in all_labels)})")
        cond_arr = np.array(all_conds)
        cond_counts = {c: int((cond_arr == c).sum()) for c in range(N_CONDITIONS)}
        print(f"  Condition distribution (9-class): {cond_counts}")
        print(f"  Task feature dim: {embed_dim} (text embeddings)")
        return data_out
