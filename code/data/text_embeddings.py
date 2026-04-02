"""
Text-based task feature embeddings for NICE-style alignment.

Instead of hand-crafted numeric vectors, each task condition is described
in natural language and encoded by a pre-trained sentence encoder, producing
semantically meaningful feature vectors — analogous to how NICE uses CLIP
image features for visual stimuli.
"""

import os
import hashlib
import numpy as np
from typing import Dict, Tuple

# ── Task descriptions keyed by (label, time_segment, perf_group) ──
# time_segment: "early" / "mid" / "late"  (thirds of the 60-s recording)
# perf_group:  0 = poor performer, 1 = good performer

TASK_DESCRIPTIONS: Dict[Tuple, str] = {
    # Resting state
    (0, "early"):
        "静息态初始阶段，被试安静闭眼坐着，大脑处于放松的基线状态，无任何认知任务",
    (0, "mid"):
        "静息态中段，持续安静休息，大脑维持低唤醒水平的基线活动",
    (0, "late"):
        "静息态后段，较长时间的安静休息可能带来轻微嗜睡或注意力游离",

    # Mental arithmetic — poor performers (high cognitive load)
    (1, "early", 0):
        "心算任务起始阶段，需要从四位数中连续减去两位数，对于计算困难的被试认知负荷较高",
    (1, "mid", 0):
        "心算任务中段，持续进行高难度连续减法运算，认知资源高度消耗，计算速度较慢",
    (1, "late", 0):
        "心算任务后段，长时间高负荷心算导致认知疲劳积累，连续减法变得更加吃力",

    # Mental arithmetic — good performers (moderate cognitive load)
    (1, "early", 1):
        "心算任务起始阶段，被试开始从四位数中连续减去两位数，计算较为流畅",
    (1, "mid", 1):
        "心算任务中段，持续进行连续减法运算，保持稳定的计算节奏和适度的认知投入",
    (1, "late", 1):
        "心算任务后段，长时间连续心算后可能出现一定疲劳，但仍维持计算效率",
}

# English fallback (used when Chinese text causes tokenizer issues)
TASK_DESCRIPTIONS_EN: Dict[Tuple, str] = {
    (0, "early"):
        "Initial resting state baseline, subject sitting quietly with eyes closed, brain in relaxed baseline with no cognitive task",
    (0, "mid"):
        "Middle resting state period, sustained quiet rest with low arousal baseline brain activity",
    (0, "late"):
        "Late resting state phase, prolonged quiet rest with possible mild drowsiness or mind wandering",

    (1, "early", 0):
        "Beginning of mental arithmetic task, serial subtraction of two-digit number from four-digit number, high cognitive load for slow performer",
    (1, "mid", 0):
        "Ongoing demanding mental arithmetic, sustained effortful serial subtraction with high cognitive resource consumption and slow computation",
    (1, "late", 0):
        "Late phase of mental arithmetic, accumulated cognitive fatigue from prolonged high-load serial subtraction",

    (1, "early", 1):
        "Beginning of mental arithmetic task, subject starts serial subtraction of two-digit from four-digit number with fluent computation",
    (1, "mid", 1):
        "Ongoing mental arithmetic, sustained serial subtraction at steady pace with moderate cognitive engagement",
    (1, "late", 1):
        "Late phase of mental arithmetic, possible mild fatigue after prolonged computation but maintaining efficiency",
}


# ── 9-class condition ID for Stage 1/2 contrastive learning ──
# Maps description key tuples to integer IDs (0-8).
# Rest conditions (no performance distinction):
#   0 = rest-early, 1 = rest-mid, 2 = rest-late
# Arithmetic conditions split by performance group:
#   3 = arith-early-good, 4 = arith-mid-good, 5 = arith-late-good
#   6 = arith-early-bad,  7 = arith-mid-bad,  8 = arith-late-bad
CONDITION_KEY_TO_ID: Dict[Tuple, int] = {
    (0, "early"): 0,
    (0, "mid"):   1,
    (0, "late"):  2,
    (1, "early", 1): 3,
    (1, "mid",   1): 4,
    (1, "late",  1): 5,
    (1, "early", 0): 6,
    (1, "mid",   0): 7,
    (1, "late",  0): 8,
}

N_CONDITIONS = len(CONDITION_KEY_TO_ID)


def _time_segment(epoch_idx: int, n_epochs: int) -> str:
    frac = epoch_idx / max(n_epochs - 1, 1)
    if frac < 0.33:
        return "early"
    elif frac < 0.67:
        return "mid"
    return "late"


def get_description_key(label: int, epoch_idx: int, n_epochs: int,
                        perf_group: int = 1) -> tuple:
    seg = _time_segment(epoch_idx, n_epochs)
    if label == 0:
        return (0, seg)
    return (1, seg, perf_group)


def get_condition_id(label: int, epoch_idx: int, n_epochs: int,
                     perf_group: int = 1) -> int:
    """Return integer condition ID (0-8) for contrastive learning pairing."""
    key = get_description_key(label, epoch_idx, n_epochs, perf_group)
    return CONDITION_KEY_TO_ID[key]


def _compute_embeddings_sentence_transformers(
    texts: list, model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False,
                              normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


def _compute_embeddings_fallback(texts: list, dim: int = 384) -> np.ndarray:
    """Deterministic hash-based embedding when no model is available."""
    embeddings = []
    for text in texts:
        h = hashlib.sha512(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-8
        embeddings.append(vec)
    return np.stack(embeddings)


def build_text_embeddings(
    lang: str = "en",
    model_name: str = "all-MiniLM-L6-v2",
    cache_dir: str = "data_cache",
) -> Dict[tuple, np.ndarray]:
    """
    Pre-compute text embeddings for all task conditions.

    Returns dict mapping condition key → embedding vector.
    """
    descs = TASK_DESCRIPTIONS_EN if lang == "en" else TASK_DESCRIPTIONS
    keys = list(descs.keys())
    texts = [descs[k] for k in keys]

    cache_tag = hashlib.md5(
        (model_name + "".join(texts)).encode()
    ).hexdigest()[:12]
    cache_path = os.path.join(cache_dir, f"text_emb_{cache_tag}.npz")

    if os.path.exists(cache_path):
        loaded = np.load(cache_path, allow_pickle=True)
        emb_dict = {}
        for k, v in zip(loaded["keys"], loaded["embeddings"]):
            emb_dict[tuple(k)] = v
        print(f"  Loaded cached text embeddings ({len(emb_dict)} conditions, "
              f"dim={v.shape[0]})")
        return emb_dict

    try:
        print(f"  Encoding task descriptions with {model_name} …")
        embeddings = _compute_embeddings_sentence_transformers(texts, model_name)
    except (ImportError, Exception) as e:
        print(f"  sentence-transformers unavailable ({e}), using hash fallback")
        embeddings = _compute_embeddings_fallback(texts, dim=384)

    emb_dict = {}
    for k, emb in zip(keys, embeddings):
        emb_dict[k] = emb

    os.makedirs(cache_dir, exist_ok=True)
    np.savez(
        cache_path,
        keys=np.array([list(k) for k in keys], dtype=object),
        embeddings=embeddings,
    )
    print(f"  Text embeddings: {len(keys)} conditions, dim={embeddings.shape[1]}")
    return emb_dict
