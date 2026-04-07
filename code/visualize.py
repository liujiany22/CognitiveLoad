"""
Visualize learned filters, intermediate activations, and embeddings.

Focus: Stage 1 (CrossEncoder) and Stage 3 (Classifier).

Usage:
    python visualize.py --run_dir checkpoints/20250401_1200 --data_batch 32

Produces figures in  <run_dir>/viz/  covering:
    1. CrossEncoder conv-filter weights  (spatial & temporal)
    2. CrossEncoder temporal-filter frequency responses
    3. CrossEncoder intermediate feature-map snapshots
    4. t-SNE of CrossEncoder embeddings
    5. Stage-3 classifier weight inspection
"""

import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from cli import add_common_args, setup
from models import DualAlign

# ──────────────────────────────────────────────────────────────
#  Hooks: capture intermediate activations
# ──────────────────────────────────────────────────────────────

class ActivationCapture:
    """Register forward hooks and collect intermediate outputs."""

    def __init__(self):
        self.activations: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._handles = []

    def register(self, module: torch.nn.Module, name: str):
        def hook(_mod, _inp, out, _name=name):
            self.activations[_name] = out.detach().cpu()
        self._handles.append(module.register_forward_hook(hook))

    def clear(self):
        self.activations.clear()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def register_cross_hooks(model: DualAlign) -> ActivationCapture:
    """Attach hooks to every layer in CrossEncoder (Stage 1)."""
    cap = ActivationCapture()
    ce = model.cross_encoder
    cap.register(ce.spatial_conv,   "cross.spatial_conv")
    cap.register(ce.time_conv,      "cross.time_conv")
    cap.register(ce.avg_pool,       "cross.avg_pool")
    cap.register(ce.depth_spatial,  "cross.depth_spatial")
    cap.register(ce.depth_temporal, "cross.depth_temporal")
    return cap


# ──────────────────────────────────────────────────────────────
#  1. Conv-filter weights
# ──────────────────────────────────────────────────────────────

def plot_spatial_filters(model: DualAlign, save_dir: str):
    """Spatial conv kernel — each filter is a (C,1) weight vector."""
    w = model.cross_encoder.spatial_conv.weight.detach().cpu()  # (n_sf, 1, C, 1)
    w = w.squeeze(1).squeeze(-1)  # (n_sf, C)
    n_filters = w.shape[0]

    cols = min(8, n_filters)
    rows = (n_filters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n_filters:
            ax.bar(range(w.shape[1]), w[i].numpy(), width=0.8)
            ax.set_title(f"SF {i}", fontsize=8)
            ax.set_xlabel("channel", fontsize=7)
        ax.tick_params(labelsize=6)
        if i >= n_filters:
            ax.axis("off")
    fig.suptitle("CrossEncoder — Spatial Filters (weight per channel)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(save_dir, "01_cross_spatial_filters.png"), dpi=150)
    plt.close(fig)


def plot_temporal_filters(model: DualAlign, save_dir: str):
    """Temporal conv kernel — each filter is a (1, K) weight vector."""
    w = model.cross_encoder.time_conv.weight.detach().cpu()  # (n_tf, 1, 1, K)
    w = w.squeeze(1).squeeze(1)  # (n_tf, K)
    n_filters = w.shape[0]

    cols = min(8, n_filters)
    rows = (n_filters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n_filters:
            ax.plot(w[i].numpy(), linewidth=0.8)
            ax.set_title(f"TF {i}", fontsize=8)
        ax.tick_params(labelsize=6)
        if i >= n_filters:
            ax.axis("off")
    fig.suptitle("CrossEncoder — Temporal Filters (time-domain)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(save_dir, "02_cross_temporal_filters.png"), dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
#  2. Frequency response of temporal filters
# ──────────────────────────────────────────────────────────────

def plot_frequency_response(model: DualAlign, sr: int, save_dir: str):
    """Plot magnitude spectrum of each CrossEncoder temporal filter."""
    w = model.cross_encoder.time_conv.weight.detach().cpu().squeeze()
    if w.dim() == 3:
        w = w.squeeze(1)
    n_filters, K = w.shape
    nfft = max(256, K)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sr)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(n_filters):
        mag = np.abs(np.fft.rfft(w[i].numpy(), n=nfft))
        ax.plot(freqs, mag, alpha=0.6, linewidth=0.7, label=f"F{i}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("CrossEncoder — Temporal Filter Frequency Response")
    ax.set_xlim(0, sr / 2)
    if n_filters <= 16:
        ax.legend(fontsize=6, ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "03_cross_freq_response.png"), dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
#  3. Intermediate feature-map snapshots
# ──────────────────────────────────────────────────────────────

def plot_activation_maps(activations: OrderedDict, save_dir: str, max_channels: int = 16):
    """Plot a selection of captured activation maps."""
    for name, act in activations.items():
        if act.dim() < 3:
            continue
        sample = act[0]  # first sample in batch
        if sample.dim() == 3:
            n_show = min(sample.shape[0], max_channels)
            cols = min(8, n_show)
            rows = (n_show + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2))
            axes = np.atleast_2d(axes)
            for i in range(rows * cols):
                ax = axes[i // cols, i % cols]
                if i < n_show:
                    fmap = sample[i].numpy()
                    if 1 in fmap.shape:
                        fmap = fmap.squeeze()
                        ax.plot(fmap, linewidth=0.7)
                    else:
                        ax.imshow(fmap, aspect="auto", cmap="RdBu_r",
                                  interpolation="nearest")
                    ax.set_title(f"ch {i}", fontsize=7)
                ax.tick_params(labelsize=5)
                if i >= n_show:
                    ax.axis("off")
            fig.suptitle(f"Activations — {name}  (sample 0)", fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            safe_name = name.replace(".", "_")
            fig.savefig(os.path.join(save_dir, f"04_act_{safe_name}.png"), dpi=150)
            plt.close(fig)


# ──────────────────────────────────────────────────────────────
#  4. t-SNE of embeddings
# ──────────────────────────────────────────────────────────────

def collect_embeddings(model: DualAlign, eeg: np.ndarray, labels: np.ndarray,
                       batch_size: int = 64, device: str = "cpu"):
    """Extract CrossEncoder embeddings."""
    model.eval()
    model.to(device)
    n = len(eeg)
    cross_feats = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = torch.from_numpy(eeg[start:end]).float().unsqueeze(1).to(device)
        with torch.no_grad():
            cross_feats.append(model.cross_encoder(x).cpu().numpy())

    return {
        "cross": np.concatenate(cross_feats),
        "labels": labels[:n],
    }


def plot_tsne(embeddings: dict, save_dir: str, label_names=None, perplexity: int = 30):
    """2-D t-SNE scatter for CrossEncoder features."""
    labels = embeddings["labels"]
    unique_labels = sorted(set(labels))
    if label_names is None:
        label_names = {l: str(l) for l in unique_labels}
    elif isinstance(label_names, (list, tuple)):
        label_names = {i: n for i, n in enumerate(label_names)}

    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    feats = embeddings["cross"]
    p = perplexity
    if feats.shape[0] < p + 1:
        p = max(5, feats.shape[0] // 4)
    tsne = TSNE(n_components=2, perplexity=p, random_state=42,
                 init="pca", learning_rate="auto")
    proj = tsne.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(proj[mask, 0], proj[mask, 1], s=8, alpha=0.6,
                   color=cmap(idx), label=label_names.get(lbl, str(lbl)))
    ax.legend(fontsize=8, markerscale=2)
    ax.set_title("t-SNE — CrossEncoder embeddings", fontsize=11)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "05_tsne_cross.png"), dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
#  5. Classifier weight inspection
# ──────────────────────────────────────────────────────────────

def plot_classifier_weights(model: DualAlign, save_dir: str):
    """Visualise first-layer classifier weights as a heatmap."""
    w = model.classifier[0].weight.detach().cpu().numpy()  # (hidden, feat_dim)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(w, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_xlabel("Input feature dimension")
    ax.set_ylabel("Hidden unit")
    ax.set_title("Stage-3 Classifier — First Linear Layer Weights")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "06_classifier_weights.png"), dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Run directory with trained checkpoints")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Specific checkpoint file (default: auto-detect best)")
    parser.add_argument("--data_batch", type=int, default=32,
                        help="Number of samples for activation / attention plots")
    parser.add_argument("--tsne_samples", type=int, default=500,
                        help="Max samples for t-SNE (sub-sampled if larger)")
    parser.add_argument("--ablation", type=str, default="",
                        choices=["", "cross_only"])
    args = parser.parse_args()

    cfg, _loaders, split_info = setup(args)
    device = cfg.device

    viz_dir = os.path.join(args.run_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    # ── Find best checkpoint ──
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        tag = f"_{cfg.ablation}" if cfg.ablation else ""
        candidates = [
            f"stage3_best{tag}.pt",
            f"stage2_best.pt",
            f"stage1_best.pt",
        ]
        ckpt_path = None
        for c in candidates:
            p = os.path.join(args.run_dir, c)
            if os.path.exists(p):
                ckpt_path = p
                break
        if ckpt_path is None:
            raise SystemExit(f"No checkpoint found in {args.run_dir}")

    model = DualAlign(cfg)
    model.load_compatible_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True),
    )
    model.to(device).eval()
    print(f"  Loaded : {ckpt_path}")
    print(f"  Output : {viz_dir}")

    # ── Grab a small batch of real data ──
    data = split_info["data"]
    n_avail = data["eeg"].shape[0]
    n_batch = min(args.data_batch, n_avail)
    idx = np.random.RandomState(cfg.seed).choice(n_avail, n_batch, replace=False)
    eeg_batch = torch.from_numpy(data["eeg"][idx]).float().unsqueeze(1).to(device)

    label_names = getattr(cfg, "label_names", None)

    # ── 1. CrossEncoder conv filter weights ──
    print("  [1/5] CrossEncoder conv filter weights …")
    plot_spatial_filters(model, viz_dir)
    plot_temporal_filters(model, viz_dir)

    # ── 2. Frequency response ──
    print("  [2/5] CrossEncoder frequency response …")
    plot_frequency_response(model, cfg.sampling_rate, viz_dir)

    # ── 3. Intermediate activations ──
    print("  [3/5] CrossEncoder intermediate activations …")
    cap = register_cross_hooks(model)
    with torch.no_grad():
        model.cross_encoder(eeg_batch)
    plot_activation_maps(cap.activations, viz_dir)
    cap.remove_hooks()

    # ── 4. t-SNE embeddings ──
    print("  [4/5] CrossEncoder t-SNE embeddings …")
    n_tsne = min(args.tsne_samples, n_avail)
    tsne_idx = np.random.RandomState(cfg.seed).choice(n_avail, n_tsne, replace=False)
    embs = collect_embeddings(model, data["eeg"][tsne_idx], data["labels"][tsne_idx],
                              device=device)
    plot_tsne(embs, viz_dir, label_names=label_names)

    # ── 5. Stage-3 classifier weights ──
    print("  [5/5] Classifier weights …")
    plot_classifier_weights(model, viz_dir)

    print(f"\nDone — {len(os.listdir(viz_dir))} figures saved to {viz_dir}")


if __name__ == "__main__":
    main()
