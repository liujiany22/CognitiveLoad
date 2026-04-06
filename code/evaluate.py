"""Evaluate a trained model on the held-out test set."""

import argparse
import json
import os

import torch

from pipeline import add_common_args, setup
from models import DualAlignModel
from trainers import Stage3Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Run directory containing stage3_best*.pt")
    parser.add_argument("--ablation", type=str, default="",
                        choices=list(DualAlignModel.branch_modes()),
                        help="Feature-selection ablation mode")
    args = parser.parse_args()

    cfg, loaders = setup(args)

    tag = f"_{cfg.ablation}" if cfg.ablation else ""
    ckpt_path = os.path.join(args.run_dir, f"stage3_best{tag}.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Error: checkpoint not found: {ckpt_path}")

    model = DualAlignModel(cfg)
    model.load_compatible_state_dict(
        torch.load(ckpt_path, map_location=cfg.device, weights_only=True),
    )
    model.to(cfg.device)
    print(f"  Loaded: {ckpt_path}")

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    metrics = Stage3Trainer(model, cfg).evaluate(loaders["test"])

    print(metrics["report"])
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"Cohen's Kappa     : {metrics['kappa']:.4f}")
    print(f"Macro F1          : {metrics['f1_macro']:.4f}")
    print(f"Per-class F1      : {metrics['f1_per_class']}")
    print(f"Per-class Prec    : {metrics['precision_per_class']}")
    print(f"Per-class Recall  : {metrics['recall_per_class']}")
    print(f"Confusion matrix  :\n{metrics['confusion_matrix']}")

    results_path = os.path.join(cfg.log_dir, f"results{tag}.json")
    with open(results_path, "w") as f:
        json.dump({
            "ablation": cfg.ablation or "full",
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "kappa": metrics["kappa"],
            "f1_macro": metrics["f1_macro"],
            "f1_per_class": metrics["f1_per_class"],
            "precision_per_class": metrics["precision_per_class"],
            "recall_per_class": metrics["recall_per_class"],
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        }, f, indent=2)
    print(f"\nResults → {results_path}")


if __name__ == "__main__":
    main()
