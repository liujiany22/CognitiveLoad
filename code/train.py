"""
DualAlign-CogLoad — unified training script.

Usage:
    python train.py --stage 1                          # Stage 1 only (creates run dir)
    python train.py --stage 2 --run_dir checkpoints/…  # Stage 2 (needs stage1_best.pt)
    python train.py --stage 3 --run_dir checkpoints/…  # Stage 3 (needs stage2_best.pt)
    python train.py --stage all                        # Full pipeline: 1 → 2 → 3
"""

import argparse

import torch

from cli import add_common_args, setup, create_run_dir
from models import DualAlign
from trainers import Stage1Trainer, Stage2Trainer, Stage3Trainer


def _load_checkpoint(model, path, device):
    model.load_compatible_state_dict(
        torch.load(path, map_location=device, weights_only=True),
    )
    print(f"  Loaded  : {path}")


def _print_header(title: str):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def run_stage1(model, cfg, loaders):
    _print_header("STAGE 1 — Cross-Subject Contrastive Pre-training")
    Stage1Trainer(model, cfg).train(loaders["pair"])


def run_stage2(model, cfg, loaders):
    _print_header("STAGE 2 — Stimulus-Task Alignment Pre-training")
    Stage2Trainer(model, cfg).train(loaders["train"], val_loader=loaders["val"])


def run_stage3(model, cfg, split_info):
    _print_header("STAGE 3 — Classification Fine-tuning")
    trainer = Stage3Trainer(model, cfg)
    de_loaders = trainer.prepare_de_loaders(
        split_info["data"],
        split_info["train_subs"],
        split_info["val_subs"],
        split_info["test_subs"],
    )
    trainer.train(de_loaders["train"], de_loaders["val"])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)

    parser.add_argument("--stage", type=str, required=True,
                        choices=["1", "2", "3", "all"],
                        help="Which stage(s) to run")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run directory (required for stage 2/3; "
                             "auto-created for stage 1 and all)")
    parser.add_argument("--stage1_epochs", type=int, default=None)
    parser.add_argument("--stage2_epochs", type=int, default=None)
    parser.add_argument("--stage3_epochs", type=int, default=None)
    parser.add_argument("--ablation", type=str, default="",
                        choices=["", "cross_only"],
                        help="cross_only: skip Stage 2, load stage1_best.pt directly")
    args = parser.parse_args()

    stage = args.stage

    if stage in ("2", "3") and args.run_dir is None:
        parser.error("--run_dir is required for stage 2 and 3")

    cfg, loaders, split_info = setup(args)

    # Resolve run directory
    if args.run_dir:
        cfg.save_dir = args.run_dir
    elif stage in ("1", "all"):
        cfg.save_dir = create_run_dir(cfg.save_dir)

    model = DualAlign(cfg)
    print(f"  Run dir : {cfg.save_dir}")
    print(f"  Params  : {sum(p.numel() for p in model.parameters()):,}")
    cfg.save()

    if stage == "1":
        run_stage1(model, cfg, loaders)

    elif stage == "2":
        _load_checkpoint(model, f"{cfg.save_dir}/stage1_best.pt", cfg.device)
        run_stage2(model, cfg, loaders)

    elif stage == "3":
        skip_stage2 = cfg.ablation == "cross_only"
        ckpt_name = "stage1_best.pt" if skip_stage2 else "stage2_best.pt"
        _load_checkpoint(model, f"{cfg.save_dir}/{ckpt_name}", cfg.device)
        if cfg.ablation:
            print(f"  Ablation: {cfg.ablation}")
        run_stage3(model, cfg, split_info)

    elif stage == "all":
        run_stage1(model, cfg, loaders)
        run_stage2(model, cfg, loaders)
        run_stage3(model, cfg, split_info)

    print(f"\nComplete.  Checkpoints → {cfg.save_dir}")


if __name__ == "__main__":
    main()
