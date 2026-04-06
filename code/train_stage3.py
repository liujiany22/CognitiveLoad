"""Stage 3 — Classification Fine-tuning.

Requires a --run_dir that contains pretrained encoder weights.
Supports --ablation to select which feature branches to use.
"""

import argparse

import torch

from pipeline import add_common_args, setup
from models import DualAlignModel
from trainers import Stage3Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Run directory containing pretrained checkpoints")
    parser.add_argument("--stage3_epochs", type=int, default=None)
    parser.add_argument("--freeze_encoders", action="store_true",
                        help="Freeze encoder weights during fine-tuning")
    parser.add_argument("--ablation", type=str, default="",
                        choices=list(DualAlignModel.branch_modes()),
                        help="Feature-selection ablation mode")
    args = parser.parse_args()

    cfg, loaders = setup(args)
    cfg.save_dir = args.run_dir

    skip_stage2 = cfg.ablation == "cross_only"
    ckpt_name = "stage1_best.pt" if skip_stage2 else "stage2_best.pt"
    ckpt_path = f"{cfg.save_dir}/{ckpt_name}"

    model = DualAlignModel(cfg)
    model.load_compatible_state_dict(
        torch.load(ckpt_path, map_location=cfg.device, weights_only=True),
    )
    print(f"  Loaded  : {ckpt_path}")
    print(f"  Params  : {sum(p.numel() for p in model.parameters()):,}")
    if cfg.ablation:
        print(f"  Ablation: {cfg.ablation}")

    print("\n" + "=" * 60)
    print("STAGE 3 — Classification Fine-tuning")
    print("=" * 60)

    Stage3Trainer(model, cfg, freeze_encoders=args.freeze_encoders) \
        .train(loaders["finetune"], loaders["val"])
    print(f"\nStage 3 complete.  Checkpoints → {cfg.save_dir}")


if __name__ == "__main__":
    main()
