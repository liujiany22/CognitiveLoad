"""Stage 2 — Stimulus-Task Alignment Pre-training.

Requires a --run_dir that already contains stage1_best.pt.
"""

import argparse

import torch

from pipeline import add_common_args, setup
from models import DualAlignModel
from trainers import Stage2Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Run directory from Stage 1 (contains stage1_best.pt)")
    parser.add_argument("--stage2_epochs", type=int, default=None)
    args = parser.parse_args()

    cfg, loaders = setup(args)
    cfg.save_dir = args.run_dir

    model = DualAlignModel(cfg)
    ckpt_path = f"{cfg.save_dir}/stage1_best.pt"
    model.load_compatible_state_dict(
        torch.load(ckpt_path, map_location=cfg.device, weights_only=True),
    )
    model.init_align_from_cross()
    print(f"  Loaded  : {ckpt_path}")
    print(f"  Params  : {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("STAGE 2 — Stimulus-Task Alignment Pre-training")
    print("=" * 60)

    Stage2Trainer(model, cfg).train(loaders["train"], val_loader=loaders["val"])
    print(f"\nStage 2 complete.  Checkpoints → {cfg.save_dir}")


if __name__ == "__main__":
    main()
