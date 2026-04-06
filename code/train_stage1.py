"""Stage 1 — Cross-Subject Contrastive Pre-training.

Creates a new timestamped run directory under checkpoints/.
Subsequent stages should pass that directory via --run_dir.
"""

import argparse

from pipeline import add_common_args, setup, create_run_dir
from models import DualAlignModel
from trainers import Stage1Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--stage1_epochs", type=int, default=None)
    args = parser.parse_args()

    cfg, loaders = setup(args)

    run_dir = create_run_dir(cfg.save_dir)
    cfg.save_dir = run_dir

    model = DualAlignModel(cfg)
    print(f"  Run dir : {run_dir}")
    print(f"  Params  : {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("STAGE 1 — Cross-Subject Contrastive Pre-training")
    print("=" * 60)

    Stage1Trainer(model, cfg).train(loaders["pair"])
    print(f"\nStage 1 complete.  Checkpoints → {run_dir}")


if __name__ == "__main__":
    main()
