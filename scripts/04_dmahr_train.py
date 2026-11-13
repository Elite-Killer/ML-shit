"""Train a leakage-controlled DM-AHR checkpoint on train-split pairs only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.diffusion.train import train  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402

log = setup_logging()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--warm-start", default=None,
                    help="checkpoint to initialise from, for sanity checks")
    ap.add_argument("--iterations", type=int, default=None,
                    help="override the configured iteration count")
    ap.add_argument("--out-name", default=None,
                    help="checkpoint subdirectory; set this for sanity runs so the "
                         "recorded Arm C checkpoint is never overwritten")
    args = ap.parse_args()
    cfg = load_config("configs/dmahr.json")
    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    train_pairs = pairs[pairs["split"] == "train"]
    val_pairs = pairs[pairs["split"] == "val"]
    if train_pairs.empty or val_pairs.empty:
        raise RuntimeError("synthetic pairs missing; run 02_hair_pipeline.py first")
    if set(train_pairs["group_id"]) & set(val_pairs["group_id"]):
        raise RuntimeError("group leakage between diffusion train and val pairs")

    default_name = "dmahr_overfit" if args.overfit else "dmahr"
    out_dir = artifact_dir("checkpoints", args.out_name or default_name)
    result = train(cfg, train_pairs, val_pairs, out_dir, overfit=args.overfit,
                   init_from=Path(args.warm_start) if args.warm_start else None,
                   max_iterations=args.iterations)
    save_json(result, artifact_dir("logs") / ("dmahr_overfit.json" if args.overfit else "dmahr_train.json"))
    record_run("dmahr_train_overfit" if args.overfit else "dmahr_train",
               gpu="A100-80GB", seconds=result["seconds"], iterations=result["iterations"])
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

