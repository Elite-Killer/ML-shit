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
    args = ap.parse_args()
    cfg = load_config("configs/dmahr.json")
    pairs = pd.read_csv(manifest_dir() / "synthetic_pairs.csv")
    train_pairs = pairs[pairs["split"] == "train"]
    val_pairs = pairs[pairs["split"] == "val"]
    if train_pairs.empty or val_pairs.empty:
        raise RuntimeError("synthetic pairs missing; run 02_hair_pipeline.py first")
    if set(train_pairs["group_id"]) & set(val_pairs["group_id"]):
        raise RuntimeError("group leakage between diffusion train and val pairs")

    out_dir = artifact_dir("checkpoints", "dmahr_overfit" if args.overfit else "dmahr")
    result = train(cfg, train_pairs, val_pairs, out_dir, overfit=args.overfit)
    save_json(result, artifact_dir("logs") / ("dmahr_overfit.json" if args.overfit else "dmahr_train.json"))
    record_run("dmahr_train_overfit" if args.overfit else "dmahr_train",
               gpu="A100-80GB", seconds=result["seconds"], iterations=result["iterations"])
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

