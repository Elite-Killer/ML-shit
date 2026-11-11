"""Screen hair, freeze parameters, build Arms B and A*, and synthetic pairs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import config_hash, load_config, save_frozen, save_json  # noqa: E402
from dm_ahr_vit.data.synthetic_hair import render_hair  # noqa: E402
from dm_ahr_vit.hair.dullrazor import apply_dullrazor  # noqa: E402
from dm_ahr_vit.hair.screening import detect_hair, tune_coverage_threshold  # noqa: E402
from dm_ahr_vit.paths import (  # noqa: E402
    artifact_dir,
    data_root,
    isic_images_dir,
    manifest_dir,
    processed_dir,
)
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()
SYNTH_SEED = 7000
SYNTH_LIMITS = {"train": 3000, "val": 800, "test": 800}


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def write_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, compress_level=3)


def tune(screen_cfg: dict, splits: pd.DataFrame) -> tuple[dict, dict]:
    """Tune the coverage threshold on train/val synthetic pairs only."""
    tune_rows = splits[splits["split"].isin(["train", "val"])]
    rng = np.random.default_rng(12345)
    sample = tune_rows.sample(n=min(400, len(tune_rows)), random_state=42)
    pairs = []
    for _, r in sample.iterrows():
        img = read_rgb(isic_images_dir() / f"{r['image_id']}.jpg")
        base = Image.fromarray(img)
        clean_cov = detect_hair(img, {**screen_cfg, "coverage_threshold": 10.0})[1]["coverage"]
        if clean_cov > 0.01:
            continue  # already hairy: not a clean control
        hairy, _, _ = render_hair(base, r["image_id"], SYNTH_SEED + 1)
        pairs.append((img, 0))
        pairs.append((np.array(hairy), 1))
    result = tune_coverage_threshold(pairs, screen_cfg)
    return result, {"tuning_pairs": len(pairs), "result": result}


def contact_sheet(items: list[tuple[np.ndarray, str]], out: Path, cols: int = 4) -> None:
    if not items:
        return
    th = 200
    thumbs = []
    for arr, label in items:
        im = Image.fromarray(arr).resize((th, th), Image.Resampling.BICUBIC)
        thumbs.append((np.array(im), label))
    rows = (len(thumbs) + cols - 1) // cols
    canvas = np.zeros((rows * (th + 22), cols * th, 3), dtype=np.uint8)
    for i, (arr, label) in enumerate(thumbs):
        r, c = divmod(i, cols)
        canvas[r * (th + 22): r * (th + 22) + th, c * th:(c + 1) * th] = arr
        cv2.putText(canvas, label[:28], (c * th + 3, r * (th + 22) + th + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out, quality=88)


def process_one(row, screen_cfg, dull_cfg) -> dict:
    """Screen one image and write every arm output. Runs in a worker thread."""
    image_id = row["image_id"]
    out = {
        "image_id": image_id,
        "split": row["split"],
        "dx": row["dx"],
        "group_id": row["group_id"],
        "screen_config_hash": screen_cfg["config_hash"],
    }
    try:
        img = read_rgb(isic_images_dir() / f"{image_id}.jpg")
    except Exception as exc:  # noqa: BLE001
        out.update({"status": "fail_decode", "error": repr(exc),
                    "hair_present": None, "hair_coverage": None})
        return out
    mask, stats = detect_hair(img, screen_cfg)
    out.update(
        {
            "width": img.shape[1],
            "height": img.shape[0],
            "hair_coverage": round(stats["coverage"], 6),
            "hair_present": stats["hair_present"],
            "status": "ok",
        }
    )
    try:
        write_png(mask * 255, processed_dir("hair_masks") / f"{image_id}.png")
        if stats["hair_present"]:
            cleaned = apply_dullrazor(img, mask, dull_cfg)
            write_png(cleaned, processed_dir("B") / f"{image_id}.png")
            star = np.array(Image.fromarray(img).resize((128, 128), Image.Resampling.BICUBIC))
            write_png(star, processed_dir("Astar") / f"{image_id}.png")
            out["arm_b_status"] = "dullrazor"
            out["arm_astar"] = "resized_128"
        else:
            # Hair-absent images are identical in every arm, per protocol.
            write_png(img, processed_dir("B") / f"{image_id}.png")
            write_png(img, processed_dir("Astar") / f"{image_id}.png")
            out["arm_b_status"] = "hair_absent_passthrough"
            out["arm_astar"] = "hair_absent_passthrough"
    except Exception as exc:  # noqa: BLE001
        out.update({"status": "fail_write", "error": repr(exc)})
    return out


def synth_one(row, split: str) -> dict:
    image_id = row["image_id"]
    img = read_rgb(isic_images_dir() / f"{image_id}.jpg")
    hairy, hmask, params = render_hair(Image.fromarray(img), image_id, SYNTH_SEED)
    base = processed_dir("synth") / split
    write_png(np.array(hairy), base / "hairy" / f"{image_id}.png")
    write_png(img, base / "reference" / f"{image_id}.png")
    write_png(np.array(hmask), base / "hair_mask" / f"{image_id}.png")
    return {
        "pair_id": f"synth_{split}_{image_id}",
        "image_id": image_id,
        "split": split,
        "dx": row["dx"],
        "group_id": row["group_id"],
        "hairy_path": str(base / "hairy" / f"{image_id}.png"),
        "reference_path": str(base / "reference" / f"{image_id}.png"),
        "hair_mask_path": str(base / "hair_mask" / f"{image_id}.png"),
        "seed": SYNTH_SEED,
        "n_hairs": params["n_hairs"],
        "blur_sigma": params["blur"],
        "source": "synthetic",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-synthetic", action="store_true")
    args = ap.parse_args()
    protocol = load_config("configs/protocol.json")
    from dm_ahr_vit.config import load_frozen

    screen_cfg = load_frozen("hair_screen", "configs/hair_screen.json")
    dull_cfg = load_frozen("dullrazor", "configs/dullrazor.json")
    splits = pd.read_csv(manifest_dir() / "splits.csv")

    if not screen_cfg.get("frozen"):
        with timed("tune_hair_screen", log) as t_tune:
            best, tune_info = tune(screen_cfg, splits)
        screen_cfg = {**screen_cfg, "coverage_threshold": best["coverage_threshold"], "frozen": True}
        screen_cfg["config_hash"] = config_hash(
            {k: v for k, v in screen_cfg.items() if k != "config_hash"}
        )
        dull_cfg = {**dull_cfg, "frozen": True}
        dull_cfg["config_hash"] = config_hash(
            {k: v for k, v in dull_cfg.items() if k != "config_hash"}
        )
        save_json(screen_cfg, "configs/hair_screen.json")
        save_json(dull_cfg, "configs/dullrazor.json")
        save_frozen("hair_screen", screen_cfg)
        save_frozen("dullrazor", dull_cfg)
        save_json(
            {"screen": screen_cfg, "dullrazor": dull_cfg, "tuning": tune_info,
             "tuning_seconds": t_tune["seconds"]},
            artifact_dir("logs") / "hair_frozen_config.json",
        )
        log.info("froze hair screening: threshold=%.4f (%s)", best["coverage_threshold"], best)

    with timed("process_arms", log) as t_proc:
        records = [r for _, r in splits.iterrows()]
        rows = []
        with ThreadPoolExecutor(max_workers=48) as ex:
            for i, row in enumerate(
                ex.map(lambda r: process_one(r, screen_cfg, dull_cfg), records), 1
            ):
                rows.append(row)
                if i % 1000 == 0:
                    log.info("screened %d/%d", i, len(records))
        df = pd.DataFrame(rows)
        df.to_csv(manifest_dir() / "hair_screening.csv", index=False)
        # Contact sheets are rebuilt from the written outputs (no re-inference).
        hairy_ids = df[df["hair_present"]].head(12)["image_id"].tolist()
        clean_ids = df[~df["hair_present"]].head(12)["image_id"].tolist()
        panels_screen = []
        for i in hairy_ids:
            panels_screen.append((read_rgb(isic_images_dir() / f"{i}.jpg"), f"{i} hairy"))
            panels_screen.append((read_rgb(processed_dir("hair_masks") / f"{i}.png"), f"{i} mask"))
        for i in clean_ids:
            panels_screen.append((read_rgb(isic_images_dir() / f"{i}.jpg"), f"{i} clean"))
        panels_b = []
        for i in hairy_ids[:8]:
            panels_b.append((read_rgb(isic_images_dir() / f"{i}.jpg"), f"{i} RAW"))
            panels_b.append((read_rgb(processed_dir("B") / f"{i}.png"), f"{i} B"))
        contact_sheet(panels_screen, artifact_dir("panels") / "hair_screening_panel.jpg")
        contact_sheet(panels_b, artifact_dir("panels") / "dullrazor_panel.jpg")

    synthetic_rows = []
    if not args.skip_synthetic:
        with timed("synthetic_pairs", log) as t_syn:
            hair_absent = df[~df["hair_present"]]
            for split, limit in SYNTH_LIMITS.items():
                sub = [r for _, r in hair_absent[hair_absent["split"] == split].head(limit).iterrows()]
                with ThreadPoolExecutor(max_workers=32) as ex:
                    for row in ex.map(lambda r, s=split: synth_one(r, s), sub):
                        synthetic_rows.append(row)
                log.info("synthetic %s: %d pairs", split, len(sub))
        pd.DataFrame(synthetic_rows).to_csv(
            manifest_dir() / "synthetic_pairs.csv", index=False
        )
    else:
        t_syn = {"seconds": 0.0}

    summary = {
        "images": int(len(df)),
        "hair_present": int(df["hair_present"].sum()),
        "hair_present_by_split": df.groupby("split")["hair_present"].sum().to_dict(),
        "hair_present_by_class": df[df["hair_present"]].groupby("dx").size().to_dict(),
        "screen_config_hash": screen_cfg["config_hash"],
        "dullrazor_config_hash": dull_cfg["config_hash"],
        "synthetic_pairs": len(synthetic_rows),
        "synthetic_by_split": pd.Series([r["split"] for r in synthetic_rows]).value_counts().to_dict()
        if synthetic_rows
        else {},
        "seconds_process": t_proc["seconds"],
        "seconds_synthetic": t_syn["seconds"],
    }
    save_json(summary, artifact_dir("logs") / "hair_pipeline_summary.json")
    record_run("hair_pipeline", gpu=None, seconds=t_proc["seconds"] + t_syn["seconds"],
               hair_present=summary["hair_present"], synthetic=summary["synthetic_pairs"])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

