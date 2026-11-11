"""DM-AHR audit smoke test: official weights, one image, batch, reload, panel."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import load_config, save_json  # noqa: E402
from dm_ahr_vit.diffusion.dmahr_adapter import (  # noqa: E402
    CHECKPOINT_NAME,
    clean_batch,
    load_model,
    load_upstream_config,
)
from dm_ahr_vit.paths import artifact_dir, data_root, isic_images_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402

log = setup_logging()


def panel(rows: list[tuple[Image.Image, str]], out: Path) -> None:
    th = 192
    canvas = Image.new("RGB", (th * len(rows), th + 18), (0, 0, 0))
    from PIL import ImageDraw

    d = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(rows):
        canvas.paste(im.convert("RGB").resize((th, th), Image.Resampling.BICUBIC), (i * th, 0))
        d.text((i * th + 3, th + 3), label[:30], fill=(255, 255, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=90)


def main() -> int:
    cfg = load_config("configs/dmahr.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = data_root() / "raw" / "dmahr_weights" / CHECKPOINT_NAME
    if not ckpt.exists():
        raise FileNotFoundError(f"missing official checkpoint {ckpt}")

    screen = pd.read_csv(manifest_dir() / "hair_screening.csv")
    test_hairy = screen[(screen["split"] == "test") & (screen["hair_present"])]
    ids = test_hairy["image_id"].head(64).tolist()
    if len(ids) < 4:
        raise RuntimeError("not enough hair-present test images for smoke test")
    images = [Image.open(isic_images_dir() / f"{i}.jpg").convert("RGB") for i in ids]

    t0 = time.time()
    net = load_model(cfg, ckpt, device=device)
    load_seconds = time.time() - t0
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None

    t1 = time.time()
    single, stats1 = clean_batch(net, images[:1], cfg, device=device, seed=42)
    single_seconds = time.time() - t1

    t2 = time.time()
    batch, stats4 = clean_batch(net, images[:4], cfg, device=device, seed=42)
    batch_seconds = time.time() - t2

    # Batch-size sweep: picks the cheapest throughput for the full Arm C run.
    bench = {}
    for bs in (8, 16, 32, 64):
        if len(images) < bs:
            continue
        tb = time.time()
        outs, s = clean_batch(net, images[:bs], cfg, device=device, seed=123)
        secs = time.time() - tb
        bench[bs] = {
            "seconds": round(secs, 2),
            "returned": s["batch_returned"],
            "seconds_per_image": round(secs / bs, 3),
        }
        if bs == 64 or (bs == 32 and len(images) < 64):
            batch = outs

    # Reload equivalence: same weights, same seed -> identical output.
    net2 = load_model(cfg, ckpt, device=device)
    again, _ = clean_batch(net2, images[:1], cfg, device=device, seed=42)
    reload_identical = bool(np.array_equal(np.array(single[0]), np.array(again[0])))

    # Geometry and range checks.
    checks = {
        "single_shape": list(np.array(single[0]).shape),
        "batch_shape": [len(batch), *np.array(batch[0]).shape],
        "geometry_preserved": all(np.array(im).shape == (128, 128, 3) for im in batch),
        "finite": stats4["finite"],
        "range_ok": all(0 <= np.array(im).min() and np.array(im).max() <= 255 for im in batch),
        "reload_identical": reload_identical,
        "single_seconds": round(single_seconds, 2),
        "batch4_seconds": round(batch_seconds, 2),
        "seconds_per_image_batch4": round(batch_seconds / 4, 2),
        "batch_sweep": bench,
        "load_seconds": round(load_seconds, 2),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if device == "cuda"
        else None,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "num_timesteps": stats4["num_timesteps"],
        "checkpoint": str(ckpt),
        "checkpoint_sha256": __import__("hashlib").sha256(ckpt.read_bytes()).hexdigest(),
        "upstream_config": load_upstream_config(),
    }
    checks["all_pass"] = bool(
        checks["geometry_preserved"] and checks["finite"] and checks["range_ok"]
        and checks["reload_identical"]
    )

    rows = []
    for i, (src, out) in enumerate(zip(images[:4], batch)):
        rows.append((src, f"{ids[i]} input"))
        rows.append((out, f"{ids[i]} DM-AHR"))
    panel(rows, artifact_dir("panels") / "dmahr_smoke_panel.jpg")
    save_json(checks, artifact_dir("logs") / "dmahr_smoke.json")
    record_run("dmahr_smoke", gpu=checks["device"], seconds=single_seconds + batch_seconds,
               seconds_per_image=checks["seconds_per_image_batch4"])
    print(json.dumps(checks, indent=2, default=str))
    return 0 if checks["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

