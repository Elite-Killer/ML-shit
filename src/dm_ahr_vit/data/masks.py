"""Lesion-mask manifests and binary mask loading (X1 released segmentations)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .ham10000 import MASKS_PROVENANCE


def build_x1_manifest(
    masks_dir: Path,
    split_df: pd.DataFrame,
    manifest: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """Join split rows with released segmentation files and QA each mask."""
    dims = manifest.set_index("image_id")[["width", "height"]].to_dict("index")
    rows = []
    for r in split_df.itertuples(index=False):
        image_id = str(r.image_id)
        mask_path = masks_dir / f"{image_id}_segmentation.png"
        row = {
            "image_id": image_id,
            "tier": "X1",
            "split": r.split,
            "dx": r.dx,
            "group_id": r.group_id,
            "has_mask": bool(mask_path.exists()),
            "mask_path": str(mask_path),
            "provenance": MASKS_PROVENANCE,
            "expert_traced_claimed": False,
            "width": None,
            "height": None,
            "binary": False,
            "non_empty": False,
            "foreground_fraction": 0.0,
            "size_matches": False,
            "status": "missing",
        }
        if row["has_mask"]:
            try:
                with Image.open(mask_path) as im:
                    arr = np.asarray(im.convert("L"))
                binary = (arr > 127).astype(np.uint8)
                row.update({
                    "width": arr.shape[1],
                    "height": arr.shape[0],
                    "binary": bool(len(np.unique(binary)) <= 2),
                    "non_empty": bool(binary.sum() > 0),
                    "foreground_fraction": float(binary.mean()),
                    "status": "ok",
                })
                img_dims = dims.get(image_id)
                row["size_matches"] = bool(
                    img_dims
                    and img_dims.get("width") == row["width"]
                    and img_dims.get("height") == row["height"]
                )
            except Exception:  # noqa: BLE001
                row["status"] = "bad_mask"
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def load_mask_224(path: Path, size: int = 224) -> np.ndarray:
    """Load a segmentation as a binary {0,1} uint8 array, nearest-neighbour."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    binary = Image.fromarray(((arr > 127).astype(np.uint8)) * 255)
    if binary.size != (size, size):
        binary = binary.resize((size, size), Image.Resampling.NEAREST)
    return (np.asarray(binary) > 0).astype(np.uint8)
