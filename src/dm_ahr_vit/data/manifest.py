"""Per-image manifest, dHash near-duplicate candidates, pixel confirmation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..config import sha256_file

_POPCOUNT_8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(img: Image.Image) -> int:
    """64-bit difference hash: 9x8 grayscale, adjacent-column differences."""
    gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    px = np.asarray(gray, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_pairs(hashes: np.ndarray, threshold: int, ids: list[str]) -> list[dict]:
    """All pairs within `threshold` bits (exact, chunked to bound memory)."""
    hashes = np.asarray(hashes, dtype=np.uint64)
    out: list[dict] = []
    block = 1024
    for start in range(0, len(hashes), block):
        stop = min(start + block, len(hashes))
        block_hashes = hashes[start:stop]
        # pairs (i, j) with i < j globally; iterate rows in this block
        for i in range(stop - start):
            xor = block_hashes[i] ^ hashes[start + i + 1:]
            dist = _popcount_rows(xor)
            for k in np.where(dist <= threshold)[0]:
                out.append({"id_a": ids[start + i], "id_b": ids[start + i + 1 + int(k)], "distance": int(dist[k])})
    return out


def _popcount_rows(values: np.ndarray) -> np.ndarray:
    """Popcount for uint64 values via an 8-bit lookup table."""
    if len(values) == 0:
        return np.zeros(0, dtype=np.int64)
    return _POPCOUNT_8[values.view(np.uint8).reshape(len(values), -1)].sum(axis=1)


def _describe(path: Path, image_id: str) -> dict:
    """Decode one image and record geometry, dhash and sha256. Never raises."""
    row = {
        "image_id": image_id,
        "path": str(path),
        "decode_status": "fail_decode",
        "width": None,
        "height": None,
        "mode": None,
        "format": None,
        "dhash": None,
        "bytes": None,
        "sha256": None,
    }
    if not path.exists():
        row["decode_status"] = "missing"
        return row
    row["bytes"] = path.stat().st_size
    try:
        with Image.open(path) as im:
            im.load()
            row["width"], row["height"] = im.size
            row["mode"], row["format"] = im.mode, im.format
            row["dhash"] = dhash(im)
    except Exception:  # noqa: BLE001
        return row
    row["decode_status"] = "ok"
    row["sha256"] = sha256_file(path)
    return row


def build_manifest(
    records_path: Path,
    images_dir: Path,
    metadata: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Join download records with metadata; quarantine C0-extra images.

    cohort "C0" marks the original 10,015 HAM10000 images; every other
    collection image is quarantined as "C0-extra" and never enters a split.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    meta = metadata.set_index("image_id")
    rows: list[dict] = []
    for rec in records:
        image_id = rec["isic_id"]
        path = images_dir / f"{image_id}.jpg"
        desc = _describe(path, image_id)
        in_c0 = str(image_id) in meta.index
        if in_c0:
            cohort = "C0" if desc["decode_status"] == "ok" else "C0-bad"
            dx, lesion_id = str(meta.loc[image_id, "dx"]), str(meta.loc[image_id, "lesion_id"])
        else:
            cohort, dx, lesion_id = "C0-extra", None, None
        rows.append({
            "image_id": image_id,
            "path": str(path),
            "in_c0": in_c0,
            "cohort": cohort,
            "license": rec.get("license"),
            "attribution": rec.get("attribution"),
            "expected_bytes": rec.get("expected_size"),
            "download_status": rec.get("status", "unknown"),
            "dx": dx,
            "lesion_id": lesion_id,
            **{k: desc[k] for k in ("width", "height", "mode", "format", "decode_status", "dhash", "bytes", "sha256")},
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "data_manifest.csv", index=False)

    excluded = manifest[manifest["cohort"] != "C0"].copy()
    excluded["reason"] = "c0_extra_quarantined"
    exclusions = excluded[["image_id", "cohort", "reason", "path"]]
    exclusions.to_csv(out_dir / "exclusions.csv", index=False)

    stats = {
        "records": len(manifest),
        "c0_ok": int(((manifest["cohort"] == "C0") & (manifest["decode_status"] == "ok")).sum()),
        "c0_extra": int((manifest["cohort"] == "C0-extra").sum()),
        "failed_decode": int((manifest["decode_status"] != "ok").sum()),
    }
    return manifest, exclusions, stats


def confirm_near_duplicates(
    candidates: pd.DataFrame,
    manifest: pd.DataFrame,
    max_mean_abs_diff: float = 4.0,
    min_correlation: float = 0.985,
    compare_size: int = 256,
) -> pd.DataFrame:
    """Pixel-confirm dHash candidate pairs on 256x256 grayscale thumbnails."""
    by_id = manifest.set_index("image_id")["path"].to_dict()

    def thumb(image_id: str) -> np.ndarray | None:
        path = by_id.get(image_id)
        if path is None or not Path(path).exists():
            return None
        with Image.open(path) as im:
            return np.asarray(im.convert("L").resize((compare_size, compare_size), Image.Resampling.BILINEAR), dtype=np.float64)

    rows = []
    for r in candidates.itertuples(index=False):
        a_id, b_id, dist = r.id_a, r.id_b, r.distance
        a, b = thumb(a_id), thumb(b_id)
        if a is None or b is None:
            rows.append({"id_a": a_id, "id_b": b_id, "dhash_distance": int(dist),
                         "mean_abs_diff": float("nan"), "correlation": float("nan"),
                         "confirmed": False})
            continue
        diff = float(np.abs(a - b).mean())
        corr = float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])
        rows.append({
            "id_a": a_id, "id_b": b_id, "dhash_distance": int(dist),
            "mean_abs_diff": round(diff, 3), "correlation": round(corr, 6),
            "confirmed": bool(diff <= max_mean_abs_diff and corr >= min_correlation),
        })
    return pd.DataFrame(rows)
