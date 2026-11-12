"""HAM10000 metadata, class accounting, Dataverse access and mask extraction."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from ..config import CLASSES, EXPECTED_C0_COUNTS, EXPECTED_C0_TOTAL

# Harvard Dataverse dataset for HAM10000. File IDs verified against the
# dataset's file listing (doi:10.7910/DVN/DBW86T).
DATAVERSE_DOI = "doi:10.7910/DVN/DBW86T"
DATAVERSE_ACCESS = "https://dataverse.harvard.edu/api/access/datafile/{fid}"
METADATA_FILE_ID = 4338392  # HAM10000_metadata.tab
MASKS_FILE_ID = 3838943  # HAM10000_segmentations_lesion_tschandl.zip
MASKS_PROVENANCE = "HAM10000_segmentations_lesion_tschandl (Harvard Dataverse doi:10.7910/DVN/DBW86T)"


def read_metadata(path: Path) -> pd.DataFrame:
    """Read HAM10000_metadata.tab (tab-separated, one row per image)."""
    df = pd.read_csv(path, sep="\t", dtype={"lesion_id": str, "image_id": str})
    if "image_id" not in df.columns or "dx" not in df.columns:
        raise ValueError(f"unexpected metadata columns: {list(df.columns)}")
    return df


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    """Counts per diagnostic class, including zero-count classes."""
    return {c: int((df["dx"] == c).sum()) for c in CLASSES}


def expected_comparison(metadata: pd.DataFrame) -> dict:
    """Compare recomputed C0 class counts with the data descriptor's values."""
    counts = class_counts(metadata)
    deltas = {c: counts[c] - EXPECTED_C0_COUNTS[c] for c in CLASSES}
    total = int(len(metadata))
    return {
        "counts": counts,
        "count_deltas": deltas,
        "total": total,
        "expected_total": EXPECTED_C0_TOTAL,
        "all_match": total == EXPECTED_C0_TOTAL and all(d == 0 for d in deltas.values()),
    }


def extract_masks(zip_path: Path, masks_dir: Path) -> dict:
    """Extract released segmentation PNGs into a flat directory.

    The source does not state that the masks were expert-drawn, so they are
    labelled X1 released masks, never "expert-traced".
    """
    masks_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/") or not name.endswith("_segmentation.png"):
                continue
            target = masks_dir / Path(name).name
            if not target.exists():
                with zf.open(name) as src, target.open("wb") as dst:
                    dst.write(src.read())
            moved += 1
    return {"dest": str(masks_dir), "masks": moved}
