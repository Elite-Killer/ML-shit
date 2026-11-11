"""Download ISIC collection 212, HAM10000 metadata, and released lesion masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import save_json, sha256_file  # noqa: E402
from dm_ahr_vit.data import ham10000, isic  # noqa: E402
from dm_ahr_vit.paths import ensure_dirs, ham_masks_dir, ham_root, isic_images_dir, manifest_dir, raw_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()


def fetch(url: str, dest: Path) -> dict:
    if dest.exists() and dest.stat().st_size > 0:
        return {"path": str(dest), "sha256": sha256_file(dest), "cached": True}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
    return {"path": str(dest), "sha256": sha256_file(dest), "cached": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", type=int, default=212)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    ensure_dirs()

    with timed("download_isic", log) as t:
        stats = isic.download_collection(
            args.collection,
            isic_images_dir(),
            raw_dir() / "isic_212" / "records.jsonl",
            workers=args.workers,
            limit=args.limit,
        )
    log.info("ISIC download: %s", stats)

    meta = fetch(
        ham10000.DATAVERSE_ACCESS.format(fid=ham10000.METADATA_FILE_ID),
        ham_root() / "HAM10000_metadata.tab",
    )
    masks_zip = fetch(
        ham10000.DATAVERSE_ACCESS.format(fid=ham10000.MASKS_FILE_ID),
        ham_root() / "HAM10000_segmentations_lesion_tschandl.zip",
    )
    extracted = ham10000.extract_masks(Path(masks_zip["path"]), ham_masks_dir())

    report = {
        "isic": stats,
        "dataverse_doi": ham10000.DATAVERSE_DOI,
        "metadata": meta,
        "masks_zip": masks_zip,
        "masks_extracted": extracted,
        "seconds": t["seconds"],
    }
    save_json(report, manifest_dir() / "download_report.json")
    record_run("download_data", gpu=None, seconds=t["seconds"], **stats)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

