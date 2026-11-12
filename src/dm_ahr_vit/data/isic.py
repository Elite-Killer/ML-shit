"""ISIC archive collection download with per-image records."""

from __future__ import annotations

import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

API_BASE = "https://api.isic-archive.com/api/v2"


def _list_images(collection_id: int, limit: int | None) -> list[dict]:
    """Page through the v2 image list for one collection."""
    items: list[dict] = []
    page_size = 100
    offset = 0
    while True:
        params = {"collection_ids": collection_id, "limit": page_size, "offset": offset}
        r = requests.get(f"{API_BASE}/images/", params=params, timeout=120)
        r.raise_for_status()
        page = r.json()
        batch = page.get("results", [])
        items.extend(batch)
        if limit is not None and len(items) >= limit:
            return items[:limit]
        if len(batch) < page_size or not batch:
            return items
        offset += page_size


def _download_one(item: dict, images_dir: Path) -> dict:
    image_id = item["isic_id"]
    dest = images_dir / f"{image_id}.jpg"
    record = {
        "isic_id": image_id,
        "license": item.get("license", {}).get("name") if isinstance(item.get("license"), dict) else item.get("license"),
        "attribution": item.get("attribution"),
        "url": item.get("url") or f"{API_BASE}/images/{image_id}/download",
        "expected_size": item.get("expected_size"),
        "status": "downloaded",
    }
    if dest.exists() and dest.stat().st_size > 0:
        return record
    r = requests.get(f"{API_BASE}/images/{image_id}/download", timeout=300)
    r.raise_for_status()
    record["expected_size"] = item.get("expected_size", len(r.content))
    record["bytes"] = len(r.content)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    record["sha256"] = hashlib.sha256(r.content).hexdigest()
    return record


def download_collection(
    collection_id: int,
    images_dir: Path,
    records_path: Path,
    workers: int = 16,
    limit: int | None = None,
) -> dict:
    """Download a collection's images and write one JSON record per line."""
    images_dir.mkdir(parents=True, exist_ok=True)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    items = _list_images(collection_id, limit)
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for record in ex.map(lambda it: _download_one(it, images_dir), items):
            records.append(record)
    with records_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    status: dict[str, int] = {}
    for record in records:
        status[record["status"]] = status.get(record["status"], 0) + 1
    return {"collection_id": collection_id, "records": len(records), "status": status}
