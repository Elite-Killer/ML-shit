"""Filesystem layout. Large data lives on Modal Volumes, never in Git."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data")).resolve()


def artifact_root() -> Path:
    return Path(os.environ.get("ARTIFACT_ROOT", REPO_ROOT / "artifacts")).resolve()


def raw_dir() -> Path:
    return data_root() / "raw"


def isic_images_dir() -> Path:
    return raw_dir() / "isic_212" / "images"


def ham_root() -> Path:
    return raw_dir() / "ham10000"


def ham_masks_dir() -> Path:
    return ham_root() / "masks"


def processed_dir(arm: str) -> Path:
    return data_root() / "processed" / arm


def manifest_dir() -> Path:
    return data_root() / "manifests"


def artifact_dir(*parts: str) -> Path:
    p = artifact_root().joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dirs() -> None:
    for p in (
        raw_dir(),
        isic_images_dir(),
        ham_root(),
        ham_masks_dir(),
        manifest_dir(),
        artifact_dir("logs"),
        artifact_dir("checkpoints"),
        artifact_dir("predictions"),
        artifact_dir("figures"),
        artifact_dir("panels"),
        artifact_dir("xai_maps"),
    ):
        p.mkdir(parents=True, exist_ok=True)

