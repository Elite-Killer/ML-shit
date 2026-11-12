"""Torch Dataset and transforms shared by every arm and seed."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..config import CLASS_TO_IDX

# ImageNet statistics; matches ViT_B_16_Weights.IMAGENET1K_V1 preprocessing.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(cfg: dict, train: bool):
    """Deterministic eval transform; matched train-only augmentation."""
    size = int(cfg["image_size"])
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ])
    aug = cfg.get("augmentation", {})
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size,
            scale=tuple(aug.get("random_resized_crop_scale", [0.75, 1.0])),
            ratio=tuple(aug.get("random_resized_crop_ratio", [0.9, 1.1111111111111112])),
        ),
        transforms.RandomHorizontalFlip(float(aug.get("horizontal_flip", 0.5))),
        transforms.RandomVerticalFlip(float(aug.get("vertical_flip", 0.5))),
        transforms.RandomRotation(float(aug.get("rotation_degrees", 15))),
        transforms.ColorJitter(*aug.get("color_jitter", [0.15, 0.15, 0.1, 0.02])),
        transforms.ToTensor(),
        normalize,
    ])


class ArmDataset(Dataset):
    """One arm's images for one split, in splits.csv row order.

    Arm A reads the raw ISIC JPEGs; every processed arm reads
    processed_root/<arm>/<image_id>.png (hair-absent images are identical
    copies in every arm, per protocol).
    """

    def __init__(
        self,
        arm: str,
        split: str,
        splits_df,
        raw_images_dir: Path,
        processed_root: Path,
        transform,
        image_ids=None,
    ) -> None:
        rows = splits_df[splits_df["split"] == split]
        if image_ids is not None:
            rows = rows[rows["image_id"].isin(set(image_ids))]
        self.rows = rows.reset_index(drop=True)
        self.arm = str(arm)
        self.raw_dir = Path(raw_images_dir)
        self.processed_root = Path(processed_root)
        self.transform = transform

    def _path(self, image_id: str) -> Path:
        if self.arm == "A":
            return self.raw_dir / f"{image_id}.jpg"
        preferred = self.processed_root / self.arm / f"{image_id}.png"
        if preferred.exists():
            return preferred
        return self.processed_root / self.arm / f"{image_id}.jpg"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        r = self.rows.iloc[index]
        image_id = str(r["image_id"])
        with Image.open(self._path(image_id)) as im:
            x = self.transform(im.convert("RGB"))
        return {
            "image": x,
            "label": int(CLASS_TO_IDX[str(r["dx"])]),
            "image_id": image_id,
            "group_id": str(r["group_id"]),
            "dx": str(r["dx"]),
        }


def collate(batch: list[dict]) -> dict:
    """Stack images and labels; keep string columns as lists."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "image_id": [b["image_id"] for b in batch],
        "group_id": [b["group_id"] for b in batch],
        "dx": [b["dx"] for b in batch],
    }
