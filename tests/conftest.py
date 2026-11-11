from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import CLASSES  # noqa: E402


@pytest.fixture(scope="session")
def tiny_images(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("imgs")
    rng = np.random.default_rng(0)
    for i in range(6):
        arr = rng.integers(0, 255, size=(64, 48, 3), dtype=np.uint8)
        Image.fromarray(arr).save(root / f"ISIC_{i:07d}.jpg", quality=92)
    (root / "corrupt.jpg").write_bytes(b"not an image")
    return root


@pytest.fixture(scope="session")
def splits_df():
    import pandas as pd

    rows = []
    for i in range(60):
        rows.append({
            "image_id": f"ISIC_{i:07d}",
            "lesion_id": f"HAM_{i // 3:05d}",
            "group_id": f"G{i // 3:04d}",
            "dx": CLASSES[i % len(CLASSES)],
            "split": ["train", "val", "test"][(i // 3) % 3],
            "sha256": f"{i:064x}",
        })
    return pd.DataFrame(rows)

