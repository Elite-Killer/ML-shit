"""The lesion mask must be carried through the same geometry as the model input."""

import numpy as np
import pytest
import torch
from PIL import Image

from dm_ahr_vit.config import load_config
from dm_ahr_vit.data.dataset import IMAGENET_MEAN, IMAGENET_STD, build_transforms
from dm_ahr_vit.data.masks import load_mask_224, load_mask_registered
from dm_ahr_vit.xai.vit_xai import (
    lesion_overlap,
    salient_in_lesion,
    top_region_centroid_distance,
)

CFG = load_config("configs/vit.json")


def _frame(size_wh, box_frac):
    """An image with a bright box at fractional coordinates, and the matching full-frame mask."""
    w, h = size_wh
    x0, y0, x1, y1 = (int(box_frac[0] * w), int(box_frac[1] * h),
                      int(box_frac[2] * w), int(box_frac[3] * h))
    img = np.zeros((h, w, 3), np.uint8)
    img[y0:y1, x0:x1] = 255
    return Image.fromarray(img), (x0, y0, x1, y1)


def _model_input_box(img: Image.Image) -> np.ndarray:
    t = build_transforms(CFG, train=False)(img)
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return ((t * std + mean).mean(0).numpy() > 0.5)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    return float((a & b).sum() / max((a | b).sum(), 1))


@pytest.mark.parametrize("size_wh", [(600, 450), (128, 128)])
def test_registered_mask_matches_the_model_input(tmp_path, size_wh):
    # a lesion-like box, centred horizontally and low in the frame, so a crop matters
    img, _ = _frame(size_wh, (0.30, 0.35, 0.70, 0.80))
    full_mask_wh = (600, 450)                       # masks are stored at the original size
    mask_img, _ = _frame(full_mask_wh, (0.30, 0.35, 0.70, 0.80))
    p = tmp_path / "m.png"
    mask_img.convert("L").save(p)
    truth = _model_input_box(img)
    registered = load_mask_registered(p, img.size, 224)
    assert _iou(registered, truth) > 0.93


def test_legacy_mask_is_misregistered_for_full_size_images(tmp_path):
    img, _ = _frame((600, 450), (0.30, 0.35, 0.70, 0.80))
    mask_img, _ = _frame((600, 450), (0.30, 0.35, 0.70, 0.80))
    p = tmp_path / "m.png"
    mask_img.convert("L").save(p)
    truth = _model_input_box(img)
    legacy = load_mask_224(p, 224)
    assert _iou(legacy, truth) < 0.6                 # documents the defect in the recorded runs
    assert legacy.mean() < truth.mean()              # and shows the mask was shrunk toward the centre


def test_precision_and_recall_definitions_differ_as_documented():
    sal = np.zeros((10, 10))
    sal[:2, :] = 1.0                                 # top 20% = the first two rows (20 pixels)
    mask = np.zeros((10, 10), np.uint8)
    mask[:4, :] = 1                                  # lesion = first four rows (40 pixels)
    assert salient_in_lesion(sal, mask, 0.2) == 1.0  # all salient pixels are inside the lesion
    assert lesion_overlap(sal, mask, 0.2) == 0.5     # but they cover half of the lesion
    small = np.zeros((10, 10), np.uint8)
    small[:1, :] = 1
    assert salient_in_lesion(sal, small, 0.2) == 0.5
    assert lesion_overlap(sal, small, 0.2) == 1.0


def test_centroid_distance_zero_when_top_region_sits_on_the_lesion():
    sal = np.zeros((20, 20))
    sal[5:10, 5:10] = 1.0
    mask = np.zeros((20, 20), np.uint8)
    mask[5:10, 5:10] = 1
    assert top_region_centroid_distance(sal, mask, 25 / 400) == pytest.approx(0.0, abs=1e-6)
    mask2 = np.zeros((20, 20), np.uint8)
    mask2[5:10, 10:15] = 1
    assert top_region_centroid_distance(sal, mask2, 25 / 400) == pytest.approx(5 / 20, abs=1e-6)
