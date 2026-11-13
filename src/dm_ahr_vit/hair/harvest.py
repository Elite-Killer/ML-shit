"""Harvest real hair from dermoscopic images as thin-structure masks.

The frozen detector (`screening.py`) is built to flag *whether* hair is present and is
deliberately generous: its mask is dilated and picks up lesion texture. For compositing real
hair onto clean images we need the opposite: only long, thin, dark strands. Components are
kept when their skeleton is long and their mean width is small, so blobs, ink dots, lesion
borders and short pigment-network segments are rejected.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .screening import hair_response

MIN_COMPONENT_AREA = 40
MIN_SKELETON_LENGTH = 40
MAX_MEAN_WIDTH = 7.0
MAX_FRACTION = 0.25


def thin_hair_mask(image_rgb: np.ndarray, cfg: dict,
                   min_skeleton: int = MIN_SKELETON_LENGTH,
                   max_width: float = MAX_MEAN_WIDTH) -> tuple[np.ndarray, dict]:
    """Return a uint8 {0,1} mask of long thin dark components and a stats dict."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    dark, _ = hair_response(gray, {**cfg, "light_hair": False})
    mask = (dark > int(cfg["blackhat_threshold"])).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    skel = skeletonize(mask > 0)
    skel_len = np.bincount(labels[skel], minlength=n)
    area = stats[:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if area[i] < MIN_COMPONENT_AREA or skel_len[i] < min_skeleton:
            continue
        if area[i] / float(skel_len[i]) <= max_width:
            keep[i] = True
    out = keep[labels].astype(np.uint8)
    fraction = float(out.mean())
    info = {"components_total": int(n - 1), "components_kept": int(keep.sum()),
            "hair_pixels": int(out.sum()), "fraction": fraction}
    if fraction > MAX_FRACTION:
        out = np.zeros_like(out)
        info.update(components_kept=0, hair_pixels=0, fraction=0.0, rejected="too_much")
    return out, info
