"""Deterministic synthetic hair rendering for paired hairy/hairless images."""

from __future__ import annotations

import hashlib

import cv2
import numpy as np
from PIL import Image


def _rng_for(image_id: str, seed: int) -> np.random.Generator:
    """Stable per-image seed: PYTHONHASHSEED-independent."""
    digest = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def render_hair(img: Image.Image, image_id: str, seed: int):
    """Render synthetic hairs over a clean image.

    Returns (hairy PIL image, hair mask PIL image with {0,255} values,
    params dict). Fully deterministic given (image_id, seed).
    """
    rng = _rng_for(image_id, seed)
    base = np.asarray(img.convert("RGB"), dtype=np.float64)
    h, w = base.shape[:2]

    n_hairs = int(rng.integers(8, 18))
    blur = float(round(rng.uniform(0.4, 1.2), 3))
    alpha_layer = np.zeros((h, w), dtype=np.float64)
    color_layer = np.zeros((h, w, 3), dtype=np.float64)
    for _ in range(n_hairs):
        thickness = int(rng.integers(1, 4))
        darkness = int(rng.integers(20, 90))
        color = np.array([darkness, int(darkness * 0.6), int(darkness * 0.45)], dtype=np.float64)
        alpha = float(rng.uniform(0.55, 0.85))
        # gentle curved stroke: control points walked across the image
        edge = int(rng.integers(0, 4))
        if edge == 0:  # top-left region heading down-right
            pts = [(int(rng.uniform(0, w * 0.3)), int(rng.uniform(0, h * 0.2)))]
        elif edge == 1:
            pts = [(int(rng.uniform(0, w * 0.3)), int(rng.uniform(h * 0.8, h)))]
        elif edge == 2:
            pts = [(int(rng.uniform(w * 0.7, w)), int(rng.uniform(0, h * 0.2)))]
        else:
            pts = [(int(rng.uniform(w * 0.7, w)), int(rng.uniform(h * 0.8, h)))]
        steps = int(rng.integers(3, 6))
        dx = float(rng.uniform(-0.3, 0.3)) * w / steps
        dy = float(rng.uniform(0.15, 0.35)) * h / steps
        for s in range(1, steps + 1):
            jitter = rng.uniform(-0.05, 0.05) * w
            pts.append((int(pts[-1][0] + dx + jitter), int(pts[-1][1] + dy + jitter * 0.4)))
        pts_i = np.array(pts, dtype=np.int32)
        canvas = np.zeros((h, w), dtype=np.float64)
        cv2.polylines(canvas, [pts_i], False, 1.0, thickness, cv2.LINE_AA)
        alpha_layer = np.maximum(alpha_layer, canvas * alpha)
        color_layer = np.where(canvas[..., None] > 0, color, color_layer)

    if blur > 0:
        alpha_layer = cv2.GaussianBlur(alpha_layer, (0, 0), blur)

    a = alpha_layer[..., None]
    hairy = np.clip(base * (1 - a) + color_layer * a, 0, 255).astype(np.uint8)
    hair_mask = ((alpha_layer > 0.1) * 255).astype(np.uint8)
    params = {"n_hairs": n_hairs, "blur": blur, "seed": int(seed)}
    return Image.fromarray(hairy), Image.fromarray(hair_mask), params
