"""Multi-orientation black-hat/top-hat hair detector.

Parameters are tuned on train/validation synthetic pairs only and frozen before
any test image is processed.
"""

from __future__ import annotations

import cv2
import numpy as np


def _oriented_kernel(length: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.uint8)
    c = length // 2
    rad = np.deg2rad(angle_deg)
    dx, dy = int(round(c * np.cos(rad))), int(round(c * np.sin(rad)))
    cv2.line(kernel, (c - dx, c - dy), (c + dx, c + dy), 1, 1)
    return kernel


def hair_response(gray: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    dark = np.zeros_like(gray)
    light = np.zeros_like(gray)
    for i in range(int(cfg["orientations"])):
        angle = 180.0 * i / int(cfg["orientations"])
        k = _oriented_kernel(int(cfg["kernel_length"]), angle)
        dark = np.maximum(dark, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k))
        if cfg.get("light_hair", True):
            light = np.maximum(light, cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k))
    return dark, light


def detect_hair(image_rgb: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    """Return a binary hair mask and detection statistics."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    dark, light = hair_response(gray, cfg)
    mask = (dark > int(cfg["blackhat_threshold"])).astype(np.uint8)
    if cfg.get("light_hair", True):
        mask |= (light > int(cfg["tophat_threshold"])).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= int(cfg["min_component_area"]):
            keep[labels == i] = 1
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(cfg["dilate_kernel"]), int(cfg["dilate_kernel"]))
    )
    keep = cv2.dilate(keep, k, iterations=int(cfg["dilate_iterations"]))
    coverage = float(keep.mean())
    stats_out = {
        "coverage": coverage,
        "hair_present": bool(coverage >= float(cfg["coverage_threshold"])),
        "dark_max": int(dark.max()),
        "light_max": int(light.max()),
    }
    return keep, stats_out


def tune_coverage_threshold(
    pairs: list[tuple[np.ndarray, int]], cfg: dict, grid: np.ndarray | None = None
) -> dict:
    """Pick the coverage threshold maximizing balanced accuracy on labelled pairs.

    `pairs` is [(image_rgb, is_hairy)] from train/validation synthetic pairs.
    """
    grid = grid if grid is not None else np.round(np.arange(0.002, 0.081, 0.002), 4)
    coverages = np.array([detect_hair(img, cfg)[1]["coverage"] for img, _ in pairs])
    labels = np.array([lab for _, lab in pairs])
    best = None
    for thr in grid:
        pred = (coverages >= thr).astype(int)
        tpr = float((pred[labels == 1] == 1).mean()) if (labels == 1).any() else 0.0
        tnr = float((pred[labels == 0] == 0).mean()) if (labels == 0).any() else 0.0
        score = 0.5 * (tpr + tnr)
        if best is None or score > best["balanced_accuracy"]:
            best = {
                "coverage_threshold": float(thr),
                "balanced_accuracy": score,
                "sensitivity": tpr,
                "specificity": tnr,
            }
    return best

