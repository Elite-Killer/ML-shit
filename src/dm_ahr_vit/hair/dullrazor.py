"""DullRazor-style morphology + Telea inpainting baseline (Arm B)."""

from __future__ import annotations

import cv2
import numpy as np


def apply_dullrazor(image_rgb: np.ndarray, hair_mask: np.ndarray, cfg: dict) -> np.ndarray:
    mask = (hair_mask > 0).astype(np.uint8) * 255
    if cfg.get("mask_dilate_kernel", 0):
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (int(cfg["mask_dilate_kernel"]), int(cfg["mask_dilate_kernel"])),
        )
        mask = cv2.dilate(mask, k, iterations=int(cfg.get("mask_dilate_iterations", 1)))
    method = cv2.INPAINT_TELEA if cfg.get("inpaint_method", "telea") == "telea" else cv2.INPAINT_NS
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, mask, float(cfg.get("inpaint_radius", 3)), method)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

