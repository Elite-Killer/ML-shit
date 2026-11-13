"""The one place that defines the model's view of an image and its lesion mask.

The ViT never sees the stored photo: it sees the photo resized so the shorter side is
``int(size * 1.14)`` and then centre-cropped to ``size`` x ``size``. Saliency maps live in that
cropped frame, so a lesion mask has to be carried through the identical resize and crop before
it is compared with a map. The classifier's evaluation transform (``data.dataset``) and every
explanation script call the functions below, so the two cannot drift apart.

Stored image sizes differ by arm: the original and DullRazor arms keep the 600 x 450 photo; the
diffusion-based arms and the 128 px control are stored as 128 x 128 (an aspect-ratio squash).
Masks are stored at 600 x 450 and are squashed to the arm's stored size first.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

EVAL_SCALE = 1.14
ARM_STORED_SIZE: dict[str, tuple[int, int]] = {
    "A": (600, 450), "B": (600, 450),
    "Astar": (128, 128), "C": (128, 128), "Creleased": (128, 128), "Cv2": (128, 128),
}
FRAME_VERSION = "model_frame_v2"


def eval_resize_size(size: int = 224) -> int:
    return int(size * EVAL_SCALE)


def eval_ops(size: int = 224, interpolation=InterpolationMode.BILINEAR) -> list:
    """The resize and centre-crop of the classifier's evaluation transform."""
    return [transforms.Resize(eval_resize_size(size), interpolation=interpolation),
            transforms.CenterCrop(size)]


def to_model_frame(img: Image.Image, size: int = 224, is_mask: bool = False) -> Image.Image:
    """Carry a PIL image (or mask, with nearest-neighbour resampling) into the model's frame."""
    mode = InterpolationMode.NEAREST if is_mask else InterpolationMode.BILINEAR
    for op in eval_ops(size, mode):
        img = op(img)
    return img


def mask_to_model_frame(mask: Image.Image, stored_size_wh: tuple[int, int] | None,
                        size: int = 224) -> np.ndarray:
    """Binary {0,1} lesion mask in the model's frame, for an arm whose images are stored at
    ``stored_size_wh``. ``None`` means the mask is already at the stored size."""
    m = mask.convert("L")
    if stored_size_wh is not None and m.size != tuple(stored_size_wh):
        m = m.resize(tuple(int(v) for v in stored_size_wh), Image.Resampling.NEAREST)
    m = to_model_frame(m, size, is_mask=True)
    return (np.asarray(m) > 127).astype(np.uint8)


def shift_zero_fill(a, dx: int, dy: int):
    """Translate an (..., H, W) numpy array or tensor by (dx, dy) pixels, filling with zeros."""
    is_t = isinstance(a, torch.Tensor)
    out = torch.zeros_like(a) if is_t else np.zeros_like(a)
    h, w = a.shape[-2], a.shape[-1]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    if abs(dx) < w and abs(dy) < h:
        out[..., yd, xd] = a[..., ys, xs]
    return out
