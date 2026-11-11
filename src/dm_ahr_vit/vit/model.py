"""torchvision ViT-B/16 with a fixed seven-class head."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from ..config import CLASSES


def build_model(pretrained: bool = True, num_classes: int = len(CLASSES)) -> nn.Module:
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def save_checkpoint(
    path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_f1: float,
    meta: dict,
) -> None:
    tmp = str(path) + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val_macro_f1": best_f1,
            "meta": meta,
        },
        tmp,
    )
    import os

    os.replace(tmp, path)


def load_checkpoint(path, model: nn.Module, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt


@torch.no_grad()
def predict_proba(model: nn.Module, loader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probs, labels = [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        labels.append(batch["label"])
    return torch.cat(probs), torch.cat(labels)

