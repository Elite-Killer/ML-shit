"""Matched ViT-B/16 training with resumable checkpoints and early stopping."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from ..config import CLASS_TO_IDX, CLASSES, save_json
from ..data.dataset import ArmDataset, build_transforms, collate
from ..utils import setup_logging
from .model import build_model, save_checkpoint

log = setup_logging()


def class_weights(splits_df, class_order=CLASSES) -> torch.Tensor:
    """Inverse-frequency weights computed from TRAIN labels only."""
    counts = splits_df[splits_df["split"] == "train"]["dx"].value_counts()
    total = counts.sum()
    w = [total / (len(class_order) * max(counts.get(c, 0), 1)) for c in class_order]
    return torch.tensor(w, dtype=torch.float32)


def cosine_with_warmup(optimizer, warmup_epochs: int, total_epochs: int, steps_per_epoch: int):
    warmup = max(1, warmup_epochs * steps_per_epoch)
    total = max(warmup + 1, total_epochs * steps_per_epoch)

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate(model, loader, device: str, criterion=None) -> dict:
    model.eval()
    probs, labels = [], []
    loss_sum, n = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
        if criterion is not None:
            loss_sum += float(criterion(logits.float(), y)) * y.numel()
            n += y.numel()
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        labels.append(y.cpu())
    prob = torch.cat(probs)
    label = torch.cat(labels)
    pred = prob.argmax(dim=1)
    return {
        "prob": prob,
        "label": label,
        "pred": pred,
        "loss": loss_sum / max(n, 1),
        "macro_f1": float(f1_score(label, pred, average="macro", zero_division=0)),
        "accuracy": float((pred == label).float().mean()),
    }


def train_arm(
    arm: str,
    seed: int,
    cfg: dict,
    splits_df,
    raw_images_dir: Path,
    processed_root: Path,
    run_dir: Path,
    device: str = "cuda",
    max_epochs: int | None = None,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_ds = ArmDataset(arm, "train", splits_df, raw_images_dir, processed_root,
                          build_transforms(cfg, train=True))
    val_ds = ArmDataset(arm, "val", splits_df, raw_images_dir, processed_root,
                        build_transforms(cfg, train=False))
    bs = int(cfg["batch_size"])
    nw = int(cfg["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False,
                              num_workers=nw, pin_memory=True, collate_fn=collate,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                            pin_memory=True, collate_fn=collate, persistent_workers=nw > 0)

    model = build_model(pretrained=True).to(device)
    weights = class_weights(splits_df).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=float(cfg.get("label_smoothing", 0.0))
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )
    epochs = int(max_epochs or cfg["epochs"])
    scheduler = cosine_with_warmup(optimizer, int(cfg["warmup_epochs"]), epochs, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    meta = {
        "arm": arm,
        "seed": seed,
        "config": cfg,
        "class_weights": weights.detach().cpu().tolist(),
        "class_order": list(CLASSES),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    }
    save_json(meta, run_dir / "run_meta.json")

    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    start_epoch, best_f1, bad_epochs, history = 0, -1.0, 0, []
    if last_path.exists():
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt["best_val_macro_f1"]
        bad_epochs = ckpt["meta"].get("bad_epochs", 0)
        history = ckpt["meta"].get("history", [])
        log.info("resumed %s/%d at epoch %d", arm, seed, start_epoch)

    patience = int(cfg["early_stopping_patience"])
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        run_loss, seen = 0.0, 0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x)
                loss = criterion(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += float(loss) * y.numel()
            seen += y.numel()
        val = evaluate(model, val_loader, device, criterion)
        rec = {
            "epoch": epoch,
            "train_loss": run_loss / max(seen, 1),
            "val_loss": val["loss"],
            "val_macro_f1": val["macro_f1"],
            "val_accuracy": val["accuracy"],
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        log.info("%s/%d ep %d: train %.4f val %.4f f1 %.4f", arm, seed, epoch,
                 rec["train_loss"], rec["val_loss"], rec["val_macro_f1"])

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_f1, meta)
        else:
            bad_epochs += 1
        meta_epoch = {**meta, "bad_epochs": bad_epochs, "history": history}
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_f1, meta_epoch)
        if bad_epochs >= patience:
            log.info("%s/%d early stop at epoch %d (best %.4f)", arm, seed, epoch, best_f1)
            break

    save_checkpoint(run_dir / "final.pt", model, optimizer, scheduler, epoch, best_f1,
                    {**meta, "bad_epochs": bad_epochs, "history": history})
    result = {
        "arm": arm,
        "seed": seed,
        "best_val_macro_f1": best_f1,
        "epochs_run": len(history),
        "seconds": round(time.time() - t0, 1),
        "history": history,
    }
    save_json(result, run_dir / "train_result.json")
    return result

