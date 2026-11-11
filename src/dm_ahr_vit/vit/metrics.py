"""Classification metrics with explicit handling of undefined values."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ..config import CLASSES


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    labels = list(range(len(CLASSES)))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {},
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    for i, c in enumerate(CLASSES):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")
        auc, auc_note = _auc_for(y_true, y_prob, i)
        out["per_class"][c] = {
            "support": int(support[i]),
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "auc": auc,
            "auc_note": auc_note,
        }
    if y_prob is not None:
        out["auc_macro"] = _macro_auc(y_true, y_prob)
        out["auc_weighted"] = _weighted_auc(y_true, y_prob)
    else:
        out["auc_macro"] = float("nan")
        out["auc_weighted"] = float("nan")
    out["confusion_matrix"] = cm.tolist()
    out["confusion_matrix_normalized"] = (
        (cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).round(6).tolist()
    )
    return out


def _auc_for(y_true: np.ndarray, y_prob: np.ndarray | None, idx: int):
    if y_prob is None:
        return float("nan"), "no_probabilities"
    binary = (y_true == idx).astype(int)
    if binary.sum() == 0 or binary.sum() == len(binary):
        return float("nan"), "single_class_present"
    return float(roc_auc_score(binary, y_prob[:, idx])), ""


def _macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    vals = [v for v in (_auc_for(y_true, y_prob, i)[0] for i in range(len(CLASSES)))
            if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _weighted_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    vals, weights = [], []
    for i in range(len(CLASSES)):
        v = _auc_for(y_true, y_prob, i)[0]
        if not np.isnan(v):
            vals.append(v)
            weights.append(int((y_true == i).sum()))
    if not vals:
        return float("nan")
    return float(np.average(vals, weights=weights))


def metrics_table(metrics_by_key: dict[tuple, dict]) -> pd.DataFrame:
    """Flatten many metric dicts into one CSV-friendly table."""
    rows = []
    for key, m in metrics_by_key.items():
        arm, seed, subset = key
        base = {
            "arm": arm,
            "seed": seed,
            "subset": subset,
            "n": m["n"],
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "auc_macro": m["auc_macro"],
            "auc_weighted": m["auc_weighted"],
        }
        for c in CLASSES:
            pc = m["per_class"][c]
            for k in ("support", "precision", "recall", "f1", "sensitivity",
                      "specificity", "auc"):
                base[f"{c}_{k}"] = pc[k]
            base[f"{c}_auc_note"] = pc["auc_note"]
        rows.append(base)
    return pd.DataFrame(rows)

