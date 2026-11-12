"""Macro-F1 by hair-coverage decile, and residual hair after each arm.

Exploratory only, computed from existing predictions on CPU. No retraining and
no GPU. Tests whether Arm B's deficit is concentrated on images that barely had
hair, which would point to false-positive damage rather than hair removal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import CLASSES  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, manifest_dir  # noqa: E402
from dm_ahr_vit.stats.bootstrap import holm  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging  # noqa: E402

log = setup_logging()
ARMS = ["A", "Astar", "B", "C"]
SEEDS = [42, 123, 2026]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for k in range(len(CLASSES)):
        tp = float(((y_pred == k) & (y_true == k)).sum())
        fp = float(((y_pred == k) & (y_true != k)).sum())
        fn = float(((y_pred != k) & (y_true == k)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def load(arm: str) -> pd.DataFrame | None:
    frames = []
    for seed in SEEDS:
        p = artifact_dir("predictions") / f"pred_{arm}_seed{seed}_full.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def cluster_bootstrap(diffs: np.ndarray, groups: np.ndarray,
                      reps: int = 10000, seed: int = 20260101) -> dict:
    """Bootstrap a mean difference by resampling lesion groups."""
    uniq = np.unique(groups)
    idx = {g: i for i, g in enumerate(uniq)}
    gi = np.array([idx[g] for g in groups])
    sums = np.zeros(len(uniq))
    cnts = np.zeros(len(uniq))
    np.add.at(sums, gi, diffs)
    np.add.at(cnts, gi, 1.0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(reps, len(uniq)))
    num = sums[draws].sum(axis=1)
    den = cnts[draws].sum(axis=1)
    ok = den > 0
    dist = np.full(reps, np.nan)
    dist[ok] = num[ok] / den[ok]
    dist = dist[np.isfinite(dist)]
    lo, hi = np.percentile(dist, [2.5, 97.5])
    p = 2 * min(float((dist <= 0).mean()), float((dist >= 0).mean()))
    return {"diff": float(diffs.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_raw": float(min(1.0, p)), "n_groups": int(len(uniq))}


def main() -> int:
    hair = pd.read_csv(manifest_dir() / "hair_screening.csv")
    test_hair = hair[hair["split"] == "test"].set_index("image_id")["hair_coverage"]

    preds = {a: load(a) for a in ARMS}
    preds = {a: d for a, d in preds.items() if d is not None}
    if "A" not in preds:
        log.error("Arm A predictions missing; run 07_eval_vit.py first")
        return 1

    # Coverage deciles from the test split's own distribution.
    cov = test_hair.dropna()
    edges = np.unique(np.quantile(cov, np.linspace(0, 1, 11)))
    edges = edges[edges >= 0]
    if len(edges) < 3:
        log.error("not enough coverage variation to form deciles")
        return 1
    bins = pd.cut(cov, bins=edges, include_lowest=True, duplicates="drop")
    decile_of = {img: int(c) for img, c in bins.cat.codes.items()}

    rows = []
    group_rows = []
    for arm, df in preds.items():
        d = df.copy()
        d["coverage"] = d["image_id"].map(test_hair)
        d["decile"] = d["image_id"].map(decile_of)
        d = d.dropna(subset=["coverage", "decile"])
        for dec, sub in d.groupby("decile"):
            per_seed = [macro_f1(g["label"].map(CLASSES.index).to_numpy(),
                                 g["pred"].map(CLASSES.index).to_numpy())
                        for _, g in sub.groupby("seed")]
            rows.append({
                "arm": arm, "decile": int(dec),
                "coverage_lo": float(sub["coverage"].min()),
                "coverage_hi": float(sub["coverage"].max()),
                "n_images": int(sub["image_id"].nunique()),
                "macro_f1": float(np.mean(per_seed)),
            })
        # Per-group correctness, averaged over seeds, for the gap analysis.
        d["correct"] = (d["label"] == d["pred"]).astype(float)
        agg = d.groupby(["group_id", "seed"])["correct"].mean().reset_index()
        group_rows.append(agg.groupby("group_id")["correct"].mean()
                          .rename(arm).reset_index())

    table = pd.DataFrame(rows).sort_values(["arm", "decile"])
    table.to_csv(artifact_dir("logs") / "hair_coverage_by_decile.csv", index=False)
    log.info("per-decile macro-F1\n%s", table.to_string(index=False))

    # Gaps versus Arm A, per decile, with lesion-cluster CIs.
    merged = group_rows[0]
    for g in group_rows[1:]:
        merged = merged.merge(g, on="group_id", how="inner")

    # One decile per lesion group: the mean coverage of its test images.
    splits = pd.read_csv(manifest_dir() / "splits.csv")
    test_map = splits[splits["split"] == "test"][["image_id", "group_id"]].copy()
    test_map["coverage"] = test_map["image_id"].map(test_hair)
    group_cov = test_map.dropna(subset=["coverage"]).groupby("group_id")["coverage"].mean()
    group_decile = {
        g: int(pd.cut(pd.Series([c]), bins=edges, include_lowest=True,
                      labels=False).iloc[0])
        for g, c in group_cov.items()
    }
    merged["decile"] = merged["group_id"].map(group_decile)
    merged = merged.dropna(subset=["decile"])

    gap_rows = []
    for arm in [a for a in ARMS if a in merged.columns and a != "A"]:
        for dec, sub in merged.groupby("decile"):
            if len(sub) < 20:
                continue
            diffs = (sub[arm] - sub["A"]).to_numpy(dtype=float)
            groups = sub["group_id"].to_numpy()
            r = cluster_bootstrap(diffs, groups)
            gap_rows.append({"comparison": f"{arm}-A", "decile": int(dec),
                             "n_groups": r["n_groups"], **{k: v for k, v in r.items()
                                                           if k != "n_groups"}})
    gaps = pd.DataFrame(gap_rows)
    if not gaps.empty:
        gaps["p_holm"] = np.nan
        for _, grp in gaps.groupby("comparison"):
            gaps.loc[grp.index, "p_holm"] = holm(grp["p_raw"].tolist())
        gaps = gaps.round(4)
        gaps.to_csv(artifact_dir("logs") / "hair_coverage_gaps.csv", index=False)
        log.info("per-decile gaps vs Arm A\n%s", gaps.to_string(index=False))

    # Residual hair: re-screen each arm's test outputs with the frozen detector.
    residual: list[dict] = []
    try:
        import cv2  # noqa: F401
        from PIL import Image

        from dm_ahr_vit.config import load_config
        from dm_ahr_vit.hair.screening import detect_hair
        from dm_ahr_vit.paths import data_root, processed_dir

        cfg = load_config("configs/hair_screen.json")
        test_ids = list(test_hair.index)
        raw = data_root() / "raw" / "isic_212" / "images"
        for arm in ("A", "B", "C"):
            vals = []
            for image_id in test_ids[:400]:
                if arm == "A":
                    p = raw / f"{image_id}.jpg"
                else:
                    p = processed_dir(arm) / f"{image_id}.png"
                if not p.exists():
                    continue
                img = np.asarray(Image.open(p).convert("RGB"))
                _, stats = detect_hair(img, cfg)
                vals.append(stats["coverage"])
            if vals:
                residual.append({"arm": arm, "n": len(vals),
                                 "mean_coverage": float(np.mean(vals)),
                                 "present_rate": float(np.mean(
                                     [v >= cfg["coverage_threshold"] for v in vals]))})
    except Exception as exc:  # noqa: BLE001
        log.warning("residual-hair screening skipped: %s", exc)
        residual.append({"arm": "skipped", "reason": repr(exc)[:160]})

    resid = pd.DataFrame(residual)
    if "mean_coverage" in resid.columns:
        resid.to_csv(artifact_dir("logs") / "hair_residual.csv", index=False)
        log.info("residual hair after each arm\n%s", resid.to_string(index=False))

    record_run("hair_coverage", gpu=None, seconds=0.0,
               deciles=int(table["decile"].nunique()), arms=len(preds))
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
