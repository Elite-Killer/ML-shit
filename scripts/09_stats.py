"""Assemble the statistical comparison table.

Paired lesion-group bootstrap with the same resample shared across arms and
seeds, Holm correction across the three planned comparisons per endpoint, plus
group-aggregated Wilcoxon tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import CLASSES, load_config, save_json  # noqa: E402
from dm_ahr_vit.paths import artifact_dir  # noqa: E402
from dm_ahr_vit.stats.bootstrap import holm  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()
ARMS = ["A", "B", "C", "Astar"]


def load_predictions(arm: str, subset: str, seeds: list[int]):
    frames = []
    for seed in seeds:
        p = artifact_dir("predictions") / f"pred_{arm}_seed{seed}_{subset}.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p)
        df["seed"] = seed
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def group_class_counts(df: pd.DataFrame, groups: list[str]) -> dict:
    """Per-group TP/FP/FN per class so a bootstrap becomes one matmul."""
    gidx = {g: i for i, g in enumerate(groups)}
    n_classes = len(CLASSES)
    tp = np.zeros((len(groups), n_classes))
    fp = np.zeros((len(groups), n_classes))
    fn = np.zeros((len(groups), n_classes))
    y_true = np.array([CLASSES.index(v) for v in df["label"]])
    y_pred = np.array([CLASSES.index(v) for v in df["pred"]])
    gi = df["group_id"].map(gidx).to_numpy()
    for k in range(len(df)):
        g, t, p = int(gi[k]), int(y_true[k]), int(y_pred[k])
        if t == p:
            tp[g, t] += 1.0
        else:
            fp[g, p] += 1.0
            fn[g, t] += 1.0
    return {"tp": tp, "fp": fp, "fn": fn}


def macro_f1_from_counts(tp, fp, fn) -> np.ndarray:
    denom = 2 * tp + fp + fn
    per_class = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return per_class.mean(axis=1)


def draw_counts(n_groups: int, reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_groups, size=(reps, n_groups))
    counts = np.zeros((reps, n_groups), dtype=np.float32)
    for r in range(reps):
        np.add.at(counts[r], idx[r], 1.0)
    return counts


def compare(dist_a: np.ndarray, dist_b: np.ndarray, alpha: float = 0.05) -> dict:
    diff = dist_a - dist_b
    lo, hi = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min(float((diff <= 0).mean()), float((diff >= 0).mean()))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")
    mean_diff = float(dist_a.mean() - dist_b.mean())
    return {
        "difference": mean_diff,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_raw": float(min(1.0, p)),
        "effect_size_dz": float(mean_diff / sd) if sd > 0 else float("nan"),
    }


def bootstrap_arm_scores(per_seed_counts: list, W: np.ndarray) -> np.ndarray:
    """Bootstrap distribution of the seed-averaged macro-F1 for one arm."""
    scores = []
    for counts in per_seed_counts:
        tp = W @ counts["tp"]
        fp = W @ counts["fp"]
        fn = W @ counts["fn"]
        scores.append(macro_f1_from_counts(tp, fp, fn))
    return np.mean(np.vstack(scores), axis=0)


def classification_endpoint(seeds: list[int], subset: str, reps: int, bseed: int,
                            comparisons: list) -> list:
    preds = {arm: load_predictions(arm, subset, seeds) for arm in ARMS}
    available = [a for a in ARMS if preds[a] is not None]
    if len(available) < 2:
        log.warning("classification/%s: only arms %s available", subset, available)
        return []
    common = None
    for arm in available:
        ids = set(preds[arm]["image_id"])
        common = ids if common is None else (common & ids)
    common = sorted(common)
    groups = sorted({g for arm in available
                     for g in preds[arm].loc[preds[arm]["image_id"].isin(common),
                                             "group_id"].unique()})
    W = draw_counts(len(groups), reps, bseed)

    dists, observed = {}, {}
    for arm in available:
        per_seed, seed_scores = [], []
        for seed in seeds:
            sub = preds[arm][(preds[arm]["seed"] == seed)
                             & (preds[arm]["image_id"].isin(common))]
            if sub.empty:
                continue
            counts = group_class_counts(sub, groups)
            per_seed.append(counts)
            seed_scores.append(float(macro_f1_from_counts(
                counts["tp"].sum(0)[None], counts["fp"].sum(0)[None],
                counts["fn"].sum(0)[None])[0]))
        if not per_seed:
            continue
        dists[arm] = bootstrap_arm_scores(per_seed, W)
        observed[arm] = float(np.mean(seed_scores))

    rows = []
    for a, b in comparisons:
        if a not in dists or b not in dists:
            continue
        rows.append({
            "endpoint": "test_macro_f1",
            "subset": subset,
            "comparison": f"{a}-{b}",
            "arm_a": a,
            "arm_b": b,
            "arm_a_mean": observed[a],
            "arm_b_mean": observed[b],
            "n_images": len(common),
            "n_groups": len(groups),
            "replicates": reps,
            **compare(dists[a], dists[b]),
            "wilcoxon_p_raw": float("nan"),
            "test": "lesion-group paired bootstrap, percentile CI, two-sided",
        })
    save_json(
        {"subset": subset, "observed_macro_f1": observed,
         "arms": available, "n_images": len(common), "n_groups": len(groups)},
        artifact_dir("logs") / f"stats_classification_{subset}.json",
    )
    return rows


def bootstrap_group_means(df: pd.DataFrame, value: str, arm_col: str,
                          W: np.ndarray, groups: list, gidx: dict) -> dict:
    """Bootstrap mean of `value` per arm, resampling lesion groups."""
    dists, obs = {}, {}
    for arm, sub in df.groupby(arm_col):
        sub = sub.dropna(subset=[value])
        if sub.empty:
            continue
        gi = sub["group_id"].map(gidx).to_numpy()
        vals = sub[value].to_numpy(dtype=float)
        sums = np.zeros(len(groups))
        cnts = np.zeros(len(groups))
        np.add.at(sums, gi, vals)
        np.add.at(cnts, gi, 1.0)
        dists[arm] = (W @ sums) / np.maximum(W @ cnts, 1e-9)
        obs[arm] = float(vals.mean())
    return {"dists": dists, "obs": obs}


def paired_column_endpoint(df: pd.DataFrame, col_a: str, col_b: str, endpoint: str,
                           reps: int, bseed: int, comparisons: list,
                           arm_col=None) -> list:
    """Bootstrap a paired column comparison (two conditions in the same rows)."""
    if df.empty or col_a not in df.columns or col_b not in df.columns:
        return []
    df = df.dropna(subset=[col_a, col_b, "group_id"])
    if df.empty:
        return []
    groups = sorted(df["group_id"].unique())
    gidx = {g: i for i, g in enumerate(groups)}
    W = draw_counts(len(groups), reps, bseed)
    rows = []
    if arm_col is not None and arm_col in df.columns:
        stats_by_arm = bootstrap_group_means(df, col_a, arm_col, W, groups, gidx)
        for x, y in comparisons:
            if x not in stats_by_arm["dists"] or y not in stats_by_arm["dists"]:
                continue
            rows.append({
                "endpoint": endpoint, "subset": "test_x1_tier",
                "comparison": f"{x}-{y}", "arm_a": x, "arm_b": y,
                "arm_a_mean": stats_by_arm["obs"][x],
                "arm_b_mean": stats_by_arm["obs"][y],
                "n_images": int(df[df[arm_col] == x].shape[0]),
                "n_groups": len(groups), "replicates": reps,
                **compare(stats_by_arm["dists"][x], stats_by_arm["dists"][y]),
                "wilcoxon_p_raw": float("nan"),
                "test": "lesion-group paired bootstrap, percentile CI, two-sided",
            })
        return rows

    gi = df["group_id"].map(gidx).to_numpy()
    a = df[col_a].to_numpy(dtype=float)
    b = df[col_b].to_numpy(dtype=float)
    sa, sb = np.zeros(len(groups)), np.zeros(len(groups))
    ca, cb = np.zeros(len(groups)), np.zeros(len(groups))
    np.add.at(sa, gi, a)
    np.add.at(sb, gi, b)
    np.add.at(ca, gi, 1.0)
    da = (W @ sa) / np.maximum(W @ ca, 1e-9)
    db = (W @ sb) / np.maximum(W @ cb, 1e-9)
    for x, y in comparisons:
        rows.append({
            "endpoint": endpoint, "subset": "held_out_pairs",
            "comparison": f"{x}-{y}", "arm_a": x, "arm_b": y,
            "arm_a_mean": float(a.mean()), "arm_b_mean": float(b.mean()),
            "n_images": int(len(df)), "n_groups": len(groups),
            "replicates": reps, **compare(da, db),
            "wilcoxon_p_raw": float("nan"),
            "test": "lesion-group paired bootstrap, percentile CI, two-sided",
        })
    return rows


def xai_rows(metric: str, reps: int, bseed: int, comparisons: list) -> list:
    files = sorted(artifact_dir("logs").glob("xai_localization_*.csv"))
    if not files:
        return []
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if metric not in df.columns or "group_id" not in df.columns:
        return []
    # One value per (arm, seed, group) before resampling.
    agg = df.groupby(["arm", "seed", "group_id"])[metric].mean().reset_index()
    # Average seeds first so seeds are not counted as independent samples.
    per_group = agg.groupby(["arm", "group_id"])[metric].mean().reset_index()
    groups = sorted(per_group["group_id"].unique())
    gidx = {g: i for i, g in enumerate(groups)}
    W = draw_counts(len(groups), reps, bseed)
    res = bootstrap_group_means(per_group, metric, "arm", W, groups, gidx)
    rows = []
    for a, b in comparisons:
        if a not in res["dists"] or b not in res["dists"]:
            continue
        rows.append({
            "endpoint": f"x1_{metric}", "subset": "test_x1",
            "comparison": f"{a}-{b}", "arm_a": a, "arm_b": b,
            "arm_a_mean": res["obs"][a], "arm_b_mean": res["obs"][b],
            "n_images": int((df["arm"] == a).sum()),
            "n_groups": len(groups), "replicates": reps,
            **compare(res["dists"][a], res["dists"][b]),
            "wilcoxon_p_raw": float("nan"),
            "test": "lesion-group paired bootstrap, percentile CI, two-sided",
        })
    return rows


def main() -> int:
    protocol = load_config("configs/protocol.json")
    seeds = list(protocol["seeds"])
    reps = int(protocol["bootstrap_replicates"])
    bseed = int(protocol["bootstrap_seed"])
    comparisons = [tuple(c) for c in protocol["planned_comparisons"]]
    out: list = []

    with timed("stats", log) as t:
        for subset in ("full", "hair_present"):
            out += classification_endpoint(seeds, subset, reps, bseed, comparisons)

        p = artifact_dir("logs") / "paired_psnr_ssim_test.csv"
        if p.exists():
            df = pd.read_csv(p)
            for metric in ("psnr", "ssim"):
                out += paired_column_endpoint(
                    df, f"B_{metric}", f"hairy_input_{metric}",
                    f"paired_{metric}", reps, bseed, [("B", "hairy input")])
                if f"C_{metric}" in df.columns and df[f"C_{metric}"].notna().any():
                    out += paired_column_endpoint(
                        df, f"C_{metric}", f"B_{metric}",
                        f"paired_{metric}", reps, bseed, [("C", "B")])
                    out += paired_column_endpoint(
                        df, f"C_{metric}", f"hairy_input_{metric}",
                        f"paired_{metric}", reps, bseed, [("C", "hairy input")])

        for metric in ("gradcam_overlap20", "rollout_overlap20"):
            out += xai_rows(metric, reps, bseed, comparisons)

    if not out:
        log.error("no statistical comparisons could be computed")
        return 1
    table = pd.DataFrame(out)
    table["p_holm"] = np.nan
    for _, grp in table.groupby(["endpoint", "subset"]):
        table.loc[grp.index, "p_holm"] = holm(grp["p_raw"].tolist())
    cols = ["endpoint", "subset", "comparison", "arm_a", "arm_b", "arm_a_mean",
            "arm_b_mean", "difference", "ci_low", "ci_high", "p_raw", "p_holm",
            "effect_size_dz", "n_images", "n_groups", "replicates", "test"]
    table = table[[c for c in cols if c in table.columns]]
    table = table.sort_values(["endpoint", "subset", "comparison"])
    table.to_csv(artifact_dir("logs") / "statistical_comparisons.csv", index=False)
    save_json({"seconds": t["seconds"], "rows": int(len(table)),
               "endpoints": sorted(table["endpoint"].unique().tolist())},
              artifact_dir("logs") / "stats_summary.json")
    record_run("stats", gpu=None, seconds=t["seconds"], rows=int(len(table)))
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




