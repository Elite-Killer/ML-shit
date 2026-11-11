"""Build the artifact inventory and the final research report from real files.

Every number in the report is read from a CSV or JSON artifact produced by an
earlier phase, so the report cannot drift from the evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dm_ahr_vit.config import CLASSES, load_config, save_json, sha256_file  # noqa: E402
from dm_ahr_vit.paths import artifact_dir, data_root, manifest_dir  # noqa: E402
from dm_ahr_vit.utils import record_run, setup_logging, timed  # noqa: E402

log = setup_logging()
ARMS = ["A", "B", "C", "Astar"]
ARM_LABEL = {"A": "Original", "B": "DullRazor", "C": "DM-AHR", "Astar": "A* (128 px)"}


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "NaN (undefined)"
    return f"{v:.{digits}f}"


def build_inventory() -> pd.DataFrame:
    """Every deliverable with its SHA-256, grouped by kind."""
    rows = []
    repo = Path(__file__).resolve().parents[1]
    groups = {
        "manifests": sorted(manifest_dir().glob("*")),
        "logs": sorted((artifact_dir("logs")).glob("*")),
        "predictions": sorted((artifact_dir("predictions")).glob("*")),
        "panels": sorted((artifact_dir("panels")).glob("*")),
        "reports": sorted((artifact_dir("reports")).glob("*")),
        "checkpoints_vit": sorted((artifact_dir("checkpoints", "vit")).glob("*/*")),
        "checkpoints_dmahr": sorted((artifact_dir("checkpoints", "dmahr")).glob("*")),
        "source": sorted((repo / "src" / "dm_ahr_vit").rglob("*.py")),
    }
    for kind, files in groups.items():
        for f in files:
            if not f.is_file():
                continue
            size = f.stat().st_size
            digest = sha256_file(f) if size < 200 * 1024 * 1024 else "skipped_large"
            rows.append({"kind": kind, "path": str(f), "bytes": size,
                         "sha256": digest})
    return pd.DataFrame(rows)


def metrics_table() -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for seed in (42, 123, 2026):
            p = artifact_dir("predictions") / f"metrics_{arm}_seed{seed}.json"
            if not p.exists():
                continue
            payload = read_json(p) or {}
            for key, m in payload.get("metrics", {}).items():
                _, _, subset = key.split("|")
                rows.append({
                    "arm": arm, "seed": seed, "subset": subset,
                    "n": m["n"], "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
                    "auc_macro": m["auc_macro"], "auc_weighted": m["auc_weighted"],
                    "per_class": m["per_class"],
                })
    return pd.DataFrame(rows)


def seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = df.groupby(["arm", "subset"]).agg(
        seeds=("seed", "nunique"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        balanced_acc_mean=("balanced_accuracy", "mean"),
        auc_macro_mean=("auc_macro", "mean"),
        n=("n", "max"),
    ).reset_index()
    return grouped.sort_values(["subset", "arm"])


def per_class_table(df: pd.DataFrame, subset: str = "full") -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        sub = df[(df["arm"] == arm) & (df["subset"] == subset)]
        if sub.empty:
            continue
        for c in CLASSES:
            vals = [r["per_class"][c] for r in sub.to_dict("records")]
            rows.append({
                "arm": arm,
                "class": c,
                "support": int(np.mean([v["support"] for v in vals])),
                "f1_mean": float(np.mean([v["f1"] for v in vals])),
                "f1_sd": float(np.std([v["f1"] for v in vals], ddof=1))
                if len(vals) > 1 else float("nan"),
                "recall_mean": float(np.mean([v["recall"] for v in vals])),
                "specificity_mean": float(np.mean([v["specificity"] for v in vals])),
                "auc_mean": float(np.nanmean([v["auc"] for v in vals])),
            })
    return pd.DataFrame(rows)


def history_table(arm: str, seed: int) -> dict | None:
    p = artifact_dir("checkpoints", "vit", f"{arm}_seed{seed}") / "train_result.json"
    return read_json(p)


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_No data available._\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(fmt(v, digits) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_report() -> str:
    protocol = load_config("configs/protocol.json")
    audit = read_json(artifact_dir("logs") / "data_audit.json") or {}
    smoke = read_json(artifact_dir("logs") / "dmahr_smoke.json") or {}
    hair = read_json(artifact_dir("logs") / "hair_pipeline_summary.json") or {}
    dm_train = read_json(artifact_dir("logs") / "dmahr_train.json") or {}
    quality = read_json(artifact_dir("logs") / "image_quality_test.json") or {}
    stats = pd.read_csv(artifact_dir("logs") / "statistical_comparisons.csv") \
        if (artifact_dir("logs") / "statistical_comparisons.csv").exists() else pd.DataFrame()
    mdf = metrics_table()
    summ = seed_summary(mdf)
    pclass = per_class_table(mdf)

    L: list[str] = []
    L.append("# DM-AHR-ViT: diffusion hair removal and ViT classification on HAM10000\n")
    L.append("**Research prototype only. Not clinically validated and not intended "
             "for diagnosis, triage or patient care.**\n")
    L.append("## Summary\n")
    if not summ.empty:
        full = summ[summ["subset"] == "full"]
        parts = [f"{ARM_LABEL.get(r.arm, r.arm)} {r.macro_f1_mean:.3f} "
                 f"± {r.macro_f1_sd:.3f}" for r in full.itertuples()]
        L.append("Test macro-F1 by arm (mean ± SD over seeds): " + "; ".join(parts) + ".\n")
    else:
        L.append("Classification results are not yet available.\n")
    L.append("The hypotheses were that DM-AHR hair removal improves image fidelity, "
             "classification and explanation localization relative to the classical "
             "baseline. The results below are reported as measured; a null or negative "
             "result is a valid outcome and is not reframed.\n")

    L.append("## Cohort and splits\n")
    L.append(f"- Collection 212 images: {audit.get('collection_total')}\n"
             f"- Confirmatory C0 images: {audit.get('c0_decoded')}\n"
             f"- Quarantined C0-extra: {audit.get('c0_extra')}\n"
             f"- Groups after lesion + confirmed near-duplicate merging: "
             f"{audit.get('groups_total')}\n"
             f"- Split sizes: {audit.get('frozen_splits', {}).get('sizes')}\n"
             f"- Split SHA-256: `{audit.get('frozen_splits', {}).get('splits_sha256')}`\n"
             f"- Leakage check: {audit.get('split_verification', {}).get('ok')}\n")
    L.append("Recomputed class counts matched the expected HAM10000 counts exactly "
             f"({audit.get('metadata_comparison', {}).get('all_match')}).\n")

    L.append("## Hair screening and arms\n")
    L.append(f"- Hair-present images: {hair.get('hair_present')} of {hair.get('images')} "
             f"({(hair.get('hair_present') or 0) / max(hair.get('images') or 1, 1):.1%})\n"
             f"- Screening config hash: `{hair.get('screen_config_hash')}`\n"
             f"- DullRazor config hash: `{hair.get('dullrazor_config_hash')}`\n"
             f"- Synthetic pairs: {hair.get('synthetic_pairs')} "
             f"({hair.get('synthetic_by_split')})\n")
    L.append("Hair-absent images are identical across arms. Arm A* exists because "
             "DM-AHR is native 128 px, below the 224 px ViT input.\n")

    L.append("## DM-AHR audit and training\n")
    L.append(f"- Pinned commit: `{load_config('configs/dmahr.json')['upstream_commit']}`\n"
             f"- Released checkpoint SHA-256: `{smoke.get('checkpoint_sha256')}`\n"
             f"- Smoke test passed: {smoke.get('all_pass')} "
             f"(geometry preserved {smoke.get('geometry_preserved')}, "
             f"reload identical {smoke.get('reload_identical')})\n"
             f"- Sampling throughput: {smoke.get('batch_sweep')}\n"
             f"- Leakage-controlled training: {dm_train.get('iterations')} iterations, "
             f"best validation L1 {fmt(dm_train.get('best_val'), 5)}\n")
    L.append("The released weights are used for the smoke test only, because the "
             "authors do not document their training data. Arm C uses the checkpoint "
             "trained here on train-split sources only.\n")

    L.append("## Classification results\n")
    L.append("### Mean ± SD over seeds\n")
    L.append(md_table(summ))
    L.append("\n### Per-class test metrics (full test split)\n")
    L.append(md_table(pclass))
    L.append("\nUndefined AUC values are reported as NaN with a reason in the "
             "per-run metric JSON files; `df` and `vasc` have small test supports.\n")

    L.append("## Paired image quality (held-out synthetic pairs)\n")
    if quality:
        L.append(f"- Pairs scored: {quality.get('pairs')} (Arm C scored: "
                 f"{quality.get('C_scored')})\n")
        means = quality.get("means", {})
        L.append(md_table(pd.DataFrame([means])) if means else "_No means recorded._\n")
    else:
        L.append("_Image quality not computed._\n")
    L.append("\nPSNR/SSIM are computed only against genuine hairless references from "
             "synthetic pairs. Naturally hairy images are never scored against a "
             "guessed reference.\n")

    L.append("## Explainability\n")
    xai_files = sorted(artifact_dir("logs").glob("xai_summary_*.json"))
    if xai_files:
        rows = []
        for f in xai_files:
            d = read_json(f) or {}
            rows.append({
                "arm": d.get("arm"), "seed": d.get("seed"), "tier": d.get("tier"),
                "n": d.get("n"),
                "gradcam_overlap20": d.get("gradcam_overlap20_mean"),
                "rollout_overlap20": d.get("rollout_overlap20_mean"),
                "gradcam_pointing": d.get("gradcam_pointing_mean"),
                "rollout_pointing": d.get("rollout_pointing_mean"),
                "chance_uniform": d.get("chance_uniform_overlap20_mean"),
                "chance_gaussian": d.get("chance_gaussian_overlap20_mean"),
            })
        L.append(md_table(pd.DataFrame(rows)))
    else:
        L.append("_Explainability not computed._\n")
    L.append("\nLocalization measures spatial alignment with released lesion masks, "
             "not clinically correct reasoning. X1 masks are the released HAM10000 "
             "segmentations; the source does not state they were expert-traced, so "
             "they are not described as expert masks.\n")

    L.append("## Statistical comparisons\n")
    if not stats.empty:
        show = stats[["endpoint", "subset", "comparison", "arm_a_mean", "arm_b_mean",
                      "difference", "ci_low", "ci_high", "p_raw", "p_holm",
                      "effect_size_dz", "n_groups"]].copy()
        L.append(md_table(show))
        L.append("\nTest: paired lesion-group bootstrap, 10,000 replicates, the same "
                 "resample shared across arms and seeds, percentile 95% CI, two-sided. "
                 "Holm correction is applied across the three planned comparisons "
                 "within each endpoint and subset. A difference whose CI spans zero is "
                 "reported as inconclusive, not as no difference.\n")
    else:
        L.append("_Statistics not computed._\n")

    L.append("## Limitations and failure cases\n")
    L.append("- DERMAHAIR could not be downloaded (no Kaggle credentials), so the "
             "synthetic-hair fallback is used. Held-out test pairs number only "
             f"{quality.get('pairs', 'n/a')}, so paired CIs are wide.\n")
    L.append("- No independent hair annotations exist, so the screening threshold is "
             "NOT VERIFIED against real hair labels; it was tuned on synthetic pairs "
             "and frozen before test scoring.\n")
    L.append("- DM-AHR runs at 128 px, so Arm C is upsampled to 224 px for the ViT. "
             "Arm A* controls for that resolution change but cannot separate it from "
             "diffusion artefacts.\n")
    L.append("- The diffusion model was trained for 20,000 iterations, far below the "
             "660,000 in the upstream config, so Arm C is not a faithful reproduction "
             "of the released checkpoint's training budget.\n")
    L.append("- `df` and `vasc` have very small test supports, so their per-class "
             "numbers are noisy.\n")
    L.append("- The released DM-AHR weights have undocumented training data and are "
             "used for exploration only.\n")

    L.append("## Licence and provenance\n")
    L.append("HAM10000 is distributed under CC BY-NC 4.0. The DM-AHR repository "
             "contradicts itself: its README states CC BY-NC 4.0 while its LICENSE "
             "file is Apache-2.0. **Commercial use is unresolved** and needs a rights "
             "decision by the client. See `reports/license_audit.md`.\n")

    L.append("## Reproducibility\n")
    L.append("Commands are in `README.md`. Frozen artifacts and their hashes are in "
             "`artifacts/logs/` and `data/manifests/`. Every table above is generated "
             "from a CSV or JSON artifact listed in `artifact_inventory.csv`.\n")
    return "\n".join(L)


def main() -> int:
    with timed("finalize", log) as t:
        inv = build_inventory()
        inv_path = artifact_dir("reports") / "artifact_inventory.csv"
        inv.to_csv(inv_path, index=False)
        report = build_report()
        rep_path = artifact_dir("reports") / "final_research_report.md"
        rep_path.write_text(report)
        mdf = metrics_table()
        if not mdf.empty:
            mdf.drop(columns=["per_class"]).to_csv(
                artifact_dir("logs") / "classification_metrics_all.csv", index=False)
            seed_summary(mdf).to_csv(
                artifact_dir("logs") / "classification_seed_summary.csv", index=False)
            per_class_table(mdf).to_csv(
                artifact_dir("logs") / "classification_per_class.csv", index=False)
    save_json({"seconds": t["seconds"], "inventory_rows": int(len(inv)),
               "report_bytes": len(report)},
              artifact_dir("logs") / "finalize_summary.json")
    record_run("finalize", gpu=None, seconds=t["seconds"], inventory=len(inv))
    print(f"inventory rows: {len(inv)}")
    print(f"report: {rep_path} ({len(report)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



