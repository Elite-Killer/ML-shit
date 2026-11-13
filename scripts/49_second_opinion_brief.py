"""Assemble reports/SECOND_OPINION_BRIEF.md.

Reads the hand-written reports/second_opinion_brief.template.md and replaces the
`<!-- APPENDIX -->` marker with tables generated from the files in artifacts/, so every number
in the appendix traces to a saved artifact. `{{HEAD}}` and `{{DATE}}` are filled from git.
"""

from __future__ import annotations

import collections
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "artifacts" / "logs"
DIAG = ROOT / "artifacts" / "diagnosis"
PRED = ROOT / "artifacts" / "predictions"
REPORTS = ROOT / "reports"
SEEDS = (42, 123, 2026)


def md(df: pd.DataFrame, digits: dict | int = 3) -> str:
    cols = list(df.columns)
    d = digits if isinstance(digits, dict) else {}
    default = digits if isinstance(digits, int) else 3

    def cell(c, v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else f"{v:.{d.get(c, default)}f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(cell(c, r[c]) for c in cols) + " |")
    return "\n".join(lines)


def cv2_rows() -> pd.DataFrame:
    rows = []
    for s in SEEDS:
        m = json.loads((PRED / f"metrics_Cv2_seed{s}.json").read_text())["metrics"]
        for subset in ("full", "hair_present"):
            x = m[f"Cv2|{s}|{subset}"]
            rows.append({"arm": "Cv2", "seed": s, "subset": subset, "n": x["n"],
                         "macro_f1": x["macro_f1"], "accuracy": x["accuracy"],
                         "balanced_accuracy": x["balanced_accuracy"], "auc_macro": x["auc_macro"]})
    return pd.DataFrame(rows)


def a1_a2() -> str:
    per = pd.read_csv(LOGS / "classification_metrics_all.csv")
    per = per[["arm", "seed", "subset", "n", "macro_f1", "accuracy", "balanced_accuracy",
               "auc_macro"]]
    per = pd.concat([per, cv2_rows()], ignore_index=True)
    order = {"A": 0, "Astar": 1, "B": 2, "Cv2": 3, "C": 4, "Creleased": 5}
    per["o"] = per["arm"].map(order)
    per = per.sort_values(["subset", "o", "seed"]).drop(columns="o")
    g = per.groupby(["subset", "arm"], sort=False)
    summ = g.agg(seeds=("seed", "count"), n=("n", "first"),
                 macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
                 accuracy_mean=("accuracy", "mean"),
                 balanced_acc_mean=("balanced_accuracy", "mean"),
                 auc_macro_mean=("auc_macro", "mean")).reset_index()
    summ["o"] = summ["arm"].map(order)
    summ = summ.sort_values(["subset", "o"]).drop(columns="o")
    return ("### A1. Classification, mean over seeds 42/123/2026 (SD with ddof=1)\n\n"
            + md(summ, {"macro_f1_mean": 4, "macro_f1_sd": 4, "accuracy_mean": 4,
                        "balanced_acc_mean": 4, "auc_macro_mean": 4})
            + "\n\n### A2. Classification, every seed\n\n"
            + md(per, 4))


def a3() -> str:
    parts = ["### A3. Statistical comparisons (paired lesion-group bootstrap, 10,000 replicates, "
             "percentile CI; Holm within endpoint, subset and family)\n"]
    cols = ["endpoint", "subset", "family", "comparison", "arm_a_mean", "arm_b_mean",
            "difference", "ci_low", "ci_high", "p_raw", "p_holm", "effect_size_dz"]
    for f, title in (("statistical_comparisons.csv", "A3a. Original protocol, arms A, B, C, A*"),
                     ("statistical_comparisons_exploratory.csv",
                      "A3b. Exploratory, released checkpoint (Creleased)"),
                     ("statistical_comparisons_Cv2.csv",
                      "A3c. Stage C, rebuilt arm Cv2 in the role of C (A*-control rows for the "
                      "explanation endpoints were added after the planned rows were seen)")):
        d = pd.read_csv(LOGS / f)
        c = [x for x in cols if x in d.columns]
        parts.append(f"**{title}** ({len(d)} rows)\n\n"
                     + md(d[c], {"p_raw": 4, "p_holm": 4, "effect_size_dz": 3, "arm_a_mean": 4,
                                 "arm_b_mean": 4, "difference": 4, "ci_low": 4, "ci_high": 4}))
    return "\n\n".join(parts)


def a4() -> str:
    t = json.loads((LOGS / "image_quality_test.json").read_text())["means"]
    c = json.loads((LOGS / "image_quality_test_Cv2.json").read_text())["means"]
    v = json.loads((LOGS / "image_quality_val_Cv2.json").read_text())["means"]
    rows = [
        ["hairy input", t["hairy_input_psnr"], t["hairy_input_ssim"], v["hairy_input_psnr"],
         v["hairy_input_ssim"]],
        ["B DullRazor", t["B_psnr"], t["B_ssim"], v["B_psnr"], v["B_ssim"]],
        ["C first run (defective trainer, 1,000 steps)", t["C_psnr"], t["C_ssim"], np.nan,
         np.nan],
        ["Creleased (released ckpt, 25-step respaced)", t["Creleased_psnr"],
         t["Creleased_ssim"], np.nan, np.nan],
        ["Cv2 rebuilt", c["Cv2_psnr"], c["Cv2_ssim"], v["Cv2_psnr"], v["Cv2_ssim"]],
    ]
    df = pd.DataFrame(rows, columns=["input", "test PSNR dB", "test SSIM", "val PSNR dB",
                                     "val SSIM"])
    return ("### A4. Image quality on synthetic pairs (scored at 128 px against the hairless "
            "reference; test n=186, validation n=182)\n\n" + md(df, 3))


def a5() -> str:
    fs = [f for f in sorted(LOGS.glob("xai_localization_*_X1.csv")) if "_v2_" not in f.name]
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    cols = ["gradcam_overlap20", "rollout_overlap20", "chance_uniform_overlap20",
            "chance_gaussian_overlap20"]
    cols = [c for c in cols if c in d.columns]
    order = {"A": 0, "Astar": 1, "B": 2, "Cv2": 3, "C": 4}
    by_seed = d.groupby(["arm", "seed"])[cols].mean().reset_index()
    by_seed["o"] = by_seed["arm"].map(order)
    by_seed = by_seed.sort_values(["o", "seed"]).drop(columns="o")
    return ("### A5. Explanation overlap, X1 mask subset (share of the top 20% attention inside "
            "the lesion), image-level mean per arm and seed\n\n" + md(by_seed, 4)
            + f"\n\nRows per arm-seed: {int(d.groupby(['arm', 'seed']).size().iloc[0])}.")


def _stage_rows(path: Path) -> pd.DataFrame:
    j = json.loads(path.read_text())
    r = pd.DataFrame(j["rows"])
    r.insert(0, "run", path.stem.replace("_partial_noise", ""))
    r.insert(1, "ckpt_sha8", j.get("checkpoint_sha256", "")[:8])
    return r[["run", "ckpt_sha8", "t0", "pairs", "seeds", "mean_psnr", "hairy_floor_psnr",
              "mean_margin_db", "fraction_runs_below_floor", "sd_of_seed_means", "mean_ssim"]]


def a6() -> str:
    files = sorted(DIAG.glob("stageA_*_partial_noise.json")) + [
        LOGS / "stageB_v2_synth_partial_noise.json"]
    df = pd.concat([_stage_rows(f) for f in files], ignore_index=True)
    return ("### A6. Partial-noise sampler on validation synthetic pairs (Stage A; last rows are the "
            "Stage B fine-tuned model). t0=1999 is a pure-noise start. Margin is over the "
            "hairy-input PSNR floor\n\n" + md(df, 3))


def a7() -> str:
    out = ["### A7. Real-photo hair-removal check (frozen detector at 128 px; ratio = mean change "
           "inside the detected hair mask over outside it)\n"]
    for f, title in (("real_hair_check_stageB0.json", "Train-split photos (n=40), Stage B0"),
                     ("real_hair_check_stageB_val_baseline.json",
                      "Validation photos (n=40), 50k model baseline"),
                     ("real_hair_check_stageB_val.json",
                      "Validation photos (n=40), gate set: 50k model and v2 fine-tune"),
                     ("real_hair_check_stageB_dense_val.json",
                      "Validation photos with the most harvested hair (n=12)")):
        j = json.loads((DIAG / f).read_text())
        rows = [{"arm": k, "ratio": v["inside_over_outside_ratio"],
                 "median_coverage_remaining": v["median_fraction_of_hair_coverage_remaining"],
                 "images_under_half_remaining": v["images_with_less_than_half_remaining"],
                 "mean_change_inside": v["mean_absdiff_inside_hair_mask"],
                 "mean_change_outside": v["mean_absdiff_outside_hair_mask"]}
                for k, v in j["summary"].items()]
        out.append(f"**{title}**; median original coverage {j['median_orig_coverage']:.3f}\n\n"
                   + md(pd.DataFrame(rows), 3))
    return "\n\n".join(out)


def a8() -> str:
    j = json.loads((LOGS / "stageB_val_composite_eval.json").read_text())
    df = pd.DataFrame(j["rows"])
    df["ckpt_sha8"] = df["checkpoint_sha256"].str[:8]
    df = df[["model", "ckpt_sha8", "t0", "items", "seeds", "mean_psnr", "floor_psnr",
             "ceiling_psnr", "margin_db", "fraction_of_gap_closed", "fraction_runs_below_floor",
             "hair_removal_native"]]
    return ("### A8. Ground-truth test on pasted real strands (validation composites, seed 100)\n\n"
            + md(df, 3))


def a9() -> str:
    out = ["### A9. Stage D paste-back, validation synthetic pairs only (n=182)\n"]
    for f, name in (("stageD_val_tuning.json", "variant coarse (fails D1 and D2)"),
                    ("stageD_val_tuning_thin.json", "variant thin (passes D1, fails D2)")):
        j = json.loads((LOGS / f).read_text())
        g = pd.DataFrame(j["grid"]).sort_values("psnr128", ascending=False).head(6)
        out.append(f"**{name}**; gates {j['gates']}; baselines: "
                   + ", ".join(f"{k} {v['psnr']:.2f} dB / {v['ssim']:.3f}"
                               for k, v in j["baselines"].items()) + "\n\n" + md(g, 3))
    return "\n\n".join(out)


def a10() -> str:
    rows = []
    hf = json.loads((LOGS / "hair_frozen_config.json").read_text())
    rows.append(["hair screening config sha256", hf["screen"].get("config_hash")])
    rows.append(["DullRazor config sha256", hf["dullrazor"].get("config_hash")])
    for f in ("stageA_fixed50k_partial_noise.json", "stageA_released_partial_noise.json"):
        j = json.loads((DIAG / f).read_text())
        rows.append([f"checkpoint {j['name']}", j["checkpoint_sha256"]])
    j = json.loads((LOGS / "stageB_val_composite_eval.json").read_text())
    for r in j["rows"]:
        rows.append([f"checkpoint {r['model']}", r["checkpoint_sha256"]])
    tm = json.loads((DIAG / "dmahr_v2_training_manifest.json").read_text())
    rows.append(["v2 training manifest sha256", tm["train_manifest_sha256"]])
    rows.append(["v2 init_from", tm["init_from"]])
    for s in SEEDS:
        m = json.loads((PRED / f"metrics_Cv2_seed{s}.json").read_text())
        rows.append([f"Cv2 ViT seed {s} checkpoint sha256", m["checkpoint_sha256"]])
        rows.append([f"Cv2 ViT seed {s} splits sha256", m["splits_sha256"]])
    return "### A10. Provenance hashes\n\n" + md(pd.DataFrame(rows, columns=["item", "value"]))


def a11() -> str:
    rows = [json.loads(line) for line in (LOGS / "run_registry.jsonl").read_text().splitlines()
            if line.strip()]
    agg = collections.defaultdict(lambda: [0, 0.0, set()])
    for r in rows:
        a = agg[r.get("job")]
        a[0] += 1
        a[1] += float(r.get("seconds") or 0)
        a[2].add(str(r.get("gpu")))
    df = pd.DataFrame([[k, n, s / 60, ", ".join(sorted(g))] for k, (n, s, g) in sorted(agg.items())],
                      columns=["job", "records", "minutes_recorded", "gpu"])
    return ("### A11. Run registry (`artifacts/logs/run_registry.jsonl`, " + str(len(rows))
            + " records). Many records carry seconds=0, so this under-records; see "
            "`reports/compute_plan.md` for the measured per-stage times\n\n" + md(df, 1))


def a12() -> str:
    text = (REPORTS / "prereg_continuation_arm_c.md").read_text()
    i = text.index("## Outcome")
    return ("### A12. Verbatim: outcome of the fixed-trainer continuation (profile on 12 "
            "validation pairs, one noise draw, CPU)\n\n" + text[i:].strip())


def a13() -> str:
    heads = [ln[3:] for ln in (REPORTS / "decision_log.md").read_text().splitlines()
             if ln.startswith("## D")]
    return "### A13. Decision-log index (`reports/decision_log.md`)\n\n" + "\n".join(
        f"- {h}" for h in heads)


def main() -> int:
    tpl = (REPORTS / "second_opinion_brief.template.md").read_text()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, cwd=ROOT).stdout.strip()
    date = subprocess.run(["date", "-u", "+%Y-%m-%d"], capture_output=True, text=True).stdout.strip()
    appendix = "\n\n".join(f() for f in (a1_a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13))
    out = tpl.replace("<!-- APPENDIX -->", appendix).replace("{{HEAD}}", head).replace(
        "{{DATE}}", date)
    assert "{{" not in out and "<!-- APPENDIX -->" not in out
    dest = REPORTS / "SECOND_OPINION_BRIEF.md"
    dest.write_text(out)
    print(f"wrote {dest} ({len(out.splitlines())} lines, {len(out)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
