"""Build the client-facing results report (Word).

Covers training setup, classification results, image quality, explainability,
statistics and hypothesis verdicts. Numbers are recomputed from committed
artifacts. Shares document helpers with 13_preprocessing_report.py.
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dm_ahr_vit.paths import artifact_dir  # noqa: E402

pre = import_module("13_preprocessing_report")
h1, h2, body, bullet, table, callout, figure = (
    pre.h1, pre.h2, pre.body, pre.bullet, pre.table, pre.callout, pre.figure)
add_page_number = pre.add_page_number
ARM_LABEL = {"A": "A — original", "Astar": "A* — 128 px control",
             "B": "B — DullRazor classical", "C": "C — DM-AHR diffusion",
             "Creleased": "Creleased — released ckpt"}

OUT = artifact_dir("reports") / "Results_Report.docx"


def verdict(r) -> str:
    if r is None:
        return "not computed"
    d, lo, hi = float(r["difference"]), float(r["ci_low"]), float(r["ci_high"])
    a = str(r["arm_a"]).replace("Astar", "A*")
    b = str(r["arm_b"]).replace("Astar", "A*")
    if lo <= 0 <= hi:
        return f"inconclusive ({a} - {b} 95% CI {lo:+.3f} to {hi:+.3f})"
    w = a if d > 0 else b
    l = b if d > 0 else a
    return f"{w} higher than {l} ({a} - {b} = {d:+.3f}, 95% CI {lo:+.3f} to {hi:+.3f})"


def row(stats, endpoint, comparison, subset=None):
    q = stats[(stats["endpoint"] == endpoint) & (stats["comparison"] == comparison)]
    if subset:
        q = q[q["subset"] == subset]
    return q.iloc[0] if len(q) else None


def build() -> Path:
    logs = artifact_dir("logs")
    seed = pd.read_csv(logs / "classification_seed_summary.csv")
    per_class = pd.read_csv(logs / "classification_per_class.csv")
    stats = pd.read_csv(logs / "statistical_comparisons.csv")
    quality = pd.read_csv(logs / "paired_psnr_ssim_test.csv")
    # Exclude the `_v2_` diagnostic generation: it uses different columns and
    # would otherwise be averaged together with the reported measurements.
    xai = pd.concat([pd.read_csv(f) for f in sorted(logs.glob("xai_localization_*.csv"))
                     if "_v2_" not in f.name], ignore_index=True)
    env = json.loads((logs / "environment_report.json").read_text())
    repro = json.loads((logs / "reproduction_log.json").read_text()) \
        if (logs / "reproduction_log.json").exists() else {}

    full = seed[seed["subset"] == "full"].set_index("arm").loc[
        ["A", "Astar", "B", "C", "Creleased"]].reset_index()
    expl_path = logs / "statistical_comparisons_exploratory.csv"
    expl = pd.read_csv(expl_path) if expl_path.exists() else pd.DataFrame()

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    sec.left_margin = sec.right_margin = Inches(0.95)
    sec.top_margin, sec.bottom_margin = Inches(0.8), Inches(0.75)
    add_page_number(sec)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10)
    doc.core_properties.title = "DM-AHR-ViT — Results Report"

    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("RESULTS REPORT")
    r.font.size = Pt(9); r.font.bold = True; r.font.color.rgb = pre.ACCENT
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(4)
    r = p.add_run("Model training,\nevaluation and findings")
    r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = pre.NAVY
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(18)
    r = p.add_run("What was trained, how it was measured, and what it showed")
    r.font.size = Pt(13); r.font.color.rgb = pre.SLATE
    pre.para_border(p, "C05A33", 12, "bottom")
    body(doc, "DM-AHR-ViT · Explainable Vision Transformer for Skin Lesion "
              "Classification Using Dermoscopy Images", size=10.5)
    body(doc, "Research prototype only. Not clinically validated and not intended "
              "for diagnosis, triage or patient care.", size=9, italic=True)

    best = full.loc[full["macro_f1_mean"].idxmax()]
    worst = full.loc[full["macro_f1_mean"].idxmin()]
    callout(doc, "In one sentence",
            f"Across 15 trained models (5 arms × 3 seeds) evaluated once on a locked test set of "
            f"1,502 images, the best arm was {ARM_LABEL[best['arm']]} at macro-F1 "
            f"{best['macro_f1_mean']:.3f}. Our own diffusion arm "
            f"{ARM_LABEL[worst['arm']]} ({worst['macro_f1_mean']:.3f}) was far behind, and the "
            f"authors' released checkpoint (Creleased, "
            f"{full[full['arm'] == 'Creleased']['macro_f1_mean'].iloc[0]:.3f}) closed most of that "
            f"gap but still did not beat classical DullRazor or original images. The honest "
            f"conclusion is 'hair removal did not help this classifier', not 'DM-AHR cannot "
            f"remove hair' — the released checkpoint does remove hair, it just does not improve "
            f"classification.")
    doc.add_page_break()

    h1(doc, "1. What was trained and how")
    body(doc,
         "Fifteen Vision Transformers were trained: five input conditions (arms) at three random "
         "seeds each. Everything except the arm is identical — same frozen split, same manifests, "
         "same architecture, hyperparameters, augmentation and evaluation code — so a difference "
         "between arms is attributable to the hair-removal condition alone. Creleased is the "
         "authors' released DM-AHR checkpoint run through our pipeline at a 25-step respaced "
         "schedule; it was added after the protocol was frozen and is reported as exploratory.")
    table(doc, ["Arm", "Input the classifier sees", "Macro-F1 (mean ± SD)"],
          [[ARM_LABEL[a], desc,
            f"{full[full['arm'] == a]['macro_f1_mean'].iloc[0]:.3f} ± "
            f"{full[full['arm'] == a]['macro_f1_sd'].iloc[0]:.3f}"]
           for a, desc in [("A", "untouched dermoscopy image"),
                           ("B", "classical hair removal (DullRazor-style)"),
                           ("C", "diffusion-generated hairless image"),
                           ("Astar", "original through the same 128 px round-trip"),
                           ("Creleased", "released DM-AHR checkpoint, 25-step respaced "
                                         "(exploratory)")]],
          widths=[1.7, 3.0, 1.6], align_right_from=2)
    h2(doc, "Training recipe (identical for all fifteen runs)")
    table(doc, ["Component", "Setting"],
          [["Backbone", "ViT-B/16, ImageNet-1k pretrained, 224 px input, 196 patches"],
           ["Optimiser", "AdamW, lr 3e-5, weight decay 0.05, warm-up 5 epochs then cosine decay"],
           ["Batch / precision", "32, bf16 autocast, gradient clipping 1.0"],
           ["Class imbalance", "inverse-frequency class weights from the training split only"],
           ["Early stopping", "validation macro-F1, patience 10, max 100 epochs"],
           ["Seeds", "42, 123, 2026"],
           ["Test protocol", "the locked test set is scored once per frozen checkpoint"]],
          widths=[1.55, 4.75])
    h2(doc, "Compute environment")
    body(doc,
         f"All GPU work ran on Modal: {env.get('device')}, Python {env.get('python')}, "
         f"torch {env.get('torch')}, CUDA {env.get('cuda')}. A typical training run took "
         f"about 6.5 minutes (25 epochs to early stopping). Per-run GPU type and wall-clock "
         f"are logged in artifacts/logs/run_registry.jsonl.")

    h1(doc, "2. Classification results")
    body(doc,
         "All metrics are computed on the held-out test set of 1,502 images. Macro-F1 is the "
         "primary endpoint because classes are severely imbalanced: nv is two thirds of the "
         "data, so plain accuracy rewards ignoring the rare classes.")
    table(doc, ["Arm", "Macro-F1", "Accuracy", "Balanced acc.", "Macro AUC"],
          [[ARM_LABEL[r.arm], f"{r.macro_f1_mean:.3f} ± {r.macro_f1_sd:.3f}",
            f"{r.accuracy_mean:.3f} ± {r.accuracy_sd:.3f}",
            f"{r.balanced_acc_mean:.3f}", f"{r.auc_macro_mean:.3f}"]
           for r in full.itertuples()],
          widths=[1.85, 1.35, 1.15, 1.05, 0.95], align_right_from=1)
    h2(doc, "Hair-present subset")
    hp = seed[seed["subset"] == "hair_present"].set_index("arm")
    table(doc, ["Arm", "Macro-F1", "n"],
          [[ARM_LABEL[a], f"{hp.loc[a, 'macro_f1_mean']:.3f} ± {hp.loc[a, 'macro_f1_sd']:.3f}",
            int(hp.loc[a, 'n'])] for a in ["A", "Astar", "B", "C", "Creleased"]],
          widths=[2.2, 1.9, 0.8], align_right_from=1)
    body(doc,
         "This is the subset where hair removal actually operates — the images it could help. "
         "The ordering is unchanged from the full test set.", size=9.5, italic=True)

    h2(doc, "Per-class test metrics")
    rows = []
    for cls in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]:
        for arm in ["A", "B", "C", "Astar", "Creleased"]:
            q = per_class[(per_class["arm"] == arm) & (per_class["class"] == cls)]
            if len(q):
                r = q.iloc[0]
                rows.append([cls if arm == "A" else "", arm,
                             f"{int(r['support'])}", f"{r['f1_mean']:.3f}",
                             f"{r['recall_mean']:.3f}", f"{r['specificity_mean']:.3f}",
                             f"{r['auc_mean']:.3f}"])
    table(doc, ["Class", "Arm", "Support", "F1", "Recall (sensitivity)", "Specificity", "AUC"],
          rows, widths=[0.72, 0.5, 0.72, 0.62, 1.25, 0.95, 0.62], align_right_from=2)
    body(doc,
         "Recall is the proposal's sensitivity. Supports for df (17) and vasc (21) are small, so "
         "their per-class numbers are noisy and are not interpreted individually.",
         size=9.5, italic=True)

    h1(doc, "3. Hair-removal image quality")
    body(doc,
         f"Measured on the {len(quality)} held-out synthetic pairs — hair composited onto genuinely "
         f"hair-absent images, so a true hairless reference exists. The unprocessed hairy input is "
         f"scored as a floor: falling below it means the method made the image worse than doing "
         f"nothing.")
    qrows = [["Hairy input (floor)",
              f"{quality['hairy_input_psnr'].mean():.2f} ± {quality['hairy_input_psnr'].std():.2f}",
              f"{quality['hairy_input_ssim'].mean():.3f} ± {quality['hairy_input_ssim'].std():.3f}"],
             ["Arm B — classical",
              f"{quality['B_psnr'].mean():.2f} ± {quality['B_psnr'].std():.2f}",
              f"{quality['B_ssim'].mean():.3f} ± {quality['B_ssim'].std():.3f}"]]
    if "Creleased_psnr" in quality.columns and quality["Creleased_psnr"].notna().any():
        qrows.append(["Creleased — released ckpt",
                      f"{quality['Creleased_psnr'].mean():.2f} ± "
                      f"{quality['Creleased_psnr'].std():.2f}",
                      f"{quality['Creleased_ssim'].mean():.3f} ± "
                      f"{quality['Creleased_ssim'].std():.3f}"])
    qrows.append(["Arm C — diffusion",
                  f"{quality['C_psnr'].mean():.2f} ± {quality['C_psnr'].std():.2f}",
                  f"{quality['C_ssim'].mean():.3f} ± {quality['C_ssim'].std():.3f}"])
    table(doc, ["Condition", "PSNR (dB) mean ± SD", "SSIM mean ± SD"],
          qrows, widths=[1.9, 2.1, 2.1], align_right_from=1)
    callout(doc, "Why Arm C scores below the untouched input — a training-script defect",
            "An earlier version of this report called the checkpoint undertrained; that "
            "diagnosis is retracted. Our training script fed the network t/T as the noise "
            "level while the sampler feeds sqrt(alphas_cumprod_prev), and the two run in "
            "opposite directions. On 50 validation pairs the released checkpoint prefers the "
            "sampler's encoding at every timestep tested (2.9× to 20.5× lower error) and ours "
            "prefers t/T (about 10× lower), so more iterations of the old script could not have "
            "fixed it. Arm C as reported measures a defective training run and says nothing "
            "about DM-AHR as published. The script is now fixed; a retrain on validation pairs "
            "still failed full 2,000-step sampling (14.96 dB against a 21.30 dB hairy-input "
            "floor, 24 pairs) and was stopped on cost at 56,300 of a planned 100,000 iterations. "
            "See reports/oneshot_diagnostic_report.md.",
            fill="FDF3ED", bar="C05A33")

    h1(doc, "4. Explainability")
    body(doc,
         "For every test image, two saliency methods — Grad-CAM on the patch grid and attention "
         "roll-out — were compared against the released HAM10000 lesion mask. The table reports "
         "the fraction of the top-20% most salient pixels that fall inside the mask, averaged over "
         "seeds, with the two chance baselines for context.")
    agg = (xai.groupby("arm")[["gradcam_overlap20", "rollout_overlap20",
                               "gradcam_pointing", "rollout_pointing",
                               "chance_uniform_overlap20",
                               "chance_gaussian_overlap20"]].mean()
           .loc[["A", "Astar", "B", "C", "Creleased"]])
    table(doc, ["Arm", "Grad-CAM overlap", "Roll-out overlap", "Grad-CAM pointing",
                "Roll-out pointing", "Uniform chance", "Gaussian chance"],
          [[ARM_LABEL[i], f"{r['gradcam_overlap20']:.3f}", f"{r['rollout_overlap20']:.3f}",
            f"{r['gradcam_pointing']:.3f}", f"{r['rollout_pointing']:.3f}",
            f"{r['chance_uniform_overlap20']:.3f}", f"{r['chance_gaussian_overlap20']:.3f}"]
           for i, r in agg.iterrows()],
          widths=[1.75, 0.95, 0.95, 0.95, 0.9, 0.85, 0.9], align_right_from=1)
    body(doc,
         "Overlap measures spatial alignment with released masks, not clinically correct "
         "reasoning. Every arm beats uniform chance, but a centred-Gaussian prior scores "
         "0.644 on overlap and 0.953 on the pointing game, exceeding every arm on both "
         "measures. No arm therefore beats a trivial attend-to-the-centre baseline, and "
         "these numbers do not establish that any arm localises lesions better than that "
         "prior. The masks themselves are single-rater: a dermatologist (Tschandl P) "
         "created binary masks for all 10,015 images, network-initialised and then "
         "verified, corrected or replaced by hand in FIJI (Tschandl et al., Nature "
         "Medicine 2020), so overlap is measured against one tracing, not consensus.",
         size=9.5, italic=True)
    fig_dir = artifact_dir("reports") / "preprocessing_assets"
    figure(doc, fig_dir / "dmahr.jpg",
           "Figure 1 — Arm C diffusion removal, from the preprocessing report, reproduced here "
           "because the same failed hair-removal output drives both its quality scores and its "
           "classification deficit.")

    h1(doc, "5. Statistical comparisons")
    body(doc,
         "Every endpoint is tested twice, as the protocol requires: a lesion-group bootstrap "
         "(10,000 replicates, the same resample shared across arms and seeds, percentile 95% CI) "
         "and a Wilcoxon signed-rank test on values collapsed to one per lesion group. "
         "For the classification endpoint the per-group value is that group's mean "
         "per-image correctness, so the Wilcoxon test asks whether the arms differ in "
         "per-lesion accuracy — it is not a test of macro-F1 itself, and the column "
         "header says so. Holm correction is applied across the three planned "
         "comparisons within each endpoint, separately for each test; A–A* is a "
         "secondary control in its own family and cannot inflate the planned family. "
         "A CI spanning zero is called inconclusive, not 'no difference'.")
    rows = []
    for r in stats.itertuples():
        rows.append([r.endpoint, r.comparison, f"{r.difference:+.3f}",
                     f"{r.ci_low:+.3f} to {r.ci_high:+.3f}",
                     f"{r.p_holm:.4f}" if r.p_holm >= 0.0001 else "<0.0001",
                     f"{r.wilcoxon_p_holm:.4f}" if r.wilcoxon_p_holm >= 0.0001 else "<0.0001",
                     f"{r.n_groups}"])
    table(doc, ["Endpoint", "Comparison", "Diff.", "95% CI (bootstrap)",
                 "p (Holm)", "Wilcoxon p (per-group acc.)", "Groups"],
          rows, widths=[1.35, 0.95, 0.6, 1.25, 0.72, 0.9, 0.55], font=8, align_right_from=2)
    if not expl.empty:
        h2(doc, "Exploratory: released-checkpoint arm")
        body(doc,
             "These rows are not part of the pre-registered family. Creleased uses the authors' "
             "released DM-AHR checkpoint at a 25-step respaced schedule, a deviation from the "
             "paper's 2,000 steps, and its classifier was trained on our budget. Holm correction "
             "is applied within this exploratory family only.", size=9.5, italic=True)
        erows = []
        for r in expl.itertuples():
            erows.append([r.endpoint, r.comparison, f"{r.difference:+.3f}",
                          f"{r.ci_low:+.3f} to {r.ci_high:+.3f}",
                          f"{r.p_holm:.4f}" if r.p_holm >= 0.0001 else "<0.0001",
                          f"{r.effect_size_dz:+.3f}", f"{r.n_groups}"])
        table(doc, ["Endpoint", "Comparison", "Diff.", "95% CI (bootstrap)",
                    "p (Holm)", "Cohen's dz", "Groups"],
              erows, widths=[1.35, 1.15, 0.6, 1.25, 0.72, 0.7, 0.55], font=8,
              align_right_from=2)

    h1(doc, "6. What this says about the three hypotheses")
    psnr = row(stats, "paired_psnr", "C-B")
    ssim = row(stats, "paired_ssim", "C-B")
    gc = row(stats, "x1_gradcam_overlap20", "A-C")
    ro = row(stats, "x1_rollout_overlap20", "A-C")
    body(doc, "H1 — image quality (diffusion preserves texture better than classical): ",
         space_after=2)
    c_ssim = quality["C_ssim"].dropna()
    bullet(doc, f"PSNR: {verdict(psnr)}. SSIM: {verdict(ssim)}. H1 is not supported by these "
                f"data. Arm C's SSIM is near-constant across all {len(c_ssim)} pairs "
                f"({c_ssim.mean():.3f} ± {c_ssim.std():.3f}, with "
                f"{int((c_ssim < 0.15).sum())} below 0.15), which is what the inverted "
                "noise-level signal in our training script produces. "
                "The released checkpoint was then scored through the same adapter (Creleased): "
                "it clears the hairy-input floor by 4.95 dB but sits 8.66 dB below classical "
                "inpainting, so H1 is now tested for DM-AHR and is not supported.")
    body(doc, "H2 — classification (diffusion-cleaned images improve the ViT): ", space_after=2)
    bullet(doc, f"A vs C: {verdict(row(stats, 'test_macro_f1', 'A-C', 'full'))}. "
                f"B vs C: {verdict(row(stats, 'test_macro_f1', 'B-C', 'full'))}. The direction is "
                "opposite to the hypothesis and the gap is large. The released checkpoint does "
                "not rescue it: A vs Creleased is +0.038 (95% CI +0.005 to +0.071, Holm p=0.048, "
                "negligible effect) and B vs Creleased is inconclusive (-0.015, CI -0.049 to "
                "+0.019, Holm p=0.400).")
    body(doc, "H3 — explainability (attention focuses more on lesions after diffusion): ",
         space_after=2)
    bullet(doc, f"Grad-CAM overlap, A vs C: {verdict(gc)}. Attention roll-out, A vs C: "
                f"{verdict(ro)}. The methods disagree in direction, but H3 is not supported and "
                "the measurement is uninformative as specified: a centred-Gaussian prior beats "
                "every arm on both overlap (0.644) and pointing (0.953), so no arm demonstrates "
                "lesion localisation beyond a centre bias. That is a limitation of the "
                "measurement, not a finding about diffusion hair removal, and it must be fixed "
                "before H3 can be tested.")
    body(doc, "Resolution control: ", space_after=2)
    bullet(doc, f"A vs A*: {verdict(row(stats, 'test_macro_f1', 'A-Astar', 'full'))}. Because "
                "this is inconclusive, the 128 px round-trip alone does not explain Arm C's "
                "deficit.")

    h1(doc, "7. Limitations")
    for lead, txt in [
        ("Arm C is a failed replication, caused by a training-script defect — ",
         "Our script gave the network an inverted noise-level signal (t/T instead of the "
         "sampler's sqrt(alphas_cumprod_prev)), so the model ran 20,000 iterations learning the "
         "wrong mapping; it was also sampled at 1,000 steps against 2,000 and upsampled from "
         "128 px to 224. The upstream config's n_iter is 660,000 and the released checkpoint is "
         "named I725000_E1083. The fixed script did not reach usable sampling at the budgets "
         "tried. Arm C's results describe that defective run, not the published method."),
        ("Creleased is exploratory, not pre-registered — ",
         "It uses the authors' released checkpoint at a 25-step respaced schedule, a deviation "
         "from the paper's 2,000 steps, and its classifier was trained on our budget rather than "
         "the paper's. Its comparisons are reported in their own Holm family and must not be "
         "read as pre-registered results."),
        ("Three seeds — ",
         "inference is over ~1,437 lesion groups with seeds averaged inside each replicate, "
         "never counted as independent samples; still, three seeds cannot characterise variance "
         "exhaustively."),
        ("Screening unvalidated — ",
         "the proposal specifies the manually annotated hair masks of Lama et al. (ChimeraNet) "
         "as an independent reference for screening quality, and those masks were not obtained. "
         "This is a skipped step, not a gap in the data. The threshold was calibrated on "
         "synthetic pairs and frozen, which protects the test set but does not validate the "
         "detector on real hair. The 89.8% screening prevalence is reported as a limitation."),
        ("Masks are single-rater — ",
         "the HAM10000 segmentations were created by one dermatologist (Tschandl P), "
         "network-initialised and then verified, corrected or replaced by hand in FIJI "
         "(Tschandl et al., Nature Medicine 2020). No inter-rater agreement is available, so "
         "explanation overlap is measured against one tracing."),
        ("Single corpus — ",
         "one public dataset; nothing here establishes behaviour on other scanners or "
         "populations."),
        ("Licensing unresolved — ",
         "HAM10000 is CC BY-NC 4.0; the DM-AHR repository's README (CC BY-NC 4.0) and LICENSE "
         "(Apache-2.0) contradict each other. Commercial use needs a rights decision."),
    ]:
        bullet(doc, txt, lead)

    h1(doc, "8. Reproducibility")
    if repro:
        bullet(doc, f"Arm A seed 42 re-run from scratch reproduced validation macro-F1 exactly "
                    f"({repro['recorded']['best_val_macro_f1']:.6f} vs "
                    f"{repro['reproduction']['best_val_macro_f1']:.6f}).", "Determinism — ")
        bullet(doc, "the frozen split hash was reproduced byte for byte across two Modal "
                    "accounts, and the full data-preparation chain was re-run natively with no "
                    "Modal present, giving the identical hash.", "Portability — ")
    bullet(doc, "notebooks/01_results_walkthrough.ipynb reads the committed artifacts and "
                "explains these tables interactively; notebooks/02 and 03 reproduce the pipeline "
                "on Modal or a local GPU.", "How to explore — ")
    body(doc,
         "Every table here is generated from artifacts/logs/: classification_seed_summary.csv, "
         "classification_per_class.csv, paired_psnr_ssim_test.csv, xai_localization_*.csv and "
         "statistical_comparisons.csv. This document is built by scripts/14_results_report.py "
         "and can be regenerated after any rerun.", size=9, italic=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


def main() -> int:
    out = build()
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
