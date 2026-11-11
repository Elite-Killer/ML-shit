---
name: stats-and-final-report
description: Run the paired lesion-cluster bootstrap, Holm correction and Wilcoxon tests, compute image-quality summaries, then build the final report, artifact inventory, acceptance check and handover. Use for Phases 9, 11 and 12.
---

# Statistics, final report and handover

## Image quality (Phase 9)
For each held-out pair compute PSNR and SSIM for: hairy input, Arm B output, Arm C output, all against the hairless reference. One declared policy: same size, same data range (`data_range=1.0` on float RGB, or 255 on uint8), SSIM with `channel_axis`. Pair-level CSV, then mean ± SD overall and per class, for DERMAHAIR pairs and own-synthetic pairs in separate tables. Statistics: paired bootstrap over source `group_id`, Holm across B vs input, C vs input, C vs B (state the family you chose before the run).

## Planned comparisons (Phase 11)
Pairs: A–B, A–C, B–C (and A\*–C labelled as an extra). Holm correction over the three planned pairs per primary endpoint.
Primary endpoints (fixed in `experiment_protocol.md` before the test run): classification macro-F1 on the full test set; PSNR; SSIM; lesion-overlap at k=20% for Grad-CAM and for rollout separately on X1. All else is secondary.

## Lesion-cluster bootstrap (classification and X1 localization)
1. Group ids from `splits.csv`. Precompute per-group sufficient statistics: for classification a class×class confusion matrix per arm and seed; for localization the per-group sum of overlap and count.
2. For replicate b = 1…10,000 with a fixed RNG seed: resample groups with replacement; sum the statistics; compute the metric for each arm and seed; **average over seeds**; compute the difference for each planned pair.
3. 95% percentile CI. Two-sided p = `2·min(P(d≤0), P(d≥0))` using (count+1)/(B+1). Record B, seed, failed replicates (a resample missing a class → metric undefined: count it, do not impute).
4. Holm over the three p-values per endpoint; report raw p, adjusted p, effect (mean difference), CI, test name, bootstrap config.
5. Proposal-faithful companion: aggregate each group's mean correctness (averaged over seeds) per arm; Wilcoxon signed-rank on paired group-level differences; Holm. Report beside the bootstrap. If they disagree, say so.
6. External X2 images: bootstrap over images. Never pool tiers.
Tests: bootstrap is reproducible for a fixed seed; Holm gives the known textbook result on a hand example; the cluster bootstrap never splits a group.
Output: `artifacts/metrics/statistical_comparisons.csv` with columns `endpoint, comparison, estimate, ci_low, ci_high, test, p_raw, p_holm, n_groups, n_images, bootstrap_B, bootstrap_seed, notes`.

## Words you may and may not use
"Significant" only with test name and adjusted p. A CI that includes zero is "no detectable difference at this sample size", not "no difference". Never write H1–H3 as findings unless the numbers support them; say which were supported, not supported, or inconclusive.

## Final report (`reports/final_research_report.md`, plus PDF/DOCX export if tools exist)
Sections: abstract; objective and hypotheses; data (collection 212 vs C0 vs extras, actual counts, licences, audit); split and leakage checks with hashes; hair screening and DullRazor-style parameters; DM-AHR provenance, config, training, leakage control, resolution control (A\*); ViT recipe and seeds; classification results (full and hair-present); image quality; XAI (tiers separate, chance baselines); statistics; failure cases (diffusion hallucinations, hair-detector errors, XAI on artefacts); limitations; compute and runtime actually used; non-clinical statement; licence caveats (CC BY-NC data and DM-AHR README/LICENSE conflict); reproduction commands and versions; references and third-party notices.
Every table/figure cites its CSV or image path. No placeholder numbers. State plainly which hypotheses held.

## Artifact inventory and acceptance
- `reports/artifact_inventory.csv`: path, description, SHA-256, size, producing script, git commit.
- Run `scripts/check_acceptance.py`; fix failures or document blockers; save `artifacts/logs/acceptance_report.json`.
- README: step-by-step Modal commands from a clean checkout, secrets needed, volume names, and `modal volume get` commands for checkpoints.
- `scripts/export_handover.py` bundles small artifacts; large ones stay on volumes with exact retrieval commands.

## Final message to the user
Short: what was produced, where, which gates passed or are blocked, the headline findings with their CIs, the licence caveat, and the one next step if any. No recap of the process.
