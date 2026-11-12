# DM-AHR-ViT: diffusion hair removal and ViT classification on HAM10000

**Research prototype only. Not clinically validated and not intended for diagnosis, triage or patient care.**

## Summary

Test macro-F1 by arm (mean ± SD over seeds): Original 0.750 ± 0.005; A* (128 px) 0.732 ± 0.008; DullRazor 0.698 ± 0.009; DM-AHR 0.426 ± 0.013.

The hypotheses were that DM-AHR hair removal improves image fidelity, classification and explanation localization relative to the classical baseline. The results below are reported as measured; a null or negative result is a valid outcome and is not reframed.

## Cohort and splits

- Collection 212 images: 11720
- Confirmatory C0 images: 10015
- Quarantined C0-extra: 1705
- Groups after lesion + confirmed near-duplicate merging: 7470
- Split sizes: {'test': 1502, 'train': 7011, 'val': 1502}
- Split SHA-256: `fda56cf3c555e6d22f823510670e1392cd9b1b27778a50a831e68f1dd8d5be2b`
- Leakage check: True

Recomputed class counts matched the expected HAM10000 counts exactly (True).

## Hair screening and arms

- Hair-present images: 8993 of 10015 (89.8%)
- Screening config hash: `e232efe4705133a175ec50d5147954a6545456d469b6353fb42d348387704273`
- DullRazor config hash: `0af616f50f77e1c57c4fff1e1f6bb8c2ab99100052c703d937bccbbf109a9d70`
- Synthetic pairs: 1022 ({'test': 186, 'train': 654, 'val': 182})

Hair-absent images are identical across arms. Arm A* exists because DM-AHR is native 128 px, below the 224 px ViT input.

## DM-AHR audit and training

- Pinned commit: `0aee1bbabf788a59be5e1eecddc20f533b5006d8`
- Released checkpoint SHA-256: `c99e406050d3aaf36473f76477bcfb8ba05d80a420d50d57533eac75b8c8d2fe`
- Smoke test passed: True (geometry preserved True, reload identical True)
- Sampling throughput: {'8': {'returned': 8, 'seconds': 48.8, 'seconds_per_image': 6.1}, '16': {'returned': 16, 'seconds': 75.28, 'seconds_per_image': 4.705}, '32': {'returned': 32, 'seconds': 138.96, 'seconds_per_image': 4.343}, '64': {'returned': 64, 'seconds': 261.65, 'seconds_per_image': 4.088}}
- Leakage-controlled training: 20000 iterations, best validation L1 0.01703

The released weights are used for the smoke test only, because the authors do not document their training data. Arm C uses the checkpoint trained here on train-split sources only.

## Classification results

**Arm C was run as a budget-forced ablation: 1,000 sampling steps instead of the reference 2,000.** The account balance could not cover the full-step run, so Arm C is half the reference sampling budget. Its results are reported as an ablation and must not be read as DM-AHR at the paper's settings. See decision log D12.

### Mean ± SD over seeds

| arm | subset | seeds | macro_f1_mean | macro_f1_sd | accuracy_mean | accuracy_sd | balanced_acc_mean | auc_macro_mean | n |
|---|---|---|---|---|---|---|---|---|---|
| A | full | 3 | 0.750 | 0.005 | 0.858 | 0.012 | 0.735 | 0.969 | 1502 |
| Astar | full | 3 | 0.732 | 0.008 | 0.851 | 0.006 | 0.726 | 0.969 | 1502 |
| B | full | 3 | 0.698 | 0.009 | 0.842 | 0.005 | 0.687 | 0.965 | 1502 |
| C | full | 3 | 0.426 | 0.013 | 0.685 | 0.019 | 0.452 | 0.878 | 1502 |
| A | hair_present | 3 | 0.749 | 0.002 | 0.843 | 0.013 | 0.736 | 0.966 | 1316 |
| Astar | hair_present | 3 | 0.735 | 0.013 | 0.836 | 0.008 | 0.735 | 0.966 | 1316 |
| B | hair_present | 3 | 0.691 | 0.009 | 0.824 | 0.006 | 0.680 | 0.960 | 1316 |
| C | hair_present | 3 | 0.411 | 0.017 | 0.651 | 0.018 | 0.440 | 0.864 | 1316 |


### Per-class test metrics (full test split)

| arm | class | support | f1_mean | f1_sd | recall_mean | specificity_mean | auc_mean |
|---|---|---|---|---|---|---|---|
| A | akiec | 49 | 0.625 | 0.034 | 0.694 | 0.982 | 0.956 |
| A | bcc | 77 | 0.750 | 0.035 | 0.706 | 0.991 | 0.975 |
| A | bkl | 165 | 0.676 | 0.036 | 0.644 | 0.967 | 0.955 |
| A | df | 17 | 0.733 | 0.078 | 0.706 | 0.998 | 0.987 |
| A | mel | 167 | 0.634 | 0.019 | 0.615 | 0.959 | 0.939 |
| A | nv | 1006 | 0.943 | 0.006 | 0.956 | 0.855 | 0.981 |
| A | vasc | 21 | 0.889 | 0.015 | 0.825 | 1.000 | 0.991 |
| B | akiec | 49 | 0.589 | 0.056 | 0.639 | 0.982 | 0.947 |
| B | bcc | 77 | 0.724 | 0.021 | 0.740 | 0.984 | 0.979 |
| B | bkl | 165 | 0.628 | 0.009 | 0.590 | 0.965 | 0.947 |
| B | df | 17 | 0.591 | 0.066 | 0.529 | 0.997 | 0.978 |
| B | mel | 167 | 0.561 | 0.039 | 0.529 | 0.957 | 0.935 |
| B | nv | 1006 | 0.942 | 0.002 | 0.958 | 0.844 | 0.980 |
| B | vasc | 21 | 0.853 | 0.014 | 0.825 | 0.998 | 0.991 |
| C | akiec | 49 | 0.286 | 0.019 | 0.408 | 0.952 | 0.911 |
| C | bcc | 77 | 0.315 | 0.043 | 0.290 | 0.971 | 0.881 |
| C | bkl | 165 | 0.356 | 0.024 | 0.461 | 0.863 | 0.791 |
| C | df | 17 | 0.140 | 0.073 | 0.137 | 0.990 | 0.863 |
| C | mel | 167 | 0.400 | 0.024 | 0.373 | 0.939 | 0.820 |
| C | nv | 1006 | 0.871 | 0.018 | 0.827 | 0.855 | 0.922 |
| C | vasc | 21 | 0.615 | 0.090 | 0.667 | 0.993 | 0.955 |
| Astar | akiec | 49 | 0.556 | 0.042 | 0.673 | 0.974 | 0.961 |
| Astar | bcc | 77 | 0.747 | 0.051 | 0.719 | 0.989 | 0.982 |
| Astar | bkl | 165 | 0.677 | 0.015 | 0.630 | 0.971 | 0.949 |
| Astar | df | 17 | 0.719 | 0.046 | 0.627 | 0.999 | 0.985 |
| Astar | mel | 167 | 0.613 | 0.011 | 0.611 | 0.952 | 0.925 |
| Astar | nv | 1006 | 0.943 | 0.002 | 0.949 | 0.872 | 0.982 |
| Astar | vasc | 21 | 0.868 | 0.024 | 0.873 | 0.998 | 0.999 |


Undefined AUC values are reported as NaN with a reason in the per-run metric JSON files; `df` and `vasc` have small test supports.

## Paired image quality (held-out synthetic pairs)

- Pairs scored: 186 (Arm C scored: 186)

| B_psnr | B_ssim | C_psnr | C_ssim | hairy_input_psnr | hairy_input_ssim |
|---|---|---|---|---|---|
| 35.732 | 0.918 | 16.312 | 0.130 | 22.127 | 0.520 |


PSNR/SSIM are computed only against genuine hairless references from synthetic pairs. Naturally hairy images are never scored against a guessed reference.


**Arm C image quality is poor and this is a real finding, not a scoring artefact.** The leakage-controlled DM-AHR checkpoint was trained for 20,000 iterations, whereas the upstream config specifies 660,000. Its outputs are visibly over-smoothed: on the same 186 held-out pairs the output standard deviation collapses relative to the input, so it scores below the unprocessed hairy image. Arm C therefore cannot be read as a fair test of DM-AHR as published; it tests an undertrained model. A faithful Arm C needs the full iteration budget.

## Explainability

| arm | seed | tier | n | gradcam_overlap20 | rollout_overlap20 | gradcam_pointing | rollout_pointing | chance_uniform | chance_gaussian |
|---|---|---|---|---|---|---|---|---|---|
| A | 123 | X1 | 1502 | 0.439 | 0.514 | 0.565 | 0.754 | 0.200 | 0.644 |
| A | 2026 | X1 | 1502 | 0.389 | 0.556 | 0.525 | 0.867 | 0.200 | 0.644 |
| A | 42 | X1 | 1502 | 0.223 | 0.601 | 0.303 | 0.901 | 0.200 | 0.644 |
| Astar | 123 | X1 | 1502 | 0.477 | 0.540 | 0.637 | 0.784 | 0.200 | 0.644 |
| Astar | 2026 | X1 | 1502 | 0.357 | 0.607 | 0.446 | 0.926 | 0.200 | 0.644 |
| Astar | 42 | X1 | 1502 | 0.378 | 0.585 | 0.524 | 0.864 | 0.200 | 0.644 |
| B | 123 | X1 | 1502 | 0.411 | 0.547 | 0.463 | 0.851 | 0.200 | 0.644 |
| B | 2026 | X1 | 1502 | 0.354 | 0.538 | 0.458 | 0.797 | 0.200 | 0.644 |
| B | 42 | X1 | 1502 | 0.390 | 0.605 | 0.519 | 0.871 | 0.200 | 0.644 |
| C | 123 | X1 | 1502 | 0.381 | 0.322 | 0.572 | 0.212 | 0.200 | 0.644 |
| C | 2026 | X1 | 1502 | 0.406 | 0.342 | 0.608 | 0.260 | 0.200 | 0.644 |
| C | 42 | X1 | 1502 | 0.448 | 0.347 | 0.670 | 0.255 | 0.200 | 0.644 |


Localization measures spatial alignment with released lesion masks, not clinically correct reasoning. X1 masks are the released HAM10000 segmentations; the source does not state they were expert-traced, so they are not described as expert masks.

## Statistical comparisons

| endpoint | subset | comparison | arm_a_mean | arm_b_mean | difference | ci_low | ci_high | p_raw | p_holm | effect_size_dz | n_groups |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paired_psnr | held_out_pairs | B-hairy input | 35.732 | 22.127 | 13.604 | 13.261 | 13.943 | 0.000 | 0.000 | 5.759 | 186 |
| paired_psnr | held_out_pairs | C-B | 16.312 | 35.732 | -19.422 | -19.768 | -19.087 | 0.000 | 0.000 | -8.190 | 186 |
| paired_psnr | held_out_pairs | C-hairy input | 16.312 | 22.127 | -5.819 | -6.152 | -5.490 | 0.000 | 0.000 | -2.508 | 186 |
| paired_ssim | held_out_pairs | B-hairy input | 0.918 | 0.520 | 0.398 | 0.385 | 0.410 | 0.000 | 0.000 | 4.666 | 186 |
| paired_ssim | held_out_pairs | C-B | 0.130 | 0.918 | -0.788 | -0.791 | -0.784 | 0.000 | 0.000 | -31.204 | 186 |
| paired_ssim | held_out_pairs | C-hairy input | 0.130 | 0.520 | -0.390 | -0.404 | -0.377 | 0.000 | 0.000 | -4.138 | 186 |
| test_macro_f1 | full | A-Astar | 0.750 | 0.732 | 0.018 | -0.008 | 0.045 | 0.167 | 0.167 | 0.048 | 1437 |
| test_macro_f1 | full | A-B | 0.750 | 0.698 | 0.052 | 0.019 | 0.087 | 0.001 | 0.001 | 0.056 | 1437 |
| test_macro_f1 | full | A-C | 0.750 | 0.426 | 0.324 | 0.283 | 0.366 | 0.000 | 0.000 | 0.469 | 1437 |
| test_macro_f1 | full | B-C | 0.698 | 0.426 | 0.272 | 0.225 | 0.320 | 0.000 | 0.000 | 0.430 | 1437 |
| test_macro_f1 | hair_present | A-Astar | 0.749 | 0.735 | 0.014 | -0.014 | 0.041 | 0.322 | 0.322 | 0.049 | 1251 |
| test_macro_f1 | hair_present | A-B | 0.749 | 0.691 | 0.058 | 0.022 | 0.095 | 0.001 | 0.001 | 0.066 | 1251 |
| test_macro_f1 | hair_present | A-C | 0.749 | 0.411 | 0.337 | 0.291 | 0.383 | 0.000 | 0.000 | 0.505 | 1251 |
| test_macro_f1 | hair_present | B-C | 0.691 | 0.411 | 0.279 | 0.224 | 0.333 | 0.000 | 0.000 | 0.456 | 1251 |


Test: paired lesion-group bootstrap, 10,000 replicates, the same resample shared across arms and seeds, percentile 95% CI, two-sided. Holm correction is applied across the three planned comparisons within each endpoint and subset. A difference whose CI spans zero is reported as inconclusive, not as no difference.

## Limitations and failure cases

- DERMAHAIR could not be downloaded (no Kaggle credentials), so the synthetic-hair fallback is used. Held-out test pairs number only 186, so paired CIs are wide.

- No independent hair annotations exist, so the screening threshold is NOT VERIFIED against real hair labels; it was tuned on synthetic pairs and frozen before test scoring.

- DM-AHR runs at 128 px, so Arm C is upsampled to 224 px for the ViT. Arm A* controls for that resolution change but cannot separate it from diffusion artefacts.

- The diffusion model was trained for 20,000 iterations, far below the 660,000 in the upstream config, so Arm C is not a faithful reproduction of the released checkpoint's training budget.

- `df` and `vasc` have very small test supports, so their per-class numbers are noisy.

- The released DM-AHR weights have undocumented training data and are used for exploration only.

## Licence and provenance

HAM10000 is distributed under CC BY-NC 4.0. The DM-AHR repository contradicts itself: its README states CC BY-NC 4.0 while its LICENSE file is Apache-2.0. **Commercial use is unresolved** and needs a rights decision by the client. See `reports/license_audit.md`.

## Reproducibility

Arm A, seed 42 was re-run from scratch through the same entrypoint (gate G13). Recorded validation macro-F1 0.778336 vs reproduced 0.778336, absolute difference 0.000000 against a 0.02 tolerance. Separate runs on two different Modal accounts also reproduced the frozen split hash byte for byte.

Commands are in `README.md`. Frozen artifacts and their hashes are in `artifacts/logs/` and `data/manifests/`. Every table above is generated from a CSV or JSON artifact listed in `artifact_inventory.csv`.
