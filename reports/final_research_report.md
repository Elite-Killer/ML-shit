# DM-AHR-ViT: diffusion hair removal and ViT classification on HAM10000

**Research prototype only. Not clinically validated and not intended for diagnosis, triage or patient care.**

## Summary

Test macro-F1 by arm (mean ± SD over seeds): Original 0.750 ± 0.005; A* (128 px) 0.732 ± 0.008; DullRazor 0.698 ± 0.009; DM-AHR 0.426 ± 0.013; Creleased (released ckpt) 0.713 ± 0.028.

The hypotheses were that DM-AHR hair removal improves image fidelity, classification and explanation localization relative to the classical baseline. The results below are reported as measured; a null or negative result is a valid outcome and is not reframed.

## Cohort and splits

- Collection 212 images: 11720
- Confirmatory C0 images: 10015
- Quarantined C0-extra: 1705
- Lesion groups: 7470
- Confirmed near-duplicate pairs: 21 (all already shared a lesion_id, so grouping is unchanged by them)
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
| Creleased | full | 3 | 0.713 | 0.028 | 0.848 | 0.009 | 0.696 | 0.967 | 1502 |
| A | hair_present | 3 | 0.749 | 0.002 | 0.843 | 0.013 | 0.736 | 0.966 | 1316 |
| Astar | hair_present | 3 | 0.735 | 0.013 | 0.836 | 0.008 | 0.735 | 0.966 | 1316 |
| B | hair_present | 3 | 0.691 | 0.009 | 0.824 | 0.006 | 0.680 | 0.960 | 1316 |
| C | hair_present | 3 | 0.411 | 0.017 | 0.651 | 0.018 | 0.440 | 0.864 | 1316 |
| Creleased | hair_present | 3 | 0.712 | 0.029 | 0.833 | 0.010 | 0.695 | 0.962 | 1316 |


Creleased is the authors' released DM-AHR checkpoint run through our pipeline at a 25-step respaced schedule. It was added after the protocol was frozen, so it is exploratory and is not part of the pre-registered planned family; its comparisons are reported separately in `artifacts/logs/statistical_comparisons_exploratory.csv`.


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
| Creleased | akiec | 49 | 0.562 | 0.048 | 0.633 | 0.979 | 0.954 |
| Creleased | bcc | 77 | 0.726 | 0.012 | 0.736 | 0.984 | 0.976 |
| Creleased | bkl | 165 | 0.692 | 0.021 | 0.679 | 0.965 | 0.946 |
| Creleased | df | 17 | 0.620 | 0.104 | 0.471 | 1.000 | 0.980 |
| Creleased | mel | 167 | 0.609 | 0.028 | 0.617 | 0.949 | 0.936 |
| Creleased | nv | 1006 | 0.942 | 0.004 | 0.941 | 0.884 | 0.980 |
| Creleased | vasc | 21 | 0.840 | 0.053 | 0.794 | 0.999 | 0.994 |


Undefined AUC values are reported as NaN with a reason in the per-run metric JSON files; `df` and `vasc` have small test supports.

## Paired image quality (held-out synthetic pairs)

- Pairs scored: 186 (Arm C scored: 186)

| B_psnr | B_ssim | C_psnr | C_ssim | Creleased_psnr | Creleased_ssim | hairy_input_psnr | hairy_input_ssim |
|---|---|---|---|---|---|---|---|
| 35.732 | 0.918 | 14.292 | 0.074 | 27.076 | 0.703 | 22.127 | 0.520 |


PSNR/SSIM are computed only against genuine hairless references from synthetic pairs. Naturally hairy images are never scored against a guessed reference.


**Arm C image quality is poor, and the cause has been diagnosed.** The leakage-controlled checkpoint was trained for 20,000 iterations where the upstream config specifies 660,000. A diagnosis on validation pairs (see `reports/arm_c_diagnosis.md`) separates the two possible causes: a hairless reference round-trips through the same inference path at PSNR 34.6 / SSIM 0.990, and the released DM-AHR checkpoint reaches PSNR 26.8 / SSIM 0.684 through that path, above the hairy-input floor of 22.1. Our checkpoint reaches PSNR 15.0 / SSIM 0.079, below the floor. The adapter is sound; the checkpoint is undertrained. Its output is not over-smoothed — it carries roughly 15x the reference's high-frequency energy with only a modest drop in global contrast, which is the signature of a model that has not learned the mapping rather than one that has learned to blur. Arm C therefore cannot be read as a fair test of DM-AHR as published.

## Explainability

| arm | seed | tier | n | gradcam_overlap20 | rollout_overlap20 | gradcam_pointing | rollout_pointing | chance_uniform | chance_gaussian |
|---|---|---|---|---|---|---|---|---|---|
| C | 123 | X1 | 1502 | 0.381 | 0.322 | 0.572 | 0.212 | 0.200 | 0.644 |
| C | 2026 | X1 | 1502 | 0.406 | 0.342 | 0.608 | 0.260 | 0.200 | 0.644 |
| C | 42 | X1 | 1502 | 0.448 | 0.347 | 0.670 | 0.255 | 0.200 | 0.644 |
| Creleased | 123 | X1 | 1502 | 0.411 | 0.547 | 0.575 | 0.838 | 0.200 | 0.644 |
| Creleased | 2026 | X1 | 1502 | 0.352 | 0.564 | 0.467 | 0.917 | 0.200 | 0.644 |
| Creleased | 42 | X1 | 1502 | 0.382 | 0.592 | 0.531 | 0.878 | 0.200 | 0.644 |


Localization measures spatial alignment with released lesion masks, not clinically correct reasoning. The X1 masks are the HAM10000 segmentations released on Harvard Dataverse (doi:10.7910/DVN/DBW86T). Their description states that a single dermatologist (Tschandl P) created binary masks for all 10,015 images, initialised with a segmentation network and then verified, corrected or replaced by hand in FIJI (Tschandl et al., Nature Medicine 2020). They are therefore single-rater, network-initialised and dermatologist-corrected. They are not multi-rater consensus masks, and no inter-rater agreement is available, so overlap is reported against one dermatologist's tracing.

## Statistical comparisons

_Statistics not computed._


### Exploratory: released-checkpoint arm

These rows are **not** part of the pre-registered family. Creleased uses the authors' released DM-AHR checkpoint at a 25-step respaced schedule, a deviation from the paper's 2,000 steps, and its classifier was trained on our budget. Holm correction is applied within this exploratory family only.

| endpoint | subset | comparison | arm_a_mean | arm_b_mean | difference | ci_low | ci_high | p_raw | p_holm | effect_size_dz | n_groups |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paired_psnr | held_out_pairs | Creleased-B | 27.076 | 35.732 | -8.657 | -9.063 | -8.254 | 0.000 | 0.000 | -3.073 | 186 |
| paired_psnr | held_out_pairs | Creleased-hairy input | 27.076 | 22.127 | 4.946 | 4.503 | 5.388 | 0.000 | 0.000 | 1.639 | 186 |
| paired_ssim | held_out_pairs | Creleased-B | 0.703 | 0.918 | -0.215 | -0.228 | -0.204 | 0.000 | 0.000 | -2.558 | 186 |
| paired_ssim | held_out_pairs | Creleased-hairy input | 0.703 | 0.520 | 0.182 | 0.166 | 0.199 | 0.000 | 0.000 | 1.595 | 186 |
| test_macro_f1 | full | A-Creleased | 0.750 | 0.713 | 0.038 | 0.005 | 0.071 | 0.024 | 0.048 | 0.053 | 1437 |
| test_macro_f1 | full | B-Creleased | 0.698 | 0.713 | -0.015 | -0.049 | 0.019 | 0.400 | 0.400 | -0.011 | 1437 |
| test_macro_f1 | hair_present | A-Creleased | 0.749 | 0.712 | 0.037 | 0.002 | 0.073 | 0.038 | 0.077 | 0.060 | 1251 |
| test_macro_f1 | hair_present | B-Creleased | 0.691 | 0.712 | -0.021 | -0.057 | 0.016 | 0.261 | 0.261 | -0.016 | 1251 |


A significant but negligible effect (|Cohen's dz| < 0.2) is flagged in the CSV's `negligible` column and is not described as a meaningful difference.

## Compute environment

- All GPU work ran on **Modal**, not on Google Colab: NVIDIA A100 80GB PCIe, Python 3.11.12, torch 2.5.1+cu124, CUDA 12.4.
- The proposal names Google Colab as the implementation environment. That wording is superseded: the project instructions require all GPU jobs to run on Modal, and this study's Arm C sampling budget exceeds free-tier Colab limits. Colab was used only to retrieve artifacts. See decision log D14.
- Per-run GPU type and wall-clock are in `artifacts/logs/run_registry.jsonl`.

## How the results bear on the three hypotheses

The proposal states three hypotheses. Each is answered below only from the tables above; where the evidence does not settle the question it is called inconclusive rather than resolved either way.
**H1 (image quality): diffusion removal preserves lesion boundaries and texture better than classical inpainting.** On 0 held-out synthetic pairs, not computed on PSNR, and not computed on SSIM. H1 is not supported by these data. The caveat is material and unresolved: Arm C's output scores below the unprocessed hairy input, and its SSIM is near-constant across all pairs (0.130 +/- 0.010, with 185 of 186 below 0.15). A uniform collapse of that kind is consistent with either an undertrained model or a defect in the inference path, and the two have not been separated. The released checkpoint has not been scored through this adapter, so H1 is best read as untested for DM-AHR rather than as evidence against it.

**H2 (classification): a ViT trained on diffusion-cleaned images outperforms the original and classically cleaned arms.** A vs C: not computed. B vs C: not computed. The direction is opposite to the hypothesis and the gap is large, but the same unresolved caveat applies: Arm C's image quality is anomalous in a way that has not been attributed to training budget or to the inference path. This is a true statement about the model that was run, not a refutation of the published method.

**H3 (explainability): attention and Grad-CAM focus more on lesions after diffusion removal.** H3 is **not supported, and the measurement is uninformative as specified**. Grad-CAM lesion overlap, A vs C: not computed. Attention roll-out lesion overlap, A vs C: not computed. The two methods disagree in direction, but the decisive problem is the baseline: a centred-Gaussian prior scores 0.644 on overlap and 0.953 on the pointing game, exceeding **every arm on both measures**. No arm beats a trivial attend-to-the-centre prior, so the saliency measures do not demonstrate lesion localisation at all, in any arm. This is a property of the measurement, not a finding about diffusion hair removal, and neither measure establishes clinically correct reasoning. A stronger design would use an empirical random-map baseline per image and a saliency method that can beat the centre prior before H3 is testable.

**Resolution control.** Arm A* isolates the 128 px round-trip that Arm C introduces: A-A* macro-F1 difference not computed. Because that comparison is inconclusive, the resolution change alone does not explain Arm C's large deficit.

## Limitations and failure cases

- DERMAHAIR could not be downloaded (no Kaggle credentials), so the synthetic-hair fallback is used. Held-out test pairs number only 186, so paired CIs are wide.

- The screening threshold is NOT VERIFIED against real hair labels. This is a skipped step, not a property of the data: the proposal specifies the manually annotated hair masks of Lama et al. (ChimeraNet) as an independent reference for screening quality, and those masks were not obtained. The threshold was tuned on synthetic pairs and frozen before test scoring, which protects the test set but does not validate the detector on real hair.

- DM-AHR runs at 128 px, so Arm C is upsampled to 224 px for the ViT. Arm A* controls for that resolution change but cannot separate it from diffusion artefacts.

- The diffusion model was trained for 20,000 iterations. The 660,000 figure is the `n_iter` value in the vendored upstream config `third_party/dmahr/config/akiec.json` at commit `0aee1bb`, and the released checkpoint is named `I725000_E1083` (725,000 iterations), so the two upstream sources agree in magnitude. The project kit separately specifies 20,000 iterations as the reference setting; that conflict is recorded in decision log D1. Either way, Arm C is not a reproduction of the released checkpoint's training budget.

- `df` and `vasc` have very small test supports, so their per-class numbers are noisy.

- The released DM-AHR weights have undocumented training data and are used for exploration only.

- Augmentation differs from the proposal's wording: rotation is ±15° and uses a resized crop, where the proposal asks for ±90° rotation and explicit zoom/shift. The recipe is identical across arms, so the comparison stays internally valid, but it is not the stated recipe. See decision log D15.

- Class imbalance is handled by inverse-frequency loss weights only. The proposal also mentions class-balanced sampling; combining both is not recommended, and the dev kit makes weighted loss primary. This is recorded in decision log D16 rather than left implicit.

- Arm C was sampled at 1,000 timesteps rather than 2,000 for budget reasons (decision log D12). Every per-image row carries its `num_timesteps` value so the deviation is auditable.

## Licence and provenance

HAM10000 is distributed under CC BY-NC 4.0. The DM-AHR repository contradicts itself: its README states CC BY-NC 4.0 while its LICENSE file is Apache-2.0. **Commercial use is unresolved** and needs a rights decision by the client. See `reports/license_audit.md`.

## Reproducibility

Arm A, seed 42 was re-run from scratch through the same entrypoint (gate G13). Recorded validation macro-F1 0.778336 vs reproduced 0.778336, absolute difference 0.000000 against a 0.02 tolerance. Separate runs on two different Modal accounts also reproduced the frozen split hash byte for byte.

Commands are in `README.md`. Frozen artifacts and their hashes are in `artifacts/logs/` and `data/manifests/`. Every table above is generated from a CSV or JSON artifact listed in `artifact_inventory.csv`.
